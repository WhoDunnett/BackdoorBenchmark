'''
Reconstructive Neuron Pruning for Backdoor Defense
This file is modified based on the following source:
link : https://github.com/bboylyg/RNP

@article{li2023reconstructive,
    title={Reconstructive Neuron Pruning for Backdoor Defense},
    author={Li, Yige and Lyu, Xixiang and Ma, Xingjun and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Jiang, Yu-Gang},
    journal={arXiv preprint arXiv:2305.14876},
    year={2023}}

basic structure:
1. config args, save_path, fix random seed
2. load the backdoor attack data and backdoor test data
3. load the backdoor model
4. rnp defense:
    a. unlearn the backdoor model and save the unlearned model
    b. recover the unlearned model and record the mask value
    c. prune the backdoor model by the mask value
5. test the result and get ASR, ACC, RC

'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import math
import shutil

os.chdir(sys.path[0])
sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from copy import deepcopy
import torch.nn.utils.prune as prune
import pandas as pd
import copy
from torch.utils.data import DataLoader
import pickle

from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import ModelTrainerCLS_v2, BackdoorModelTrainer, Metric_Aggregator, given_dataloader_test, general_plot_for_epoch, given_dataloader_test_v2
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform, spc_choose_poisoned_sample
from tqdm import tqdm

from utils.defense_utils.rnp.preact import PreActResNet18
from utils.defense_utils.rnp.vgg import vgg19_bn
from utils.defense_utils.rnp.mobilenet import mobilenet_v3_large
from utils.defense_utils.rnp.efficientnet import efficientnet_b3

from utils.defense_utils.rnp.mask_batchnorm import MaskBatchNorm2d

from collections import OrderedDict

def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

def get_rnp_network(
    model_name: str,
    num_classes: int = 10,
    norm_layer = nn.BatchNorm2d,
    **kwargs,
):
    
    if model_name == 'preactresnet18':
        net = PreActResNet18(num_classes = num_classes, norm_layer = norm_layer, **kwargs)
    elif model_name == 'vgg19_bn':
        net = vgg19_bn(num_classes = num_classes, norm_layer = norm_layer,  **kwargs)
    elif model_name == 'mobilenet_v3_large':
        net = mobilenet_v3_large(num_classes= num_classes, norm_layer = norm_layer, **kwargs)
    elif model_name == 'efficientnet_b3':
        net = efficientnet_b3(num_classes= num_classes, norm_layer = norm_layer, **kwargs)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net

def train_step_unlearning(args, model, criterion, optimizer, data_loader):
    
    try:
        model.train()
        total_correct = 0
        total_loss = 0.0
        for i, batch in enumerate(data_loader):
            
            images, labels = batch[0].to(args.device), batch[1].to(args.device)
            optimizer.zero_grad()
       
            output = model(images)
            loss = criterion(output, labels)

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            (-loss).backward()
            optimizer.step()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc

    except Exception as e:
        # Propagate the exception so that the learning rate can be reduced
        raise e

def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)

def train_step_recovering(args, unlearned_model, criterion, mask_opt, data_loader):
    
    try:
        unlearned_model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, batch in enumerate(data_loader):
            images, labels = batch[0].to(args.device), batch[1].to(args.device)
            nb_samples += images.size(0)

            mask_opt.zero_grad()
            output = unlearned_model(images)
            loss = criterion(output, labels)
            loss = loss

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            mask_opt.step()
            clip_mask(unlearned_model)

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc

    except Exception as e:
        # Propagate the exception so that the learning rate can be reduced
        raise e

def get_train_acc(model, data_loader):
    
    model.eval()
    total_correct = 0
    nb_samples = 0
    for i, batch in enumerate(data_loader):
        images, labels = batch[0].to(args.device), batch[1].to(args.device)
        nb_samples += images.size(0)
        output = model(images)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
    
    acc = float(total_correct) / nb_samples
    return acc


class RNP(defense):

    def __init__(self):
        pass

    def set_args(self, parser):
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument("--dataset_path", type=str)

        parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny')
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--input_height", type=int)
        parser.add_argument("--input_width", type=int)
        parser.add_argument("--input_channel", type=int)

        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)

        parser.add_argument('--attack', type=str)
        parser.add_argument('--poison_rate', type=float)
        parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel')
        parser.add_argument('--target_label', type=int)
        parser.add_argument('--trigger_type', type=str,
                            help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--result_base', type=str, help='the location of result base path', default = "../record")

        parser.add_argument('--yaml_path', type=str, help='the path of yaml')

        # set the parameter for the mmdf defense
        parser.add_argument('--spc', type=int, help='the samples per class used for training')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        
        # RNP Arguments
        parser.add_argument('--weight_decay', type=float, help='the weight decay for optimizer')
        parser.add_argument('--momentum', type=float, help='the momentum for optimizer')
        
        parser.add_argument('--unlearn_epochs', type=int, help='the number of epochs for unlearning')
        parser.add_argument('--unlearn_lr', type=float, help='the learning rate for unlearning')
        parser.add_argument('--unlearn_threshold', type=float, help='the threshold for unlearning')
        
        parser.add_argument('--recovery_lr', type=float, help='the learning rate for recovery')
        parser.add_argument('--recovery_epochs', type=int, help='the number of epochs for recovery')
        parser.add_argument('--recovery_drop', type=float, help='the threshold for recovery')
        
        return parser

    def add_yaml_to_args(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults

    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        if args.spc is None and args.ratio is None:
            raise Exception("Either spc or ratio must be specified")
        
        # #######################################
        # Modified to be compatible with the new result_base and SPC
        # #######################################
        if args.spc is not None:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "rnp" + os.path.sep + f'spc_{args.spc}' + os.path.sep + str(args.random_seed)
        else:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "rnp" + os.path.sep + f'ratio_{args.ratio}' + os.path.sep + str(args.random_seed)
        
        os.makedirs(defense_save_path, exist_ok = True)
        args.defense_save_path = defense_save_path
        return args

    def prepare(self, args):

        ### set the logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        # file Handler
        fileHandler = logging.FileHandler(
            args.defense_save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log will be recorded.
        logger.setLevel(0)
        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        logging.info(pformat(args.__dict__))

        logging.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        # record the git infomation for debug (if available.)
        try:
            logging.debug(pformat(get_git_info()))
        except:
            logging.debug('Getting git info fails.')

        fix_random(args.random_seed)
        self.args = args

        '''
                load_dict = {
                        'model_name': load_file['model_name'],
                        'model': load_file['model'],
                        'clean_train': clean_train_dataset_with_transform,
                        'clean_test' : clean_test_dataset_with_transform,
                        'bd_train': bd_train_dataset_with_transform,
                        'bd_test': bd_test_dataset_with_transform,
                    }
                '''
        self.attack_result = load_attack_result(args.result_base + os.path.sep + self.args.result_file + os.path.sep +'attack_result.pt')
        
    def load_model_replica(self, weights=None, mask_bn=False):
        
        if mask_bn:
            model = get_rnp_network(self.args.model, self.args.num_classes, norm_layer = MaskBatchNorm2d)
        else:
            model = get_rnp_network(self.args.model, self.args.num_classes)
            
        if weights is not None:
            load_state_dict(model, weights)
            
        model.to(self.args.device)
        model.eval()
        model.requires_grad_(False)
        
        return model

    def defense(self):

        args = self.args
        attack_result = self.attack_result

        # ------------------------------- Code from FP -------------------------------
        # Setup dataloaders
        
        clean_train_dataset = attack_result['clean_train']
        clean_train_wrapper = clean_train_dataset.wrapped_dataset
        clean_train_wrapper = prepro_cls_DatasetBD_v2(clean_train_wrapper)
        
        # #######################################
        # Modified to be compatible with SPC
        # Note: Some methods require validation and therefore SPC cannot be 1
        # #######################################
        if args.spc is not None:
            spc_use = args.spc
            if args.spc < 1: 
                raise Exception("SPC must be greater than 1")
            if args.spc == 1: spc_use = 2
            train_idx, _ = spc_choose_poisoned_sample(clean_train_wrapper, spc_use, val_ratio=0)
        else:
            ran_idx = choose_index(args, len(clean_train_wrapper))
            train_idx = np.random.choice(len(ran_idx), int(len(ran_idx) * (1-args.val_ratio)), replace=False)
        
        clean_train_wrapper.subset(train_idx)
    
        clean_train_dataset.wrapped_dataset = clean_train_wrapper

        logging.info(f'Len of train dataset: {len(clean_train_dataset)}')
            
        train_loader = DataLoader(clean_train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
        
        clean_test_dataset_with_transform = attack_result['clean_test']
        data_clean_testset = clean_test_dataset_with_transform


        bd_test_dataset_with_transform = attack_result['bd_test']
        data_bd_testset = bd_test_dataset_with_transform
           
        # ------------------------------- Training Loop -------------------------------
        # https://github.com/bboylyg/RNP
        # NOTE: We added RNP to our own codebase before it was integrated into the official codebase
        
        # Unlearning step
        max_tries = 10
        for i in range(max_tries):
            
            # Try and run the unlearning step
            # NOTE: The original codebase has a bug where the learning rate is to large
            # We therefore reduce the learning rate by a factor of 10 if an exception is thrown
            try:
                unlearn_model = self.load_model_replica(weights=self.attack_result['model'], mask_bn=True)
                unlearn_model.train()
                unlearn_model.requires_grad_(True)
                
                criterion = nn.CrossEntropyLoss()
                
                args.weight_decay = float(args.weight_decay)
                
                optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=args.unlearn_lr, momentum=args.momentum, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.unlearn_epochs)
                
                unlearn_acc, unlearn_asr, unlearn_ra = [], [], []
                for epoch in range(args.unlearn_epochs):
                    
                    train_loss, train_acc = train_step_unlearning(args, unlearn_model, criterion, optimizer, train_loader)
                    
                    #acc, asr, ra = given_dataloader_test_v2(unlearn_model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), args)
                    #logging.info(f'UNLEARN | Epoch {epoch} train_loss:{train_loss}  train_acc:{train_acc}  test_acc:{acc}  test_asr:{asr}  test_ra:{ra}')
                    #unlearn_acc.append(acc), unlearn_asr.append(asr), unlearn_ra.append(ra)
                    
                    logging.info(f'UNLEARN | Epoch {epoch} train_loss:{train_loss}  train_acc:{train_acc}')
                
                    if train_acc <= args.unlearn_threshold:
                        break
                
                    scheduler.step()   
                  
                # Save the unlearning acc, asr, ra to a csv file
                unlearn_result = {
                    "unlearn_acc": unlearn_acc,
                    "unlearn_asr": unlearn_asr,
                    "unlearn_ra": unlearn_ra,
                }
                
                unlearn_result_df = pd.DataFrame(unlearn_result, columns=["unlearn_acc", "unlearn_asr", "unlearn_ra"])
                unlearn_result_df.to_csv(os.path.join(args.defense_save_path, "unlearn_result.csv"))
                
                # -------------------------------------
                # Recovering step  
                recover_model = self.load_model_replica(weights=unlearn_model.state_dict(), mask_bn=True)
                mask_params = [param for name, param in recover_model.named_parameters() if 'neuron_mask' in name]
                
                # Make the mask parameters require grad
                for param in mask_params:
                    param.requires_grad = True
                
                mask_opt = torch.optim.SGD(mask_params, lr=args.recovery_lr, momentum=args.momentum)
                
                recover_acc, recover_asr, recover_ra = [], [], []
                for epoch in range(args.recovery_epochs):
                    
                    train_loss, train_acc = train_step_recovering(args, recover_model, criterion, mask_opt, train_loader)
                    #acc, asr, ra = given_dataloader_test_v2(recover_model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), args)
                    #logging.info(f'RECOVER | Epoch {epoch} train_loss:{train_loss}  train_acc:{train_acc}  test_acc:{acc}  test_asr:{asr}  test_ra:{ra}')
                    #recover_acc.append(acc), recover_asr.append(asr), recover_ra.append(ra)
                    
                    logging.info(f'RECOVER | Epoch {epoch} train_loss:{train_loss}  train_acc:{train_acc}')
                    
                # Save the recovering acc, asr, ra to a csv file
                recover_result = {
                    "recover_acc": recover_acc,
                    "recover_asr": recover_asr,
                    "recover_ra": recover_ra,
                }
                
                recover_result_df = pd.DataFrame(recover_result, columns=["recover_acc", "recover_asr", "recover_ra"])
                recover_result_df.to_csv(os.path.join(args.defense_save_path, "recover_result.csv"))
                
                save_mask_scores(recover_model.state_dict(), os.path.join(args.defense_save_path, 'mask_scores.txt'))
                
                break
            
            # Catch the exception and try again with a lower learning rate
            except Exception as e:               
                logging.error(f'Error: {e}')
                args.unlearn_lr *= 0.1
                logging.info(f'Unlearning learning rate reduced to {args.unlearn_lr}')
                continue
        
        # -------------------------------------
        # Model Pruning
        final_model = self.load_model_replica(weights=self.attack_result['model'], mask_bn=False)
        
        mask_values = read_data(os.path.join(args.defense_save_path, 'mask_scores.txt'))
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        
        # Get the initial accuracy of the train data
        initial_train_acc = get_train_acc(final_model, train_loader)
        best_model = copy.deepcopy(final_model)
        best_model.load_state_dict(final_model.state_dict())
        
        logging.info(f'Initial train_acc:{initial_train_acc}')
        
        threshold_acc, threshold_asr, threshold_ra = [], [], []
        
        start_threshold = 0.0
        threshold_step_size = 0.02
        accuracy_drop = args.recovery_drop
        
        while start_threshold < 1.0:
            
            for neuron in mask_values:
                
                if float(neuron[2]) < start_threshold:
                    pruning(final_model, neuron)
            
            train_acc = get_train_acc(final_model, train_loader)
            
            #acc, asr, ra = given_dataloader_test_v2(final_model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), args)
            #logging.info(f'PRUNING | Threshold {start_threshold}  train_acc:{initial_train_acc}  test_acc:{acc}  test_asr:{asr}  test_ra:{ra}')
            #threshold_acc.append(acc), threshold_asr.append(asr), threshold_ra.append(ra)
            
            logging.info(f'PRUNING | Threshold {start_threshold}  train_acc:{train_acc}')
            
            
            if train_acc < initial_train_acc - accuracy_drop:
                break
            else:
                best_model.load_state_dict(final_model.state_dict())
                start_threshold += threshold_step_size
                
        # Save the threshold acc, asr, ra to a csv file
        threshold_result = {
            "threshold_acc": threshold_acc,
            "threshold_asr": threshold_asr,
            "threshold_ra": threshold_ra,
        }
        
        threshold_result_df = pd.DataFrame(threshold_result, columns=["threshold_acc", "threshold_asr", "threshold_ra"])
        threshold_result_df.to_csv(os.path.join(args.defense_save_path, "threshold_result.csv"))
        
        # Save the mask values
        save_mask_scores(final_model.state_dict(), os.path.join(args.defense_save_path, 'final_mask_scores.txt'))
        
        # ------------------------------- Final Test -------------------------------
        test_acc, test_asr, test_ra = given_dataloader_test_v2(best_model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), self.args)
        logging.info(f'Final test_acc:{test_acc}  test_asr:{test_asr}  test_ra:{test_ra}')
        
        # save the result to a csv file in the defense_save_path
        final_result = {
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        }

        final_result_df = pd.DataFrame(final_result, columns=["test_acc", "test_asr", "test_ra"], index=[0])
        final_result_df.to_csv(os.path.join(self.args.defense_save_path, "final_result.csv"))

        save_defense_result(
            model_name = args.model,
            num_classes = args.num_classes,
            model = best_model.cpu().state_dict(),
            save_path = self.args.defense_save_path,
        )

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    rnp = RNP()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = rnp.set_args(parser)
    args = parser.parse_args()
    rnp.add_yaml_to_args(args)
    args = rnp.process_args(args)
    rnp.prepare(args)
    rnp.defense()
