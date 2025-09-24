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

import torch.nn.functional as F

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
from torch.autograd import Variable
from collections import OrderedDict


# ---------- MODIFIED
# This was modified to allow models without BatchNorm2d layers after Conv2d layers.
# ----------
def initialize(model, rank_list):
    # Get an ordered list of named modules from the model.
    modules = list(model.named_modules())
    conv_idx = 0  # To index into rank_list

    # Iterate over modules in the order returned
    for i, (name, module) in enumerate(modules):
        if isinstance(module, nn.Conv2d):
            # Get the rank list for this Conv2d layer
            helper = rank_list[conv_idx]
            
            # Prune the Conv2d layer by setting the weights (and bias if available) to zero 
            for fmidx in helper:
                module.weight.data[fmidx] = torch.zeros_like(module.weight.data[fmidx])
                if module.bias is not None:
                    module.bias.data[fmidx] = torch.zeros_like(module.bias.data[fmidx])
            
            # Check if the next module is a BatchNorm2d layer
            if i + 1 < len(modules):
                next_name, next_module = modules[i + 1]
                if isinstance(next_module, nn.BatchNorm2d):
                    # Prune the BatchNorm2d layer weights (and bias if available)
                    for fmidx in helper:
                        next_module.weight.data[fmidx] = torch.zeros_like(next_module.weight.data[fmidx])
                        if next_module.bias is not None:
                            next_module.bias.data[fmidx] = torch.zeros_like(next_module.bias.data[fmidx])
            
            conv_idx += 1

    return model

def ranking(model,adv_list,label, device):
    rank_list = []
    for data in adv_list:
        result_list = []
        for image in data:
            pred = model(image.to(device))
            pred = np.argmax(pred.cpu().detach(), axis=-1)
            correct = pred == label.cpu().detach()
            correct = np.sum(correct.numpy(), axis=-1)
            result_list.append(correct/image.shape[0])

        rank_list.append(result_list)
    return rank_list

def dynamiccluster(arrays):
    score_list = list()
    arrays = np.array(arrays)
    arrays = np.squeeze(arrays)
    return np.argsort(arrays)[:len(arrays)//(2**6)]

# ------- MODIFED
# This code is modified to use only the first 32 images from the dataset.
# This is hardcoded in the original code.
# -------
def obtain_adv_dataset(args, model, train_loader):

    for idx, (data, label, *_) in enumerate(train_loader):
        x = data.to(args.device)
        y = label.to(args.device)

        x = x[0:args.adv_dataset_size]
        y = y[0:args.adv_dataset_size]

        # Originally the code had both eps and alpha. However, eps was not in the config. We have used eps only.
        adv_images = x + args.eps*torch.empty_like(x).uniform_(-args.eps, args.eps).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach().to(args.device)
        break

    conv_outputs = []
    def get_conv_output_hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            conv_outputs.append(output)

    # Register the hook on all convolutional layers
    def model_hook(model):
        handles = []
        for module in model.modules():
            handle = module.register_forward_hook(get_conv_output_hook)
            handles.append(handle)
        return handles

    def remove_hook(handles):
        for handle in handles:
            handle.remove

    handles = model_hook(model)
    x = x.to(args.device)
    output = model(x)
    num_conv_layer = len(conv_outputs)
    total_feature = 0
    inter_node = [0]
    for layer in conv_outputs:
        total_feature+=layer.shape[1]
        inter_node.append(total_feature)
    for handle in handles:
        handle.remove()
    
    def adv_sample_generation(idx,x, adv_images):
        image_list = []
        for each_adv_idx in range(1):
            conv_outputs = []
            def get_conv_output_hook(module, input, output):
                if isinstance(module, nn.Conv2d):
                    conv_outputs.append(output)

            # Register the hook on all convolutional layers
            def model_hook(model):
                handles = []
                for module in model.modules():
                    handle = module.register_forward_hook(get_conv_output_hook)
                    handles.append(handle)
                return handles

            def remove_hook(handles):
                for handle in handles:
                    handle.remove
            handles = model_hook(model)
            x = x.to(args.device)
            adv_images = adv_images.requires_grad_(True).to(args.device)
            output = model(x)
            adv_output = model(adv_images)
            clean_feature_maps = []
            adv_feature_maps = []
            start = 0
            for i, conv_output in enumerate(conv_outputs):
                if i<num_conv_layer:
                    if start<=idx and start+conv_output.shape[1]>idx:
                        clean_feature_map = conv_output[:,idx-start,:,:]
                    start = start+conv_output.shape[1]
                    continue

                if i==num_conv_layer:
                    start = 0
                    if start<=idx and start+conv_output.shape[1]>idx:
                        adv_feature_map = conv_output[:,idx-start,:,:]
                    start = start+conv_output.shape[1]
                    continue
                if i>num_conv_layer:
                    if start<=idx and start+conv_output.shape[1]>idx:
                        adv_feature_map = conv_output[:,idx-start,:,:]
                    start = start+conv_output.shape[1]

            loss = F.mse_loss(clean_feature_map,adv_feature_map)
            grad = torch.autograd.grad(loss, adv_images,
                                        retain_graph=True, create_graph=False)[0]

            adv_images = adv_images.detach() + args.eps*grad.sign()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            for handle in handles:
                handle.remove()
            image_list.append(adv_images.cpu())
        return image_list

    adv_list = []

    rank_list = []
    rank_acc_list = []
    layer_idx = 1
    for i in tqdm(range(total_feature)):
        if i<inter_node[layer_idx]:
            adv_list.append(adv_sample_generation(i, x, adv_images))
        else:
            layer_idx+=1
            rank_acc = ranking(model,adv_list,y, args.device)
            rank_list.append(dynamiccluster(rank_acc))
            adv_list = []
            adv_list.append(adv_sample_generation(i, x, adv_images))
            rank_acc_list.append(rank_acc)
    rank_acc_list.append(ranking(model,adv_list,y, args.device))
    rank_list.append(dynamiccluster(ranking(model,adv_list,y, args.device)))


    return rank_list, rank_acc_list

def train(args, model,train_loader,test_loader,rank_list):

    # Set the model to training mode
    model.train()
    for param in model.parameters():
        param.requires_grad = True
            
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.eta_min)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        for i, (inputs,labels, *_) in enumerate(train_loader):
            model.train()
            model.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


class FMP(defense):

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

        parser.add_argument('--yaml_path', type=str, default="./config/defense/nft/config.yaml", help='the path of yaml')

        # set the parameter for the mmdf defense
        parser.add_argument('--spc', type=int, help='the samples per class used for training')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        
        # nft Arguments
        parser.add_argument('--momentum', type=float, help='the momentum for optimizer')
        parser.add_argument('--lr', type=float, help='the learning rate for optimizer')
        parser.add_argument('--epochs', type=int, help='the number of epochs for training')
        parser.add_argument('--eta_min', type=float, help='the eta min for scheduler')

        parser.add_argument('--eps', type=float, help='the eps for adv sample generation')
        parser.add_argument('--adv_dataset_size', type=int, help='the number of adv samples for training')
        
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

        # #######################################
        # Modified to be compatible with the new result_base and SPC
        # #######################################
        if args.spc is not None:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "fmp" + os.path.sep + f'spc_{args.spc}' + os.path.sep + str(args.random_seed)
        else:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "fmp" + os.path.sep + f'ratio_{args.ratio}' + os.path.sep + str(args.random_seed)
        
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
        self.result = load_attack_result(args.result_base + os.path.sep + self.args.result_file + os.path.sep +'attack_result.pt')

    def defense(self):

        args = self.args
        result = self.result

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)

        if args.spc is not None:
            if args.spc == 1: args.spc = 2
            ran_idx, _ = spc_choose_poisoned_sample(clean_dataset, args.spc, 0)

        else:
            data_all_length = len(clean_dataset)
            ran_idx = choose_index(args, data_all_length)

        clean_dataset.subset(ran_idx)
        logging.info(f'Using {len(clean_dataset)} samples for training')
        
        data_set_without_tran = clean_dataset

        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran

        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        trainloader = data_loader
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        ##############################################       
        
        model = generate_cls_model(args.model, args.num_classes)
        model = model.to(args.device)
        model.load_state_dict(self.result['model'])

        logging.info(f'Obtaining the adv dataset')
        rank_list, rank_acc_list = obtain_adv_dataset(args, model, trainloader)

        logging.info(f'Apply pruning')
        model = initialize(model, rank_list)
        model = model.to(args.device)

        # Test the model after pruning
        test_acc, test_asr, test_ra = given_dataloader_test_v2(model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), self.args)
        logging.info(f'Pruning test_acc:{test_acc}  test_asr:{test_asr}  test_ra:{test_ra}')

        # Save the initial pruning result to a csv file in the defense_save_path
        initial_result = {
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        }

        initial_result_df = pd.DataFrame(initial_result, columns=["test_acc", "test_asr", "test_ra"], index=[0])
        initial_result_df.to_csv(os.path.join(self.args.defense_save_path, "initial_result.csv"))

        logging.info(f'Training the model')
        model = train(args, model, trainloader, data_clean_loader, rank_list)               
        
        ##############################################
        # Get the final results for
        test_acc, test_asr, test_ra = given_dataloader_test_v2(model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), self.args)
        logging.info(f'Final test_acc:{test_acc}  test_asr:{test_asr}  test_ra:{test_ra}')
        
        # save the result to a csv file in the defense_save_path
        final_result = {
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        }

        final_result_df = pd.DataFrame(final_result, columns=["test_acc", "test_asr", "test_ra"], index=[0])
        final_result_df.to_csv(os.path.join(self.args.defense_save_path, "final_result.csv"))
        ##############################################

        save_defense_result(
            model_name = args.model,
            num_classes = args.num_classes,
            model = model.cpu().state_dict(),
            save_path = self.args.defense_save_path,
        )

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    fmp = FMP()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = fmp.set_args(parser)
    args = parser.parse_args()
    fmp.add_yaml_to_args(args)
    args = fmp.process_args(args)
    fmp.prepare(args)
    fmp.defense()
