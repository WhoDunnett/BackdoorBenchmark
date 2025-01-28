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

class FST(defense):

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
        
        # FST Arguments
        parser.add_argument('--training_epochs', type=int, help='number of training epochs', default=15)
        parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
        parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
        parser.add_argument('--alpha', type=float, help='alpha', default=0.2)
        
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
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "fst" + os.path.sep + f'spc_{args.spc}' + os.path.sep + str(args.random_seed)
        else:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "fst" + os.path.sep + f'ratio_{args.ratio}' + os.path.sep + str(args.random_seed)
        
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

        model = generate_cls_model(args.model, args.num_classes)
        model.load_state_dict(self.attack_result['model'])
        model.to(args.device)
        model.eval()

        self.model = model

    def defense(self):

        model = self.model

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
        # https://github.com/AISafety-HKUST/Backdoor_Safety_Tuning/blob/main/fine_tune/ft.py
        training_acc, training_asr, trianing_ra = [], [], []
        
        # Save a reference so that the weights can be accessed later
        original_weights = None
        new_layer = None
        
        # Ensure each parameter requires a gradient
        for name, param in model.named_parameters():
            param.requires_grad = True   
        
        if args.model == 'preactresnet18':
            original_weights = [model.linear.weight.data.clone()]
            model.linear = nn.Linear(model.linear.in_features, args.num_classes).to(args.device)
            new_layer = [model.linear]
        elif args.model == 'vgg19_bn':
            original_weights = [model.classifier[0].weight.data.clone(), model.classifier[3].weight.data.clone(), model.classifier[6].weight.data.clone()]
            model.classifier[0] = nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features).to(args.device)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, model.classifier[3].out_features).to(args.device)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.num_classes).to(args.device)
            new_layer = [model.classifier[0], model.classifier[3], model.classifier[6]]
        elif args.model == "mobilenet_v3_large":
            original_weights = [model.classifier[0].weight.data.clone(), model.classifier[3].weight.data.clone()]
            model.classifier[0] = nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features).to(args.device)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, args.num_classes).to(args.device)
            new_layer = [model.classifier[0], model.classifier[3]]
        elif args.model == "efficientnet_b3":
            original_weights = [model.classifier[1].weight.data.clone()]
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, args.num_classes).to(args.device)
            new_layer = [model.classifier[1]]
        else:
            raise Exception("Model not supported")
        
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.training_epochs)
        critierion = nn.CrossEntropyLoss()
        
        acc, asr, ra = given_dataloader_test_v2(model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), args)
        logging.info(f'Initial acc:{acc}  asr:{asr}  ra:{ra}')
        
        for epoch in range(args.training_epochs):
            
            epoch_loss = 0
            
            model.train()
            for batch in tqdm(train_loader):
                inputs, labels = batch[0].to(args.device), batch[1].to(args.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Loss 1 is the inner product of the original weights and the new weights
                inner_product = torch.tensor(0.0).to(args.device)
                for i in range(len(original_weights)):
                    inner_product += torch.sum(original_weights[i] * new_layer[i].weight)
                
                # Loss 2 is the cross entropy loss
                ce_loss = critierion(outputs, labels)
                
                # Total loss and backpropagation
                loss = (args.alpha * inner_product) + ce_loss
                loss.backward()
                
                optimizer.step()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                
                # Normalize the weights
                for i in range(len(original_weights)):
                   new_layer[i].weight.data = (new_layer[i].weight.data / torch.norm(new_layer[i].weight.data, p=2)) * torch.norm(original_weights[i], p=2)
                
            scheduler.step()
            
            acc, asr, ra = given_dataloader_test_v2(model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), args)
            logging.info(f'Epoch {epoch}  loss:{epoch_loss}  acc:{acc}  asr:{asr}  ra:{ra}')
            training_acc.append(acc), training_asr.append(asr), trianing_ra.append(ra)
        
        # ------------------------------- Final Test -------------------------------
        test_acc, test_asr, test_ra = given_dataloader_test_v2(model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), self.args)
        logging.info(f'Final test_acc:{test_acc}  test_asr:{test_asr}  test_ra:{test_ra}')
        
        # Save the training results to a csv file
        training_result = {
            "training_acc": training_acc,
            "training_asr": training_asr,
            "training_ra": trianing_ra,
        }
        
        training_result_df = pd.DataFrame(training_result, columns=["training_acc", "training_asr", "training_ra"])
        training_result_df.to_csv(os.path.join(self.args.defense_save_path, "training_result.csv"))
        
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
            model = model.cpu().state_dict(),
            save_path = self.args.defense_save_path,
        )

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    fst = FST()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = fst.set_args(parser)
    args = parser.parse_args()
    fst.add_yaml_to_args(args)
    args = fst.process_args(args)
    fst.prepare(args)
    fst.defense()
