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
from torch.autograd import Variable
from collections import OrderedDict

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# Mask model helper

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

def get_nft_network(
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

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# Training methods from https://github.com/nazmul-karim170/NFT/blob/main/src/Remove_Backdoor.py#L252

## Mask Regularization
def Regularization(model):
    L1=0
    L2=0
    L_inf = 0
    for name, param in model.named_parameters():
        if 'neuron_mask' in name:
            L1 += torch.sum(torch.abs(1-param))
            L2 += torch.norm(param, 2)
            L_inf += torch.max(torch.abs(1-param))
    # for name, module in model.named_parameters():
    return L1, L2, L_inf

## Clip the mask within [mu, 1]
def mask_clip(args, model, upper=1):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    count_layer = 1
    with torch.no_grad():
        for param in params:
            param.clamp_(args.alpha*math.exp(-args.beta*count_layer), upper)
            count_layer += 1

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def NFT_Train(args, N_c, model, criterion, mask_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss    = 0.0
    nb_samples    = 0

    ## Train the model for 1 epoch
    for i, (images, labels, _, _, _) in enumerate(data_loader):
        nb_samples += images.size(0)
        inputs, targets = images.cuda(), labels.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       alpha=1, use_cuda=True)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        mask_opt.zero_grad()
        L1, L2, L_inf = Regularization(model)
        tot_loss     = loss + args.l1_weight * L1/N_c
        
        tot_loss.backward()
        mask_opt.step()
        mask_clip(args,model)

        ## Claculate the train accuracy 
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                          + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        nb_samples += inputs.size(0)

        total_loss += tot_loss.item()


    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

class NFT(defense):

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
        parser.add_argument('--weight_decay', type=float, help='the weight decay for optimizer')
        parser.add_argument('--momentum', type=float, help='the momentum for optimizer')
        parser.add_argument('--lr', type=float, help='the learning rate for optimizer')
        parser.add_argument('--lr_min', type=float, help='the minimum learning rate for optimizer')
        parser.add_argument('--epochs', type=int, help='the number of epochs for training')

        parser.add_argument('--l1_weight', type=float, help='the weight for the l1 regularization')
        parser.add_argument('--alpha', type=float, help='the alpha for the mask clipping')
        parser.add_argument('--beta', type=float, help='the beta for the mask clipping')
        
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
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "nft" + os.path.sep + f'spc_{args.spc}' + os.path.sep + str(args.random_seed)
        else:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "nft" + os.path.sep + f'ratio_{args.ratio}' + os.path.sep + str(args.random_seed)
        
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
        
    def load_model_replica(self, weights=None, mask_bn=False):
        
        if mask_bn:
            model = get_nft_network(self.args.model, self.args.num_classes, norm_layer = MaskBatchNorm2d)
        else:
            model = get_nft_network(self.args.model, self.args.num_classes)
            
        if weights is not None:
            load_state_dict(model, weights)
            
        model.to(self.args.device)
        model.eval()
        model.requires_grad_(False)
        
        return model

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
        
        mask_model = self.load_model_replica(weights=self.result['model'], mask_bn=True)
        mask_params = []

        for name, param in mask_model.named_parameters():
            if 'neuron_mask' in name:
                param.requires_grad = True
                mask_params.append(param)
            else:
                param.requires_grad = False

        print(f"Mask parameters: {len(mask_params)}")

        criterion = torch.nn.CrossEntropyLoss().cuda()
        mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=args.momentum)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mask_optimizer, args.epochs, eta_min=args.lr_min)

        if args.spc is None:
            spc = int(len(data_set_o) / args.num_classes)
            self.args.spc = spc

        for i in range(args.epochs):
            train_loss, train_acc = NFT_Train(args, args.spc, model=mask_model, criterion=criterion, data_loader=trainloader, mask_opt=mask_optimizer)
            logging.info(f'Iteration {i} / {args.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}')

            scheduler.step()
                
        # Make the mask parameters require grad
        for param in mask_params:
            param.requires_grad = True

        # Save the mask values
        save_mask_scores(mask_model.state_dict(), os.path.join(args.defense_save_path, 'final_mask_scores.txt'))
        
        ##############################################
        # Get the final results for
        test_acc, test_asr, test_ra = given_dataloader_test_v2(mask_model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), self.args)
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
            model = mask_model.cpu().state_dict(),
            save_path = self.args.defense_save_path,
        )

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    nft = NFT()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = nft.set_args(parser)
    args = parser.parse_args()
    nft.add_yaml_to_args(args)
    args = nft.process_args(args)
    nft.prepare(args)
    nft.defense()
