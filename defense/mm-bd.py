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

class Layer_Mask:
    
    def __init__(self, mask, is_max=True, apply_relu_first=False):
        self.mask = mask
        self.is_max = is_max
        self.apply_relu_first = apply_relu_first
        
        self.save_activation = False
        
    def hook(self, module, input, output):
        
        if self.save_activation:
            self.activation = output
            return output
        
        if self.is_max:
            
            # NOTE: This is a hacky way resolve issues with the Preact Resnet model
            # The model has a ReLU layer after it however this is defined in the forward function
            # This means that the ReLU layer is not part of the model.children() and so is not
            # This will have no impact on the activation as the ReLU applied after will not change the output
            if self.apply_relu_first:
                output = torch.relu(output)
            
            output = torch.clamp(output, max=self.mask)
        else:
            output = torch.clamp(output, min=self.mask)    
        
        return output

def get_relu_layers(modules):
    
    relu_layers = []
    out_channels = []
    for name, layer in modules:
        if isinstance(layer, nn.ReLU):
            relu_layers.append(layer)
            out_channels.append(previous_output_channels)
            
        if hasattr(layer, 'out_channels'):
            previous_output_channels = layer.out_channels
            
    return relu_layers, out_channels

def get_batch_bn_layers(modules):
        
    bn_layers = []
    output_channels = []
    for name, layer in modules:

        if isinstance(layer, nn.BatchNorm2d):
            bn_layers.append(layer)
            output_channels.append(layer.num_features)
                
    return bn_layers, output_channels

def get_silu_layers(modules):
        
    silu_layers = []
    for name, layer in modules:
            
        if isinstance(layer, nn.SiLU):
            silu_layers.append(layer)
            
    return silu_layers

def get_hardswish_layers(modules):
        
    hardswish_layers = []
    for name, layer in modules:
            
        if isinstance(layer, nn.Hardswish):
            hardswish_layers.append(layer)
                
    return hardswish_layers

def get_layer_min_max_activation(model, hook_object, train_loader, get_output_channels=False):
    
    largest_layer_activation = -torch.tensor(float('inf'))
    smalled_layer_activation = torch.tensor(float('inf'))
    
    hook_object.save_activation = True
    for idx, batch in enumerate(train_loader):
        img, label = batch[0].to(args.device), batch[1].to(args.device)
        model(img)
                
        activations = hook_object.activation
        
        max_act = torch.max(activations)
        min_act = torch.min(activations)
                
        if max_act > largest_layer_activation:
            largest_layer_activation = max_act
            
        if min_act < smalled_layer_activation:
            smalled_layer_activation = min_act
            
        if get_output_channels:
            output_channels = activations.shape[1]
            
        del img, label, activations, max_act, min_act
            
    hook_object.save_activation = False
    
    del hook_object.activation
    hook_object.activation = None
    
    if get_output_channels:
        return largest_layer_activation.item(), smalled_layer_activation.item(), output_channels
    return largest_layer_activation.item(), smalled_layer_activation.item()           


# --------------------------------------------------------------------------------
# Model Specific Setup

def setup_mask_mobilenet(model, args, train_loader):
    
    modules = model.named_modules()
    hardswish_layers = get_hardswish_layers(modules)
    
    # Remove the last hardswish layer as this is the final layer
    hardswish_layers = hardswish_layers[:-1]
    
    relu_layers, out_channels = get_relu_layers(modules)
    
    mask_dict = {}
    index = 0
    
    # Do the hardswish layers first
    for layer in hardswish_layers:
            
        placeholder = torch.ones((1, 1, 1)).to(args.device)
        hook_object = Layer_Mask(placeholder, is_max=True)
        
        hook_token = layer.register_forward_hook(hook_object.hook)
            
        largest_layer_activation, _, output_channels = get_layer_min_max_activation(model, hook_object, train_loader, get_output_channels=True)
        mask = torch.full((output_channels, 1, 1), largest_layer_activation).to(args.device)
        mask.requires_grad = True
        
        hook_object.mask = mask
                
        mask_dict[index] = (layer, hook_object, hook_token)
        index += 1
        
        # Remove the hook_token and delete the hook_object
        hook_token.remove()
        del hook_object
    
    # Do the ReLU layers next
    for layer, output_channels in zip(relu_layers, out_channels):
                
        mask = torch.ones((output_channels, 1, 1)).to(args.device)
        mask.requires_grad = True
    
        hook_object = Layer_Mask(mask, is_max=True)
        hook_token = layer.register_forward_hook(hook_object.hook)
                
        largest_layer_activation, _ = get_layer_min_max_activation(model, hook_object, train_loader)
        mask = torch.full((output_channels, 1, 1), largest_layer_activation).to(args.device)
        mask.requires_grad = True
                
        hook_object.mask = mask
                    
        mask_dict[index] = (layer, hook_object, hook_token)
        index += 1
        
        # Remove the hook_token and delete the hook_object
        hook_token.remove()
        del hook_object
        
    return mask_dict

def setup_mask_efficientnet(model, args, train_loader):
    
    modules = model.named_modules()
    silu_layers = get_silu_layers(modules)
    
    mask_dict = {}
    
    index = 0
    for layer in silu_layers:
            
        placeholder = torch.ones((1, 1, 1)).to(args.device)
        hook_object = Layer_Mask(placeholder, is_max=True)
        
        hook_token = layer.register_forward_hook(hook_object.hook)
            
        largest_layer_activation, _, output_channels = get_layer_min_max_activation(model, hook_object, train_loader, get_output_channels=True)
        mask = torch.full((output_channels, 1, 1), largest_layer_activation).to(args.device)
        mask.requires_grad = True
        
        hook_object.mask = mask
                
        mask_dict[index] = (layer, hook_object, hook_token)
        index += 1
        
        # Remove the hook_token and delete the hook_object
        hook_token.remove()
        del hook_object
        
    return mask_dict

def setup_mask_preactresnet18(model, args, train_loader):
    
    modules = model.named_modules()
    bn_layers, out_channels = get_batch_bn_layers(modules)
    
    mask_dict = {}
    
    index = 0
    for layer, output_channels in zip(bn_layers, out_channels):
            
        mask = torch.ones((output_channels, 1, 1)).to(args.device)
        mask.requires_grad = True

        hook_object = Layer_Mask(mask, is_max=True, apply_relu_first=True)
        hook_token = layer.register_forward_hook(hook_object.hook)
            
        largest_layer_activation, _ = get_layer_min_max_activation(model, hook_object, train_loader)
        mask = torch.full((output_channels, 1, 1), largest_layer_activation).to(args.device)
        mask.requires_grad = True
            
        hook_object.mask = mask
                
        mask_dict[index] = (layer, hook_object, hook_token)
        index += 1
        
        # Remove the hook_token and delete the hook_object
        hook_token.remove()
        del hook_object
        
    return mask_dict

def setup_mask_vgg19_bn(model, args, train_loader):
    
    modules = model.named_modules()
    relu_layers, out_channels = get_relu_layers(modules)
    
    # Remove last two layers as these are dense layers
    relu_layers = relu_layers[:-2]
    out_channels = out_channels[:-2]
    
    mask_dict = {}
    
    index = 0
    for layer, output_channels in zip(relu_layers, out_channels):
            
        mask = torch.ones((output_channels, 1, 1)).to(args.device)
        mask.requires_grad = True

        hook_object = Layer_Mask(mask, is_max=True)
        hook_token = layer.register_forward_hook(hook_object.hook)
            
        largest_layer_activation, _ = get_layer_min_max_activation(model, hook_object, train_loader)
        
        mask = torch.full((output_channels, 1, 1), largest_layer_activation).to(args.device)
        mask.requires_grad = True
            
        hook_object.mask = mask
                
        mask_dict[index] = (layer, hook_object, hook_token)
        index += 1
        
        # Remove the hook_token and delete the hook_object
        hook_token.remove()
        del hook_object
        
    return mask_dict    

class MMBD(defense):

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
        parser.add_argument('--lr', type=float)

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
        
        # Training step
        parser.add_argument('--training_lr', type=float, help='learning rate for training the mask', default = 1e-3)
        parser.add_argument('--training_epochs', type=int, help='number of epochs for training the mask', default = 50)
        parser.add_argument('--test_save_frequency', type=int, help='frequency to save the test results, 0 is never', default = 10)
        parser.add_argument('--alpha', type=float, help='alpha value to reduce lamba by', default = 1.2)
        parser.add_argument('--lambda_hyper', type=float, help='lambda value for mask regularisation', default = 0.5)
        parser.add_argument('--lambda_reduction_num', type=int, help='number of epochs to reduce lambda by', default = 10)
        parser.add_argument('--accuracy_reduction', type=float, help='accuracy reduction threshold', default = 0.05)
        
        
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
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "mm-bd" + os.path.sep + f'spc_{args.spc}' + os.path.sep + str(args.random_seed)
        else:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "mm-bd" + os.path.sep + f'ratio_{args.ratio}' + os.path.sep + str(args.random_seed)
        
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
        model.requires_grad_(False)

        self.model = model
    
    def get_correct_index(self, model, data_wrapper):
    
        correct_idx = []

        data_loader = torch.utils.data.DataLoader(data_wrapper, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            for idx, (batch) in enumerate(data_loader):
                img, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
                output = model(img)
                pred = torch.argmax(output, dim = 1)
                
                # Get the correct index
                correct_idx.extend(np.where((pred == label).cpu().numpy())[0])
                
                del img, label, output, pred
                
        return correct_idx

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
        # https://github.com/wanghangpsu/MM-BD
        
        original_model_copy = copy.deepcopy(model)
        original_model_copy.to(args.device)
        original_model_copy.load_state_dict(model.state_dict())
        original_model_copy.eval()
        
        # Setup model with mask layers
        if args.model == 'vgg19_bn':
            mask_dict = setup_mask_vgg19_bn(model, args, train_loader)
        elif args.model == 'preactresnet18':
            mask_dict = setup_mask_preactresnet18(model, args, train_loader)
        elif args.model == 'efficientnet_b3':
            mask_dict = setup_mask_efficientnet(model, args, train_loader)
        elif args.model == 'mobilenet_v3_large':
            mask_dict = setup_mask_mobilenet(model, args, train_loader)
        else:
            raise ValueError(f'Model {args.model} not supported')
        
        acc, asr, ra = given_dataloader_test_v2(model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), self.args)
        logging.info(f'Initial Acc: {acc} ASR: {asr} RA: {ra}')
        
        save_epoch, training_acc, training_asr, training_ra = [], [], [], []
        save_epoch.append(0), training_acc.append(acc), training_asr.append(asr), training_ra.append(ra)
        
        mask_optimizer = torch.optim.Adam([mask_object.mask for (layer, mask_object, hook) in mask_dict.values()], lr = self.args.training_lr)
        
        # Get the initial training accuracy
        correct = 0
        total = 0
        for idx, batch in enumerate(train_loader):
            model.eval()
            with torch.no_grad():
                img, label = batch[0].to(args.device), batch[1].to(args.device)
                output = model(img)
                pred = torch.argmax(output, dim = 1)
                correct += torch.sum(pred == label).item()
                total += len(label)
                
                del img, label, output, pred
                
        initial_accuracy = correct / total
        
        for epoch in range(args.training_epochs):
            
            correct = 0
            total = 0
            for idx, batch in enumerate(train_loader):
                
                mask_optimizer.zero_grad()
                img, label = batch[0].to(args.device), batch[1].to(args.device)

                original_model_copy.eval()
                model.eval()
                
                reference_output = original_model_copy(img)
                new_output = model(img)
                
                mse_loss = torch.nn.functional.mse_loss(new_output, reference_output)
                
                # Get the L2 norm of the mask
                mask_loss = 0
                for (layer, mask_object, hook) in mask_dict.values():
                    mask_loss += torch.norm(mask_object.mask, p = 2)
                
                # Total Loss
                loss = mse_loss + (self.args.lambda_hyper * mask_loss)
                
                loss.backward()
                mask_optimizer.step()
                
                # Check the accuracy
                pred = torch.argmax(new_output, dim = 1)
                correct += torch.sum(pred == label).item()
                total += len(label)
                
                del img, label, reference_output, new_output, mse_loss, mask_loss, loss, pred
                
            if self.args.test_save_frequency != 0 and epoch % self.args.test_save_frequency == 0:
                acc, asr, ra = given_dataloader_test_v2(model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), self.args)
                save_epoch.append(epoch), training_acc.append(acc), training_asr.append(asr), training_ra.append(ra)
                logging.info(f'Epoch: {epoch} Acc: {acc} ASR: {asr} RA: {ra}')
                
            if epoch % self.args.lambda_reduction_num == 0 and epoch != 0:
                if (correct / total) > initial_accuracy - self.args.accuracy_reduction:
                    args.lambda_hyper *= self.args.alpha
                    logging.info(f'Lambda Hyper: {args.lambda_hyper}')
                    
                else:
                    args.lambda_hyper /= self.args.alpha
                    logging.info(f'Lambda Hyper: {args.lambda_hyper}')

        # ------------------------------- Final Test -------------------------------
        test_acc, test_asr, test_ra = given_dataloader_test_v2(model, data_clean_testset, data_bd_testset, nn.CrossEntropyLoss(), self.args)
        logging.info(f'Final test_acc:{test_acc}  test_asr:{test_asr}  test_ra:{test_ra}')
        
        # Save the training results
        training_results = {
            "round": save_epoch,
            "train_acc": training_acc,
            "train_asr": training_asr,
            "train_ra": training_ra,
        }
        
        training_results = pd.DataFrame.from_dict(training_results)
        training_results.columns = ['round', 'train_acc', 'train_asr', 'train_ra']
        training_results.to_csv(os.path.join(self.args.defense_save_path, "training_results.csv"))
        
        # save the result to a csv file in the defense_save_path
        final_result = {
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        }

        final_result_df = pd.DataFrame(final_result, columns=["test_acc", "test_asr", "test_ra"], index=[0])
        final_result_df.to_csv(os.path.join(self.args.defense_save_path, "final_result.csv"))
        
        # Pickle the mask_dict
        pickle.dump(mask_dict, open(self.args.defense_save_path + '/mask_dict.pkl', 'wb'))
        
        for (layer, mask, hook) in mask_dict.values():
            hook.remove()

        save_defense_result(
            model_name = args.model,
            num_classes = args.num_classes,
            model = model.cpu().state_dict(),
            save_path = self.args.defense_save_path,
        )

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    mmbd = MMBD()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = mmbd.set_args(parser)
    args = parser.parse_args()
    mmbd.add_yaml_to_args(args)
    args = mmbd.process_args(args)
    mmbd.prepare(args)
    mmbd.defense()
    
    # Remove all allocated GPU memory
    torch.cuda.empty_cache()