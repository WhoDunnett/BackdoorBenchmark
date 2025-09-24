import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

os.chdir(sys.path[0])
sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense

import sys
import os

# Add ../ to the sys.path
sys.path.append('../')
sys.path.append(os.getcwd())

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer, given_dataloader_test_v2
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, spc_choose_poisoned_sample

import random
from copy import deepcopy
from tqdm import tqdm

from utils.defense.utils_btidbf.unet_model import UNet
from utils.defense.utils_btidbf.model_wrapper import PreActResNetWrapper, EfficientNetWrapper, MobileNetV3Wrapper, VGGWrapper


# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Modified evaluation code from utils.trainer_cls.py

def given_dataloader_mask_test(
        model,
        test_dataloader,
        criterion,
        mask,
        non_blocking : bool = False,
        device = "cpu",
        verbose : int = 0
):
    model.to(device, non_blocking=non_blocking)
    model.eval()
    metrics = {
        'test_correct': 0,
        'test_loss_sum_over_batch': 0,
        'test_total': 0,
    }
    criterion = criterion.to(device, non_blocking=non_blocking)

    if verbose == 1:
        batch_predict_list, batch_label_list = [], []

    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_dataloader):
            
            x = x.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            
            # -------- Modification --------
            # Apply the mask to the input
            x_feat = model.from_input_to_features(x)
            mask_x_pred = model.from_features_to_output(mask*x_feat)

            loss = criterion(mask_x_pred, target)
            # ------------------------------

            _, predicted = torch.max(mask_x_pred, -1)
            correct = predicted.eq(target).sum()

            if verbose == 1:
                batch_predict_list.append(predicted.detach().clone().cpu())
                batch_label_list.append(target.detach().clone().cpu())

            metrics['test_correct'] += correct.item()
            metrics['test_loss_sum_over_batch'] += loss.item()
            metrics['test_total'] += target.size(0)

    metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch']/len(test_dataloader)
    metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

    if verbose == 0:
        return metrics, None, None
    elif verbose == 1:
        return metrics, torch.cat(batch_predict_list), torch.cat(batch_label_list)

def given_dataloader_test_mask_v2(
    model,
    test_dataset,
    bd_test_dataset,
    criterion,
    args,
    mask,
):

    # ----------------- Clean -----------------

    clean_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    test_acc = given_dataloader_mask_test(
        model,
        clean_dataloader,
        criterion,
        mask,
        args.non_blocking,
        args.device,
    )[0]['test_acc']

    # # ----------------- BD -----------------
    
    bd_dataloader = torch.utils.data.DataLoader(
        bd_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    bd_test_acc = given_dataloader_mask_test(
        model,
        bd_dataloader,
        criterion,
        mask,
        args.non_blocking,
        args.device,
    )[0]['test_acc']

    # ----------------- RA -----------------
    ra_dataset = deepcopy(bd_test_dataset)
    ra_dataset.wrapped_dataset.getitem_all_switch = True
    ra_dataset.wrapped_dataset.getitem_all = True

    ra_dataloader = torch.utils.data.DataLoader(
        ra_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    bd_test_ra = given_dataloader_mask_test(
        model,
        ra_dataloader,
        criterion,
        mask,
        args.non_blocking,
        args.device,
    )[0]['test_acc']

    return test_acc, bd_test_acc, bd_test_ra

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Train UNet

def train_unet(args, classifier, cln_trainloader, save_path, device):

    # Add wrapper to classifier
    if args.model == 'preactresnet18':
        classifier = PreActResNetWrapper(classifier)
    elif args.model == 'efficientnet_b3':
        classifier = EfficientNetWrapper(classifier)
    elif args.model == 'mobilenet_v3_large':
        classifier = MobileNetV3Wrapper(classifier)
    elif args.model == 'vgg19_bn':
        classifier = VGGWrapper(classifier)
    else:
        raise NotImplementedError(f"Model {args.model} not supported")

    generator = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.gen_lr)

    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax()

    for p in range(args.preround):
        pbar = tqdm(cln_trainloader, desc="Pretrain Generator")
        generator.train()
        classifier.eval()
        tloss = 0
        tloss_pred = 0
        tloss_feat = 0
        tloss_norm = 0
        for batch_idx, (cln_img, targets, _, _, _) in enumerate(pbar):
            cln_img = cln_img.to(device)
            pur_img = generator(cln_img)

            cln_feat = classifier.from_input_to_features(cln_img)
            pur_feat = classifier.from_input_to_features(pur_img)
            cln_out = classifier.from_features_to_output(cln_feat)
            pur_out = classifier.from_features_to_output(pur_feat)

            loss_pred = ce(softmax(cln_out), softmax(pur_out))
            loss_feat = mse(cln_feat, pur_feat)
            loss_norm = mse(cln_img, pur_img)

            if loss_norm > 0.1:
                loss = 1*loss_pred + 1*loss_feat + 100*loss_norm
            else:
                loss = loss_pred + 1*loss_feat + 0.01*loss_norm

            opt_g.zero_grad()
            loss.backward()
            opt_g.step()
            
            tloss += loss.item()
            tloss_pred += loss_pred.item()
            tloss_feat += loss_feat.item()
            tloss_norm += loss_norm.item()

            pbar.set_postfix({"epoch": "{:d}".format(p), 
                              "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                              "loss_pred": "{:.4f}".format(tloss_pred/(batch_idx+1)),
                              "loss_feat": "{:.4f}".format(tloss_feat/(batch_idx+1)),
                              "loss_norm": "{:.4f}".format(tloss_norm/(batch_idx+1))})
    
    # Check if pretrain folder exists
    if not os.path.exists(os.path.join(save_path, "pretrain")):
        os.makedirs(os.path.join(save_path, "pretrain"))

    save_path = os.path.join(save_path, "pretrain/init_generator.pt")

    torch.save(generator.state_dict(), save_path)
    return generator

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Mitigation Code

class MaskGenerator(nn.Module):
    def __init__(self, init_mask, classifier) -> None:
        super().__init__()
        self._EPSILON = 1e-7
        self.classifier = classifier
        self.mask_tanh = nn.Parameter(init_mask.clone().detach().requires_grad_(True))
    
    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        return bounded

def get_target_label(args, device, loader, testmodel, midmodel = None):
    model = deepcopy(testmodel)
    model.eval()        
    reg = np.zeros([args.num_classes])
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, _, _) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if not midmodel is None:
                tmodel = deepcopy(midmodel)
                tmodel.eval()
                gnoise = 0.03 * torch.randn_like(inputs, device=device)
                inputs = tmodel(inputs + gnoise)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(inputs.shape[0]):
                p = predicted[i]
                reg[p] += 1
                    
    return np.argmax(reg)

def run_mitigation(args, classifier, cln_trainloader, bd_gen, save_path, device, clean_test, bd_test):

    # Add wrapper to classifier
    if args.model == 'preactresnet18':
        classifier = PreActResNetWrapper(classifier)
    elif args.model == 'efficientnet_b3':
        classifier = EfficientNetWrapper(classifier)
    elif args.model == 'mobilenet_v3_large':
        classifier = MobileNetV3Wrapper(classifier)
    elif args.model == 'vgg19_bn':
        classifier = VGGWrapper(classifier)
    else:
        raise NotImplementedError(f"Model {args.model} not supported")

    opt_cls = torch.optim.Adam(classifier.parameters(), lr = args.cls_lr)
    bd_gen = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4)
    bd_gen.load_state_dict(torch.load(os.path.join(save_path, "pretrain/init_generator.pt"), map_location=torch.device('cpu'), weights_only=False))
    bd_gen = bd_gen.to(device)
    opt_bd = torch.optim.Adam(bd_gen.parameters(), lr=args.gen_lr)

    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax()

    detected_tlabel = None

    def reverse(model, bd_gen):
        inv_classifier = deepcopy(model)
        inv_classifier.eval()
        tmp_img = torch.ones([1, 3, args.size, args.size], device=device)
        tmp_feat = inv_classifier.from_input_to_features(tmp_img)
        feat_shape = tmp_feat.shape
        init_mask = torch.randn(feat_shape).to(device)
        m_gen = MaskGenerator(init_mask=init_mask, classifier=inv_classifier)
        opt_m = torch.optim.Adam([m_gen.mask_tanh], lr=0.01)
        for m in range(args.mround):
            tloss = 0
            tloss_pos_pred = 0
            tloss_neg_pred = 0
            m_gen.train()
            inv_classifier.train()
            pbar = tqdm(cln_trainloader, desc="Decoupling Benign Features")
            for batch_idx, (cln_img, targets, _, _, _) in enumerate(pbar):
                opt_m.zero_grad()
                cln_img = cln_img.to(device)
                targets = targets.to(device)
                feat_mask = m_gen.get_raw_mask()

                cln_feat = inv_classifier.from_input_to_features(cln_img)
                mask_pos_pred = inv_classifier.from_features_to_output(feat_mask*cln_feat)
                remask_neg_pred = inv_classifier.from_features_to_output((1-feat_mask)*cln_feat)

                mask_norm = torch.norm(feat_mask, 1)

                loss_pos_pred = ce(mask_pos_pred, targets)
                loss_neg_pred = ce(remask_neg_pred, targets)            
                loss = loss_pos_pred - loss_neg_pred

                loss.backward()
                opt_m.step()

                tloss += loss.item()
                tloss_pos_pred += loss_pos_pred.item()
                tloss_neg_pred += loss_neg_pred.item()
                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(m),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_pos_pred": "{:.4f}".format(tloss_pos_pred/(batch_idx+1)),
                                "loss_neg_pred": "{:.4f}".format(tloss_neg_pred/(batch_idx+1)),
                                "mask_norm": "{:.4f}".format(mask_norm)})
                
        feat_mask = m_gen.get_raw_mask().detach()

        for u in range(args.uround):
            tloss = 0
            tloss_benign_feat = 0
            tloss_backdoor_feat = 0
            tloss_norm = 0
            m_gen.eval()
            bd_gen.train()
            inv_classifier.eval()
            pbar = tqdm(cln_trainloader, desc="Training Backdoor Generator")
            for batch_idx, (cln_img, targets, _, _, _) in enumerate(pbar):
                cln_img = cln_img.to(device)
                bd_gen_img = bd_gen(cln_img)

                cln_feat = inv_classifier.from_input_to_features(cln_img)
                bd_gen_feat = inv_classifier.from_input_to_features(bd_gen_img)

                loss_benign_feat = mse(feat_mask*cln_feat, feat_mask*bd_gen_feat)
                loss_backdoor_feat = mse((1-feat_mask)*cln_feat, (1-feat_mask)*bd_gen_feat)
                loss_norm = mse(cln_img, bd_gen_img)

                if loss_norm > args.norm_bound or loss_benign_feat > args.feat_bound:
                    loss = loss_norm
                else:
                    loss = -loss_backdoor_feat + 0.01*loss_benign_feat
                    
                if n > 0:
                    inv_tlabel = torch.ones_like(targets, device=device)*detected_tlabel
                    bd_gen_pred = inv_classifier(bd_gen_img)
                    loss += ce(bd_gen_pred, inv_tlabel)

                opt_bd.zero_grad()
                loss.backward()
                opt_bd.step()
                
                tloss += loss.item()
                tloss_benign_feat += loss_benign_feat.item()
                tloss_backdoor_feat += loss_backdoor_feat.item()
                tloss_norm += loss_norm.item()

                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(u),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_bengin_feat": "{:.4f}".format(tloss_benign_feat/(batch_idx+1)),
                                "loss_backdoor_feat": "{:.4f}".format(tloss_backdoor_feat/(batch_idx+1)),
                                "loss_norm": "{:.4f}".format(tloss_norm/(batch_idx+1))})

    def unlearn(model, bd_gen):
        classifier = model    
        for ul in range(args.ul_round):
            tloss = 0
            tloss_pred = 0
            tloss_feat = 0
            bd_gen.eval()
            classifier.train()
            pbar = tqdm(cln_trainloader, desc="Unlearning")
            for batch_idx, (cln_img, targets, _, _, _) in enumerate(pbar):
                targets = targets.to(device)
                bd_gen_num = int(0.1*cln_img.shape[0] + 1)
                bd_gen_list = random.sample(range(cln_img.shape[0]), bd_gen_num)
                cln_img = cln_img.to(device)
                bd_gen_img = deepcopy(cln_img).to(device)
                bd_gen_img[bd_gen_list] = bd_gen(bd_gen_img[bd_gen_list])

                cln_feat = classifier.from_input_to_features(cln_img)
                bd_gen_feat = classifier.from_input_to_features(bd_gen_img)
                bd_gen_pred = classifier.from_features_to_output(bd_gen_feat)

                loss_pred = ce(bd_gen_pred, targets)
                loss_feat = mse(cln_feat, bd_gen_feat)
                loss = loss_pred + loss_feat

                opt_cls.zero_grad()
                loss.backward()
                opt_cls.step()
            
                tloss += loss.item()
                tloss_pred += loss_pred.item()
                tloss_feat += loss_feat.item()
                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(ul),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_pred": "{:.4f}".format(tloss_pred/(batch_idx+1)),
                                "loss_feat": "{:.4f}".format(tloss_feat/(batch_idx+1))})

    for n in range(args.nround):
        reverse(classifier, bd_gen)
        if n == 0:
            detected_tlabel = get_target_label(args, device, cln_trainloader, classifier, bd_gen)
        elif args.earlystop:
            checked_tlabel = get_target_label(args, device, cln_trainloader, classifier, bd_gen)
            if checked_tlabel != detected_tlabel:
                break
        unlearn(classifier, bd_gen)

    # Return the original model and the mask
    return classifier.model 

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Main Class

class BTIDBF_U(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--result_base', type=str, help='the location of result base path', default = "../record")
    
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--model', type=str, help='resnet18')

        parser.add_argument('--gen_lr', type=float, help='learning rate of generator')
        parser.add_argument('--cls_lr', type=float, help='learning rate of classifier')
        parser.add_argument('--preround', type=int, help='pretrain round')
        parser.add_argument('--mround', type=int, help='mask round')
        parser.add_argument('--uround', type=int, help='unlearn round')
        parser.add_argument('--nround', type=int, help='number of rounds')
        parser.add_argument('--ul_round', type=int, help='unlearn round')
        parser.add_argument('--earlystop', type=lambda x: str(x) in ['True', 'true', '1'], help='early stop')
        parser.add_argument('--norm_bound', type=float, help='norm bound')
        parser.add_argument('--feat_bound', type=float, help='feature bound')
        parser.add_argument('--size', type=int, help='size')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/btidbf_u/config.yaml", help='the path of yaml')

        #set the parameter for the ft defense
        parser.add_argument('--spc', type=int, help='the samples per class used for training')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')

    def set_result(self, result_file):
        attack_file = args.result_base + os.path.sep + result_file


        # #######################################
        # Modified to be compatible with the new result_base and SPC
        # #######################################
        if args.spc is not None:
            save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "btidbf_u" + os.path.sep + f'spc_{args.spc}' + os.path.sep + str(args.random_seed)
        else:
            save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "btidbf_u" + os.path.sep + f'ratio_{args.ratio}' + os.path.sep + str(args.random_seed)
        
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
       
        self.set_trainer(model)

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)

        # MODIFICATION MADE HERE
        if args.spc is not None:
            # Given grad prune uses validation set an spc of 1 is not possible
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

        generator = train_unet(self.args, model, trainloader, self.args.save_path, self.device)
        final_model = run_mitigation(self.args, model, trainloader, generator, self.args.save_path, self.device, data_clean_testset, data_bd_testset)

        ##############################################
        # Get the final results for
        criterion = nn.CrossEntropyLoss()
        test_acc, test_asr, test_ra = given_dataloader_test_v2(final_model, data_clean_testset, data_bd_testset, criterion, self.args)
        logging.info(f'Final test_acc:{test_acc}  test_asr:{test_asr}  test_ra:{test_ra}')

        # save the result to a csv file in the defense_save_path
        final_result = {
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        }

        final_result_df = pd.DataFrame(final_result, columns=["test_acc", "test_asr", "test_ra"], index=[0])
        final_result_df.to_csv(os.path.join(self.args.save_path, "final_result.csv"))
        ##############################################

        result = {}
        result['model'] = model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )

        return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    BTIDBF_U.add_arguments(parser)
    args = parser.parse_args()
    btidbf_u = BTIDBF_U(args)
    result = btidbf_u.defense(args.result_file)