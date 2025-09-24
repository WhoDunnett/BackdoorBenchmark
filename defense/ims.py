import argparse
import os,sys

import torch
import torch.nn as nn
import numpy as np

os.chdir(sys.path[0])
sys.path.append('../')
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

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
from torch.utils.checkpoint import checkpoint

from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import ModelTrainerCLS_v2, BackdoorModelTrainer, Metric_Aggregator, given_dataloader_test, general_plot_for_epoch, given_dataloader_test_v2
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_normalization, get_dataset_denormalization
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform, spc_choose_poisoned_sample
from tqdm import tqdm

# ---------------------------- IMS Defense Helper Functions ----------------------------

class InfiniteRandomBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, batch_size, num_batches=None):
        """
        A custom BatchSampler that yields batches of random indices.
        
        Parameters:
          dataset     : the dataset object (used to get the length)
          batch_size  : the number of unique samples per batch.
          num_batches : total number of batches to generate (if None, runs infinitely).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        batch_count = 0
        while self.num_batches is None or batch_count < self.num_batches:
            # Generate a random permutation and take the first 'batch_size' indices.
            # This guarantees unique indices within the batch.
            indices = torch.randperm(len(self.dataset))[:self.batch_size].tolist()
            yield indices
            batch_count += 1

    def __len__(self):
        if self.num_batches is not None:
            return self.num_batches
        else:
            raise ValueError("This sampler has no defined length because it is infinite.")

def get_conv_modules(model):
    conv = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv.append(module)
    
    return conv

def get_perturbed_image(images, pert, norm, denorm, single_pert=False):

    images_wo_trans = denorm(images) 
    
    if single_pert:
        pert = pert.unsqueeze(0)
        images_wo_trans = images_wo_trans + pert
        
    else:
        images_wo_trans = images_wo_trans + pert
    
    images_with_trans = norm(images_wo_trans)
    return images_with_trans

def get_mask_value(channel_mask, channel_selection, apply_sigmoid=False, k=30):
    if apply_sigmoid:
        # Shift the sigmoid function to map 0 input to 0 output
        channel_mask = torch.sigmoid(k * (channel_mask - 0.5))
    final_mask = channel_mask + (channel_selection * (1 - channel_mask))
    return final_mask

# ---------------------------- IMS Defense Hook Functions ----------------------------

class Prune_Mask_Inverse:
    
    def __init__(self):
        
        self.apply_inverse = True
        self.channel_mask = None
        self.channel_selection = None
        self.k = 20
        
    def hook_fn(self, module, input, output):
        
        modified_output = output
        
        if self.apply_inverse:
            channel_mask = get_mask_value((1 - self.channel_mask), self.channel_selection, apply_sigmoid=True, k=self.k)
        else:
            channel_mask = get_mask_value(self.channel_mask, self.channel_selection, apply_sigmoid=True, k=self.k)
            
        modified_output = modified_output * channel_mask.unsqueeze(-1).unsqueeze(-1)
        return modified_output     
    
def setup_hooks(conv_modules, device):
    
    hooks = []
    for i in range(len(conv_modules)):
        hook = Prune_Mask_Inverse()
        hook.channel_mask = torch.ones(conv_modules[i].out_channels, device=device)
        hooks_handle = conv_modules[i].register_forward_hook(hook.hook_fn)
        hooks.append((hook, hooks_handle))
        
    return hooks  

def set_model_eval(model):
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

def set_mask_grad(hooks, requires_grad=False):
    for hook, _ in hooks:
        hook.channel_mask.requires_grad = requires_grad
        hook.channel_selection.requires_grad = requires_grad

def set_mask(hooks, apply_inverse=False):
    for hook, _ in hooks:
        hook.apply_inverse = apply_inverse

# ---------------------------- IMS Defense Training Functions ----------------------------

def disagree_loss(logits1, logits2, eps=1e-8):

    # Convert logits to probability distributions
    p1 = F.softmax(logits1, dim=1)  # shape: (batch_size, C)
    p2 = F.softmax(logits2, dim=1)  # shape: (batch_size, C)
    
    # Overlap = sum_{c} p1(c) * p2(c)  (done per sample in the batch)
    overlap = torch.sum(p1 * p2, dim=1)  # shape: (batch_size,)

    # Disagreement = 1 - overlap
    # Then take negative log to make it a loss we can minimize
    loss = -torch.log(1.0 - overlap + eps).mean()
    
    return loss

def agree_loss(logits1, logits2):

    # Convert logits to probability vectors
    p1 = F.softmax(logits1, dim=1)  # shape: (batch_size, C)
    p2 = F.softmax(logits2, dim=1)  # shape: (batch_size, C)
    
    # sum_{c} p1(c) * p2(c)  (done per sample in the batch)
    overlap = torch.sum(p1 * p2, dim=1)  # shape: (batch_size,)
    
    # - log( overlap )
    loss = -torch.log(overlap + 1e-8).mean()  # add epsilon for numerical stability
    
    return loss

def find_batch_peturb(args, model_mask, model_ref, hooks, x_batch, label_batch, device, norm, denorm):
    
    set_mask_grad(hooks, requires_grad=False)
    set_mask(hooks, apply_inverse=False)
        
    set_model_eval(model_mask)
    set_model_eval(model_ref)

    # Setup peturb and optimizer
    pert = torch.zeros_like(x_batch, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([pert], lr=args.lr_peturb)

    round_agree_loss, round_disagree_loss = [], []
    for n in range(args.num_steps_peturb):

        optimizer.zero_grad()
        peturb_x = get_perturbed_image(x_batch, pert, norm, denorm)

        # Get the logits of the reference model for peturb 
        pred_peturb_ref = model_ref(peturb_x)
        pred_clean_ref = model_ref(x_batch)

        # Get the logits of the masked
        set_mask(hooks, apply_inverse=True)
        pred_peturb_inv = model_mask(peturb_x)

        set_mask(hooks, apply_inverse=False)
        pred_peturb_masked = model_mask(peturb_x)

        # Compute the clean and bd mask loss
        agreement_loss = agree_loss(pred_peturb_inv, pred_peturb_ref)
        disagreement_loss = disagree_loss(pred_peturb_ref, pred_clean_ref)

        total_loss = agreement_loss + disagreement_loss

        total_loss.backward()
        optimizer.step()

        # Clip the pert to be in the range of the dataset
        pert.data = torch.clamp(pert.data, -args.norm_bound, args.norm_bound)

        round_agree_loss.append(agreement_loss.item())
        round_disagree_loss.append(disagreement_loss.item())

        del peturb_x, pred_peturb_ref, pred_clean_ref, pred_peturb_inv, pred_peturb_masked
        del agreement_loss, disagreement_loss, total_loss
        
    # Detach the pert
    pert = pert.detach()
    return pert, round_agree_loss, round_disagree_loss

def test_batch_peturb(model_mask, model_ref, x_batch, label_batch, peturb_batch, hooks, norm, denorm, target_label=0):

    # Test the distribution of a peturb label
    target_count = 0
    
    pred_peturb_inv_agree = 0
    pred_peturb_ref_disagree = 0

    pred_peturb_mask_correct = 0
    pred_clean_mask_correct = 0
    pred_clean_ref_correct = 0

    peturb_x = get_perturbed_image(x_batch, peturb_batch, norm, denorm)

    set_model_eval(model_mask), set_model_eval(model_ref)
    set_mask(hooks, apply_inverse=True)

    with torch.no_grad():
        pred_peturb_ref = model_ref(peturb_x)
        softmax_peturb_ref = F.softmax(pred_peturb_ref, dim=1)
        peturb_ref_label = torch.argmax(softmax_peturb_ref, dim=1)

        pred_peturb_inv = model_mask(peturb_x)
        softmax_peturb_inv = F.softmax(pred_peturb_inv, dim=1)
        peturb_inv_label = torch.argmax(softmax_peturb_inv, dim=1)

    set_mask(hooks, apply_inverse=False)

    with torch.no_grad():
        pred_peturb_masked = model_mask(peturb_x)
        softmax_peturb_masked = F.softmax(pred_peturb_masked, dim=1)
        peturb_masked_label = torch.argmax(softmax_peturb_masked, dim=1)

        pred_clean_masked = model_mask(x_batch)
        softmax_clean_masked = F.softmax(pred_clean_masked, dim=1)
        clean_masked_label = torch.argmax(softmax_clean_masked, dim=1)

        pred_clean_ref = model_ref(x_batch)
        softmax_clean_ref = F.softmax(pred_clean_ref, dim=1)
        clean_ref_label = torch.argmax(softmax_clean_ref, dim=1)

    # Summm all peturb_ref_label == 0
    target_count += (peturb_ref_label == target_label).sum().item()

    # Compare peturb_ref_label and peturb_inv_label
    pred_peturb_inv_agree += (peturb_ref_label == peturb_inv_label).sum().item()

    # Compare peturb_ref_label and clean_ref_label
    pred_peturb_ref_disagree += (peturb_ref_label != clean_ref_label).sum().item()

    # Compare peturb_masked_label and clean_masked_label to get the correct predictions
    pred_peturb_mask_correct += (peturb_masked_label == label_batch).sum().item()
    pred_clean_mask_correct += (clean_masked_label == label_batch).sum().item()
    pred_clean_ref_correct += (clean_ref_label == label_batch).sum().item()    
        
    # Calculate the percentages
    total_samples = x_batch.size(0)
    target_count_percentage = (target_count / total_samples) * 100
    pred_peturb_inv_agree_percentage = (pred_peturb_inv_agree / total_samples) * 100
    pred_peturb_ref_disagree_percentage = (pred_peturb_ref_disagree / total_samples) * 100
    pred_peturb_mask_correct_percentage = (pred_peturb_mask_correct / total_samples) * 100
    pred_clean_mask_correct_percentage = (pred_clean_mask_correct / total_samples) * 100
    pred_clean_ref_correct_percentage = (pred_clean_ref_correct / total_samples) * 100

    del pred_peturb_ref, pred_peturb_inv, pred_peturb_masked, pred_clean_masked, pred_clean_ref
    del softmax_peturb_ref, softmax_peturb_inv, softmax_peturb_masked, softmax_clean_masked, softmax_clean_ref

    percentages = (
        target_count_percentage,
        pred_peturb_inv_agree_percentage,
        pred_peturb_ref_disagree_percentage,
        pred_peturb_mask_correct_percentage,
        pred_clean_mask_correct_percentage,
        pred_clean_ref_correct_percentage
    )

    return percentages

def optimize_mask_initial(args, model_mask, model_ref, hooks, x_batch, device, mask_optimizer):
        
    set_model_eval(model_mask), set_model_eval(model_ref)
    set_mask_grad(hooks, requires_grad=True)

    pred_clean_ref = model_ref(x_batch)

    # Get the masked forward
    set_mask(hooks, apply_inverse=False)
    pred_clean_masked = model_mask(x_batch)

    set_mask(hooks, apply_inverse=True)
    pred_clean_inv = model_mask(x_batch)

    # Compute the agreement loss between the inv and ref model
    agreement_loss = agree_loss(pred_clean_masked, pred_clean_ref)
    disagreement_loss = disagree_loss(pred_clean_inv, pred_clean_ref)

    # Compute the L1 loss
    total_l1 = torch.tensor(0.0).to(device)
    num_elements = torch.tensor(0.0).to(device)

    for hook, _ in hooks:
        l1 = torch.norm((1 - hook.channel_selection), 1) 
        total_l1 += l1
        num_elements += hook.channel_selection.numel()

    total_l1 = (total_l1 / num_elements) * args.l1_weight
    total_loss = agreement_loss + disagreement_loss + total_l1

    mask_optimizer.zero_grad()
    total_loss.backward()
    mask_optimizer.step()

    # Clip the mask values to be within [0, 1]
    for hook, _ in hooks:
        hook.channel_mask.data = torch.clamp(hook.channel_mask.data, 0, 1)
        hook.channel_selection.data = torch.clamp(hook.channel_selection.data, 0, 1)

    return agreement_loss.item(), disagreement_loss.item(), total_l1.item()

def optimize_minimize(args, model_mask, model_ref, hooks, x_batch, peturb_batch, device, mask_optimizer, dataset_norm, dataset_denorm):

    # Set the peturb to not require grad
    peturb_batch.requires_grad = False
        
    set_model_eval(model_mask), set_model_eval(model_ref)
    set_mask_grad(hooks, requires_grad=True)

    with torch.no_grad():
        peturb_x = get_perturbed_image(x_batch, peturb_batch, dataset_norm, dataset_denorm)
        peturb_x._requires_grad = False

    # Get the logits of the reference model for peturb 
    pred_peturb_ref = model_ref(peturb_x)
    pred_clean_ref = model_ref(x_batch)

    # Get the logits of the masked
    set_mask(hooks, apply_inverse=True)
    pred_peturb_inv = model_mask(peturb_x)
    pred_clean_inv = model_mask(x_batch)

    set_mask(hooks, apply_inverse=False)
    pred_peturb_masked = model_mask(peturb_x)
    pred_clean_masked = model_mask(x_batch)

    # Compute the clean and bd mask loss
    clean_mask_loss = agree_loss(pred_clean_masked, pred_clean_ref)
    bd_mask_loss = agree_loss(pred_peturb_masked, pred_clean_ref) 

    # Compute the agreement loss between the inv and ref model
    agreement_loss = agree_loss(pred_peturb_inv, pred_peturb_ref)
    disagree_peturb_loss = disagree_loss(pred_peturb_inv, pred_clean_ref)
    disagreement_clean_loss = disagree_loss(pred_clean_inv, pred_clean_ref)

    # Compute the L1 loss
    total_l1 = torch.tensor(0.0).to(device)
    num_elements = torch.tensor(0.0).to(device)

    for hook, _ in hooks:
        l1 = torch.norm((1 - hook.channel_selection), 1)
        total_l1 += l1

        num_elements += hook.channel_selection.numel()

    total_l1 = (total_l1 / num_elements) * args.l1_weight
    total_loss = clean_mask_loss + bd_mask_loss + agreement_loss + disagreement_clean_loss + disagree_peturb_loss + total_l1

    mask_optimizer.zero_grad()
    total_loss.backward()
    mask_optimizer.step()

    # Clip the mask values to be within [0, 1]
    for hook, _ in hooks:
        hook.channel_mask.data = torch.clamp(hook.channel_mask.data, 0, 1)
        hook.channel_selection.data = torch.clamp(hook.channel_selection.data, 0, 1)

    return clean_mask_loss.item(), bd_mask_loss.item(), agreement_loss.item(), disagree_peturb_loss.item(), disagreement_clean_loss.item(), total_l1.item()
    
def train_mask_initial(args, model_mask, model_ref, hooks, train_loader, device, dataset_norm, dataset_denorm):

    mask_params = []
    for hook, _ in hooks:
        mask_params.append(hook.channel_selection)
        mask_params.append(hook.channel_mask)

    mask_optimizer = torch.optim.AdamW(mask_params, lr=args.lr_mask)

    round_loss_inner, round_loss_outer = [], []
    round_predictions = []

    batch_iter = iter(train_loader)

    best_loss = torch.inf
    current_patience = 0
    patience = args.patience
    improvement_threshold = args.improvement_threshold

    logging.info(f"Starting initial rounds with max rounds: {args.max_initial_rounds} and patience: {patience}")

    for round_i in range(args.max_initial_rounds):

        # Get a batch of data
        batch = next(batch_iter)
        x_batch, label_batch = batch[0], batch[1]

        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)

        peturb, peturb_agree_loss, peturb_disagree_loss = find_batch_peturb(
            args,
            model_mask,
            model_ref,
            hooks,
            x_batch,
            label_batch,
            device,
            dataset_norm,
            dataset_denorm
        )

        round_loss_inner.append((peturb_agree_loss[-1], peturb_disagree_loss[-1]))

        agreement_loss, disagreement_loss, total_l1 = optimize_mask_initial(
            args,
            model_mask,
            model_ref,
            hooks,
            x_batch,
            device,
            mask_optimizer
        )

        round_loss_outer.append((agreement_loss, disagreement_loss, total_l1))

        with torch.no_grad():
            percentage = test_batch_peturb(
                model_mask,
                model_ref,
                x_batch,
                label_batch,
                peturb,
                hooks,
                dataset_norm,
                dataset_denorm
            )
        
        round_predictions.append(percentage)

        outer_sum_loss = agreement_loss + disagreement_loss + total_l1
        if outer_sum_loss < (best_loss * improvement_threshold):
            best_loss = outer_sum_loss
            current_patience = 0
        else:
            current_patience += 1
            if current_patience > patience:
                logging.info(f"Early stopping at round {round_i} with patience {patience}")
                break

        if round_i % 20 == 0:
            logging.info(f"Round {round_i} (INNER): Disagree Loss: {peturb_disagree_loss[-1]:.4f}, Agree Loss: {peturb_agree_loss[-1]:.4f}")
            logging.info(f"Round {round_i} (OUTER): Clean Mask Loss: {agreement_loss:.4f}, BD Mask Loss: {disagreement_loss:.4f}, L1 Loss: {total_l1:.4f}")
            logging.info(f"Round {round_i} (PREDICTIONS): Target Count: {percentage[0]:.4f}, Peturb Inv Agree: {percentage[1]:.4f}, Peturb Ref Disagree: {percentage[2]:.4f}, Peturb Mask Correct: {percentage[3]:.4f}, Clean Mask Correct: {percentage[4]:.4f}, Clean Ref Correct: {percentage[5]:.4f}")

        del peturb, peturb_agree_loss, peturb_disagree_loss
        del x_batch, label_batch
        del agreement_loss, disagreement_loss, total_l1        

    logging.info(f"Finished initial rounds with max rounds: {args.max_initial_rounds} and patience: {patience}")

    # Save a csv file of the loss
    inner_loss_df = pd.DataFrame(round_loss_inner, columns=['Peturb Agree Loss', 'Peturb Disagree Loss'])
    outer_loss_df = pd.DataFrame(round_loss_outer, columns=['Clean Mask Loss', 'BD Mask Loss', 'L1 Loss'])
    predictions_df = pd.DataFrame(round_predictions, columns=['Target Count', 'Peturb Inv Agree', 'Peturb Ref Disagree', 'Peturb Mask Correct', 'Clean Mask Correct', 'Clean Ref Correct'])
    inner_loss_df.to_csv(args.defense_save_path + '/initial_inner_loss.csv', index=False)
    outer_loss_df.to_csv(args.defense_save_path + '/initial_outer_loss.csv', index=False)
    predictions_df.to_csv(args.defense_save_path + '/initial_predictions.csv', index=False)

def train_mask_minimize(args, model_mask, model_ref, hooks, train_loader, device, dataset_norm, dataset_denorm, with_l1=True):

    mask_params = []
    for hook, _ in hooks:
        mask_params.append(hook.channel_selection)
        mask_params.append(hook.channel_mask)

    mask_optimizer = torch.optim.AdamW(mask_params, lr=args.lr_mask)

    round_loss_inner, round_loss_outer = [], []
    round_predictions = []

    batch_iter = iter(train_loader)

    best_loss = torch.inf
    current_patience = 0
    patience = args.patience
    improvement_threshold = args.improvement_threshold

    if not with_l1:
        saved_l1_weight = args.l1_weight
        args.l1_weight = 0.0

    logging.info(f'Starting minimize rounds with max rounds: {args.max_minimize_rounds} and patience: {patience} and l1 weight: {args.l1_weight}')

    for round_i in range(args.max_minimize_rounds):

        # Get a batch of data
        batch = next(batch_iter)
        x_batch, label_batch = batch[0], batch[1]

        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)

        peturb, peturb_agree_loss, peturb_disagree_loss = find_batch_peturb(
            args,
            model_mask,
            model_ref,
            hooks,
            x_batch,
            label_batch,
            device,
            dataset_norm,
            dataset_denorm
        )

        round_loss_inner.append((peturb_agree_loss[-1], peturb_disagree_loss[-1]))

        clean_mask_loss, bd_mask_loss, agreement_loss, disagree_peturb_loss, disagreement_clean_loss, total_l1 = optimize_minimize(
            args,
            model_mask,
            model_ref,
            hooks,
            x_batch,
            peturb,
            device,
            mask_optimizer,
            dataset_norm,
            dataset_denorm
        )

        round_loss_outer.append((clean_mask_loss, bd_mask_loss, agreement_loss, disagree_peturb_loss, disagreement_clean_loss, total_l1))

        with torch.no_grad():
            percentage = test_batch_peturb(
                model_mask,
                model_ref,
                x_batch,
                label_batch,
                peturb,
                hooks,
                dataset_norm,
                dataset_denorm
            )
        
        round_predictions.append(percentage)

        outer_sum_loss = clean_mask_loss + bd_mask_loss + agreement_loss + disagree_peturb_loss + disagreement_clean_loss + total_l1
        if outer_sum_loss < (best_loss * improvement_threshold):
            best_loss = outer_sum_loss
            current_patience = 0
        else:
            current_patience += 1
            if current_patience > patience:
                logging.info(f"Early stopping at round {round_i} with patience {patience}")
                break

        if round_i % 20 == 0:
            logging.info(f"Round {round_i} (INNER): Disagree Loss: {peturb_disagree_loss[-1]:.4f}, Agree Loss: {peturb_agree_loss[-1]:.4f}")
            logging.info(f"Round {round_i} (OUTER): Clean Mask Loss: {clean_mask_loss:.4f}, BD Mask Loss: {bd_mask_loss:.4f}, Agree Loss: {agreement_loss:.4f}, Disagree Clean Loss: {disagree_peturb_loss:.4f}, Disagree BD Loss: {disagreement_clean_loss:.4f}, L1 Loss: {total_l1:.4f}")
            logging.info(f"Round {round_i} (PREDICTIONS): Target Count: {percentage[0]:.4f}, Peturb Inv Agree: {percentage[1]:.4f}, Peturb Ref Disagree: {percentage[2]:.4f}, Peturb Mask Correct: {percentage[3]:.4f}, Clean Mask Correct: {percentage[4]:.4f}, Clean Ref Correct: {percentage[5]:.4f}")

        del peturb, peturb_agree_loss, peturb_disagree_loss
        del x_batch, label_batch
        del clean_mask_loss, bd_mask_loss, agreement_loss, disagree_peturb_loss, disagreement_clean_loss, total_l1

    logging.info(f"Finished minimize rounds with max rounds: {args.max_minimize_rounds} and patience: {patience} and l1 weight: {args.l1_weight}")

    # Save a csv file of the loss
    inner_loss_df = pd.DataFrame(round_loss_inner, columns=['Peturb Agree Loss', 'Peturb Disagree Loss'])
    outer_loss_df = pd.DataFrame(round_loss_outer, columns=['Clean Mask Loss', 'BD Mask Loss', 'Agree Loss', 'Disagree Clean Loss', 'Disagree BD Loss', 'L1 Loss'])
    predictions_df = pd.DataFrame(round_predictions, columns=['Target Count', 'Peturb Inv Agree', 'Peturb Ref Disagree', 'Peturb Mask Correct', 'Clean Mask Correct', 'Clean Ref Correct'])
    
    if with_l1:
        inner_loss_df.to_csv(args.defense_save_path + '/minimize_inner_loss.csv', index=False)
        outer_loss_df.to_csv(args.defense_save_path + '/minimize_outer_loss.csv', index=False)
        predictions_df.to_csv(args.defense_save_path + '/minimize_predictions.csv', index=False)
    else:
        inner_loss_df.to_csv(args.defense_save_path + '/minimize_inner_loss_without_l1.csv', index=False)
        outer_loss_df.to_csv(args.defense_save_path + '/minimize_outer_loss_without_l1.csv', index=False)
        predictions_df.to_csv(args.defense_save_path + '/minimize_predictions_without_l1.csv', index=False)
        args.l1_weight = saved_l1_weight

# ---------------------------- IMS Defense Main Class ----------------------------

class IMS(defense):

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

        parser.add_argument('--yaml_path', type=str, default="./config/defense/ims/default.yaml", help='the yaml path for the defense')

        # set the parameter for the mmdf defense
        parser.add_argument('--spc', type=int, help='the samples per class used for training')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        
        # IMS defense parameters
        parser.add_argument('--k', type=int, help='the k value for the IMS defense')

        parser.add_argument('--l1_weight', type=float, help='the weight for the l1 loss')
        parser.add_argument('--lr_mask', type=float, help='the learning rate for the mask')
        parser.add_argument('--lr_peturb', type=float, help='the learning rate for the peturb')
        parser.add_argument('--num_steps_peturb', type=int, help='the number of steps for the peturb')
        parser.add_argument('--max_initial_rounds', type=int, help='the max rounds for the initial training')
        parser.add_argument('--max_minimize_rounds', type=int, help='the max rounds for the minimize training')

        parser.add_argument('--norm_bound', type=float, help='the norm value for the peturb IMS defense')
        
        parser.add_argument('--patience', type=int, help='the patience for the early stopping')
        parser.add_argument('--improvement_threshold', type=float, help='the improvement threshold for the early stopping')

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
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "ims" + os.path.sep + f'spc_{args.spc}' + os.path.sep + str(args.random_seed)
        else:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "ims" + os.path.sep + f'ratio_{args.ratio}' + os.path.sep + str(args.random_seed)
        
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

        # a. train the mask of old model
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)

        # Get the datasets
        clean_train_dataset = attack_result['clean_train']
        clean_train_wrapper = copy.deepcopy(clean_train_dataset.wrapped_dataset)
        clean_train_wrapper = prepro_cls_DatasetBD_v2(clean_train_wrapper)
        
        if args.spc is not None:
            spc_use = args.spc
            if args.spc == 1: spc_use = 2
            train_idx, _ = spc_choose_poisoned_sample(clean_train_wrapper, spc_use, val_ratio=0)
        else:
            ran_idx = choose_index(args, len(clean_train_wrapper))
            train_idx = np.random.choice(len(ran_idx), int(len(ran_idx) * (1-args.val_ratio)), replace=False)
        
        clean_train_wrapper.subset(train_idx)
    
        data_set_clean = dataset_wrapper_with_transform(clean_train_wrapper, train_tran)
        data_set_clean.wrapped_dataset = clean_train_wrapper
        data_set_clean.wrap_img_transform = train_tran

        logging.info(f'len of train dataset: {len(data_set_clean)}')

        sampler = InfiniteRandomBatchSampler(
            data_set_clean, 
            args.batch_size
        )
            
        train_loader = DataLoader(
            data_set_clean, 
            num_workers=args.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
        )
        
        clean_test_dataset_with_transform = attack_result['clean_test']
        data_clean_testset = clean_test_dataset_with_transform

        bd_test_dataset_with_transform = attack_result['bd_test']
        data_bd_testset = bd_test_dataset_with_transform
        
        model_reference = generate_cls_model(args.model, args.num_classes)
        model_reference.load_state_dict(model.state_dict())
        model_reference.to(args.device)
        
        model_mask = generate_cls_model(args.model, args.num_classes)
        model_mask.load_state_dict(model.state_dict())
        model_mask.to(args.device)
        
        dataset_norm = get_dataset_normalization(args.dataset)
        dataset_denorm = get_dataset_denormalization(dataset_norm)
        
        # Setup the hooks
        conv_modules = get_conv_modules(model_mask)
        hooks = setup_hooks(conv_modules, args.device)

        for hook, _ in hooks:
            hook.channel_mask = torch.ones_like(hook.channel_mask, device=args.device, requires_grad=False)
            hook.channel_selection = torch.ones_like(hook.channel_mask, device=args.device, requires_grad=False)
            hook.apply_mask = True
            hook.k = args.k

        train_mask_initial(
            args,
            model_mask,
            model_reference,
            hooks,
            train_loader,
            args.device,
            dataset_norm,
            dataset_denorm
        )

        initial_acc, initial_asr, initial_ra = given_dataloader_test_v2(
            model_mask,
            data_clean_testset,
            data_bd_testset,
            torch.nn.CrossEntropyLoss(),
            args
        )[:3]

        logging.info(f'Initial test_acc:{initial_acc}  test_asr:{initial_asr}  test_ra:{initial_ra}')

        initial_result = {
            "test_acc": initial_acc,
            "test_asr": initial_asr,
            "test_ra": initial_ra,
        }

        initial_result_df = pd.DataFrame(initial_result, columns=["test_acc", "test_asr", "test_ra"], index=[0])
        initial_result_df.to_csv(args.defense_save_path + '/initial_result.csv', index=False)

        train_mask_minimize(
            args,
            model_mask,
            model_reference,
            hooks,
            train_loader,
            args.device,
            dataset_norm,
            dataset_denorm,
            with_l1=False
        )

        minimize_acc, minimize_asr, minimize_ra = given_dataloader_test_v2(
            model_mask,
            data_clean_testset,
            data_bd_testset,
            torch.nn.CrossEntropyLoss(),
            args
        )[:3]

        logging.info(f'Minimize test_acc:{minimize_acc}  test_asr:{minimize_asr}  test_ra:{minimize_ra}')

        minimize_result = {
            "test_acc": minimize_acc,
            "test_asr": minimize_asr,
            "test_ra": minimize_ra,
        }

        minimize_result_df = pd.DataFrame(minimize_result, columns=["test_acc", "test_asr", "test_ra"], index=[0])
        minimize_result_df.to_csv(args.defense_save_path + '/without_l1_result.csv', index=False)

        train_mask_minimize(
            args,
            model_mask,
            model_reference,
            hooks,
            train_loader,
            args.device,
            dataset_norm,
            dataset_denorm,
            with_l1=True
        )

        torch.cuda.empty_cache()

        minimize_acc, minimize_asr, minimize_ra = given_dataloader_test_v2(
            model_mask,
            data_clean_testset,
            data_bd_testset,
            torch.nn.CrossEntropyLoss(),
            args
        )[:3]

        logging.info(f'Minimize (with l1) test_acc:{minimize_acc}  test_asr:{minimize_asr}  test_ra:{minimize_ra}')

        minimize_result = {
            "test_acc": minimize_acc,
            "test_asr": minimize_asr,
            "test_ra": minimize_ra,
        }

        minimize_result_df = pd.DataFrame(minimize_result, columns=["test_acc", "test_asr", "test_ra"], index=[0])
        minimize_result_df.to_csv(args.defense_save_path + '/final_result.csv', index=False)
        
        # Save the mask to the output directory
        mask_dir = {}
        channel_selection_dir = {}
        for i, (hook, _) in enumerate(hooks):
            mask_dir[f'conv_{i}'] = hook.channel_mask.cpu().detach().numpy()
            channel_selection_dir[f'conv_{i}'] = hook.channel_selection.cpu().detach().numpy()
            
        with open(args.defense_save_path + '/mask.pkl', 'wb') as f:
            pickle.dump(mask_dir, f)
            
        with open(args.defense_save_path + '/channel_selection.pkl', 'wb') as f:
            pickle.dump(channel_selection_dir, f)

if __name__ == '__main__':
    ims = IMS()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = ims.set_args(parser)
    args = parser.parse_args()
    ims.add_yaml_to_args(args)
    args = ims.process_args(args)
    ims.prepare(args)
    ims.defense()