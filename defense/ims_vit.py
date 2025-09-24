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

# ---------------------------- IMP Defense Helper Functions ----------------------------

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

# ---------------------------- IMP Defense Hook Functions ----------------------------

def plot_mask(hooks, save_path):

    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5))
    counts_selection = 0
    counts_mask = 0
    counts_inverse_mask = 0

    for i, (hook, _) in enumerate(hooks):

        channel_selection = hook.channel_selection

        # Apply sigmoid to the channel selection
        channel_selection = channel_selection.cpu().detach().numpy()

        ax[0].boxplot(channel_selection, positions=[i], widths=0.6, showfliers=True)

        mask = get_mask_value(hook.channel_mask, hook.channel_selection, apply_sigmoid=True, k=hook.k).cpu().detach().numpy()
        final_mask = []
        for m, s in zip(mask, channel_selection):
            if s < 0.5:
                final_mask.append(m)

        # Add the number of channel_selection values < 0.5 to the counts
        counts_selection += (channel_selection < 0.5).sum().item()
        counts_mask += (mask < 0.5).sum().item()

        x = np.random.normal(i, 0.05, len(final_mask))
        x = np.clip(x, i - 0.1, i + 0.1)
        ax[1].scatter(x, final_mask, alpha=0.2, c='blue')

        inverse_mask = get_mask_value((1 - hook.channel_mask), hook.channel_selection, apply_sigmoid=True, k=hook.k).cpu().detach().numpy()
        final_inverse_mask = []
        for m, s in zip(inverse_mask, channel_selection):
            if s < 0.5:
                final_inverse_mask.append(m)

        # Add the number of channel_selection values < 0.5 to the counts
        counts_inverse_mask += (inverse_mask < 0.5).sum().item()

        x = np.random.normal(i, 0.05, len(final_inverse_mask))
        x = np.clip(x, i - 0.1, i + 0.1)

        ax[2].scatter(x, final_inverse_mask, alpha=0.2, c='blue')

    ax[0].set_title('Channel Selection')
    ax[0].set_ylabel('Channel Selection Value')
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_xticks([])

    ax[1].set_title('Mask')
    ax[1].set_ylabel('Mask Value')
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_xticks([])

    ax[2].set_title('Inverse Mask')
    ax[2].set_xlabel('Layer')
    ax[2].set_ylabel('Mask Value')
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_xticks([])
    plt.savefig(save_path)
    plt.close()


# ---------------- Hook that masks the *block output* ----------------
class Prune_BlockOut:
    """
    Forward hook for an encoder block that gates the *final output* of the block
    (after attention, MLP, and residuals). Mask is length E (embed_dim).
    """
    def __init__(self, embed_dim, device, k=20):
        self.embed_dim = embed_dim
        self.device = device
        self.k = k

        # Learnable masks (same as your original class)
        self.channel_mask = nn.Parameter(torch.ones(embed_dim, device=device))
        self.channel_selection = nn.Parameter(torch.ones(embed_dim, device=device))

        self.apply_inverse = False  # toggle externally if needed

    def hook_fn(self, module, inputs, output):
        # output is the block's final tensor: shape (..., E)
        E = output.shape[-1]
        if E != self.embed_dim:
            raise RuntimeError(f"Embed dim mismatch: hook has {self.embed_dim}, block output has {E}")

        # Build mask (inverse if requested)
        base = (1 - self.channel_mask) if self.apply_inverse else self.channel_mask
        mask = get_mask_value(base, self.channel_selection, apply_sigmoid=True, k=self.k)  # shape [E]

        # Broadcast mask across leading dims (batch, tokens)
        # If output is (B, N, E) -> (1,1,E); if (N, B, E) -> (1,1,E); both broadcast fine.
        for _ in range(output.dim() - 1):
            mask = mask.unsqueeze(0)

        return output * mask  # element-wise gating along the channel dim


# ---------------- Find encoder blocks in a few common libraries ----------------
def get_encoder_blocks(model):
    """
    Collects modules that look like encoder blocks.
    Compatible with:
      - torchvision.models.vision_transformer.EncoderBlock (class name 'EncoderBlock')
      - timm ViT blocks (often class name 'Block')
      - torch.nn.TransformerEncoderLayer
    You can extend the predicates below if needed.
    """
    blocks = []
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in {"EncoderBlock", "Block"} or isinstance(module, nn.TransformerEncoderLayer):
            # Heuristic sanity check: must have a LayerNorm and MLP/FFN inside
            if hasattr(module, "ln_1") or hasattr(module, "norm1"):
                blocks.append(module)
    return blocks


# ---------------- Set up hooks on blocks instead of MHAs ----------------
def setup_block_output_hooks(blocks, device, k=20):
    """
    Register a forward hook on each encoder block to gate its final output.
    Returns a list of (hook_object, handle).
    """
    hooks = []
    total_embed_dim = 0

    for block in blocks:
        # Try to infer embed_dim from common attributes:
        embed_dim = None

        # torchvision ViT: first MLP Linear is 768->3072, so in_features is E.
        if hasattr(block, "mlp") and isinstance(block.mlp, nn.Sequential) and isinstance(block.mlp[0], nn.Linear):
            embed_dim = block.mlp[0].in_features

        # torch.nn.TransformerEncoderLayer: has self_attn with embed_dim
        if embed_dim is None and hasattr(block, "self_attn") and hasattr(block.self_attn, "embed_dim"):
            embed_dim = block.self_attn.embed_dim

        # Fallback: from first LayerNorm's normalized_shape
        if embed_dim is None:
            ln = getattr(block, "ln_1", getattr(block, "norm1", None))
            if ln is not None and hasattr(ln, "normalized_shape"):
                # normalized_shape can be tuple; last element is E
                ns = ln.normalized_shape
                embed_dim = ns[-1] if isinstance(ns, (tuple, list)) else int(ns)

        if embed_dim is None:
            # Could not infer; skip this module
            continue

        hook_obj = Prune_BlockOut(embed_dim, device, k=k)
        handle = block.register_forward_hook(hook_obj.hook_fn)
        hooks.append((hook_obj, handle))
        total_embed_dim += embed_dim

    print(f"Registered {len(hooks)} block-output hooks. Sum of embed_dims: {total_embed_dim}")
    return hooks

def set_model_eval(model):
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

# You can keep using your existing helpers verbatim:
def set_mask_grad(hooks, requires_grad=False):
    for hook, _ in hooks:
        hook.channel_mask.requires_grad = requires_grad
        hook.channel_selection.requires_grad = requires_grad

def set_mask(hooks, apply_inverse=False):
    for hook, _ in hooks:
        hook.apply_inverse = apply_inverse

# ---------------------------- IMP Defense Training Functions ----------------------------

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

def forward_with_mask(model_mask, hooks, x, apply_inverse: bool):
    # make sure hooks are in the intended mode
    set_mask(hooks, apply_inverse=apply_inverse)
    return model_mask(x)

def find_batch_peturb(args, model_mask, model_ref, hooks, x_batch, label_batch, device, norm, denorm):
    # masks are frozen in this phase (as in your code)
    set_mask_grad(hooks, requires_grad=False)
    set_mask(hooks, apply_inverse=False)

    set_model_eval(model_mask)
    set_model_eval(model_ref)

    # perturbation we optimize
    pert = torch.zeros_like(x_batch, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([pert], lr=args.lr_peturb)

    round_agree_loss, round_disagree_loss = [], []
    for n in range(args.num_steps_peturb):
        optimizer.zero_grad()

        peturb_x = get_perturbed_image(x_batch, pert, norm, denorm)

        pred_peturb_ref = model_ref(peturb_x)
        pred_clean_ref = model_ref(x_batch)

        # masked model: direct calls (no checkpoint)
        pred_peturb_inv = forward_with_mask(model_mask, hooks, peturb_x, apply_inverse=True)
        pred_peturb_masked = forward_with_mask(model_mask, hooks, peturb_x, apply_inverse=False)

        agreement_loss = agree_loss(pred_peturb_inv,  pred_peturb_ref)
        disagreement_loss = disagree_loss(pred_peturb_ref, pred_clean_ref)
        total_loss = agreement_loss + disagreement_loss

        total_loss.backward()

        # Check if the pert has gradients
        # if pert.grad is None:
        #     logging.warning("Perturbation has no gradients. Check your model and hooks.")

        # Get the max and min gradient values
        max_grad = pert.grad.abs().max().item()
        min_grad = pert.grad.abs().min().item()

        #print(f"Step {n}: Max Grad: {max_grad}, Min Grad: {min_grad}")

        optimizer.step()

        # clamp pert
        with torch.no_grad():
            pert.clamp_(-args.norm_bound, args.norm_bound)

        # Print the max and min values pert
        max_pert = pert.abs().max().item()
        min_pert = pert.abs().min().item()

        #print(f"Step {n}: Max Pert: {max_pert}, Min Pert: {min_pert}")

        #print(f"Step {n}: Agreement Loss: {agreement_loss.item()}, Disagreement Loss: {disagreement_loss.item()}, Total Loss: {total_loss.item()}")

        round_agree_loss.append(agreement_loss.item())
        round_disagree_loss.append(disagreement_loss.item())

        del peturb_x, pred_peturb_ref, pred_clean_ref, pred_peturb_inv, pred_peturb_masked
        del agreement_loss, disagreement_loss, total_loss

    #raise ValueError("Peturbation optimization did not converge within the specified number of steps.")

    return pert.detach(), round_agree_loss, round_disagree_loss

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
    set_model_eval(model_mask); set_model_eval(model_ref)
    set_mask_grad(hooks, requires_grad=True)

    pred_clean_ref = model_ref(x_batch)

    # direct forwards
    pred_clean_inv = forward_with_mask(model_mask, hooks, x_batch, apply_inverse=True)
    pred_clean_masked = forward_with_mask(model_mask, hooks, x_batch, apply_inverse=False)

    agreement_loss = agree_loss(pred_clean_masked, pred_clean_ref)
    disagreement_loss = disagree_loss(pred_clean_inv, pred_clean_ref)

    # L1
    total_l1, num_elements = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    for hook, _ in hooks:
        l1 = torch.norm((1 - hook.channel_selection), 1)
        total_l1 += l1
        num_elements += hook.channel_selection.numel()
    total_l1 = (total_l1 / num_elements) * args.l1_weight

    total_loss = agreement_loss + disagreement_loss + total_l1

    mask_optimizer.zero_grad()
    total_loss.backward()

    # Check if the mask has gradients
    for i, (hook, _) in enumerate(hooks):
        if hook.channel_mask.grad is None or hook.channel_selection.grad is None:
            logging.warning("Mask has no gradients. Check your model and hooks.")

        # Print the max and min gradient values
        max_grad_mask = hook.channel_mask.grad.abs().max().item()
        min_grad_mask = hook.channel_mask.grad.abs().min().item()

        #print(f"Hook {i}: Max Grad Mask: {max_grad_mask}, Min Grad Mask: {min_grad_mask}")

    mask_optimizer.step()

    # clamp masks in-place
    with torch.no_grad():
        for i, (hook, _) in enumerate(hooks):
            hook.channel_mask.clamp_(0, 1)
            hook.channel_selection.clamp_(0, 1)

            # Print the max and min values of the mask
            max_mask = hook.channel_mask.abs().max().item()
            min_mask = hook.channel_mask.abs().min().item()
            #print(f"Hook {i}: Max Mask: {max_mask}, Min Mask: {min_mask}")

    return agreement_loss.item(), disagreement_loss.item(), total_l1.item()

def optimize_minimize(args, model_mask, model_ref, hooks, x_batch, peturb_batch, device, mask_optimizer,dataset_norm, dataset_denorm):
    
    # Get allocated memory in GB
    # logging.info("In optimize_minimize 1")
    # allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    # logging.info(f"Allocated memory: {allocated_memory:.2f} GB")
    # torch.cuda.empty_cache()
    # allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    # logging.info(f"Allocated memory after emptying cache: {allocated_memory:.2f} GB")

    # Set the peturb to not require grad
    peturb_batch.requires_grad = False
        
    set_model_eval(model_mask), set_model_eval(model_ref)
    set_mask_grad(hooks, requires_grad=True)

    with torch.no_grad():
        peturb_x = get_perturbed_image(x_batch, peturb_batch, dataset_norm, dataset_denorm)
        peturb_x._requires_grad = False

        pred_peturb_ref = model_ref(peturb_x)
        pred_clean_ref = model_ref(x_batch)

    def checkpointed_mask_forward(x, hooks, apply_inverse):
        # Make sure to set mask state before each forward
        set_mask(hooks, apply_inverse=apply_inverse)
        return model_mask(x)

    # Inverse masked forward (checkpointed)
    inv_outputs = checkpoint(
        lambda x: checkpointed_mask_forward(x, hooks, apply_inverse=True),
        torch.cat([peturb_x, x_batch], dim=0),
        use_reentrant=False,
    )

    pred_peturb_inv, pred_clean_inv = torch.chunk(inv_outputs, 2)

    # Regular masked forward (checkpointed)
    masked_outputs = checkpoint(
        lambda x: checkpointed_mask_forward(x, hooks, apply_inverse=False),
        torch.cat([peturb_x, x_batch], dim=0),
        use_reentrant=False,
    )
    pred_peturb_masked, pred_clean_masked = torch.chunk(masked_outputs, 2)

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

        # Find the peturb
        start_time = time.time()
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

        end_time = time.time()
        logging.info(f"Time taken to find peturb: {end_time - start_time:.4f} seconds")

        round_loss_inner.append((peturb_agree_loss[-1], peturb_disagree_loss[-1]))

        start_time = time.time()
        agreement_loss, disagreement_loss, total_l1 = optimize_mask_initial(
            args,
            model_mask,
            model_ref,
            hooks,
            x_batch,
            device,
            mask_optimizer
        )
        end_time = time.time()
        logging.info(f"Time taken to optimize mask: {end_time - start_time:.4f} seconds")

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

        if round_i % 10 == 0:
            logging.info(f"Round {round_i} (INNER): Disagree Loss: {peturb_disagree_loss[-1]:.4f}, Agree Loss: {peturb_agree_loss[-1]:.4f}")
            logging.info(f"Round {round_i} (OUTER): Clean Mask Loss: {agreement_loss:.4f}, BD Mask Loss: {disagreement_loss:.4f}, L1 Loss: {total_l1:.4f}")
            logging.info(f"Round {round_i} (PREDICTIONS): Target Count: {percentage[0]:.4f}, Peturb Inv Agree: {percentage[1]:.4f}, Peturb Ref Disagree: {percentage[2]:.4f}, Peturb Mask Correct: {percentage[3]:.4f}, Clean Mask Correct: {percentage[4]:.4f}, Clean Ref Correct: {percentage[5]:.4f}")
            torch.cuda.empty_cache()

        del peturb, peturb_agree_loss, peturb_disagree_loss
        del x_batch, label_batch
        del agreement_loss, disagreement_loss, total_l1        

    # Save the loss plots
    plt.figure(figsize=(20, 3))
    plt.subplot(1, 4, 1)
    plt.plot([x[0] for x in round_loss_outer], label='Clean Mask')
    plt.plot([x[1] for x in round_loss_outer], label='BD Mask')
    plt.plot([x[2] for x in round_loss_outer], label='L1 Loss')
    plt.legend(loc='upper left')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title("Outer Loss")

    plt.subplot(1, 4, 2)
    plt.plot([x[0] for x in round_loss_inner], label='Agree')
    plt.plot([x[1] for x in round_loss_inner], label='Disagree')
    plt.legend(loc='upper left')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title("Inner Loss")

    plt.subplot(1, 4, 3)
    plt.plot([x[0] for x in round_predictions], label='Target Count')
    plt.plot([x[1] for x in round_predictions], label='Peturb Inv Agree')
    plt.plot([x[2] for x in round_predictions], label='Peturb Ref Disagree')
    plt.plot([x[3] for x in round_predictions], label='Peturb Mask Correct')
    plt.plot([x[4] for x in round_predictions], label='Clean Mask Correct')
    plt.plot([x[5] for x in round_predictions], label='Clean Ref Correct')
    plt.legend(loc='upper left')
    plt.ylim(0, 105)
    plt.xlabel('Round')
    plt.ylabel('Percentage')
    plt.title("Peturb Distribution")    

    plt.subplot(1, 4, 4)
    sum_outer_loss = np.array([x[0] + x[1] + x[2] for x in round_loss_outer])
    plt.plot(sum_outer_loss, label='Total Loss')
    plt.legend(loc='upper left')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title("Total Loss | Patience: " + str(current_patience))
    plt.savefig(args.defense_save_path + '/mask_initial_loss.png')
    plt.close()

    # Save a csv file of the loss
    inner_loss_df = pd.DataFrame(round_loss_inner, columns=['Peturb Agree Loss', 'Peturb Disagree Loss'])
    outer_loss_df = pd.DataFrame(round_loss_outer, columns=['Clean Mask Loss', 'BD Mask Loss', 'L1 Loss'])
    predictions_df = pd.DataFrame(round_predictions, columns=['Target Count', 'Peturb Inv Agree', 'Peturb Ref Disagree', 'Peturb Mask Correct', 'Clean Mask Correct', 'Clean Ref Correct'])
    inner_loss_df.to_csv(args.defense_save_path + '/initial_inner_loss.csv', index=False)
    outer_loss_df.to_csv(args.defense_save_path + '/initial_outer_loss.csv', index=False)
    predictions_df.to_csv(args.defense_save_path + '/initial_predictions.csv', index=False)

    plot_mask(hooks, args.defense_save_path + '/mask_initial.png')

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

        # Find the peturb
        start_time = time.time()
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
        end_time = time.time()

        logging.info(f"(MIN) Time taken to find peturb: {end_time - start_time:.4f} seconds")

        round_loss_inner.append((peturb_agree_loss[-1], peturb_disagree_loss[-1]))

        start_time = time.time()
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
        end_time = time.time()
        logging.info(f"(MIN) Time taken to optimize mask: {end_time - start_time:.4f} seconds")

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

        if round_i % 10 == 0:
            logging.info(f"Round {round_i} (INNER): Disagree Loss: {peturb_disagree_loss[-1]:.4f}, Agree Loss: {peturb_agree_loss[-1]:.4f}")
            logging.info(f"Round {round_i} (OUTER): Clean Mask Loss: {clean_mask_loss:.4f}, BD Mask Loss: {bd_mask_loss:.4f}, Agree Loss: {agreement_loss:.4f}, Disagree Clean Loss: {disagree_peturb_loss:.4f}, Disagree BD Loss: {disagreement_clean_loss:.4f}, L1 Loss: {total_l1:.4f}")
            logging.info(f"Round {round_i} (PREDICTIONS): Target Count: {percentage[0]:.4f}, Peturb Inv Agree: {percentage[1]:.4f}, Peturb Ref Disagree: {percentage[2]:.4f}, Peturb Mask Correct: {percentage[3]:.4f}, Clean Mask Correct: {percentage[4]:.4f}, Clean Ref Correct: {percentage[5]:.4f}")
            torch.cuda.empty_cache()

        del peturb, peturb_agree_loss, peturb_disagree_loss
        del x_batch, label_batch
        del clean_mask_loss, bd_mask_loss, agreement_loss, disagree_peturb_loss, disagreement_clean_loss, total_l1

    # Save the loss plots
    plt.figure(figsize=(20, 3))
    plt.subplot(1, 4, 1)
    plt.plot([x[0] for x in round_loss_outer], label='Clean Mask')
    plt.plot([x[1] for x in round_loss_outer], label='BD Mask')
    plt.plot([x[2] for x in round_loss_outer], label='Agree')
    plt.plot([x[3] for x in round_loss_outer], label='Disagree Clean')
    plt.plot([x[4] for x in round_loss_outer], label='Disagree BD')
    plt.plot([x[5] for x in round_loss_outer], label='L1 Loss')
    plt.legend(loc='upper left')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title("Outer Loss")

    plt.subplot(1, 4, 2)
    plt.plot([x[0] for x in round_loss_inner], label='Agree')
    plt.plot([x[1] for x in round_loss_inner], label='Disagree')
    plt.legend(loc='upper left')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title("Inner Loss")

    plt.subplot(1, 4, 3)
    plt.plot([x[0] for x in round_predictions], label='Target')
    plt.plot([x[1] for x in round_predictions], label='Inv Agree')
    plt.plot([x[2] for x in round_predictions], label='Ref Disagree')
    plt.plot([x[3] for x in round_predictions], label='BD Mask')
    plt.plot([x[4] for x in round_predictions], label='C Mask')
    plt.plot([x[5] for x in round_predictions], label='C Ref')
    plt.legend(loc='upper left')
    plt.ylim(0, 105)
    plt.xlabel('Round')
    plt.ylabel('Percentage')
    plt.title("Peturb Distribution")    

    plt.subplot(1, 4, 4)
    sum_outer_loss = np.array([x[0] + x[1] + x[2] + x[3] + x[4] + x[5] for x in round_loss_outer])
    plt.plot(sum_outer_loss, label='Total Loss')
    plt.legend(loc='upper left')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title("Total Loss | Patience: " + str(current_patience))
    
    if with_l1:
        plt.savefig(args.defense_save_path + '/mask_minimize_loss.png')
    else:
        plt.savefig(args.defense_save_path + '/mask_minimize_loss_without_l1.png')
    plt.close()

    # Save a csv file of the loss
    inner_loss_df = pd.DataFrame(round_loss_inner, columns=['Peturb Agree Loss', 'Peturb Disagree Loss'])
    outer_loss_df = pd.DataFrame(round_loss_outer, columns=['Clean Mask Loss', 'BD Mask Loss', 'Agree Loss', 'Disagree Clean Loss', 'Disagree BD Loss', 'L1 Loss'])
    predictions_df = pd.DataFrame(round_predictions, columns=['Target Count', 'Peturb Inv Agree', 'Peturb Ref Disagree', 'Peturb Mask Correct', 'Clean Mask Correct', 'Clean Ref Correct'])
    
    if with_l1:
        inner_loss_df.to_csv(args.defense_save_path + '/minimize_inner_loss.csv', index=False)
        outer_loss_df.to_csv(args.defense_save_path + '/minimize_outer_loss.csv', index=False)
        predictions_df.to_csv(args.defense_save_path + '/minimize_predictions.csv', index=False)
        plot_mask(hooks, args.defense_save_path + '/mask_minimize.png')
    else:
        inner_loss_df.to_csv(args.defense_save_path + '/minimize_inner_loss_without_l1.csv', index=False)
        outer_loss_df.to_csv(args.defense_save_path + '/minimize_outer_loss_without_l1.csv', index=False)
        predictions_df.to_csv(args.defense_save_path + '/minimize_predictions_without_l1.csv', index=False)
        plot_mask(hooks, args.defense_save_path + '/mask_minimize_without_l1.png')
        args.l1_weight = saved_l1_weight

# ---------------------------- IMP Defense Main Class ----------------------------

class IMP(defense):

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

        parser.add_argument('--yaml_path', type=str, default="./config/defense/imp/default.yaml", help='the yaml path for the defense')

        # set the parameter for the mmdf defense
        parser.add_argument('--spc', type=int, help='the samples per class used for training')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        
        # IMP defense parameters
        parser.add_argument('--k', type=int, help='the k value for the IMP defense')

        parser.add_argument('--l1_weight', type=float, help='the weight for the l1 loss')
        parser.add_argument('--lr_mask', type=float, help='the learning rate for the mask')
        parser.add_argument('--lr_peturb', type=float, help='the learning rate for the peturb')
        parser.add_argument('--num_steps_peturb', type=int, help='the number of steps for the peturb')
        parser.add_argument('--max_initial_rounds', type=int, help='the max rounds for the initial training')
        parser.add_argument('--max_minimize_rounds', type=int, help='the max rounds for the minimize training')

        parser.add_argument('--norm_bound', type=float, help='the norm value for the peturb IMP defense')
        
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
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "ims_vit" + os.path.sep + f'spc_{args.spc}' + os.path.sep + str(args.random_seed)
        else:
            defense_save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "ims_vit" + os.path.sep + f'ratio_{args.ratio}' + os.path.sep + str(args.random_seed)
        
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

        ##############################################
        
        model_reference = generate_cls_model(args.model, args.num_classes)
        model_reference.load_state_dict(model.state_dict())
        model_reference.to(args.device)
        
        model_mask = generate_cls_model(args.model, args.num_classes)
        model_mask.load_state_dict(model.state_dict())
        model_mask.to(args.device)
        
        dataset_norm = get_dataset_normalization(args.dataset)
        dataset_denorm = get_dataset_denormalization(dataset_norm)
        
        # Setup the hooks
        mha_modules = get_encoder_blocks(model_mask)
        hooks = setup_block_output_hooks(
            mha_modules,
            args.device
        )

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

        # Get allocated memory in GB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"Allocated memory: {allocated_memory:.2f} GB")
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"Allocated memory after emptying cache: {allocated_memory:.2f} GB")

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

        # Get allocated memory in GB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"Allocated memory: {allocated_memory:.2f} GB")
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"Allocated memory after emptying cache: {allocated_memory:.2f} GB")

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

        # Get allocated memory in GB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"Allocated memory: {allocated_memory:.2f} GB")
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"Allocated memory after emptying cache: {allocated_memory:.2f} GB")

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

        # Get allocated memory in GB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"Allocated memory: {allocated_memory:.2f} GB")
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"Allocated memory after emptying cache: {allocated_memory:.2f} GB")

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

        # Set the model to evaluation mode and set the mask to apply
        set_model_eval(model_mask)
        set_mask(hooks, apply_inverse=False)

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
    imp = IMS()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = imp.set_args(parser)
    args = parser.parse_args()
    imp.add_yaml_to_args(args)
    args = imp.process_args(args)
    imp.prepare(args)
    imp.defense()