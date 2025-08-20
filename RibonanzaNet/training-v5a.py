## Owned by Ryan Mehra. Licensed for free use. 
# Purpose: Main training script for RibonanzaNet v5a, combining Transformer and EGNN for RNA 3D structure prediction.
## Custom Architecure 

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm
import os
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

config = {
    "seed": 0,
    "cutoff_date": "2020-01-01", # Keep or adjust based on new data splits if needed
    "test_cutoff_date": "2022-05-01", # Keep or adjust based on new data splits if needed
    "max_len": 1024, # Increased max_len (adjust based on H100 memory)
    "batch_size": 4, # Increased batch_size (adjust based on H100 memory)
    "learning_rate": 1e-4, # May need tuning for larger batch size/dataset
    "weight_decay": 0.0,
    "mixed_precision": "bf16", # Suitable for H100
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 50, # Increased epochs for larger dataset
    "cos_epoch": 40, # Adjusted cosine annealing start epoch
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1, # Consider adjusting if gradients explode/vanish
    "gradient_accumulation_steps": 1, # Increase if batch_size needs to be effectively larger
    "d_clamp": 30,
    "max_len_filter": 9999999, # Keep filtering logic for now
    "min_len_filter": 10,
    "structural_violation_epoch": 50, # Re-evaluate if needed
    "balance_weight": False,
}

# # III. Data Prepration

# Load data

# --- TODO: Update these paths to your new, larger datasets ---
train_sequences=pd.read_csv("./kaggle-data/new_training_sequences.csv") # Your new 90% training split
train_labels=pd.read_csv("./kaggle-data/new_training_labels.csv") # Ensure this corresponds to the new train_sequences

validation_sequences=pd.read_csv("./kaggle-data/new_validation_sequences.csv") # Your new 10% validation split
validation_labels=pd.read_csv("./kaggle-data/new_validation_labels.csv") # Ensure this corresponds to the new validation_sequences

test_sequences=pd.read_csv("./kaggle-data/test_sequences.csv") # Keep original test set path
# --- End of path updates ---

# Subset to a single sample for quick end-to-end testing
# sample_tid = train_sequences['target_id'].iloc[0]
# train_sequences = train_sequences[train_sequences['target_id'] == sample_tid].reset_index(drop=True)
# train_labels    = train_labels[train_labels['target_id']   == sample_tid].reset_index(drop=True)

# val_tid = validation_sequences['target_id'].iloc[0]
# validation_sequences = validation_sequences[validation_sequences['target_id'] == val_tid].reset_index(drop=True)
# validation_labels    = validation_labels[validation_labels['target_id']   == val_tid].reset_index(drop=True)

print("\Shapes", train_sequences.shape, train_labels.shape, validation_sequences.shape, validation_labels.shape, test_sequences.shape)


## Build all coordinates as list per target_id

# build dicts of raw xyz arrays per target_id
train_xyz_dict = {
    tid: grp[['x_1','y_1','z_1']].to_numpy(dtype='float32')
    for tid, grp in train_labels.groupby('target_id')
}
val_xyz_dict = {
    tid: grp[['x_1','y_1','z_1']].to_numpy(dtype='float32')
    for tid, grp in validation_labels.groupby('target_id')
}

# process sequences in one pass each
all_xyz_coord_trng = []
for tid in tqdm(train_sequences['target_id'], desc='proc train'):
    xyz = train_xyz_dict[tid]
    mask = (xyz < -1e17) & np.isfinite(xyz)
    xyz[mask] = np.nan
    all_xyz_coord_trng.append(xyz)

all_xyz_coord_val = []
for tid in tqdm(validation_sequences['target_id'], desc='proc val'):
    xyz = val_xyz_dict[tid]
    mask = (xyz < -1e17) & np.isfinite(xyz)
    xyz[mask] = np.nan
    all_xyz_coord_val.append(xyz)


print("\nLength of processed coords", len(all_xyz_coord_trng), len(all_xyz_coord_val))


"""
Filter and process data
	•	finds and prints the maximum coordinate-sequence length.
	•	keeps only those RNAs whose coordinate arrays have
    	1.	≤ 50% missing values,
    	2.	length within your configured min(10), max(9999) bounds.
	•	It then filters your sequence labels and coordinate data down to that clean subset.
"""

#### Process is required for only Training Data, expected to have clean Validaton Data

# initialize stats
lengths = [len(xyz) for xyz in all_xyz_coord_trng]
max_len = max(lengths)
min_len = min(lengths)
total = len(all_xyz_coord_trng)

# build filter mask
filter_mask = []
for xyz in all_xyz_coord_trng:
    frac_nan = np.isnan(xyz).mean()
    seq_len = len(xyz)
    keep = (
        (frac_nan <= 0.5) and
        (seq_len < config['max_len_filter']) and
        (seq_len > config['min_len_filter'])
    )
    filter_mask.append(keep)

filter_mask = np.array(filter_mask)
kept_indices = np.nonzero(filter_mask)[0]
dropped = total - len(kept_indices)

# apply filter
train_sequences = train_sequences.loc[kept_indices].reset_index(drop=True)
all_xyz_coord_trng = [all_xyz_coord_trng[i] for i in kept_indices]

# Check max length AFTER filtering
filtered_lengths = [len(xyz) for xyz in all_xyz_coord_trng]
if filtered_lengths:
    max_len_after_filter = max(filtered_lengths)
    min_len_after_filter = min(filtered_lengths)
else:
    max_len_after_filter = 0
    min_len_after_filter = 0

# print stats
print(f"Total sequences initially : {total}")
print(f" Kept                    : {len(kept_indices)}")
print(f" Dropped                 : {dropped}")
print(f"Shortest sequence length (original) : {min_len}")
print(f"Longest sequence length (original)  : {max_len}")
print(f"Shortest sequence length (kept)     : {min_len_after_filter}")
print(f"Longest sequence length (kept)      : {max_len_after_filter}")
print(f"Configured max_len for cropping     : {config['max_len']}")


#pack data into a dictionary

training_data={
      "sequence":train_sequences['sequence'].to_list(),
      "xyz": all_xyz_coord_trng
}

validation_data={
      "sequence":validation_sequences['sequence'].to_list(),
      "xyz": all_xyz_coord_val
}


print(next(iter(training_data['sequence'])), next(iter(training_data['xyz'])))


print(next(iter(validation_data['sequence'])), next(iter(validation_data['xyz'])))


# # IV. Training Data Prepration


## No need to split from the training set, as we have validaton set 
# all_index = np.arange(len(data['sequence']))
# cutoff_date = pd.Timestamp(config['cutoff_date'])
# test_cutoff_date = pd.Timestamp(config['test_cutoff_date'])
# train_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) <= cutoff_date]
# test_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(d) > cutoff_date and pd.Timestamp(d) <= test_cutoff_date]


# print(f"Train size: {len(train_index)}")
# print(f"Test size: {len(test_index)}")

print(f"Train size: {len(training_data['sequence'])}")
print(f"Validation size: {len(validation_data['sequence'])}")


# **Pytorch Dataset**


from torch.utils.data import Dataset, DataLoader
from ast import literal_eval

def get_ct(bp,s):
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0]-1,b[1]-1]=1
    return ct_matrix


class RNA3D_Dataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, config: dict):
        """
        data: dict of lists, keys include:
              'sequence' (list of str), 'xyz' (list of Nx3 arrays), etc.
        config: dict with at least 'max_len' key
        """
        self.data   = data
        self.config = config

        # build token map for known nucleotides
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}
        # assign an ID for unknown tokens
        self.UNK_ID = len(self.tokens)

    def __len__(self):
        return len(self.data['sequence'])
    
    def __getitem__(self, idx):
        # --- sequence to IDs, unknown → UNK_ID ---
        seq_str = self.data['sequence'][idx]
        seq_ids = [ self.tokens.get(nt, self.UNK_ID) for nt in seq_str ]
        sequence = torch.tensor(seq_ids, dtype=torch.long)

        # --- xyz list → tensor ---
        xyz_arr = np.array(self.data['xyz'][idx], dtype=np.float32)
        xyz     = torch.tensor(xyz_arr,   dtype=torch.float32)

        # --- optional random crop if too long ---
        max_len = self.config['max_len']
        if len(sequence) > max_len:
            start = np.random.randint(0, len(sequence) - max_len + 1)
            end   = start + max_len
            sequence = sequence[start:end]
            xyz       = xyz[start:end]
        
        return {
            'sequence': sequence,
            'xyz':       xyz
        }

train_dataset=RNA3D_Dataset(training_data, config)
val_dataset=RNA3D_Dataset(validation_data, config)


print(train_dataset.__getitem__(0), val_dataset.__getitem__(0))



# V. Create Custom Model Instance

#We will add a linear layer to predict xyz of C1' atoms on the base /kaggle/input/ribonanzanet2d-final 


import sys

sys.path.append("./ribonanzanet2d-final")

from Network import *
import yaml



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)



class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout=0.1
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load("./ribonanzanet-weights/RibonanzaNet.pt",map_location='cpu'))
        self.dropout=nn.Dropout(0.0)
        self.xyz_predictor=nn.Linear(256,3)


    
    def forward(self,src):
        
        #with torch.no_grad():
        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))


        xyz=self.xyz_predictor(sequence_features)

        return xyz


## Available GPUs 
print("GPUs available:", torch.cuda.device_count())


from pprint import pprint
cfg = load_config_from_yaml("./ribonanzanet2d-final/configs/pairwise.yaml")

## Update the batch size to new value
_batch_size = config['batch_size'] # Use value from config

cfg.batch_size = _batch_size
cfg.entries['batch_size'] = _batch_size

## Update the GPUs to multiple if multiple available 
if torch.cuda.device_count() > 1:
    cfg.gpu_id = "0,1"
    cfg.entries['gpu_id'] = "0,1"
    
pprint(vars(cfg))


## Create dataloader instances 

import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    # batch is a list of dicts, e.g. {'sequence': Tensor[L], 'xyz': Tensor[L,3], …}
    seqs = [torch.tensor(item['sequence']) for item in batch]
    xyzs = [torch.tensor(item['xyz'], dtype=torch.float32) for item in batch]

    # pad to the max length in this batch
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)       # or pad_token
    xyzs_padded = pad_sequence(xyzs, batch_first=True, padding_value=float('nan'))

    return {
        'sequence': seqs_padded,
        'xyz':       xyzs_padded,
    }




# then in your DataLoader:
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size, # Use config value
    shuffle=True,
    num_workers=4, # Increase num_workers if I/O is a bottleneck (e.g., 4, 8)
    pin_memory=True,
    collate_fn=pad_collate
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size, # Use config value
    shuffle=False,
    num_workers=4, # Increase num_workers if I/O is a bottleneck
    pin_memory=True,
    collate_fn=pad_collate
)

# instantiate on CPU first
model = finetuned_RibonanzaNet(cfg, pretrained=True)

# wrap in DataParallel (uses all available GPUs by default)
model = torch.nn.DataParallel(model)

# then move to CUDA
model = model.cuda()

# after wrapping in DataParallel
# print("Model sees config:", model.module.cfg.batch_size, model.module.cfg.gpu_id)

print("GPUs visible:", torch.cuda.device_count())

print("DataParallel device IDs:", model.device_ids)
print("First parameter on device:", next(model.parameters()).device)


# **Define Loss Function**
# 
# we will use dRMSD loss on the predicted xyz. the loss function is invariant to translations, rotations, and reflections. because dRMSD is invariant to reflections, it cannot distinguish chiral structures, so there may be better loss functions


def calculate_distance_matrix(X,Y,epsilon=1e-4):
    return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()


def dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    if d_clamp is not None:
        rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
    else:
        rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

    return rmsd.sqrt().mean()/Z

def local_dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=30):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=(~torch.isnan(gt_dm))*(gt_dm<d_clamp)
    mask[torch.eye(mask.shape[0]).bool()]=False



    rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon
    # rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).sqrt()/Z
    #rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])/Z
    return rmsd.sqrt().mean()/Z

def dRMAE(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=None):
    # align lengths
    if pred_x.shape[0] != gt_x.shape[0]:
        k = min(pred_x.shape[0], gt_x.shape[0])
        pred_x = pred_x[:k]
        pred_y = pred_y[:k]
        gt_x   = gt_x[:k]
        gt_y   = gt_y[:k]

    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)

    mask=~torch.isnan(gt_dm)
    mask[torch.eye(mask.shape[0]).bool()]=False

    rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])

    return rmsd.mean()/Z

import torch

def align_svd_mae(input, target, Z=10):
    """
    Align input (Nx3) to target (Nx3) via Procrustes (SVD) in float32,
    then compute MAE / Z.
    """
    # if lengths differ, trim to the minimum
    if input.shape[0] != target.shape[0]:
        k = min(input.shape[0], target.shape[0])
        input  = input[:k]
        target = target[:k]

    # 1) Mask out NaNs
    mask = ~torch.isnan(target.sum(-1))
    inp = input[mask].float()   # cast to float32
    tgt = target[mask].float()  # cast to float32

    # 2) Compute and remove centroids
    c_inp = inp.mean(dim=0, keepdim=True)
    c_tgt = tgt.mean(dim=0, keepdim=True)
    inp_c = inp - c_inp
    tgt_c = tgt - c_tgt

    # 3) Covariance matrix
    cov = inp_c.t() @ tgt_c

    # 4) SVD in float32
    #    Detach so no gradients flow through the SVD
    with torch.no_grad():
        U, S, Vt = torch.svd(cov)
        R = Vt @ U.t()
        # fix potential reflection
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt @ U.t()

    # 5) Rotate back and re-add centroid
    #    (R is already float32, inp_c is float32)
    aligned = inp_c @ R.t() + c_tgt

    # 6) MAE loss (float32)
    loss = torch.abs(aligned - tgt).mean() / Z

    return loss

def compute_rmsd(pred, true, eps=1e-6):
    # mask out any NaN rows
    mask = ~torch.isnan(true).any(-1)
    P = pred[mask].float()
    T = true[mask].float()
    if P.numel() == 0:
        return float('nan')
    # center
    mu_P = P.mean(0, keepdim=True)
    mu_T = T.mean(0, keepdim=True)
    P0 = P - mu_P
    T0 = T - mu_T
    # covariance
    C = P0.t() @ T0
    U, S, Vt = torch.svd(C)
    R = Vt @ U.t()
    if torch.det(R) < 0:
        Vt = Vt.clone(); Vt[-1] *= -1
        R = Vt @ U.t()
    # align and compute RMSD
    P_aligned = P0 @ R.t() + mu_T
    diff2 = (P_aligned - T).pow(2).sum(dim=-1)
    return torch.sqrt(diff2.mean() + eps).item()

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Initialize scores log for metrics
scores_log = open('training-v5a-scores.log', 'w')
scores_log.write('type,step_or_epoch,dRMSD,dRMAE,RMSD\n')

def evaluate_metrics():
    model.eval()
    total_dRMSD = total_dRMAE = total_RMSD = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            seq = batch['sequence'].cuda(non_blocking=True)
            gt  = batch['xyz'].cuda(non_blocking=True)
            if gt.ndim == 2: gt = gt.unsqueeze(0)
            if seq.ndim == 1: seq = seq.unsqueeze(0)
            with autocast(dtype=torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float16):
                pred = model(seq)
            for i in range(pred.shape[0]):
                L = min(pred[i].shape[0], gt[i].shape[0])
                pred_i = pred[i, :L, :]
                gt_i   = gt[i, :L, :]
                nan_mask = ~torch.any(torch.isnan(gt_i), dim=-1)
                pred_i = pred_i[nan_mask]
                gt_i   = gt_i[nan_mask]
                if pred_i.shape[0] < 2: continue
                drmsd = dRMSD(pred_i, pred_i, gt_i, gt_i).item()
                drmae = dRMAE(pred_i, pred_i, gt_i, gt_i).item()
                rmsd_score = compute_rmsd(pred_i, gt_i)
                total_dRMSD += drmsd
                total_dRMAE += drmae
                total_RMSD += rmsd_score
                count += 1
    if count > 0:
        return total_dRMSD / count, total_dRMAE / count, total_RMSD / count
    else:
        return float('nan'), float('nan'), float('nan')

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.learning_rate, # Use config value
    weight_decay=cfg.weight_decay # Use config value
)


epochs    = config['epochs'] # Use config value
cos_epoch = config['cos_epoch'] # Use config value
T_max_val = (epochs - cos_epoch) * (len(train_loader) // config['gradient_accumulation_steps'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, T_max_val)
)
scaler = GradScaler(enabled=(config['mixed_precision'] == 'fp16'))

# ---- TRAIN & VALIDATION LOOP ----
best_val_loss = float('inf')

# start log file
log_path = 'training_log.csv'
log_f = open(log_path, 'w')
log_f.write('epoch,train_loss,val_loss,lr\n')

for epoch in range(1, epochs + 1):
    # TRAINING
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    running_loss = 0.0
    accum_steps = config['gradient_accumulation_steps']

    for idx, batch in enumerate(train_bar, start=1):
        seq = batch['sequence'].cuda(non_blocking=True)
        gt  = batch['xyz'].cuda(non_blocking=True) # Keep batch dimension

        # Ensure gt has the correct shape (B, L, 3) even if batch size is 1
        if gt.ndim == 2:
             gt = gt.unsqueeze(0)
        if seq.ndim == 1:
             seq = seq.unsqueeze(0)

        # Use autocast with the configured precision
        with autocast(dtype=torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float16):
            pred = model(seq) # Keep batch dimension (B, L, 3)
            # --- Loss Calculation per Sample in Batch ---
            batch_dR_loss = 0.0
            batch_rot_loss = 0.0
            actual_batch_size = pred.shape[0] # Handle last partial batch
            valid_samples = 0
            for i in range(actual_batch_size):
                # --- ensure equal length first ---
                L = min(pred[i].shape[0], gt[i].shape[0])
                pred_i = pred[i, :L, :]
                gt_i   = gt[i, :L, :]
                # 1) drop padded/missing rows
                mask_valid = ~torch.isnan(gt_i[:, 0])
                pred_i = pred_i[mask_valid]
                gt_i   = gt_i[mask_valid]
                if pred_i.shape[0] < 2: continue # Need at least 2 points for distance matrix

                # Calculate losses for this sample
                # dRMAE expects (N, 3) inputs
                sample_dR_loss = dRMAE(pred_i, pred_i, gt_i, gt_i)

                # Alignment loss needs fp32 for SVD
                with autocast(enabled=False):
                    pred_i_fp32 = pred_i.float()
                    gt_i_fp32 = gt_i.float()
                    sample_rot_loss = align_svd_mae(pred_i_fp32, gt_i_fp32)

                batch_dR_loss += sample_dR_loss
                batch_rot_loss += sample_rot_loss
                valid_samples += 1

            if valid_samples == 0: continue # Skip batch if no valid samples

            # Average loss over valid samples in the batch
            dR_loss = batch_dR_loss / valid_samples
            rot_loss = batch_rot_loss / valid_samples
            # --- End of Per Sample Loss ---

            loss = (dR_loss + rot_loss) / accum_steps # Normalize loss for accumulation

        # Accumulate gradients
        if config['mixed_precision'] == 'fp16':
             scaler.scale(loss).backward()
        else: # For bf16 or fp32
             loss.backward()

        # Optimizer step
        if (idx % accum_steps == 0) or (idx == len(train_loader)):
            if config['mixed_precision'] == 'fp16':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else: # For bf16 or fp32
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)


        # Use item() for logging scalar loss
        running_loss += (dR_loss.item() + rot_loss.item()) # Log non-normalized loss
        if idx % 10 == 0: # Log every 10 steps
             current_avg_loss = running_loss / idx
             train_bar.set_postfix(loss=current_avg_loss)

        # Evaluate metrics every 500 steps
        if idx % 500 == 0:
            dRMSD_score, dRMAE_score, RMSD_score = evaluate_metrics()
            scores_log.write(f"step,{idx},{dRMSD_score:.6f},{dRMAE_score:.6f},{RMSD_score:.6f}\n")


    # LR SCHEDULER STEP (after epoch)
    if epoch >= cos_epoch: # Start stepping scheduler *after* cos_epoch
        scheduler.step()

    # VALIDATION
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc="Validation", unit="batch")
    with torch.no_grad():
        for batch in val_bar:
            seq = batch['sequence'].cuda(non_blocking=True)
            gt  = batch['xyz'].cuda(non_blocking=True) # Keep batch dim (B, L, 3)

            if gt.ndim == 2: gt = gt.unsqueeze(0)
            if seq.ndim == 1: seq = seq.unsqueeze(0)

            # Use autocast consistent with training, but no gradient needed
            with autocast(dtype=torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float16):
                pred = model(seq) # (B, L, 3)

                # --- Loss Calculation per Sample in Batch ---
                batch_vloss = 0.0
                actual_batch_size = pred.shape[0]
                valid_samples = 0
                for i in range(actual_batch_size):
                    # equalize lengths
                    L = min(pred[i].shape[0], gt[i].shape[0])
                    pred_i = pred[i, :L, :]
                    gt_i   = gt[i, :L, :]
                    if L == 0: continue

                    # mask out any row containing NaN in gt_i
                    nan_mask = ~torch.any(torch.isnan(gt_i), dim=-1)
                    pred_i = pred_i[nan_mask]
                    gt_i   = gt_i[nan_mask]
                    if pred_i.shape[0] < 2: continue

                    # Calculate dRMAE only for validation metric
                    sample_vloss = dRMAE(pred_i, pred_i, gt_i, gt_i)

                    batch_vloss += sample_vloss
                    valid_samples += 1

                if valid_samples > 0:
                    val_loss += (batch_vloss / valid_samples).item() # Accumulate average loss for the batch
                # --- End of Per Sample Loss ---


    # Average validation loss over all batches
    val_loss /= len(val_loader)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} Validation Loss: {val_loss:.4f} LR: {current_lr:.2e}")

    # compute average training loss
    train_avg = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    # record metrics
    log_f.write(f"{epoch},{train_avg:.6f},{val_loss:.6f},{current_lr:.2e}\n")

    # Evaluate metrics after each epoch
    dRMSD_score, dRMAE_score, RMSD_score = evaluate_metrics()
    scores_log.write(f"epoch,{epoch},{dRMSD_score:.6f},{dRMAE_score:.6f},{RMSD_score:.6f}\n")

    # SAVE BEST MODEL
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(save_state, 'RibonanzaNet-best-kg.pt')
        print(f"  ✨ Saved new best model (val_loss={val_loss:.4f})")

# close log file
log_f.close()
scores_log.close()

# FINAL SAVE
final_save_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
torch.save(final_save_state, 'RibonanzaNet-final-kg.pt')
print("Training complete. Final model saved.")





