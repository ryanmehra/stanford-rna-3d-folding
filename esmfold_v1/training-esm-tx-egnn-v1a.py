## Owned by Ryan Mehra. Licensed for free use. 
# Purpose: Main training script for ESMFold v1, combining RoPE-enhanced Transformer encoder and EGNN refinement for RNA 3D structure prediction.
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
from transformers import AutoTokenizer, EsmForProteinFolding, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from accelerate.test_utils.testing import get_backend  # for device backend
from egnn_pytorch import EGNN #, RotaryEmbedding  # Add EGNN and RotaryEmbedding imports

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)  # catch NaNs/infs during backward

config = {
    "seed": 0,
    "cutoff_date": "2020-01-01", # Keep or adjust based on new data splits if needed
    "test_cutoff_date": "2022-05-01", # Keep or adjust based on new data splits if needed
    "max_len": 1024, # Increased max_len (adjust based on H100 memory)
    "batch_size": 2, # Increased batch size (requires mixed precision & checkpointing)
    "learning_rate": 5e-5,  # Raised LR for stronger updates
    "weight_decay": 1e-2,   # Add weight decay to regularize
    "dropout": 0.2,         # Dropout for transformer head
    "mixed_precision": "fp32", # Revert to fp32 to avoid NaN errors
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 50, # Increased epochs for larger dataset
    "cos_epoch": 1,        # start cosine scheduler right away
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 1.0, # Loosen clipping to retain stronger updates
    "gradient_accumulation_steps": 1, # Increase if batch_size needs to be effectively larger
    "d_clamp": 30,
    "max_len_filter": 9999999, # Keep filtering logic for now
    "min_len_filter": 10,
    "structural_violation_epoch": 50, # Re-evaluate if needed
    "balance_weight": False,
    # weights for combining local and global losses and warmup schedule
    "alpha_loss": 0.1,
    "beta_loss": 1.0,
    # weight for local pairwise dRMSD loss
    "local_loss_weight": 0.1,
    # weight for alignment-based RMSD loss
    "align_loss_weight": 0.1,
    # weight for contrastive distance margins
    "contrastive_weight": 0.1,
    # margin for contrastive distance loss
    "contrastive_margin": 1.0,
    "warmup_epochs": 5,
    # Insert new Transformer & EGNN config parameters
    "egnn_layers": 8,
    "transformer_layers": 4,
    "transformer_heads": 16,
    "transformer_ff_dim": 512,
    "use_distance_bias": True,
    # unfreeze entire ESM trunk
    "train_trunk_layers": 4,  # 0 to unfreeze all trunk layers
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

# Replace U with T in sequences and resname fields to match ESM alphabet
# for df in [train_sequences, validation_sequences, test_sequences]:
#     df['sequence'] = df['sequence'].str.replace('U', 'T')
# for df in [train_labels, validation_labels]:
#     if 'resname' in df.columns:
#         df['resname'] = df['resname'].str.replace('U', 'T')

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


# print(next(iter(training_data['sequence'])), next(iter(training_data['xyz'])))


# print(next(iter(validation_data['sequence'])), next(iter(validation_data['xyz'])))


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
        # reserve 0 for padding; map nucleotides to IDs starting from 1
        self.tokens = {nt: i+1 for i, nt in enumerate('ACGU')}
        # unknown token ID follows mapped tokens
        self.UNK_ID = len(self.tokens) + 1
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


# print(train_dataset.__getitem__(0), val_dataset.__getitem__(0))

# Create DataLoaders for training and validation
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    seqs = [item['sequence'] for item in batch]
    xyzs = [item['xyz'] for item in batch]
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    xyzs_padded = pad_sequence(xyzs, batch_first=True, padding_value=float('nan'))
    pad_mask = seqs_padded == 0
    return {'sequence': seqs_padded, 'xyz': xyzs_padded, 'pad_mask': pad_mask}

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=pad_collate
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=pad_collate
)

# V. Create Custom Model Instance

# Instantiate ESM-based fine-tuner instead of RibonanzaNet
DEVICE, _, _ = get_backend()
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esm_full = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1", low_cpu_mem_usage=True
).to(DEVICE)
# enable gradient checkpointing for memory reduction
esm_full.gradient_checkpointing_enable()
# optional: use fp16 or bf16 for trunk if desired
if config['mixed_precision'] in ('bf16', 'fp16'):
   esm_full.esm = esm_full.esm.half()

# Add new token 'U' to tokenizer and extend model embeddings
num_added_tokens = tokenizer.add_tokens(['U'])
if num_added_tokens > 0:
    # Resize model embeddings to accommodate new token
    esm_full.resize_token_embeddings(len(tokenizer))
    # Initialize 'U' embedding by copying 'T' embedding
    t_id = tokenizer.convert_tokens_to_ids('T')
    u_id = tokenizer.convert_tokens_to_ids('U')
    with torch.no_grad():
        embedding = esm_full.get_input_embeddings()
        embedding.weight.data[u_id] = embedding.weight.data[t_id].clone()

# freeze all but last `train_trunk` folding blocks
train_trunk = config['train_trunk_layers']
if train_trunk > 0:
    total = len(esm_full.trunk.blocks)
    for i, block in enumerate(esm_full.trunk.blocks):
        if i < total - train_trunk:
            for p in block.parameters(): p.requires_grad = False
elif train_trunk == 0:
    # unfreeze all trunk layers
    pass  # default is all trainable

# === Custom Transformer + EGNN Head Definitions and Instantiation ===
class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads; self.dh = dim//heads; self.scale = self.dh**-0.5
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, pad_mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, self.heads, 3*self.dh)
        q,k,v = torch.split(qkv, self.dh, dim=-1)
        q = q.permute(0,2,1,3); k = k.permute(0,2,1,3); v = v.permute(0,2,1,3)
        attn = (q@k.transpose(-2,-1)) * self.scale
        if pad_mask is not None: attn = attn.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.softmax(attn, dim=-1); attn = self.attn_drop(attn)
        out = attn @ v; out = out.permute(0,2,1,3).reshape(B,L,D)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__(); self.ln1=nn.LayerNorm(dim); self.attn=SelfAttention(dim,heads,dropout)
        self.ln2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim,dim), nn.Dropout(dropout))
    def forward(self, x, pad_mask=None): return x + self.mlp(self.ln2(self.attn(self.ln1(x),pad_mask)))

class TransformerHead(nn.Module):
    def __init__(self, dim, n_layers, n_heads, ff_dim, dropout, use_bias):
        super().__init__(); self.layers=nn.ModuleList([TransformerBlock(dim,n_heads,ff_dim,dropout) for _ in range(n_layers)])
        self.ln=nn.LayerNorm(dim); self.predict=nn.Linear(dim,3)
    def forward(self, x, pad_mask=None):
        for l in self.layers: x = l(x, pad_mask)
        return self.predict(self.ln(x))

class TransformerWithEGNNHead(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__(); self.backbone=backbone
        dim=backbone.embeddings.word_embeddings.embedding_dim
        self.tr=TransformerHead(dim, cfg['transformer_layers'], cfg['transformer_heads'], cfg['transformer_ff_dim'], dropout=0.1, use_bias=cfg['use_distance_bias'])
        self.egnn_layers = nn.ModuleList([EGNN(dim,0,m_dim=cfg['transformer_ff_dim'],norm_feats=True,norm_coors=True,update_feats=True,update_coors=True,num_nearest_neighbors=8) for _ in range(cfg['egnn_layers'])]) if cfg['egnn_layers']>0 else None
    def forward(self, seq, coords=None, pad_mask=None):
        # build attention mask for backbone (True for actual tokens)
        attn_mask = None
        if pad_mask is not None:
            # pad_mask: True at padding positions
            attn_mask = (~pad_mask).long()
        # pass attention_mask to backbone to ignore padding
        out = self.backbone(seq, attention_mask=attn_mask)
        emb = out.last_hidden_state
        coords=self.tr(emb, pad_mask)
        if self.egnn_layers:
            feats=emb
            for l in self.egnn_layers: feats, coords = l(feats, coords)
        return coords

# Instantiate model with new head
model = TransformerWithEGNNHead(esm_full.esm, config)
model = torch.nn.DataParallel(model); model = model.to(DEVICE)
print('Using Transformer+EGNN head on', torch.cuda.device_count(), 'GPUs')


# **Define Loss Function**
# 
# we will use dRMSD loss on the predicted xyz. the loss function is invariant to translations, rotations, and reflections. because dRMSD is invariant to reflections, it cannot distinguish chiral structures, so there may be better loss functions


def calculate_distance_matrix(X, Y, epsilon=1e-4):
    """Compute pairwise distances with debug checks."""
    # squared distances plus epsilon
    d2 = torch.square(X[:, None] - Y[None, :]).sum(-1) + epsilon
    # debug: catch NaNs or Infs in squared distances
    if torch.isnan(d2).any() or torch.isinf(d2).any():
        print("[calculate_distance_matrix] NaN/Inf in d2:", d2)
        print("X:", X); print("Y:", Y)
        raise RuntimeError("NaN or Inf detected in distance computation")
    # clamp to avoid negative or zero
    d2 = torch.clamp(d2, min=epsilon)
    # compute distances
    return torch.sqrt(d2)

def dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4, Z=10, d_clamp=None):
    """
    Compute global dRMSD between pred and gt coords.
    """
    # align lengths
    if pred_x.shape[0] != gt_x.shape[0]:
        k = min(pred_x.shape[0], gt_x.shape[0])
        pred_x = pred_x[:k]; pred_y = pred_y[:k]
        gt_x   = gt_x[:k];   gt_y   = gt_y[:k]
    # distance matrices
    pred_dm = calculate_distance_matrix(pred_x, pred_y, epsilon)
    gt_dm   = calculate_distance_matrix(gt_x,   gt_y,   epsilon)
    # mask out NaNs and diagonal
    mask = ~torch.isnan(gt_dm)
    mask &= ~torch.eye(mask.shape[0], dtype=torch.bool, device=mask.device)
    # optional clamp on gt distances
    if d_clamp is not None:
        mask &= (gt_dm < d_clamp)
    # squared differences, clamp and sqrt
    diffsq = (pred_dm[mask] - gt_dm[mask]).square()
    diffsq = torch.clamp(diffsq + epsilon, min=epsilon)
    return torch.sqrt(diffsq).mean() / Z

def local_dRMSD(pred_x,
          pred_y,
          gt_x,
          gt_y,
          epsilon=1e-4,Z=10,d_clamp=30):
    pred_dm=calculate_distance_matrix(pred_x,pred_y)
    gt_dm=calculate_distance_matrix(gt_x,gt_y)



    mask=(~torch.isnan(gt_dm))*(gt_dm<d_clamp)
    mask[torch.eye(mask.shape[0]).bool()]=False



    # squared differences plus epsilon, clamp and sqrt
    raw = torch.square(pred_dm[mask] - gt_dm[mask]) + epsilon
    raw = torch.clamp(raw, min=epsilon)
    return torch.sqrt(raw).mean() / Z

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

import torch.nn.functional as F
from torch.optim import AdamW

def align_svd_mae(input, target, Z=10):
    """
    Align input (Nx3) to target (Nx3) via Procrustes SVD (differentiable), then compute RMSD/Z.
    """
    # Mask out NaNs
    mask = ~torch.isnan(target.sum(-1))
    inp = input[mask].float()
    tgt = target[mask].float()

    # Compute and remove centroids
    c_inp = inp.mean(dim=0, keepdim=True)
    c_tgt = tgt.mean(dim=0, keepdim=True)
    inp_c = inp - c_inp
    tgt_c = tgt - c_tgt

    # Covariance matrix
    cov = inp_c.t() @ tgt_c

    # Differentiable SVD
    U, S, Vt = torch.linalg.svd(cov)
    # ensure no in-place modification for reflection fix
    det_val = torch.det(Vt @ U.t())
    if det_val < 0:
        Vt = Vt.clone()
        Vt[-1, :] = -Vt[-1, :]
    R = Vt @ U.t()

    # Apply rotation and re-add centroid
    aligned = inp_c @ R.t() + c_tgt

    # Compute global RMSD loss
    diff2 = (aligned - tgt).pow(2).sum(dim=-1)
    # return mean squared RMSD (no sqrt, no Z-scaling) for stronger gradients
    return diff2.mean()

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
    val = diff2.mean() + eps
    val = torch.clamp(val, min=eps)
    return torch.sqrt(val).item()

# Utility: safe sqrt with clamping
def safe_sqrt(x, eps=1e-6):
    # clamp to avoid negatives and zeros
    return torch.sqrt(torch.clamp(x, min=eps))

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import math

# Initialize scores log for metrics (line-buffered)
scores_log = open('training-esm-v1a-scores.log', 'w', buffering=1)
scores_log.write('type,step_or_epoch,dRMSD,dRMAE,RMSD\n')

def evaluate_metrics():
    model.eval()
    total_dRMSD = total_dRAE = total_RMSD = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            seq = batch['sequence'].cuda(non_blocking=True)
            gt  = batch['xyz'].cuda(non_blocking=True)
            pad_mask = batch['pad_mask'].cuda(non_blocking=True)
            if gt.ndim == 2: gt = gt.unsqueeze(0)
            if seq.ndim == 1: seq = seq.unsqueeze(0)
            with autocast(dtype=torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float16):
                pred = model(seq, pad_mask=pad_mask)
            for i in range(pred.shape[0]):
                L = min(pred[i].shape[0], gt[i].shape[0])
                pred_i = pred[i, :L, :]
                gt_i   = gt[i, :L, :]
                # remove any rows where pred or gt have NaNs
                mask_valid = ~(torch.any(torch.isnan(gt_i), dim=-1) |
                               torch.any(torch.isnan(pred_i), dim=-1))
                pred_i = pred_i[mask_valid]
                gt_i   = gt_i[mask_valid]
                if pred_i.shape[0] < 2: continue
                drmsd = dRMSD(pred_i, pred_i, gt_i, gt_i).item()
                drmae = dRMAE(pred_i, pred_i, gt_i, gt_i).item()
                rmsd_score = compute_rmsd(pred_i, gt_i)
                total_dRMSD += drmsd
                total_dRAE += drmae
                total_RMSD += rmsd_score
                count += 1
    if count > 0:
        return total_dRMSD / count, total_dRAE / count, total_RMSD / count
    else:
        return float('nan'), float('nan'), float('nan')

# build parameter groups: lower LR for backbone, higher for head
backbone_params = [p for n,p in model.module.named_parameters() if n.startswith('module.backbone') and p.requires_grad]
head_params = [p for n,p in model.module.named_parameters() if not n.startswith('module.backbone')]
optimizer = AdamW([
    {'params': backbone_params, 'lr': config['learning_rate'] * 0.2},
    {'params': head_params,    'lr': config['learning_rate']}
], weight_decay=config['weight_decay'])
# cosine scheduler with warm-up
epochs = config['epochs']
total_steps = epochs * len(train_loader)
warmup_steps = config['warmup_epochs'] * len(train_loader)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
scaler = GradScaler(enabled=(config['mixed_precision'] == 'fp16'))

# Initialize train/val metrics log (line-buffered)
log_f = open('training-esm-tx-egnn-v1a-log.csv', 'w', buffering=1)
# log header: report train RMSD instead of raw MSE
log_f.write('epoch,train_rmsd,val_mse,val_rmsd,lr\n')
best_val_loss = float('inf')

# ---- TRAIN & VALIDATION LOOP ----
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
        pad_mask = batch['pad_mask'].cuda(non_blocking=True)

        # Ensure gt has the correct shape (B, L, 3) even if batch size is 1
        if gt.ndim == 2:
             gt = gt.unsqueeze(0)
        if seq.ndim == 1:
             seq = seq.unsqueeze(0)
        if pad_mask.ndim == 1:
             pad_mask = pad_mask.unsqueeze(0)

        # Use autocast with the configured precision
        if config['mixed_precision'] == 'fp32':
            pred = model(seq, pad_mask=pad_mask)
        else:
            with autocast(dtype=torch.bfloat16 if config['mixed_precision']=='bf16' else torch.float16):
                pred = model(seq, pad_mask=pad_mask)

        # --- Simplified Global Alignment Loss (SVD) ---
        batch_loss = 0.0
        valid = 0
        for i in range(pred.size(0)):
            L = min(pred[i].size(0), gt[i].size(0))
            p_i = pred[i, :L]
            g_i = gt[i, :L]
            mask = ~(torch.any(torch.isnan(g_i), -1) | torch.any(torch.isnan(p_i), -1))
            p_i, g_i = p_i[mask], g_i[mask]
            if p_i.size(0) < 2: continue
            # main alignment loss
            sample_loss = align_svd_mae(p_i.float(), g_i.float())
            batch_loss += sample_loss
            valid += 1
        if valid == 0: continue
        loss = (batch_loss / valid) / config['gradient_accumulation_steps']
         
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
            # step scheduler per batch
            scheduler.step()
 

         # Use item() for logging scalar loss
        running_loss += loss.item()
        if idx % 10 == 0: # Log every 10 steps
                # compute average MSE loss and convert to RMS for reporting
                current_avg_mse = running_loss / idx
                reported_loss = math.sqrt(current_avg_mse)
                train_bar.set_postfix(rmsd_loss=reported_loss)

        # Evaluate metrics every 500 steps
        if idx % 500 == 0:
            dRMSD_score, dRMAE_score, RMSD_score = evaluate_metrics()
            scores_log.write(f"step,{idx},{dRMSD_score:.6f},{dRMAE_score:.6f},{RMSD_score:.6f}\n")


    # LR SCHEDULER STEP (after epoch)
    if epoch >= config['cos_epoch']: # Start stepping scheduler *after* cos_epoch
        scheduler.step()

    # VALIDATION
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc="Validation", unit="batch")
    with torch.no_grad():
        for batch in val_bar:
            seq = batch['sequence'].cuda(non_blocking=True)
            gt  = batch['xyz'].cuda(non_blocking=True) # Keep batch dim (B, L, 3)
            pad_mask = batch['pad_mask'].cuda(non_blocking=True)

            if gt.ndim == 2: gt = gt.unsqueeze(0)
            if seq.ndim == 1: seq = seq.unsqueeze(0)
            if pad_mask.ndim == 1: pad_mask = pad_mask.unsqueeze(0)

            # Use autocast consistent with training, but no gradient needed
            if config['mixed_precision'] == 'fp32':
                pred = model(seq, pad_mask=pad_mask) # (B, L, 3)
            else:
                with autocast(dtype=torch.bfloat16 if config['mixed_precision']=='bf16' else torch.float16):
                    pred = model(seq, pad_mask=pad_mask)
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
                nan_mask = ~(torch.any(torch.isnan(gt_i), dim=-1) | torch.any(torch.isnan(pred_i), dim=-1))
                pred_i = pred_i[nan_mask]
                gt_i   = gt_i[nan_mask]
                if pred_i.shape[0] < 2: continue

                # Calculate global alignment loss for validation
                pred_i_fp32 = pred_i.float()
                gt_i_fp32 = gt_i.float()
                sample_vloss = align_svd_mae(pred_i_fp32, gt_i_fp32)

                batch_vloss += sample_vloss
                valid_samples += 1

            if valid_samples > 0:
                val_loss += (batch_vloss / valid_samples).item()  # Accumulate average loss for the batch
            # --- End of Per Sample Loss ---


    # Average validation loss (MSE) over all batches
    val_loss /= len(val_loader)
    # compute RMSD from MSE
    val_rmsd = val_loss**0.5
    current_lr = optimizer.param_groups[0]['lr']
    # Compute standard RMSD metric to compare with loss-derived RMSD
    dRMSD_score, dRMAE_score, metric_rmsd = evaluate_metrics()
    print(f"Epoch {epoch} Validation MSE: {val_loss:.4f}  RMSD(sqrt MSE): {val_rmsd:.2f}  RMSD(metric): {metric_rmsd:.2f}  LR: {current_lr:.2e}")

    # compute average training loss
    train_avg = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    # record metrics
    # write epoch, training loss, validation MSE, validation RMSD, and current learning rate
    log_f.write(f"{epoch},{train_avg:.6f},{val_loss:.6f},{val_rmsd:.6f},{current_lr:.2e}\n")

    # Evaluate metrics after each epoch
    dRMSD_score, dRMAE_score, RMSD_score = evaluate_metrics()
    scores_log.write(f"epoch,{epoch},{dRMSD_score:.6f},{dRMAE_score:.6f},{RMSD_score:.6f}\n")

    # SAVE BEST MODEL
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(save_state, 'esm-best-kg.pt')
        print(f"  ✨ Saved new best model (val_loss={val_loss:.4f})")

# close log file
log_f.close()
scores_log.close()

# FINAL SAVE
final_save_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
torch.save(final_save_state, 'esm-final-kg.pt')
print("Training complete. Final model saved.")