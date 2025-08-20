## Owned by Ryan Mehra. Licensed for free use. 
# Purpose: Main training script for RibonanzaNet v6a, with RoPE, EGNN, and memory-efficient optimizations for RNA 3D structure prediction.
## pip install rotary-embedding-torch egnn-pytorch

# full_training_mod_arch_v6.py

import os
import sys
import math
import random
import pickle
import warnings
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import autocast, GradScaler  # mixed precision support

# plotting not used directly but kept for completeness
import matplotlib.pyplot as plt

# silence annoying warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------- 1. Set Seeds & Config ---------------------

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Primary config dict (you can tweak these)
config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    # reduce batch and sequence length for memory
    "batch_size": 3,
    "max_len": 1024,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "fp16",      # Enable fp16 mixed precision
    "model_config_path": "./ribonanzanet2d-final/configs/pairwise.yaml",
    "epochs": 10,
    "cos_epoch": 40,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter": 10,
    "structural_violation_epoch": 50,
    "balance_weight": False,
    # NEW FLAGS:
    "egnn_layers": 4,
    "transformer_layers": 4,
    "transformer_heads": 8,
    "transformer_ff_dim": 1024,
    "use_distance_bias": True,      # ⬅ toggle distance-based attn bias
}

# Load base-model config
def load_config_from_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    class C: pass
    c = C()
    c.__dict__.update(data)
    return c

cfg = load_config_from_yaml(config["model_config_path"])

# --------------------- 2. Data Preparation ---------------------

# Paths to your CSVs
TRAIN_SEQ_CSV = "./kaggle-data/new_training_sequences.csv"
TRAIN_LBL_CSV = "./kaggle-data/new_training_labels.csv"
VAL_SEQ_CSV   = "./kaggle-data/new_validation_sequences.csv"
VAL_LBL_CSV   = "./kaggle-data/new_validation_labels.csv"
TEST_SEQ_CSV  = "./kaggle-data/test_sequences.csv"

# 2.1 Load CSVs
train_sequences      = pd.read_csv(TRAIN_SEQ_CSV)
train_labels         = pd.read_csv(TRAIN_LBL_CSV)
validation_sequences = pd.read_csv(VAL_SEQ_CSV)
validation_labels    = pd.read_csv(VAL_LBL_CSV)
test_sequences       = pd.read_csv(TEST_SEQ_CSV)

# Subset to a single sample for quick end-to-end testing
# sample_tid = train_sequences['target_id'].iloc[0]
# train_sequences = train_sequences[train_sequences['target_id'] == sample_tid].reset_index(drop=True)
# train_labels    = train_labels[train_labels['target_id']   == sample_tid].reset_index(drop=True)

# val_tid = validation_sequences['target_id'].iloc[0]
# validation_sequences = validation_sequences[validation_sequences['target_id'] == val_tid].reset_index(drop=True)
# validation_labels    = validation_labels[validation_labels['target_id']   == val_tid].reset_index(drop=True)

print("Shapes:", train_sequences.shape, train_labels.shape,
      validation_sequences.shape, validation_labels.shape,
      test_sequences.shape)

# 2.2 Build xyz dicts
train_xyz_dict = {
    tid: grp[['x_1','y_1','z_1']].to_numpy(dtype='float32')
    for tid, grp in train_labels.groupby('target_id')
}
val_xyz_dict = {
    tid: grp[['x_1','y_1','z_1']].to_numpy(dtype='float32')
    for tid, grp in validation_labels.groupby('target_id')
}

# 2.3 Process & mask NaNs
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

print("Processed coords lengths:", len(all_xyz_coord_trng), len(all_xyz_coord_val))

# 2.4 Filter sequences with >50% NaN or outside length bounds
lengths = [len(x) for x in all_xyz_coord_trng]
total   = len(lengths)
mask_keep = []
for xyz in all_xyz_coord_trng:
    frac_nan = np.isnan(xyz).mean()
    L = len(xyz)
    keep = (frac_nan <= 0.5) and (L < config["max_len_filter"]) and (L > config["min_len_filter"])
    mask_keep.append(keep)
mask_keep = np.array(mask_keep)
kept_idx  = np.nonzero(mask_keep)[0]
print(f"Filtering: kept {len(kept_idx)}/{total} sequences")

train_sequences = train_sequences.loc[kept_idx].reset_index(drop=True)
all_xyz_coord_trng = [all_xyz_coord_trng[i] for i in kept_idx]

# Optional: print detailed stats
L_after = [len(x) for x in all_xyz_coord_trng]
print("Length stats (original):", min(lengths), max(lengths))
print("Length stats (kept):", min(L_after), max(L_after))

# 2.5 Pack into dict for Dataset
training_data = {
    "sequence": train_sequences['sequence'].tolist(),
    "xyz": all_xyz_coord_trng
}
validation_data = {
    "sequence": validation_sequences['sequence'].tolist(),
    "xyz": all_xyz_coord_val
}
print("Train size:", len(training_data["sequence"]),
      "Val size:", len(validation_data["sequence"]))

# --------------------- 3. Dataset & Dataloader ---------------------

from torch.utils.data import Dataset, DataLoader

class RNA3D_Dataset(Dataset):
    def __init__(self, data, cfg):
        self.seq_list = data["sequence"]
        self.xyz_list = data["xyz"]
        self.max_len  = cfg["max_len"]
        # nucleotide → index
        self.tokens = {nt:i for i,nt in enumerate("ACGU")}
        self.UNK_ID = len(self.tokens)
    def __len__(self):
        return len(self.seq_list)
    def __getitem__(self, i):
        seq = self.seq_list[i]
        ids = [self.tokens.get(nt, self.UNK_ID) for nt in seq]
        seq_t = torch.tensor(ids, dtype=torch.long)
        xyz   = torch.tensor(self.xyz_list[i], dtype=torch.float32)
        # random crop if > max_len
        if len(seq_t) > self.max_len:
            start = random.randint(0, len(seq_t)-self.max_len)
            seq_t = seq_t[start:start+self.max_len]
            xyz   = xyz[start:start+self.max_len]
        return {"sequence": seq_t, "xyz": xyz}

def pad_collate(batch):
    seqs = [b["sequence"] for b in batch]
    xyzs = [b["xyz"] for b in batch]
    seqs_p = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    xyzs_p = nn.utils.rnn.pad_sequence(xyzs, batch_first=True,
                                        padding_value=float("nan"))
    return {"sequence": seqs_p, "xyz": xyzs_p}

train_ds = RNA3D_Dataset(training_data, config)
val_ds   = RNA3D_Dataset(validation_data, config)

train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                          shuffle=True, collate_fn=pad_collate,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                          shuffle=False,collate_fn=pad_collate,
                          num_workers=4, pin_memory=True)

# --------------------- 4. Model Definition ---------------------

# 4.1 Load base RibonanzaNet
sys.path.append("./ribonanzanet2d-final")
from Network import RibonanzaNet

# 4.2 Optional EGNN
try:
    from egnn_pytorch import EGNN
except ImportError:
    EGNN = None
    print("EGNN not found → refinement disabled")

# 4.3 Rotary Embedding for RoPE
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# 4.4 Self-Attention with optional distance bias + padding mask
class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim//heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bias=None, pad_mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, self.heads, 3*self.dh)
        q,k,v = torch.split(qkv, self.dh, dim=-1)
        # [B,H,S,Dh]
        q = q.permute(0,2,1,3); k = k.permute(0,2,1,3); v = v.permute(0,2,1,3)
        return q, k, v

# 4.5 Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.att = SelfAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        # self.rotary = RotaryEmbedding(dim // heads, use_xpos=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_out_proj = nn.Linear(dim, dim, bias=False)

        # only need the embedding object; we'll apply it manually below
        self.rotary_emb = RotaryEmbedding(dim // heads)

    def forward(self, x, bias=None, pad_mask=None):
        B, S, D = x.shape
        res = x
        x_ln1 = self.ln1(x)

        q, k, v = self.att(x_ln1)

        # apply RoPE via embedding API
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        scale = (D // self.att.heads) ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        if bias is not None:
            scores = scores + bias.unsqueeze(1) # No need for .to(scores.dtype) if everything is fp32
        if pad_mask is not None:
            pad_mask_bool = pad_mask.bool() if pad_mask.dtype != torch.bool else pad_mask
            scores = scores.masked_fill(
                pad_mask_bool[:,None,None,:], float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        attn_output = attn @ v
        attn_output = attn_output.permute(0,2,1,3).contiguous().view(B,S,D)
        attn_output = self.attn_out_proj(attn_output)

        x = res + attn_output

        res2 = x
        x_ln2 = self.ln2(x)
        x = res2 + self.mlp(x_ln2)
        return x

# 4.6 TransformerHead (RoPE + Residual MLP + optional distance bias)
class TransformerHead(nn.Module):
    def __init__(self, embed_dim, n_layers, n_heads, ff_dim, dropout, use_bias):
        super().__init__()
        self.use_bias = use_bias
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        # Residual MLP head
        self.mlp1 = nn.Linear(embed_dim, embed_dim*2, bias=False)
        self.mlp2 = nn.Linear(embed_dim*2, embed_dim, bias=False)
        self.act  = nn.GELU()
        self.ln_mlp = nn.LayerNorm(embed_dim)
        # final coord predictor
        self.predict = nn.Linear(embed_dim, 3)
    def forward(self, x, init_coords=None, pad_mask=None):
        # compute distance bias once if enabled
        bias = None
        if self.use_bias and (init_coords is not None):
            # fill NaN → 0
            coords0 = torch.nan_to_num(init_coords, nan=0.0)
            # ensure coords0 length matches sequence length
            if coords0.shape[1] != x.shape[1]:
                coords0 = coords0[:, :x.shape[1], :]
            diff = coords0.unsqueeze(2) - coords0.unsqueeze(1)  # [B,S,S,3]
            dist2 = torch.sum(diff*diff, dim=-1)
            bias = -dist2  # negative squared-distance
        # Transformer layers
        for i,layer in enumerate(self.layers):
            b = bias if (i==0) else None
            x = layer(x, b, pad_mask)
        # Residual MLP
        res = x
        x = self.act(self.mlp1(x))
        x = self.mlp2(x)
        x = self.ln_mlp(x + res)
        # Predict coords
        return self.predict(x)

# 4.7 Full Model with EGNN
class TransformerWithEGNNHead(nn.Module):
    def __init__(self, base_cfg, main_cfg, pretrained_path):
        super().__init__()
        # base encoder (RibonanzaNet)
        self.base = RibonanzaNet(base_cfg)
        sd = torch.load(pretrained_path, map_location='cpu')
        self.base.load_state_dict(sd, strict=False)
        for p in self.base.parameters():
            p.requires_grad = True
        embed_dim = 256 #base_cfg.embed_dim  # typically 256
        # Transformer head
        self.tr = TransformerHead(
            embed_dim,
            main_cfg["transformer_layers"],
            main_cfg["transformer_heads"],
            main_cfg["transformer_ff_dim"],
            dropout=0.1,
            use_bias=main_cfg["use_distance_bias"]
        )
        # EGNN refinement
        if EGNN is not None and main_cfg["egnn_layers"]>0:
            # no extra edge features beyond coordinate distance
            self.egnn_layers = nn.ModuleList([
                EGNN(embed_dim, 
                     0,
                     m_dim=main_cfg['transformer_ff_dim'],
                     norm_feats=True, # Example EGNN parameter
                     norm_coors=True, # Example EGNN parameter
                     update_feats=True, # Example EGNN parameter
                     update_coors=True, # Example EGNN parameter,
                     num_nearest_neighbors=8)
                for _ in range(main_cfg["egnn_layers"])
            ])
        else:
            self.egnn_layers = None

    def forward(self, seq_ids, coords_gt=None, pad_mask=None):
        # invert mask for base encoder (1=attend,0=pad)
        base_mask = (~pad_mask).long() if pad_mask is not None else torch.ones_like(seq_ids)
        # get base embeddings
        emb, _ = self.base.get_embeddings(seq_ids, base_mask.to(seq_ids.device))
        # transformer → initial coords
        coords0 = self.tr(emb, coords_gt, pad_mask)
        # refine with EGNN
        coords = coords0
        if self.egnn_layers:
            feats = emb
            for l in self.egnn_layers:
                feats, coords = l(feats, coords)
        return coords

# --------------------- 5. Instantiate Model ---------------------

print("GPUs:", torch.cuda.device_count())
model = TransformerWithEGNNHead(cfg, config, pretrained_path="./RibonanzaNet-best-rm.pt")
if torch.cuda.device_count()>1:
    model = nn.DataParallel(model)
model = model.cuda()

# Count params
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: total={total:,}, trainable={trainable:,}")

# --------------------- 6. Loss Functions ---------------------

def calculate_distance_matrix(X,Y,eps=1e-4):
    return ((X[:,None]-Y[None,:])**2 + eps).sum(-1).sqrt()

def dRMAE(px,py,gx,gy,eps=1e-4,Z=10,d_clamp=None):
    if px.shape[0]!=gx.shape[0]:
        k=min(px.shape[0],gx.shape[0])
        px,py,gx,gy = px[:k],py[:k],gx[:k],gy[:k]
    pd = calculate_distance_matrix(px,py)
    gd = calculate_distance_matrix(gx,gy)
    mask = ~torch.isnan(gd)
    mask[torch.eye(mask.size(0)).bool()] = False
    return (pd[mask]-gd[mask]).abs().mean()/Z

def align_svd_mae(inp, tgt, Z=10):
    mask = ~torch.isnan(tgt.sum(-1))
    inp_f = inp[mask].float(); tgt_f = tgt[mask].float()
    c1 = inp_f.mean(0, keepdim=True); c2 = tgt_f.mean(0, keepdim=True)
    ip = inp_f - c1; tg = tgt_f - c2
    cov = ip.t() @ tg
    U,S,Vt = torch.svd(cov)
    R = Vt @ U.t()
    if torch.det(R) < 0:
        # avoid in-place operation on Vt for proper gradient tracking
        Vt_clone = Vt.clone()
        Vt_clone[-1] = Vt_clone[-1] * -1
        R = Vt_clone @ U.t()
    aligned = ip@R.t()+c2
    return (aligned - tgt_f).abs().mean()/Z

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
    # find rotation by SVD
    C = P0.t() @ T0
    U, S, Vt = torch.svd(C)
    R = Vt @ U.t()
    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1] *= -1
        R = Vt @ U.t()
    # align and compute RMSD
    P_aligned = P0 @ R.t() + mu_T
    diff2 = (P_aligned - T).pow(2).sum(dim=-1)
    return torch.sqrt(diff2.mean() + eps).item()

# --------------------- 7. Optimizer & Scheduler ---------------------

# only train parameters requiring grads (head+EGNN, and base if unfrozen)
opt_params_gen = filter(lambda p: p.requires_grad, model.parameters())
opt_params_list = list(opt_params_gen) # Convert to list
optimizer = torch.optim.Adam(opt_params_list, # Use the list here
                             lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])

# cosine annealing after cos_epoch
steps_per_epoch = len(train_loader)//config["gradient_accumulation_steps"]
T_max = max(1, (config["epochs"]-config["cos_epoch"])*steps_per_epoch)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
# initialize gradient scaler for amp
scaler = GradScaler() if config["mixed_precision"] == "fp16" else None

# --------------------- 8. Training & Validation Loop ---------------------

best_val = float('inf')
log_f = open("training_mod_arch_v6_log.log","w")
log_f.write("epoch,train_loss,val_loss,lr\n")
log_mod = open("training_mod_arch_v6_scores.log","w")
log_mod.write("step,epoch,val_dRMAE,val_RMSD,lr,saved_epoch\n")

for epoch in range(1, config["epochs"]+1):
    model.train()
    skip_count = 0  # track how many batches were skipped due to non-finite loss
    running = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
    optimizer.zero_grad()
    for i,batch in enumerate(train_bar,1):
        seq = batch["sequence"].cuda()
        gt  = batch["xyz"].cuda()
        # pad mask
        pm  = (seq==0)

        # model forward under autocast if using fp16
        if config["mixed_precision"] == "fp16":
            with autocast():
                pred = model(seq, coords_gt=gt, pad_mask=pm)
        else:
            pred = model(seq, coords_gt=gt, pad_mask=pm)
        # per-sample loss
        dloss = 0.0; rloss = 0.0; vs=0
        for b in range(pred.size(0)):
            # true length
            L = (~pm[b]).sum().item()
            if L<2: continue
            pbs = pred[b,:L]; gbs = gt[b,:L]
            valid = ~torch.isnan(gbs[:,0])
            p_i = pbs[valid]; g_i = gbs[valid]
            if p_i.size(0)<2: continue
            d = dRMAE(p_i,p_i,g_i,g_i)
            rt= align_svd_mae(p_i, g_i)
            dloss+=d; rloss+=rt; vs+=1
        if vs==0: continue
        loss = (dloss/vs + rloss/vs) / config["gradient_accumulation_steps"]
        # guard against non-finite loss
        if not torch.isfinite(loss):
            print(f"Non-finite loss {loss.item()} at epoch {epoch} step {i}, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            skip_count += 1
            continue

        # backward with or without amp
        if config["mixed_precision"] == "fp16":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # step (without scaler)
        if (i % config["gradient_accumulation_steps"]==0) or (i==len(train_loader)):
            # unscale, clip grads and step with amp if enabled
            if config["mixed_precision"] == "fp16":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(opt_params_list, config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(opt_params_list, config["grad_clip"])
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * config["gradient_accumulation_steps"] # Adjust running loss calculation
        if i%10==0:
             # Adjust postfix calculation if using accumulation
             current_avg_loss = running / (i * config["gradient_accumulation_steps"] / config["gradient_accumulation_steps"]) # Simplified: running / i
             train_bar.set_postfix(loss=current_avg_loss)

        if i % 500 == 0:
            # periodic validation for dRMAE and RMSD
            model.eval()
            v_dRMAE = 0.0; v_rmsd = 0.0; v_cnt = 0
            with torch.no_grad():
                for vb in val_loader:
                    seq_v = vb["sequence"].cuda(); gt_v = vb["xyz"].cuda(); pm_v = (seq_v==0)
                    pred_v = model(seq_v, coords_gt=gt_v, pad_mask=pm_v)
                    for b in range(pred_v.size(0)):
                        Lb = (~pm_v[b]).sum().item()
                        if Lb < 2: continue
                        p_v = pred_v[b,:Lb]; g_v = gt_v[b,:Lb]
                        valid_v = ~torch.isnan(g_v[:,0])
                        p_i = p_v[valid_v]; g_i = g_v[valid_v]
                        if p_i.size(0) < 2: continue
                        v_dRMAE += dRMAE(p_i,p_i,g_i,g_i)
                        v_rmsd += compute_rmsd(p_i, g_i)
                        v_cnt += 1
            if v_cnt>0:
                v_dRMAE /= v_cnt; v_rmsd /= v_cnt
            lr_now = optimizer.param_groups[0]["lr"]
            log_mod.write(f"{i},{epoch},{v_dRMAE:.6f},{v_rmsd:.6f},{lr_now:.2e},\n")
            log_mod.flush()
            model.train()

    # scheduler
    if epoch>=config["cos_epoch"]:
        scheduler.step()

    # validation
    model.eval()
    vloss = 0.0; cnt = 0
    v_rmsd = 0.0; cnt_r = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            seq = batch["sequence"].cuda()
            gt  = batch["xyz"].cuda()
            pm  = (seq==0)
            pred = model(seq, coords_gt=gt, pad_mask=pm)
            for b in range(pred.size(0)):
                L = (~pm[b]).sum().item()
                if L<2: continue
                pbs = pred[b,:L]; gbs = gt[b,:L]
                valid = ~torch.isnan(gbs[:,0])
                p_i = pbs[valid]; g_i = gbs[valid]
                if p_i.size(0)<2: continue
                vloss += dRMAE(p_i,p_i,g_i,g_i)
                cnt   += 1
                v_rmsd += compute_rmsd(p_i, g_i)
                cnt_r += 1
    val_dRMAE = (vloss/cnt).item() if cnt>0 else float('nan')
    val_rmsd  = (v_rmsd/cnt_r)         if cnt_r>0 else float('nan')
    lr = optimizer.param_groups[0]["lr"]
    final_train_loss = running / len(train_loader)
    print(f"Epoch {epoch}, Train Loss: {final_train_loss:.4f}, Val dRMAE: {val_dRMAE:.4f}, Val RMSD: {val_rmsd:.4f}, LR: {lr:.2e}")
    log_f.write(f"{epoch},{final_train_loss:.6f},{val_dRMAE:.6f},{val_rmsd:.6f},{lr:.2e}\n")
    log_f.flush()
    # log end-of-epoch metrics
    saved_flag = ''
    if val_dRMAE < best_val:
        saved_flag = f"{epoch}"
    log_mod.write(f"{len(train_loader)},{epoch},{val_dRMAE:.6f},{val_rmsd:.6f},{lr:.2e},{saved_flag}\n")
    log_mod.flush()
    # save best
    if val_dRMAE < best_val:
        best_val = val_dRMAE
        sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(sd, "RibonanzaNet-TransRoPE-EGNN-best-kg.pt")
        print("✨ Saved new best model at epoch", epoch)
log_f.close()
# final save
sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
torch.save(sd, "RibonanzaNet-TransRoPE-EGNN-final-kg.pt")
print("Training complete.")

# --------------------- 9. Submission ---------------------

# (Load best model, run inference on test_loader similarly)
# ... (same as original submission code) ...