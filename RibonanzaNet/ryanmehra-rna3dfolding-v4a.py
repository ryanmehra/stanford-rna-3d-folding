## Owned by Ryan Mehra. Licensed for free use. 
# Purpose: Main training script for RibonanzaNet v4a, combining Transformer and EGNN for RNA 3D structure prediction.
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm


#set seed for everything
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter": 10, 
    "structural_violation_epoch": 50,
    "balance_weight": False,
}

# # III. Data Prepration

# Load data

train_sequences=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

validation_sequences=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
validation_labels=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")

test_sequences=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
validation_labels["pdb_id"] = validation_labels["ID"].apply(lambda x: x.split("_")[0])


train_sequences.shape, train_labels.shape, validation_sequences.shape, validation_labels.shape, test_sequences.shape


train_sequences.head(1)


train_labels.head(1)


validation_sequences.head(1)


## Validation Labels has many coordinates, we do not have this in the training set, for the first run we will ignore the rest and just pick the first XYZ set
validation_labels.head(1)


test_sequences.head(1)


_tmp = pd.DataFrame()
_tmp['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'])

year_counts = (
    _tmp
    .groupby(_tmp['temporal_cutoff'].dt.year)
    .size()
    .rename('count')
)
print(year_counts)


## Build all coordinates as list per target_id

all_xyz_coord_trng = []
all_xyz_coord_val = []

for pdb_id in tqdm(train_sequences['target_id']):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    xyz = df[['x_1','y_1','z_1']].to_numpy().astype('float32')

    # 1) Build a mask array, initialized to False
    mask = np.zeros_like(xyz, dtype=bool)
    finite_mask = np.isfinite(xyz)

    # 2) Only compare where values are finite, write into mask
    np.less(xyz, -1e17, out=mask, where=finite_mask)

    # 3) Assign NaN to all positions flagged by mask
    xyz[mask] = np.nan

    all_xyz_coord_trng.append(xyz)


for pdb_id in tqdm(validation_sequences['target_id']):
    df = validation_labels[validation_labels["pdb_id"] == pdb_id]
    xyz = df[['x_1','y_1','z_1']].to_numpy().astype('float32')

    # 1) Build a mask array, initialized to False
    mask = np.zeros_like(xyz, dtype=bool)
    finite_mask = np.isfinite(xyz)

    # 2) Only compare where values are finite, write into mask
    np.less(xyz, -1e17, out=mask, where=finite_mask)

    # 3) Assign NaN to all positions flagged by mask
    xyz[mask] = np.nan

    all_xyz_coord_val.append(xyz)


len(all_xyz_coord_trng), len(all_xyz_coord_val)


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

# print stats
print(f"Total sequences initially : {total}")
print(f" Kept                    : {len(kept_indices)}")
print(f" Dropped                 : {dropped}")
print(f"Shortest sequence length : {min_len}")
print(f"Longest sequence length  : {max_len}")


#pack data into a dictionary

training_data={
      "sequence":train_sequences['sequence'].to_list(),
      "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
      "description": train_sequences['description'].to_list(),
      "all_sequences": train_sequences['all_sequences'].to_list(),
      "xyz": all_xyz_coord_trng
}

validation_data={
      "sequence":validation_sequences['sequence'].to_list(),
      "temporal_cutoff": validation_sequences['temporal_cutoff'].to_list(),
      "description": validation_sequences['description'].to_list(),
      "all_sequences": validation_sequences['all_sequences'].to_list(),
      "xyz": all_xyz_coord_val
}


print(next(iter(training_data['sequence'])), next(iter(training_data['temporal_cutoff'])), next(iter(training_data['description'])), next(iter(training_data['all_sequences'])), next(iter(training_data['xyz'])))


print(next(iter(validation_data['sequence'])), next(iter(validation_data['temporal_cutoff'])), next(iter(validation_data['description'])), next(iter(validation_data['all_sequences'])), next(iter(validation_data['xyz'])))


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


# import plotly.graph_objects as go
# import numpy as np



# # Example: Generate an Nx3 matrix
# xyz = train_dataset[200]['xyz']  # Replace this with your actual Nx3 data
# N = len(xyz)


# for _ in range(2): #plot twice because it doesnt show up on first try for some reason
#     # Extract columns
#     x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
#     # Create the 3D scatter plot
#     fig = go.Figure(data=[go.Scatter3d(
#         x=x, y=y, z=z,
#         mode='markers+lines', #'markers',
#         marker=dict(
#             size=5,
#             color=z,  # Coloring based on z-value
#             colorscale='Viridis',  # Choose a colorscale
#             opacity=0.8
#         )
#     )])
    
#     # Customize layout
#     fig.update_layout(
#         scene=dict(
#             xaxis_title="X",
#             yaxis_title="Y",
#             zaxis_title="Z"
#         ),
#         title="3D Scatter Plot"
#     )

# fig.show()





# # V. Create Custom Model Instance
# 
# We will add a linear layer to predict xyz of C1' atoms on the base /kaggle/input/ribonanzanet2d-final 
# 
# 


# import sys

# sys.path.append("/kaggle/input/ribonanzanet2d-final")

# from Network import *
# import yaml



# class Config:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
#         self.entries=entries

#     def print(self):
#         print(self.entries)

# def load_config_from_yaml(file_path):
#     with open(file_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return Config(**config)



# class finetuned_RibonanzaNet(RibonanzaNet):
#     def __init__(self, config, pretrained=False):
#         config.dropout=0.1
#         super(finetuned_RibonanzaNet, self).__init__(config)
#         if pretrained:
#             self.load_state_dict(torch.load("/kaggle/input/ribonanzanet-weights/RibonanzaNet.pt",map_location='cpu'))
#         # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
#         #                                 nn.ReLU(),
#         #                                 nn.Linear(256,64),
#         #                                 nn.ReLU(),
#         #                                 nn.Linear(64,1)) 
#         self.dropout=nn.Dropout(0.0)
#         self.xyz_predictor=nn.Linear(256,3)


    
#     def forward(self,src):
        
#         #with torch.no_grad():
#         sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))


#         xyz=self.xyz_predictor(sequence_features)

#         return xyz


# ## Available GPUs 
# print("GPUs available:", torch.cuda.device_count())


# from pprint import pprint
# cfg = load_config_from_yaml("/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml")

# ## Update the batch size to new value
# _batch_size= 5

# cfg.batch_size = _batch_size
# cfg.entries['batch_size'] = _batch_size

# ## Update the GPUs to multiple if multiple available 
# if torch.cuda.device_count() > 1:
#     cfg.gpu_id = "0,1"
#     cfg.entries['gpu_id'] = "0,1"
    
# pprint(vars(cfg))


# ## Create dataloader instances 

# # train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
# # val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)

# import torch
# from torch.nn.utils.rnn import pad_sequence

# def pad_collate(batch):
#     # batch is a list of dicts, e.g. {'sequence': Tensor[L], 'xyz': Tensor[L,3], …}
#     seqs = [torch.tensor(item['sequence']) for item in batch]
#     xyzs = [torch.tensor(item['xyz'], dtype=torch.float32) for item in batch]

#     # pad to the max length in this batch
#     seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)       # or pad_token
#     xyzs_padded = pad_sequence(xyzs, batch_first=True, padding_value=float('nan'))

#     # collect any other fields you need, e.g. labels
#     # labels = torch.stack([item['label'] for item in batch], 0)

#     return {
#         'sequence': seqs_padded,
#         'xyz':       xyzs_padded,
#         # 'label':    labels,
#     }




# # then in your DataLoader:
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=cfg.batch_size,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=True,
#     collate_fn=pad_collate
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=cfg.batch_size,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True,
#     collate_fn=pad_collate
# )


# # model=finetuned_RibonanzaNet(load_config_from_yaml("/kaggle/input/ribonanzanet2d-final/configs/pairwise.yaml"),pretrained=True).cuda()

# # instantiate on CPU first
# model = finetuned_RibonanzaNet(cfg, pretrained=True)

# # wrap in DataParallel (uses all available GPUs by default)
# model = torch.nn.DataParallel(model)

# # then move to CUDA
# model = model.cuda()

# # after wrapping in DataParallel
# # print("Model sees config:", model.module.cfg.batch_size, model.module.cfg.gpu_id)

# print("GPUs visible:", torch.cuda.device_count())

# print("DataParallel device IDs:", model.device_ids)
# print("First parameter on device:", next(model.parameters()).device)


# # **Define Loss Function**
# # 
# # we will use dRMSD loss on the predicted xyz. the loss function is invariant to translations, rotations, and reflections. because dRMSD is invariant to reflections, it cannot distinguish chiral structures, so there may be better loss functions


# def calculate_distance_matrix(X,Y,epsilon=1e-4):
#     return (torch.square(X[:,None]-Y[None,:])+epsilon).sum(-1).sqrt()


# def dRMSD(pred_x,
#           pred_y,
#           gt_x,
#           gt_y,
#           epsilon=1e-4,Z=10,d_clamp=None):
#     pred_dm=calculate_distance_matrix(pred_x,pred_y)
#     gt_dm=calculate_distance_matrix(gt_x,gt_y)



#     mask=~torch.isnan(gt_dm)
#     mask[torch.eye(mask.shape[0]).bool()]=False

#     if d_clamp is not None:
#         rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).clip(0,d_clamp**2)
#     else:
#         rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon

#     return rmsd.sqrt().mean()/Z

# def local_dRMSD(pred_x,
#           pred_y,
#           gt_x,
#           gt_y,
#           epsilon=1e-4,Z=10,d_clamp=30):
#     pred_dm=calculate_distance_matrix(pred_x,pred_y)
#     gt_dm=calculate_distance_matrix(gt_x,gt_y)



#     mask=(~torch.isnan(gt_dm))*(gt_dm<d_clamp)
#     mask[torch.eye(mask.shape[0]).bool()]=False



#     rmsd=torch.square(pred_dm[mask]-gt_dm[mask])+epsilon
#     # rmsd=(torch.square(pred_dm[mask]-gt_dm[mask])+epsilon).sqrt()/Z
#     #rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])/Z
#     return rmsd.sqrt().mean()/Z

# def dRMAE(pred_x,
#           pred_y,
#           gt_x,
#           gt_y,
#           epsilon=1e-4,Z=10,d_clamp=None):
#     pred_dm=calculate_distance_matrix(pred_x,pred_y)
#     gt_dm=calculate_distance_matrix(gt_x,gt_y)



#     mask=~torch.isnan(gt_dm)
#     mask[torch.eye(mask.shape[0]).bool()]=False

#     rmsd=torch.abs(pred_dm[mask]-gt_dm[mask])

#     return rmsd.mean()/Z

# import torch

# def align_svd_mae(input, target, Z=10):
#     """
#     Align input (Nx3) to target (Nx3) via Procrustes (SVD) in float32,
#     then compute MAE / Z.
#     """
#     assert input.shape == target.shape, "Input and target must match"

#     # 1) Mask out NaNs
#     mask = ~torch.isnan(target.sum(-1))
#     inp = input[mask].float()   # cast to float32
#     tgt = target[mask].float()  # cast to float32

#     # 2) Compute and remove centroids
#     c_inp = inp.mean(dim=0, keepdim=True)
#     c_tgt = tgt.mean(dim=0, keepdim=True)
#     inp_c = inp - c_inp
#     tgt_c = tgt - c_tgt

#     # 3) Covariance matrix
#     cov = inp_c.t() @ tgt_c

#     # 4) SVD in float32
#     #    Detach so no gradients flow through the SVD
#     with torch.no_grad():
#         U, S, Vt = torch.svd(cov)
#         R = Vt @ U.t()
#         # fix potential reflection
#         if torch.det(R) < 0:
#             Vt[-1, :] *= -1
#             R = Vt @ U.t()

#     # 5) Rotate back and re-add centroid
#     #    (R is already float32, inp_c is float32)
#     aligned = inp_c @ R.t() + c_tgt

#     # 6) MAE loss (float32)
#     loss = torch.abs(aligned - tgt).mean() / Z

#     return loss


# from torch.cuda.amp import autocast, GradScaler
# from tqdm import tqdm

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=cfg.learning_rate,
#     weight_decay=cfg.weight_decay
# )


# epochs    = 50
# cos_epoch = 35
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=(epochs - cos_epoch) * len(train_loader) // cfg.batch_size
# )
# scaler = GradScaler()

# # ---- TRAIN & VALIDATION LOOP ----
# best_val_loss = float('inf')

# for epoch in range(1, epochs + 1):
#     # TRAINING
#     model.train()
#     optimizer.zero_grad(set_to_none=True)
#     train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
#     running_loss = 0.0

#     for idx, batch in enumerate(train_bar, start=1):
#         seq = batch['sequence'].cuda(non_blocking=True)
#         gt  = batch['xyz'].cuda(non_blocking=True).squeeze()

#         # 1) compute dRMAE in fp16
#         with autocast():
#             pred = model(seq).squeeze()
#             dR_loss = dRMAE(pred, pred, gt, gt) #+ align_svd_mae(pred, gt)

#         # 2) compute alignment loss in fp32
#         with autocast(enabled=False):
#             rot_loss = align_svd_mae(pred, gt)  # SVD runs in fp32

#         loss = dR_loss + rot_loss

#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)

#         running_loss += loss.item()
#         if idx % 10 == 0:
#             train_bar.set_postfix(loss=running_loss / idx)

#     # LR SCHEDULER STEP
#     if epoch > cos_epoch:
#         scheduler.step()

#     # VALIDATION
#     model.eval()
#     val_loss = 0.0
#     val_bar = tqdm(val_loader, desc="Validation", unit="batch")
#     with torch.no_grad():
#         for batch in val_bar:
#             seq = batch['sequence'].cuda(non_blocking=True)
#             gt  = batch['xyz'].cuda(non_blocking=True).squeeze()
#             pred = model(seq).squeeze()
#             vloss = dRMAE(pred, pred, gt, gt)
#             val_loss += vloss.item()

#     val_loss /= len(val_loader)
#     print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

#     # SAVE BEST MODEL
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'RibonanzaNet-best-rm.pt')
#         print(f"  ✨ Saved new best model (val_loss={val_loss:.4f})")

# # FINAL SAVE
# torch.save(model.state_dict(), 'RibonanzaNet-final-rm.pt')
# print("Training complete. Final model saved.")



# # # VI. Submission


# ## Load model

# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# from tqdm import tqdm

# # 1) Reconstruct model & load best checkpoint
# model = finetuned_RibonanzaNet(cfg, pretrained=False)
# model = torch.nn.DataParallel(model).cuda()
# state = torch.load('/kaggle/working/RibonanzaNet-best-rm.pt', map_location='cuda:0')
# model.load_state_dict(state)


# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# # --- assume `test_sequence` is your DataFrame,
# #     `config`, `cfg`, `model`, `pad_collate`, & `RNA3D_Dataset` are already in scope

# # 1) Build a dict of lists for the Dataset, with dummy xyz
# test_data = {
#     'sequence':      test_sequences['sequence'].tolist(),
#     'xyz':           [np.zeros((config['max_len'], 3), dtype=np.float32)]
#                        * len(test_sequences),   # dummy
# }
# # (we ignore temporal_cutoff / description / all_sequences here)

# # 2) Instantiate the Dataset + Loader
# test_ds = RNA3D_Dataset(test_data, config)
# test_loader = DataLoader(
#     test_ds,
#     batch_size=cfg.batch_size,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=True,
#     collate_fn=pad_collate
# )

# # 3) Inference
# model.eval()
# all_preds = []  # will be a list of [L_padded, 3] arrays
# with torch.no_grad():
#     for batch in tqdm(test_loader, desc="Predicting"):
#         seq   = batch['sequence'].cuda(non_blocking=True)
#         preds = model(seq).cpu().numpy()   # shape (B, L_batch, 3)
#         # append each RNA in the batch separately
#         for p in preds:
#             all_preds.append(p)

# # now all_preds[i] is the padded-prediction for test i
# # length may vary per-batch, but you'll slice to true L below

# # 4) Build submission rows
# rows = []
# for i, row in test_sequences.iterrows():
#     tid     = row['target_id']
#     seq_str = row['sequence']
#     L       = len(seq_str)
#     coords  = all_preds[i][:L]   # slice off the padding → shape [L,3]

#     for j, (x,y,z) in enumerate(coords, start=1):
#         base = {
#             'ID':      f"{tid}_{j}",
#             'resname': seq_str[j-1],
#             'resid':   j
#         }
#         # replicate each coordinate 5×
#         for k in range(1, 6):
#             base[f'x_{k}'] = x
#             base[f'y_{k}'] = y
#             base[f'z_{k}'] = z
#         rows.append(base)

# submission_df = pd.DataFrame(rows)
# print("Final submission shape:", submission_df.shape)
# submission_df.to_csv("submission.csv", index=False)





