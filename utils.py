import os
import rdkit
import gc
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem import Descriptors

print(rdkit.__version__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def compute_tanimoto(fp1, fp2):
    """
    fp1, fp2: RDKit ExplicitBitVect fingerprint objects
    return: tanimoto similarity
    """
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# def build_negative_mask(chem_emb, ms_emb, fps_batch, tau_fp=0.8, tau_ms=0.8):
#     """
#     return a boolean mask of shape (B, B)，:
#     True means (i, j) should be masked out (not treated as negative pair)
#     detail：
#       - if Tanimoto > tau_fp or MS similarity > tau_ms，flag True
#       - (i==j) flag False
#     """
#     B = chem_emb.size(0)
#     device = chem_emb.device

#     mask = torch.zeros((B, B), dtype=torch.bool, device=device)

#     for i in range(B):
#         for j in range(B):
#             if i == j:
#                 continue

#             tanimoto_sim = compute_tanimoto(fps_batch[i], fps_batch[j])

#             ms_sim = F.cosine_similarity(
#                 ms_emb[i].unsqueeze(0),
#                 ms_emb[j].unsqueeze(0),
#                 dim=-1
#             ).item()

#             # once either similarity exceeds its threshold, we mark (i, j) as True (not negative pair)
#             if (tanimoto_sim > tau_fp) or (ms_sim > tau_ms):
#                 mask[i, j] = True

#     return mask
def tanimoto_matrix(fps_batch):
    """
    compute the Tanimoto similarity matrix for a batch of fingerprints
    """
    B = len(fps_batch)
    tanimoto_mat = torch.zeros((B, B), dtype=torch.float32)

    for i, fp in enumerate(fps_batch):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_batch)
        tanimoto_mat[i] = torch.tensor(sims)
    # if isinstance(fps_batch[0], list):
    #     F = torch.tensor(fps_batch, dtype=torch.float32)
    # else:
    #     F = torch.stack([torch.tensor(fp, dtype=torch.float32) for fp in fps_batch])
    
    # intersection = F @ F.T  # (B, B)
    
    # popcount = F.sum(dim=1, keepdim=True)  # (B, 1)
    
    # union = popcount + popcount.T - intersection
    
    # sim_matrix = intersection / torch.clamp(union, min=1e-8)
    
    # sim_matrix.fill_diagonal_(1.0)
    # tanimoto_mat = sim_matrix

    return tanimoto_mat


# def build_negative_mask(chem_emb, ms_emb, fps_batch, tau_fp=0.6, tau_ms=0.8):
#     B = chem_emb.size(0)
#     device = chem_emb.device

#     # ---- 1) 批量 Tanimoto 相似度矩阵 (B, B)
#     tanimoto_mat = tanimoto_matrix(fps_batch).to(device)

#     # ---- 2) 批量 MS 余弦相似度矩阵 (B, B)
#     ms_sim_mat = F.cosine_similarity(
#         ms_emb.unsqueeze(1),   # (B,1,D)
#         ms_emb.unsqueeze(0),   # (1,B,D)
#         dim=-1
#     )  # => (B,B)

#     # ---- 3) 满足任一条件就 mask
#     mask = (tanimoto_mat > tau_fp) #| (ms_sim_mat > tau_ms)

#     # ---- 4) 主对角线设为 False
#     mask.fill_diagonal_(False)

#     return mask

def build_negative_mask(chem_emb, ms_emb=None, fps_batch=None, tau_fp=None, tau_ms=None):
    """
    return a boolean mask of shape (B, B)，:
    True means (i, j) should be masked out (not treated as negative pair)       
    """
    B = chem_emb.size(0)
    device = chem_emb.device
    
    mask = torch.zeros((B, B), dtype=torch.bool, device=device)
    
    return mask

def masked_clip_loss(chem_emb, ms_emb, mask, temperature=0.03):
    B = chem_emb.size(0)

    chem_emb = F.normalize(chem_emb, dim=1)
    ms_emb   = F.normalize(ms_emb, dim=1)

    logits = chem_emb @ ms_emb.t() / temperature
    labels = torch.arange(B).to(chem_emb.device)

    logits = logits.masked_fill(mask == 1, float('-inf'))

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)

    return (loss_i + loss_t) / 2

class BYOLPredictor(nn.Module):
    def __init__(self, dim=500, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

def byol_loss(ms_emb, chem_emb, predictor):
    pred = predictor(ms_emb)                          # (B, D)
    pred = F.normalize(pred, dim=-1)
    target = chem_emb.detach()                        # stop-gradient

    # cosine similarity loss: 1 - cos_sim
    loss = 2 - 2 * (pred * target).sum(dim=-1).mean()
    return loss

# option 2
def vicreg_loss(ms_emb, chem_emb,
                lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0):
    B, D = ms_emb.size()

    # Invariance
    inv_loss = F.mse_loss(ms_emb, chem_emb)

    # Variance
    def variance_loss(z):
        std = z.std(dim=0)
        return F.relu(1.0 - std).mean()

    var_loss = variance_loss(ms_emb) + variance_loss(chem_emb)

    # Covariance
    def covariance_loss(z):
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (B - 1)
        off_diag = cov.masked_fill(
            torch.eye(D, device=z.device).bool(), 0
        )
        return off_diag.pow(2).sum() / D

    cov_loss = covariance_loss(ms_emb) + covariance_loss(chem_emb)

    return lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss

def analyze_mask_statistics(mask, fps_batch, chem_emb, ms_emb):

    B = mask.size(0)
    

    total_pairs = B * (B - 1)
    masked_pairs = mask.sum().item()
    mask_ratio = masked_pairs / total_pairs
    
    if fps_batch is not None:
        tanimoto_mat = tanimoto_matrix(fps_batch)
        
        masked_sims = []
        unmasked_sims = []
        
        for i in range(B):
            for j in range(B):
                if i != j:
                    sim = tanimoto_mat[i, j]
                    if mask[i, j]:
                        masked_sims.append(sim)
                    else:
                        unmasked_sims.append(sim)
        
        if masked_sims:
            avg_masked_sim = np.mean(masked_sims)
        else:
            avg_masked_sim = 0
            
        if unmasked_sims:
            avg_unmasked_sim = np.mean(unmasked_sims)
        else:
            avg_unmasked_sim = 0
    
    return {
        "mask_ratio": mask_ratio,
        "avg_masked_sim": avg_masked_sim,
        "avg_unmasked_sim": avg_unmasked_sim
    }

class CLIPModel(nn.Module):
    def __init__(self, chem_encoder, ms_encoder, temperature=0.07):
        super().__init__()
        self.chem_encoder = chem_encoder
        self.ms_encoder = ms_encoder
        self.temperature = temperature

    def forward(self, cid_batch, ms_batch):
        with torch.no_grad():
            chem_emb = self.chem_encoder(cid_batch).detach()
        ms_emb = self.ms_encoder(ms_batch)
        return chem_emb, ms_emb


def train_clip(ms_encoder, dataloader, proj, device, optimizer, loss_type = "clip", predictor=None, temperature = temperature, use_sum=False):

    proj.to(device)
    proj.train()
    ms_encoder.train()
    
    epoch_loss = []

    for smiles_batch, ms_batch, fps_batch, chem_batch in tqdm(dataloader):

        # 1. MS embedding
        # ms_emb = gen_embeddings_like_model(ms_encoder, ms_batch)
        # ms_emb = torch.tensor(ms_emb, dtype=torch.float32).to(device) 
        mzs, intens, masks = ms_batch

        # 严格执行设备移动
        mzs_gpu = mzs.to(device, non_blocking=True)
        intens_gpu = intens.to(device, non_blocking=True)
        masks_gpu = masks.to(device, non_blocking=True)

        ms_inputs = (mzs_gpu, intens_gpu, masks_gpu)

        # model
        if use_sum:
            ms_emb = ms_encoder(ms_inputs, mode='emb').sum(dim=1)  # (B, L, D) → (B, D)
        else:
            ms_emb = ms_encoder(ms_inputs, mode='emb')

        # 2. chem embedding
        # chem_emb = get_chem_emb(chem_encoder, smiles_batch)
        chem_emb = torch.tensor(np.array(chem_batch), dtype=torch.float32, device=device)

        # 3. projection
        ms_emb = proj(ms_emb)   # (B, 500)

        ms_emb = F.normalize(ms_emb, p=2, dim=-1)
        chem_emb = F.normalize(chem_emb, p=2, dim=-1)

        # 4. mask
        # mask = build_negative_mask(
        #     chem_emb, ms_emb, fps_batch,
        #     tau_fp=0.20,
        #     tau_ms=0.8,
        # )

        # stats = analyze_mask_statistics(mask, fps_batch, chem_emb, ms_emb)
        # print(f"Mask ratio: {stats['mask_ratio']:.2%}")
        # print(f"Avg masked sim: {stats['avg_masked_sim']:.3f}")
        # print(f"Avg unmasked sim: {stats['avg_unmasked_sim']:.3f}")
        mask = build_negative_mask(
            chem_emb, ms_emb, fps_batch =None)

        # 5. CLIP loss
        optimizer.zero_grad()
        if loss_type == "clip":
            loss = masked_clip_loss(chem_emb, ms_emb, mask, temperature=temperature)
        elif loss_type == "byol":
            loss = byol_loss(ms_emb, chem_emb, predictor)
        elif loss_type == "vicreg":
            loss = vicreg_loss(ms_emb, chem_emb)

        loss.backward()

        optimizer.step()

        epoch_loss.append(loss.item())

        torch.cuda.empty_cache()
        gc.collect()
        
        cache_dir = './cache' 
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.endswith(".pkl"):
                    file_path = os.path.join(cache_dir, filename)
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
            print("临时硬盘缓存已清理")

    print(f"Epoch Loss = {sum(epoch_loss)/len(epoch_loss):.6f}")
    return sum(epoch_loss)/len(epoch_loss)

def validate_clip(ms_encoder, val_loader, proj, device, loss_type = "clip", predictor=None, temperature=temperature, use_sum=False):

    proj.to(device)
    ms_encoder.eval()
    proj.eval()

    losses = []
    val_auc_list = []
    all_ms_embs_list = [] 
    all_chem_embs_list = []

    all_fps_list = []

    with torch.no_grad():
        for smiles_batch, ms_batch, fps_batch, chem_batch in val_loader:

            mzs, intens, masks = ms_batch
            ms_inputs = (mzs.to(device), intens.to(device), masks.to(device))
            # mw_list = []
            # for smi in smiles_batch:
            #     mol = Chem.MolFromSmiles(smi)
            #     if mol is None:
            #         mw_list.append(0.0)
            #     else:
            #         mw_list.append(Descriptors.ExactMolWt(mol))
                    
            # mw_inputs = torch.tensor(mw_list, dtype=torch.float32, device=device).unsqueeze(1)

            if use_sum:
                ms_emb = ms_encoder(ms_inputs, mode='emb').sum(dim=1)  # (B, L, D) → (B, D)
            else:
                ms_emb = ms_encoder(ms_inputs, mode='emb')

            # chem_emb = get_chem_emb(chem_encoder, smiles_batch)
            chem_emb = torch.tensor(np.array(chem_batch), dtype=torch.float32, device=device)

            ms_emb = proj(ms_emb).to(device)   # (B, 500)

            # ms_emb = F.normalize(ms_emb, p=2, dim=-1)
            # chem_emb = F.normalize(chem_emb, p=2, dim=-1)
            if loss_type == 'vicreg':
                ms_emb_for_loss  = ms_emb   # 不 normalize
                chem_emb_for_loss = chem_emb
            else:
                ms_emb_for_loss  = F.normalize(ms_emb,   p=2, dim=-1)
                chem_emb_for_loss = F.normalize(chem_emb, p=2, dim=-1)
            
            mask = build_negative_mask(
                chem_emb, ms_emb, fps_batch =None)
            

            ms_emb = F.normalize(ms_emb, p=2, dim=-1)
            chem_emb = F.normalize(chem_emb, p=2, dim=-1)
            
            if loss_type == "clip":
                loss = masked_clip_loss(chem_emb, ms_emb, mask, temperature=temperature)
            elif loss_type == "byol":
                loss = byol_loss(ms_emb, chem_emb, predictor)
            elif loss_type == "vicreg":
                loss = vicreg_loss(ms_emb, chem_emb)
            losses.append(loss.item())

            logits = torch.matmul(chem_emb, ms_emb.t())
            batch_size = logits.size(0)
            targets = torch.eye(batch_size, device=device)
            
            y_true = targets.cpu().numpy().flatten()
            y_scores = logits.cpu().numpy().flatten()
            
            try:
                batch_auc = roc_auc_score(y_true, y_scores)
                val_auc_list.append(batch_auc)
            except ValueError:
                pass
            all_ms_embs_list.append(ms_emb.cpu())
            all_chem_embs_list.append(chem_emb.cpu())

            all_fps_list.extend(fps_batch)
        
        final_ms_embs = torch.cat(all_ms_embs_list, dim=0)
        final_chem_embs = torch.cat(all_chem_embs_list, dim=0)

    print("Validation Loss:", sum(losses)/len(losses))
    return sum(losses)/len(losses), np.mean(val_auc_list)


def plot_validation_metrics(topk_sim_history, val_loss_history, val_auc_history, epochs, save_dir="results"):
    
    epochs_range = range(1, epochs + 1)
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, topk_sim_history, marker='o', linestyle='-', color='tab:blue', label='Avg Top-10 Chem Sim')
    plt.title('MS Manifold Quality (Avg Top-10 Chem Sim)')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_loss_history, marker='s', linestyle='-', color='tab:orange', label='Validation Loss')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('CLIP Loss Value')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_auc_history, marker='^', linestyle='-', color='tab:green', label='Validation AUC')
    plt.title('Validation Batch AUC Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Batch AUC Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    plot_file = os.path.join(save_dir, "validation_metrics_plot.png")
    plt.savefig(plot_file)
    print(f"🎉 验证指标图表已保存至: {plot_file}")
    plt.show()
