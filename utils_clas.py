import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import gc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, auc, roc_curve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

class Classification(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 138, hidden_dim_ratio: float = 0.5, dropout_rate: float = 0.5):
        super(Classification, self).__init__()
        
        hidden_dim = int(input_dim * hidden_dim_ratio)
        
        if hidden_dim < 1:
            hidden_dim = 1
            
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.fc2(x)
        
        return logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        pt = targets * p + (1 - targets) * (1 - p)
        
        modulating_factor = (1.0 - pt) ** self.gamma
    
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        

        loss = alpha_weight * modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class AsymmetricMSELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):

        super(AsymmetricMSELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        loss = torch.where(diff > 0, self.alpha * diff**2, self.beta * diff**2)
        return loss.mean()
    
def train_classfication(encoder, classification_head, dataloader, optimizer, criterion, device, freeze, use_sum):
    classification_head.to(device)
    classification_head.train()
    encoder.to(device)
    criterion.to(device)
    # if freeze:
    #     encoder.eval()
    # else:

    loss_fn = nn.BCEWithLogitsLoss()
    epoch_loss = []

    for smiles_batch, ms_batch, label_batch, chem_emb in tqdm(dataloader):
        
        label_batch = label_batch.to(device, non_blocking=True)
         
        if not freeze:
            mzs, intens, masks = ms_batch

            mzs_gpu = mzs.to(device, non_blocking=True)
            intens_gpu = intens.to(device, non_blocking=True)
            masks_gpu = masks.to(device, non_blocking=True)

            ms_inputs = (mzs_gpu, intens_gpu, masks_gpu)
        

            with torch.no_grad():
                ms_emb = encoder(ms_inputs, mode='emb')
                if use_sum:
                    ms_emb = ms_emb.sum(dim=1) 
            ms_emb = ms_emb.detach()        

            features_for_head = ms_emb

        else: 
            if isinstance(chem_emb, torch.Tensor) and chem_emb.device != device:
                chem_emb = chem_emb.to(device)
            
            features_for_head = chem_emb
            
        optimizer.zero_grad()
        logits = classification_head(features_for_head).to(device, non_blocking=True)

        # loss = loss_fn(logits, label_batch)
        loss = criterion(logits, label_batch)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        # torch.cuda.empty_cache()
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
    print(f"Epoch Loss = {sum(epoch_loss)/len(epoch_loss):.6f}")
    return sum(epoch_loss)/len(epoch_loss)

def adjusted_precision_at_k(labels_np, probs_np, k):
    scores = []
    
    for i in range(labels_np.shape[0]):
        num_true_labels = labels_np[i].sum()
        
        if num_true_labels == 0:
            scores.append(0.0)
            continue
        hits = labels_np[i, np.argsort(probs_np[i])[::-1][:k]].sum()
        
        denominator = min(k, num_true_labels)
        
        scores.append(hits / denominator)
        
    return float(np.mean(scores))


def valid_classification(encoder, classification_head, dataloader, criterion, device, freeze, use_sum, 
                         pos_weight=None, k: int = 5):
    
    classification_head.to(device)
    classification_head.eval()
    encoder.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for smiles_batch, ms_batch, label_batch, chem_emb in tqdm(dataloader):
            
            label_batch = label_batch.to(device, non_blocking=True)

            if not freeze:
                mzs, intens, masks = ms_batch

                mzs_gpu = mzs.to(device, non_blocking=True)
                intens_gpu = intens.to(device, non_blocking=True)
                masks_gpu = masks.to(device, non_blocking=True)

                ms_inputs = (mzs_gpu, intens_gpu, masks_gpu)

                mw_list = []
                for smi in smiles_batch:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        mw_list.append(0.0) 
                    else:
                        mw_list.append(Descriptors.ExactMolWt(mol))
                        
                mw_inputs = torch.tensor(mw_list, dtype=torch.float32, device=device).unsqueeze(1)

                with torch.no_grad():
                    ms_emb = encoder(ms_inputs, mode='emb')
                    if use_sum:
                        ms_emb = ms_emb.sum(dim=1) 
                ms_emb = ms_emb.detach()

                features_for_head = ms_emb
            else:
                if isinstance(chem_emb, torch.Tensor) and chem_emb.device != device:
                    chem_emb = chem_emb.to(device)
                    
                features_for_head = chem_emb
                
            logits = classification_head(features_for_head).to(device, non_blocking=True)
            # loss = loss_fn(logits, label_batch)
            loss = criterion(logits, label_batch)
            losses.append(loss.item())

            all_logits.append(logits.cpu())
            all_labels.append(label_batch.cpu())
        
    avg_loss = sum(losses) / len(losses)
    
    all_logits_tensor = torch.cat(all_logits)
    all_labels_tensor = torch.cat(all_labels)
    
    all_probs_np = torch.sigmoid(all_logits_tensor).numpy() 
    labels_np = all_labels_tensor.int().numpy()
    
    try:
        per_class_auc = roc_auc_score(labels_np, all_probs_np, average=None)
        valid_mask = ~np.isnan(per_class_auc)

        micro_auc    = roc_auc_score(labels_np, all_probs_np, average='micro')
        macro_auc    = float(np.mean(per_class_auc[valid_mask]))     if valid_mask.any() else 0.0
        w            = pos_weight[valid_mask] if pos_weight is not None else np.ones(valid_mask.sum())
        weighted_auc = float(np.average(per_class_auc[valid_mask], weights=w)) if valid_mask.any() else 0.0

    except ValueError:
        micro_auc = macro_auc = weighted_auc = 0.0

    print("pos_weight is None:", pos_weight is None)
    print("w:", w)
    print("per_class_auc[valid_mask]:", per_class_auc[valid_mask])
            
    predictions_np = (all_probs_np > 0.5).astype(int)
    micro_f1 = f1_score(labels_np, predictions_np, average='micro', zero_division=0)
    prec_k = adjusted_precision_at_k(labels_np, all_probs_np, k)
    
    print(f"Val Loss: {avg_loss:.4f} | Micro-AUC: {micro_auc:.4f} | Macro-AUC: {macro_auc:.4f} | "
      f"Weighted-AUC: {weighted_auc:.4f} | Micro-F1: {micro_f1:.4f} | P@{k}: {prec_k:.4f} "
      f"| Valid classes: {valid_mask.sum()}/{len(per_class_auc)}")
    
    return avg_loss, micro_auc, weighted_auc, macro_auc,prec_k, labels_np, predictions_np 

def plot_average_roc_curve(labels_np, probs_np, title="Micro-Averaged ROC Curve", output_path="roc_curve.png"):
    
    n_classes = labels_np.shape[1] 

    fpr_micro, tpr_micro, _ = roc_curve(labels_np.ravel(), probs_np.ravel())
    
    micro_auc = auc(fpr_micro, tpr_micro)
    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr_micro,
        tpr_micro,
        label=f'Micro-AUC (Area = {micro_auc:.4f})',
        color='blue',
        lw=3,
        alpha=0.8
    )

    plt.plot([0, 1], [0, 1], 'r--', label='Chance Level', lw=1.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6) 


    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC Curve saved to {output_path}")
    return micro_auc