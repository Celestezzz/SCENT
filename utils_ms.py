import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset

import pandas as pd
import numpy as np

from copy import deepcopy

# EIMS2Vec's utils

class Spec2Emb(nn.Module):
    def __init__(self, num_emb:int=1000, emb_dim:int=500):
        super(Spec2Emb, self).__init__()
        self.max_exp = 6
        self.emb_con = nn.Embedding(
            num_embeddings=num_emb,
            embedding_dim=emb_dim,
        )
        self.emb_cen = nn.Embedding(
            num_embeddings=num_emb,
            embedding_dim=emb_dim,
        )
        self.trip_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # def _compute_embedding(self, mzs, intens, masks, power):
    #     embs = self.emb_cen(mzs)
    #     embs = embs * masks.unsqueeze(-1)
    #     intens = pt.pow(intens, power).unsqueeze(-1)
    #     embs = (embs * intens).sum(dim=1)
    #     return embs

    def _compute_embedding(self, mzs, intens, masks, power):
        device = self.emb_cen.weight.device
        
        if masks.device != device:
            masks = masks.to(device)
            
        if mzs.device != device:
            mzs = mzs.to(device)


        embs = self.emb_cen(mzs) 
        
        if intens.device != device:
            intens = intens.to(device)

        embs = embs * masks.unsqueeze(-1)
        intens = torch.pow(intens, power).unsqueeze(-1)
        embs = (embs * intens)
        return embs

    def forward(self, data, mode:str='train', power:float=0.5):
        if mode == 'train': 
            mzs_con, masks_con, poss_cen, batch_idx, negs_cen, masks_neg = data
            embs_con = self.emb_con(mzs_con)        # [batch, seq, emb_dim]
            embs_pos = self.emb_cen(poss_cen)     # [B, emb_dim]
            embs_neg = self.emb_cen(negs_cen)      # [B, neg_num, emb_dim]
            embs_neg *= masks_neg.unsqueeze(-1)
            # for every cen word its context words
            embs_con = embs_con[batch_idx] * masks_con.unsqueeze(-1)
            embs_con = embs_con.sum(dim=1) / masks_con.sum(dim=1).unsqueeze(-1) # [B, emb_dim]
            pos_score = (embs_con * embs_pos).sum(dim=-1) # 点积
            pos_score = torch.clamp(pos_score, max=self.max_exp, min=-self.max_exp)
            pos_score = -F.logsigmoid(pos_score)
            neg_score = torch.bmm(embs_neg, embs_con.unsqueeze(-1)).squeeze(-1) # 
            neg_score = torch.clamp(neg_score, max=self.max_exp, min=-self.max_exp)
            neg_score = -F.logsigmoid(-neg_score).sum(dim=-1)
            return (pos_score + neg_score).sum() 
        elif mode == 'emb': # emb模式下的masks只mask掉了padding 
            mzs_all, intens_all, masks_all = data  # [batch, seq]
            return self._compute_embedding(mzs_all, intens_all, masks_all, power)
        elif mode == 'finetune':
            data_mea, data_pre_hit, data_pre_nhit = data
            embs_mea = self._compute_embedding(*data_mea, power)
            embs_pre_hit = self._compute_embedding(*data_pre_hit, power)
            embs_pre_nhit = self._compute_embedding(*data_pre_nhit, power)
            # batchsize, emb_dim
            embs_mea = F.normalize(embs_mea, p=2, dim=-1)
            embs_pre_hit = F.normalize(embs_pre_hit, p=2, dim=-1)
            embs_pre_nhit = F.normalize(embs_pre_nhit, p=2, dim=-1)
            # batchsize
            loss = self.trip_loss(embs_mea, embs_pre_hit, embs_pre_nhit)
            return loss
        else:
            raise ValueError('mode not exist')

class MolSpec:
    def __init__(self, mz, intensities):
        self.mz = mz                 # numpy array of ints
        self.intensities = intensities   # numpy array of floats


def to_model_data(big_arr):
    mols = []
    for i in range(big_arr.shape[0]):
        spec = big_arr[i] * 999 / (max(big_arr[i]) + + 1e-12)
        mz_idx = np.where(spec > 0)[0]
        intens = spec[mz_idx]

        mols.append(MolSpec(
            mz = mz_idx.astype(int),
            intensities = intens.astype(float)
        ))
    return mols

def collate_fun(keep_prob:np.array, neg_prob:np.array, neg_num:int=5, min_len_mz:int=10, min_inten:float=0.01):
    neg_choice = np.arange(neg_prob.shape[0])
    def collate_fn(batch):
        # con: context, cen: center
        mzs_con, masks_con, poss_cen, batch_idx, negs_cen, masks_neg = [], [], [], [], [], []
        max_len = max([len(mz) for mz, _ in batch])
        idx = 0
        for mz, inten in batch:
            len_mz = len(mz)
            if len_mz >= min_len_mz: # 移除峰的数量小于阈值的质谱 
                pad_num = max_len - len_mz
                pos_cen = []
                mask_down = np.random.random(len_mz) < keep_prob[mz]
                for i in range(len_mz):
                    if mask_down[i] and inten[i] > min_inten: # 如果没有被mask掉
                        mask_pos_down = np.array(mask_down)
                        mask_pos_down[i] = False
                        if np.any(mask_pos_down): # 上下文没有被全部mask掉
                            pos_cen.append(mz[i])
                            masks_con.append(np.pad(mask_pos_down, (0, pad_num)))
                if len(pos_cen) == 0: # 整个质谱中的中心词都被mask掉了
                    continue   
                mzs_con.append(np.pad(mz, (0, pad_num)))
                poss_cen.extend(pos_cen)
                batch_idx.extend([idx] * len(pos_cen))
                idx += 1
                neg_cen = np.random.choice(neg_choice, (len(pos_cen), neg_num), p=neg_prob)
                mask_neg = neg_cen != np.array(pos_cen)[:, np.newaxis]
                negs_cen.append(neg_cen)
                masks_neg.append(mask_neg)
        if len(mzs_con) == 0:
            return None
        mzs_con = torch.tensor(np.array(mzs_con), dtype=torch.long)
        masks_con = torch.tensor(np.array(masks_con), dtype=torch.bool)
        poss_cen = torch.tensor(np.array(poss_cen), dtype=torch.long)
        batch_idx = torch.tensor(np.array(batch_idx), dtype=torch.int)
        negs_cen = torch.tensor(np.concatenate(negs_cen), dtype=torch.long)
        masks_neg = torch.tensor(np.concatenate(masks_neg), dtype=torch.bool)
        return mzs_con, masks_con, poss_cen, batch_idx, negs_cen, masks_neg
    return collate_fn

def collate_fun_emb(batch):
    mzs_con, intens_con, masks = [], [], []
    max_len = max([len(mz) for mz, _ in batch])
    for mz, inten in batch:
        len_mz = len(mz)
        pad_num = max_len - len_mz
        mz_con = np.pad(mz, (0, pad_num))
        inten_con = np.pad(inten, (0, pad_num))
        mask = np.pad(np.ones_like(mz, dtype=np.bool_), (0, pad_num))
        mzs_con.append(mz_con)
        intens_con.append(inten_con)
        masks.append(mask) 
    mzs_con = torch.tensor(np.array(mzs_con), dtype=torch.long)
    intens_con = torch.tensor(np.array(intens_con), dtype=torch.float)
    masks = torch.tensor(np.array(masks), dtype=torch.bool)
    return mzs_con, intens_con, masks
    

class Linear_Scheduler:
    def __init__(self, optimizer, epochs:int, start_lr:float=0.025, end_lr:float=2.5e-4):
        self.optimizer = optimizer
        self.epochs = epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
    
    def lr_lambda(self, cur_epoch:int, epoch_progress:float):
        progress = (cur_epoch + epoch_progress) / self.epochs
        next_lr = self.start_lr - (self.start_lr - self.end_lr) * progress
        next_lr = max(next_lr, self.end_lr)
        return next_lr

class SpecDataset(Dataset):
    def __init__(self, dataset, mapping=None):
        super(SpecDataset, self).__init__()
        if isinstance(dataset, list):
            self.spectra = dataset
            self.map = np.arange(len(dataset), dtype=np.int64)
        else:
            self.spectra = dataset.spectra
            self.map = mapping
    
    def __getitem__(self, idx):
        idx_ = self.map[idx]
        mzs = self.spectra[idx_].mz.astype(int).tolist()
        intens = self.spectra[idx_].intensities
        return deepcopy(mzs), deepcopy(intens)
    
    def __len__(self):
        return len(self.map)



class MSTransformer(nn.Module):
    def __init__(self, base_encoder, d_model=500, nhead=4, num_layers=2):
        super().__init__()
        self.base_encoder = base_encoder 
        
        # 2. [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            norm_first=True, 
            # dim_feedforward=d_model*4,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, ms_inputs, mode='emb', power = 0.5):
       
        mzs, intens, raw_masks = ms_inputs
        batch_size = mzs.size(0)
        device = mzs.device


        x = self.base_encoder(ms_inputs, mode='emb', power = 0.5) 

      
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # mw_tokens = self.mw_encoder(mw_inputs).unsqueeze(1) # (B, 1, 500)
        # x = torch.cat((cls_tokens, mw_tokens, x), dim=1) # (B, Seq+2, 500)
        x = torch.cat((cls_tokens, x), dim=1) # (B, Seq+2, 500)

        
        padding_mask = (raw_masks == 0) # (B, Seq)
    
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    
        extended_mask = torch.cat((cls_mask, padding_mask), dim=1) # (B, Seq+2)

        x_out = self.transformer(x, src_key_padding_mask=extended_mask)

        final_emb = x_out[:, 0, :]
        
        return self.out_norm(final_emb)

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = nn.functional.linear(x, self.weight, self.bias)

        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return base_out + lora_out

def inject_lora_into_mstransformer(model: nn.Module, rank: int = 4, alpha: float = 16.0):

    for layer in model.transformer.layers:
        layer.linear1 = LoRALinear(layer.linear1, rank=rank, alpha=alpha)
        layer.linear2 = LoRALinear(layer.linear2, rank=rank, alpha=alpha)
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    return model