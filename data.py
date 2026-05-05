import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from utils_ms import to_model_data


def smiles_to_canonical(smi):
    if pd.isna(smi) or smi is None or smi == "":
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

# def smiles_to_fp(smiles, radius=2):
#     # Convert SMILES to Morgen fingerprint
#     # smiles contains 2 SMILES strings (which is a pair to compare)
#     mfp_fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2)
#     mfp_fps = [mfp_fpg.GetFingerprint(mol) for mol in smiles]
#     return mfp_fps

def smiles_to_fp(smiles, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=1024)
    return fpgen.GetFingerprint(mol)  

class CLIPDataset(Dataset):
    def __init__(self, merged_df, augmentor=None, training=True):

        self.cids = merged_df["cid"].tolist()
        self.smiles = merged_df["canonical_smiles"].tolist()
        
        self.ms_raw = merged_df.iloc[:, 5:1005].values 
        
        self.chem_emb = merged_df.iloc[:, 1765:].values
        
        self.augmentor = augmentor
        self.training = training

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
 
        smiles = self.smiles[idx]
        fp = smiles_to_fp(smiles) 
        ms_raw_vector = self.ms_raw[idx]
        
        ms_mol_obj = to_model_data(ms_raw_vector.reshape(1, -1))[0] 
        
        mz = np.array(ms_mol_obj.mz)
        inten = np.array(ms_mol_obj.intensities)


        if self.training and self.augmentor:
            mz, inten = self.augmentor.augment(mz, inten)

        if len(inten) > 0 and np.max(inten) > 0:
            inten = inten * 999 / np.max(inten)
        

        ms_data = (mz, inten)

        # -------------------------------------------------------
        # Part C: Chemical Embedding
        # -------------------------------------------------------
        chem_emb = self.chem_emb[idx]

        # -------------------------------------------------------
        # Return
        # -------------------------------------------------------
        return smiles, ms_data, fp, chem_emb

def structure_split_indices(full_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):


    np.random.seed(seed)

    smiles_groups = full_df.groupby('canonical_smiles').groups
    unique_smiles = list(smiles_groups.keys())
    np.random.shuffle(unique_smiles) 

    N_unique = len(unique_smiles)

    n_train = int(N_unique * train_ratio)
    n_val = int(N_unique * val_ratio)
    

    train_smiles_set = unique_smiles[:n_train]
    val_smiles_set = unique_smiles[n_train : n_train + n_val]

    test_smiles_set = unique_smiles[n_train + n_val :]

    train_indices = []
    val_indices = []
    test_indices = []

    for smiles in train_smiles_set:

        train_indices.extend(smiles_groups[smiles].tolist())
        
    for smiles in val_smiles_set:

        val_indices.extend(smiles_groups[smiles].tolist())

    for smiles in test_smiles_set:

        test_indices.extend(smiles_groups[smiles].tolist())

    print(f"结构划分完成：训练集样本数={len(train_indices)}, 验证集样本数={len(val_indices)}, 测试集样本数={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def clip_collate(batch):
    """
    batch: list of tuples (smiles, molspec, fp)
    """
    smiles_list = []
    fps_list = []          
    ms_raw_list = [] 
    chem_list = []


    for smiles, ms_data, fp, chem_emb in batch:
        smiles_list.append(smiles)
        ms_raw_list.append(ms_data) 
        fps_list.append(fp)
        chem_list.append(chem_emb)


    max_len = max([len(x[0]) for x in ms_raw_list])
    
    mzs_padded = []
    intens_padded = []
    masks_padded = []

    for mz, inten in ms_raw_list:
        len_mz = len(mz)
        pad_num = max_len - len_mz
        
        mz_con = np.pad(mz, (0, pad_num), mode='constant', constant_values=0)
        inten_con = np.pad(inten, (0, pad_num), mode='constant', constant_values=0)
        mask = np.pad(np.ones_like(mz, dtype=bool), (0, pad_num), mode='constant', constant_values=False)

        mzs_padded.append(mz_con)
        intens_padded.append(inten_con)
        masks_padded.append(mask)

    mzs_tensor = torch.tensor(np.array(mzs_padded), dtype=torch.long)
    intens_tensor = torch.tensor(np.array(intens_padded), dtype=torch.float)
    masks_tensor = torch.tensor(np.array(masks_padded), dtype=torch.bool)

    ms_inputs = (mzs_tensor, intens_tensor, masks_tensor)

    return smiles_list, ms_inputs, fps_list, chem_list


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MSAugmentor:

    def __init__(self, 
                 min_mz=50, max_mz=180, 
                 jitter_prob=0.8, jitter_scale=0.2, 
                 ghost_prob=0.5, num_ghosts=3,
                 drop_seg_prob=0.5, max_drop_width=30):
        
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.jitter_prob = jitter_prob
        self.jitter_scale = jitter_scale
        self.ghost_prob = ghost_prob
        self.num_ghosts = num_ghosts
        self.drop_seg_prob = drop_seg_prob
        self.max_drop_width = max_drop_width

    def augment(self, mz_array, intensity_array, is_training=True):
        mz = mz_array.astype(np.float32)
        inten = intensity_array.astype(np.float32)

        if not is_training:
            return mz, inten


        valid_mask = (mz >= self.min_mz) & (mz <= self.max_mz)
        if np.sum(valid_mask) == 0:
            return np.array([self.min_mz], dtype=np.float32), np.array([0.0], dtype=np.float32)
        mz, inten = mz[valid_mask], inten[valid_mask]


        # if np.random.rand() < self.drop_seg_prob:
        #     drop_start = np.random.uniform(self.min_mz, self.max_mz - 10)
        #     drop_end = drop_start + np.random.uniform(5, self.max_drop_width)
        #     keep_mask = (mz < drop_start) | (mz > drop_end)
        #     if np.sum(keep_mask) > 0:
        #         mz, inten = mz[keep_mask], inten[keep_mask]


        # ---------------------------------------------------------
        if np.random.rand() < self.jitter_prob:
            noise = np.random.uniform(1 - self.jitter_scale, 
                                      1 + self.jitter_scale, 
                                      size=inten.shape)
            inten = inten * noise


        # ---------------------------------------------------------
        if np.random.rand() < self.ghost_prob:
            ghost_mz = np.random.uniform(self.min_mz, self.max_mz, self.num_ghosts)
     
            max_val = np.max(inten) if len(inten) > 0 else 1.0
            ghost_inten = np.random.uniform(0.01, 0.15, self.num_ghosts) * max_val
            
            mz = np.concatenate([mz, ghost_mz])
            inten = np.concatenate([inten, ghost_inten])


        sort_idx = np.argsort(mz)
        return mz[sort_idx], inten[sort_idx]

class ClassificationData(Dataset):
    def __init__(self, ms_df, label=None, chem_emb=None):
        
        if label is not None and 'nonStereoSMILES' in label.columns:
            self.smiles = label['nonStereoSMILES'].tolist()
        elif 'canonical_smiles' in ms_df.columns:
            self.smiles = ms_df['canonical_smiles'].tolist()
        else:
            self.smiles = [f"mol_{i}" for i in range(len(ms_df))]

        self.ms_raw = ms_df.iloc[:, 6 : 1006].values 

        self.has_label = label is not None
        if self.has_label:
            self.label = label.iloc[:, 3:].values
            
        self.has_chem = chem_emb is not None
        if self.has_chem:
            if isinstance(chem_emb, pd.DataFrame):
                self.chem_emb = chem_emb.values.astype(np.float32)
            else:
                self.chem_emb = chem_emb.astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]

        ms_raw_vector = self.ms_raw[index] 
        ms_mol = to_model_data(ms_raw_vector.reshape(1, -1))[0]
        out_label = self.label[index, :] if self.has_label else np.array([-1.0], dtype=np.float32)
        out_chem = self.chem_emb[index, :] if self.has_chem else np.array([-1.0], dtype=np.float32)

        return smiles, ms_mol, out_label, out_chem
    
def classification_collate(batch, augmentor = None):
    smiles_list = []
    ms_raw_list = []
    label_list = []
    chem_list = []
    processed_specs = []
    
    for smiles, ms_mol, label, chem_emb in batch:
        smiles_list.append(smiles)
        ms_raw_list.append(ms_mol)
        label_list.append(label)
        if isinstance(chem_emb, np.ndarray):
            chem_list.append(torch.from_numpy(chem_emb).float()) # 转换为 Tensor
        else:
            chem_list.append(chem_emb.float())

        mz = ms_mol.mz
        inten = ms_mol.intensities
        if augmentor:
            mz, inten = augmentor.augment(mz, inten)
        
        processed_specs.append((mz, inten))
    
    # max_len = max([len(x.mz) for x in ms_raw_list])
    if len(processed_specs) > 0:
        max_len = max([len(spec[0]) for spec in processed_specs])
    else:
        max_len = 0
    
    mzs_padded = []
    intens_padded = []
    masks_padded = []
    # for mol_spec in ms_raw_list:
    #     mz = mol_spec.mz
    #     inten = mol_spec.intensities 

        
    #     len_mz = len(mz)
    #     pad_num = max_len - len_mz
        
    #     mz_con = np.pad(mz, (0, pad_num), mode='constant', constant_values=0)
    #     inten_con = np.pad(inten, (0, pad_num), mode='constant', constant_values=0)
    #     mask = np.pad(np.ones_like(mz, dtype=bool), (0, pad_num), mode='constant', constant_values=False)

    #     mzs_padded.append(mz_con)
    #     intens_padded.append(inten_con)
    #     masks_padded.append(mask)

    # mzs_tensor = torch.tensor(np.array(mzs_padded), dtype=torch.long)
    # intens_tensor = torch.tensor(np.array(intens_padded), dtype=torch.float)
    # masks_tensor = torch.tensor(np.array(masks_padded), dtype=torch.bool)

    # ms_inputs = (mzs_tensor, intens_tensor, masks_tensor)
    # labels_np = np.stack(label_list)
    # # labels_indices_np = np.argmax(labels_np, axis=1) 
    # labels_tensor = torch.tensor(labels_np, dtype=torch.float)

    # chem_tensor = torch.stack(chem_list)

    # return smiles_list, ms_inputs, labels_tensor, chem_tensor
    for mz, inten in processed_specs:
        len_mz = len(mz)
        pad_num = max_len - len_mz
        
        if pad_num < 0:
            pad_num = 0
        

        mz_con = np.pad(mz, (0, pad_num), mode='constant', constant_values=0)
        inten_con = np.pad(inten, (0, pad_num), mode='constant', constant_values=0)
        
        mask = np.pad(np.ones(len_mz, dtype=bool), (0, pad_num), mode='constant', constant_values=False)

        mzs_padded.append(mz_con)
        intens_padded.append(inten_con)
        masks_padded.append(mask)

    mzs_tensor = torch.tensor(np.array(mzs_padded), dtype=torch.long)
    intens_tensor = torch.tensor(np.array(intens_padded), dtype=torch.float)
    masks_tensor = torch.tensor(np.array(masks_padded), dtype=torch.bool)

    ms_inputs = (mzs_tensor, intens_tensor, masks_tensor)
    
    labels_np = np.stack(label_list)
    labels_tensor = torch.tensor(labels_np, dtype=torch.float)

    chem_tensor = torch.stack(chem_list)

    return smiles_list, ms_inputs, labels_tensor, chem_tensor