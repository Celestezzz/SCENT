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


# 统一smiles表达形式
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
    return fpgen.GetFingerprint(mol)   # 注意这里是 mol

# clip的训练只需要smiles、ms、和fp
# class CLIPDataset(Dataset):
#     def __init__(self, merged_df):
#         self.cids = merged_df["cid"].tolist()
#         self.smiles = merged_df["canonical_smiles"].tolist()

#         # (1) MS raw vector
#         self.ms_raw = merged_df.iloc[:, 4+1:4+1000].values
        
#         # # (2) Convert MS to MolSpec list
#         self.ms_mol = to_model_data(self.ms_raw)

#         # (3) Fingerprints
#         self.fps = [smiles_to_fp(s) for s in self.smiles]

#         # (4) chem_emb
#         self.chem_emb = merged_df.iloc[:, 4+1760+1:].values


#     def __len__(self):
#         return len(self.cids)

#     def __getitem__(self, idx):
#         smiles = self.smiles[idx]
#         molspec = self.ms_mol[idx]         # <- Give MolSpec, NOT tensor
#         fp = self.fps[idx]
#         chem_emb = self.chem_emb[idx]

#         ms_data = (molspec.mz, molspec.intensities)

#         return smiles, ms_data, fp, chem_emb

# class CLIPDataset(Dataset):
#     def __init__(self, merged_df):
#         self.cids = merged_df["cid"].tolist()
#         self.smiles = merged_df["canonical_smiles"].tolist()

#         # (1) MS raw vector (保留，快速读取)
#         # 注意：这里假设 .values 返回的是一个 NumPy 数组
#         self.ms_raw = merged_df.iloc[:, 4+1:4+1000+1].values 
        
#         # # (2) Convert MS to MolSpec list - 移除！在 __getitem__ 中计算。
#         # self.ms_mol = to_model_data(self.ms_raw) 

#         # (3) Fingerprints - 移除！在 __getitem__ 中计算。
#         # self.fps = [smiles_to_fp(s) for s in self.smiles] 

#         # (4) chem_emb (保留，快速读取)
#         self.chem_emb = merged_df.iloc[:, 4+1760+1:].values


#     def __len__(self):
#         return len(self.cids)

#     def __getitem__(self, idx):
#         smiles = self.smiles[idx]
        
#         # =======================================================
#         # ⚡️ 关键改动：将耗时计算移到这里，由 Worker 进程并行执行
#         # =======================================================
        
#         # 1. MolSpec 转换 (Spec2Vec 预处理)
#         # 获取当前行的原始MS数据，并调用 to_model_data
#         ms_raw_vector = self.ms_raw[idx] 
#         # 假设 to_model_data 接收一个 N行 x M列 的矩阵，我们只传 1行 x M列
#         # 您可能需要根据 to_model_data 的实际需求调整输入形状
#         ms_mol = to_model_data(ms_raw_vector.reshape(1, -1))[0] # 假设返回列表，取第一个元素
        
#         # 2. Fingerprints 计算 (RDKit 预处理)
#         fp = smiles_to_fp(smiles) 
        
#         # =======================================================

#         chem_emb = self.chem_emb[idx]

#         ms_data = (ms_mol.mz, ms_mol.intensities)

#         # 注意：这里返回的 fp 仍然是 RDKit 对象，但现在是在 worker 进程中创建的
#         return smiles, ms_data, fp, chem_emb

class CLIPDataset(Dataset):
    def __init__(self, merged_df, augmentor=None, training=True):
        """
        merged_df: Pandas DataFrame
        augmentor: MSAugmentor 实例
        training: bool, 控制是否开启增强
        """
        # 1. 预先提取列到内存中 (List/Numpy)，避免在 __getitem__ 中使用 loc/iloc
        # 这比 to_dict('records') 更省内存，比 iloc 更快
        self.cids = merged_df["cid"].tolist()
        self.smiles = merged_df["canonical_smiles"].tolist()
        
        # 假设 ms_raw 是从第5列开始的 1000 维数据
        # 请确保这里的列索引切片是正确的
        self.ms_raw = merged_df.iloc[:, 5:1005].values 
        
        # 假设 chem_emb 是从第 1765 列开始的数据
        self.chem_emb = merged_df.iloc[:, 1765:].values
        
        self.augmentor = augmentor
        self.training = training

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):
        # -------------------------------------------------------
        # Part A: SMILES & Fingerprint (CPU 计算)
        # -------------------------------------------------------
        smiles = self.smiles[idx]
        fp = smiles_to_fp(smiles) # 假设这是你的外部函数

        # -------------------------------------------------------
        # Part B: MS 处理与增强 (CPU 计算)
        # -------------------------------------------------------
        # 1. 获取原始向量
        ms_raw_vector = self.ms_raw[idx]
        
        # 2. 解析为物理谱图 (mz, intensity)
        # 假设 to_model_data 返回一个对象列表，我们取第一个
        # 请根据你实际的 to_model_data 返回值调整
        ms_mol_obj = to_model_data(ms_raw_vector.reshape(1, -1))[0] 
        
        mz = np.array(ms_mol_obj.mz)
        inten = np.array(ms_mol_obj.intensities)

        # 3. 【关键插入点】数据增强
        # 只有在训练模式且有增强器时才执行
        if self.training and self.augmentor:
            mz, inten = self.augmentor.augment(mz, inten)
        
        # 4. 简单的最大值归一化 (可选，但推荐)
        if len(inten) > 0 and np.max(inten) > 0:
            inten = inten * 999 / np.max(inten)
        
        # 5. 转为 Tensor (虽然 collate_fn 可能会再转一次，这里转比较安全)
        # 或者保持 numpy，让 collate_fn 处理
        # 这里为了保持数据结构，打包成 tuple
        ms_data = (mz, inten)

        # -------------------------------------------------------
        # Part C: Chemical Embedding
        # -------------------------------------------------------
        chem_emb = self.chem_emb[idx]

        # -------------------------------------------------------
        # Return
        # -------------------------------------------------------
        # 返回结构：(smiles_str, (mz, int), fp_obj, chem_emb_vec)
        return smiles, ms_data, fp, chem_emb

def structure_split_indices(full_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    基于 canonical_smiles 结构进行划分，确保相同 SMILES 的所有行都在同一集合中。
    
    Args:
        full_df (pd.DataFrame): 完整的 DataFrame。
        train_ratio, val_ratio, test_ratio (float): 目标比例。
        seed (int): 随机种子，用于复现性。
        
    Returns:
        list, list, list: 训练集、验证集、测试集的全局索引列表。
    """
    
    # 1. 设置随机种子
    np.random.seed(seed)
    
    # 2. 按 SMILES 分组并随机打乱组的顺序
    smiles_groups = full_df.groupby('canonical_smiles').groups
    unique_smiles = list(smiles_groups.keys())
    np.random.shuffle(unique_smiles) # 打乱 SMILES 的顺序

    N_unique = len(unique_smiles)
    
    # 3. 计算划分的边界
    n_train = int(N_unique * train_ratio)
    n_val = int(N_unique * val_ratio)
    
    # 4. 划分 SMILES 结构组
    train_smiles_set = unique_smiles[:n_train]
    val_smiles_set = unique_smiles[n_train : n_train + n_val]
    # 剩余的全部作为测试集
    test_smiles_set = unique_smiles[n_train + n_val :]
    
    # 5. 提取实际的行索引
    train_indices = []
    val_indices = []
    test_indices = []

    for smiles in train_smiles_set:
        # 将该 SMILES 对应的所有行索引加入训练集
        train_indices.extend(smiles_groups[smiles].tolist())
        
    for smiles in val_smiles_set:
        # 将该 SMILES 对应的所有行索引加入验证集
        val_indices.extend(smiles_groups[smiles].tolist())

    for smiles in test_smiles_set:
        # 将该 SMILES 对应的所有行索引加入测试集
        test_indices.extend(smiles_groups[smiles].tolist())

    print(f"结构划分完成：训练集样本数={len(train_indices)}, 验证集样本数={len(val_indices)}, 测试集样本数={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def clip_collate(batch):
    """
    batch: list of tuples (smiles, molspec, fp)
    """
    smiles_list = []
    fps_list = []          # 用来存 RDKit 指纹对象
    ms_raw_list = [] 
    chem_list = []

    # --- 1. 解压 Batch ---
    for smiles, ms_data, fp, chem_emb in batch:
        smiles_list.append(smiles)
        ms_raw_list.append(ms_data) 
        fps_list.append(fp) # fp 是 RDKit 对象，直接存入列表
        chem_list.append(chem_emb)

    # --- 2. 处理 MS 数据 (保持不变) ---
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

    # --- 3. 处理 Fingerprints (关键修改！) ---
    # 不要转 Tensor！直接返回列表！
    # 因为 build_negative_mask 里的 RDKit 函数需要原始对象
    return smiles_list, ms_inputs, fps_list, chem_list
# def clip_collate(batch, mz_min=50, mz_max=180, max_len_cap=None):
#     smiles_list, fps_list, chem_list = [], [], []

#     ms_raw_list = []
#     for smiles, ms_data, fp, chem_emb in batch:
#         smiles_list.append(smiles)
#         fps_list.append(fp)
#         chem_list.append(chem_emb)
#         ms_raw_list.append(ms_data)

#     # 1) 先过滤到窗口内且 intensity>0（保持“稀疏峰token”语义）
#     filtered = []
#     lengths = []
#     for mz, inten in ms_raw_list:
#         mz = np.asarray(mz, dtype=np.int64)
#         inten = np.asarray(inten, dtype=np.float32)

#         keep = (mz >= mz_min) & (mz <= mz_max) & (inten > 0)
#         mz_k = mz[keep]
#         in_k = inten[keep]

#         # 可选：按 m/z 排序（强烈建议保持与训练一致）
#         order = np.argsort(mz_k)
#         mz_k = mz_k[order]
#         in_k = in_k[order]

#         filtered.append((mz_k, in_k))
#         lengths.append(len(mz_k))

#     # 2) 决定 pad 长度：batch 内最大，或加上一个上限 cap
#     max_len = max(lengths) if len(lengths) > 0 else 1
#     if max_len_cap is not None:
#         max_len = min(max_len, max_len_cap)
#         # 如超过 cap，可截断（建议保留强度最高或按m/z截断，需与你训练一致）
    
#     # 3) padding
#     mzs_padded, intens_padded, masks_padded = [], [], []
#     for mz_k, in_k in filtered:
#         if max_len_cap is not None and len(mz_k) > max_len:
#             # 示例：保留强度最高的 max_len 个峰（你也可以按m/z截断）
#             top = np.argsort(in_k)[-max_len:]
#             mz_k, in_k = mz_k[top], in_k[top]
#             order = np.argsort(mz_k)
#             mz_k, in_k = mz_k[order], in_k[order]

#         pad_num = max_len - len(mz_k)

#         mz_con = np.pad(mz_k, (0, pad_num), mode='constant', constant_values=0)
#         in_con = np.pad(in_k, (0, pad_num), mode='constant', constant_values=0.0)
#         mask = np.pad(np.ones(len(mz_k), dtype=bool), (0, pad_num),
#                       mode='constant', constant_values=False)

#         mzs_padded.append(mz_con)
#         intens_padded.append(in_con)
#         masks_padded.append(mask)

#     mzs_tensor = torch.tensor(np.array(mzs_padded), dtype=torch.long)
#     intens_tensor = torch.tensor(np.array(intens_padded), dtype=torch.float)
#     masks_tensor = torch.tensor(np.array(masks_padded), dtype=torch.bool)

#     ms_inputs = (mzs_tensor, intens_tensor, masks_tensor)
#     return smiles_list, ms_inputs, fps_list, chem_list

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MSAugmentor:
    """
    针对 50-200 窗口限制的专用增强器
    包含：强制窗口截断、强度抖动、假峰注入
    """
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

        # 1. 强制窗口截断 (必须在最开始做)
        valid_mask = (mz >= self.min_mz) & (mz <= self.max_mz)
        if np.sum(valid_mask) == 0:
            return np.array([self.min_mz], dtype=np.float32), np.array([0.0], dtype=np.float32)
        mz, inten = mz[valid_mask], inten[valid_mask]

        # # 2. 【核心新增】模拟断谱 (Segment Drop)
        # if np.random.rand() < self.drop_seg_prob:
        #     drop_start = np.random.uniform(self.min_mz, self.max_mz - 10)
        #     drop_end = drop_start + np.random.uniform(5, self.max_drop_width)
        #     keep_mask = (mz < drop_start) | (mz > drop_end)
        #     if np.sum(keep_mask) > 0: # 确保不会全删光
        #         mz, inten = mz[keep_mask], inten[keep_mask]

        # 2. 强度抖动 (Intensity Jitter)
        # ---------------------------------------------------------
        if np.random.rand() < self.jitter_prob:
            noise = np.random.uniform(1 - self.jitter_scale, 
                                      1 + self.jitter_scale, 
                                      size=inten.shape)
            inten = inten * noise

        # 3. 孤狼假峰注入 (Ghost Peak Injection)
        # ---------------------------------------------------------
        if np.random.rand() < self.ghost_prob:
            ghost_mz = np.random.uniform(self.min_mz, self.max_mz, self.num_ghosts)
            
            # 假峰强度：当前最大强度的 1% - 15%
            max_val = np.max(inten) if len(inten) > 0 else 1.0
            ghost_inten = np.random.uniform(0.01, 0.15, self.num_ghosts) * max_val
            
            mz = np.concatenate([mz, ghost_mz])
            inten = np.concatenate([inten, ghost_inten])

        # 4. 排序 (Transformer 输入通常不需要顺序，但排序更规范)
        sort_idx = np.argsort(mz)
        return mz[sort_idx], inten[sort_idx]

# ---------------------------------------------------------
# 集成到 Dataset 中
# ---------------------------------------------------------
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
    #     inten = mol_spec.intensities # 注意属性名是 .intensities

        
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
        
        # 防止 pad_num 小于 0 (虽然理论上有了 max_len 不会，但为了稳健)
        if pad_num < 0:
            pad_num = 0
        
        # Pad m/z 和 intensity
        mz_con = np.pad(mz, (0, pad_num), mode='constant', constant_values=0)
        inten_con = np.pad(inten, (0, pad_num), mode='constant', constant_values=0)
        
        # Create Mask (真实数据为 True/1, Padding 为 False/0)
        # 这里的 mask 逻辑取决于你模型怎么用，通常 padding 部分 mask 为 0
        mask = np.pad(np.ones(len_mz, dtype=bool), (0, pad_num), mode='constant', constant_values=False)

        mzs_padded.append(mz_con)
        intens_padded.append(inten_con)
        masks_padded.append(mask)

    # 转换为 Tensor
    mzs_tensor = torch.tensor(np.array(mzs_padded), dtype=torch.long)
    intens_tensor = torch.tensor(np.array(intens_padded), dtype=torch.float)
    masks_tensor = torch.tensor(np.array(masks_padded), dtype=torch.bool)

    ms_inputs = (mzs_tensor, intens_tensor, masks_tensor)
    
    labels_np = np.stack(label_list)
    labels_tensor = torch.tensor(labels_np, dtype=torch.float)

    chem_tensor = torch.stack(chem_list)

    return smiles_list, ms_inputs, labels_tensor, chem_tensor