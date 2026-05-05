import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import deepchem as dc
# from openpom.models.mpnn_pom import MPNNPOMModel
# from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
# from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
from utils import BYOLPredictor, train_clip, validate_clip, plot_validation_metrics
from utils_ms import Spec2Emb, MSTransformer, inject_lora_into_mstransformer
from data import clip_collate, CLIPDataset, structure_split_indices, MSAugmentor

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Fine-tuning for MS and Chem Encoders")

    # --- 路径相关参数 ---
    parser.add_argument('--data_path', type=str, default='.../multimodal_emb_clean', 
                        help='数据根目录路径')
    # parser.add_argument('--chem_csv', type=str, default='curated_GS_LF_merged_4983.csv',
    #                     help='用于初始化 Chem Encoder 的 CSV 文件名')
    parser.add_argument('--merged_csv', type=str, default='exp_clean.csv',
                        help='用于 CLIP 训练的 Merged CSV 文件名')
    parser.add_argument('--base_model_name', type=str, default='base_model.pt',
                        help='预训练的 MS Encoder 权重文件名')
    parser.add_argument('--save_model_name', type=str, default='ms_encoder_clip_best_2lr_full.pt',
                        help='保存的最佳模型文件名')
    parser.add_argument('--chem_model_type', type = str, default = 'openpom',
                        help = '化学模型')
    parser.add_argument('--chem_emb_csv', type=str, default='pom_clean_train_emb.csv')
  

    # --- 训练超参数 ---
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='training epoch')
    parser.add_argument('--lr_enc', type=float, default=5e-4, help='Encoder learning rate')
    parser.add_argument('--lr_proj', type=float, default=1e-5, help='Projector learning rate')
    parser.add_argument('--lora', action='store_true', help='Whether to use LoRA')
    parser.add_argument('--lr_lora', type=float, default=3e-4, help='LoRA learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers number')
    parser.add_argument('--loss_type', type=str, default='clip',
                    choices=['clip', 'byol', 'vicreg'], help='alignment loss type')
    parser.add_argument('--temperature', type=float, default=0.03, help='temperature for contrastive loss')
    parser.add_argument('--use_transformer', action='store_true', help='Whether to use Transformer layers on top of the base encoder')
    parser.add_argument('--use_augmentor', action='store_true', help='Whether to use augmentor for MS data')


    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)


    # input_file_chem = os.path.join(args.data_path, args.chem_csv)
    input_file_merged = os.path.join(args.data_path, args.merged_csv)
    base_model_path = os.path.join('.../multimodal_emb', args.base_model_name)
    
    server_model_dir = os.path.join(args.data_path, 'examples', 'experiments')
    server_checkpoint_path = os.path.join(server_model_dir, 'checkpoint.pt')
    save_model_path = os.path.join(args.data_path, args.save_model_name)

    
    if args.chem_model_type == 'openpom':
        file_path = os.path.join(args.data_path, args.chem_emb_csv)
        full_chem_emb = pd.read_csv(file_path)
        full_chem_emb = full_chem_emb.iloc[:, 2:]
        chem_len= full_chem_emb.shape[1]
    if args.chem_model_type == 'molformer':
        file_path = os.path.join(args.data_path, args.chem_emb_csv)
        full_chem_emb = pd.read_csv(file_path)
        full_chem_emb = full_chem_emb.iloc[:, 2:]
        chem_len= full_chem_emb.shape[1]
    


    print(">>> Initializing MS Encoder...")
    base_encoder = Spec2Emb() 
    
    # if os.path.exists(base_model_path):
    #     state_dict = torch.load(base_model_path, map_location=device)
    #     base_encoder.load_state_dict(state_dict)
    #     print(f"✅ MS Encoder weights loaded from {base_model_path}") base_encoder = Spec2Emb() # 确保你的类定义支持默认初始化
    
    if os.path.exists(base_model_path):
        state_dict = torch.load(base_model_path, map_location=device)
        base_encoder.load_state_dict(state_dict)
        print(f"✅ MS Encoder weights loaded from {base_model_path}")
    else:
        print(f"⚠️ Warning: MS base model not found at {base_model_path}")
    
    # base_model.to(device)
    if args.use_transformer:
        ms_encoder = MSTransformer(
                        base_encoder = base_encoder, 
                        d_model=500,    
                        nhead=4,       
                        num_layers=2   
                    ).to(device)
        use_sum = False
    else:
        ms_encoder = base_encoder.to(device)
        use_sum = True
    # make_rnn_contiguous(chem_encoder)
    # checkpoint = torch.load(base_model_path, map_location=device)
    # ms_encoder.load_state_dict(checkpoint['ms_encoder'])
    # else:
    #     print(f"⚠️ Warning: MS base model not found at {base_model_path}")
    
    # # base_model.to(device)
    # ms_encoder = MSTransformer(
    #                 base_encoder = base_encoder, 
    #                 d_model=500,    
    #                 nhead=4,       
    #                 num_layers=2    
    #             ).to(device)
    # # make_rnn_contiguous(chem_encoder)
    # checkpoint = torch.load(base_model_path, map_location=device)
    # ms_encoder.load_state_dict(checkpoint['ms_encoder'])
    if args.lora:
        ms_encoder = inject_lora_into_mstransformer(ms_encoder, rank=4, alpha=16.0)
    # trainable_params = sum(p.numel() for p in ms_encoder.parameters() if p.requires_grad)
    # total_params = sum(p.numel() for p in ms_encoder.parameters())
    # print(f"Trainable params: {trainable_params} || Total params: {total_params} || {100 * trainable_params / total_params:.2f}%")


    proj = nn.Linear(500, chem_len).to(device)


    for m in proj.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    predictor = None
    if args.loss_type == "byol":
        predictor = BYOLPredictor(dim=chem_len).to(device)


    print(f">>> Loading CLIP Dataset from {input_file_merged}...")
    merged_df = pd.read_csv(input_file_merged, sep=',', skipinitialspace=True, encoding='utf8')
    merged_df = merged_df.reset_index(drop=True)
    full_chem_emb = full_chem_emb.reset_index(drop=True)
    full_data = pd.concat([merged_df, full_chem_emb], axis=1)

    augmentor = MSAugmentor(min_mz=50, max_mz=180, 
                 jitter_prob=0, jitter_scale=0, 
                 ghost_prob=0, num_ghosts=0,
                 drop_seg_prob=0, max_drop_width=0)
    if args.use_augmentor:
        print(">>> Using MS Augmentor for data augmentation during training.")
        dataset = CLIPDataset(full_data, augmentor=augmentor)
    else:
        print(">>> Not using MS Augmentor. Training on original data.")
        dataset = CLIPDataset(full_data, augmentor=None)
    
    train_indices, val_indices, test_indices = structure_split_indices(
        full_df=full_data, 
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1, 
        seed=args.seed 
    )
    
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        collate_fn=clip_collate
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=clip_collate
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=clip_collate
    )
    
    SMILES_INDEX = 0

    train_smiles = set()

    for i in range(len(train_ds)):

        sample_tuple = train_ds[i]
        
        # 从元组中提取第一个元素，即 SMILES 字符串
        smiles = sample_tuple[SMILES_INDEX] 
        
        train_smiles.add(smiles)
    SMILES_INDEX = 0

    val_smiles = set()

    print(">>> extracting unique SMILES from validation set...")

    for i in range(len(val_ds)):
        sample_tuple = val_ds[i]
        
        smiles = sample_tuple[SMILES_INDEX] 
        
        val_smiles.add(smiles)

    overlap = train_smiles.intersection(val_smiles)
    print(f'check data lackage: {overlap}')

    if args.lora:
        for param in ms_encoder.parameters():
            param.requires_grad = False

        for name, param in ms_encoder.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

        for param in proj.parameters():
            param.requires_grad = True 

        lora_params = [p for n, p in ms_encoder.named_parameters() if 'lora_' in n and p.requires_grad]
        proj_params = [p for p in proj.parameters() if p.requires_grad]

        optimizer = optim.Adam([
            {'params': lora_params, 'lr': args.lr_lora, 'weight_decay': 1e-4},
        
            {'params': proj_params, 'lr': args.lr_proj, 'weight_decay': 1e-5}
        ])
    else:
        if args.loss_type == "byol":
            optimizer = optim.Adam([
                {'params': ms_encoder.parameters(), 'lr': args.lr_enc, 'weight_decay': 1e-4},
                {'params': proj.parameters(),       'lr': args.lr_proj, 'weight_decay': 1e-5},
                {'params': predictor.parameters(),  'lr': args.lr_proj, 'weight_decay': 1e-5},
            ])
        else:
            optimizer = optim.Adam([
                            {'params': ms_encoder.parameters(), 'lr': args.lr_enc, 'weight_decay': 1e-4},
                            {'params': proj.parameters(), 'lr': args.lr_proj, 'weight_decay': 1e-5}
                        ])
    
    
    scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            threshold_mode='abs',   
            threshold=0.005,      
            min_lr=1e-6
        )

    print("\n============ Start CLIP Fine-tuning ============\n")
    

    best_val_loss = float("inf")
    topk_sim_history = []
    val_auc_history = []
    val_loss_history = []

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

        train_loss = train_clip(       
            ms_encoder,
            train_loader, 
            proj=proj,
            device=device,
            optimizer = optimizer,
            loss_type = args.loss_type,
            predictor = predictor,
            temperature = args.temperature,
            use_sum = use_sum
        )
        print(f"[Epoch {epoch+1}] Train Loss = {train_loss:.6f}")

        # VALIDATE
        val_loss, val_auc = validate_clip(
            ms_encoder=ms_encoder,
            val_loader=val_loader,
            proj=proj,
            device=device,
            loss_type = args.loss_type,
            predictor = predictor,
            temperature = args.temperature,
            use_sum = use_sum
        )

        val_loss_history.append(val_loss)
        val_auc_history.append(val_auc)
        # topk_sim_history.append(avg_topk_chem_sim)

        print(f"[Epoch {epoch+1}] Val   Loss = {val_loss:.6f} | Val AUC = {val_auc:.4f} ")

        scheduler.step(val_loss)

        # SAVE BEST
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.loss_type == "byol":
                checkpoint = {
                    'ms_encoder': ms_encoder.state_dict(),
                    'proj': proj.state_dict(),
                    'predictor': predictor.state_dict()
                }
            else:
            # 保存两个模型参数
                checkpoint = {
                    'ms_encoder': ms_encoder.state_dict(),
                    'proj': proj.state_dict()
                }
            torch.save(checkpoint, save_model_path)
            print(f"  ➤ Best model updated. Saved to: {save_model_path}")

    print("\n============ Training Completed ============\n")
    
    print(f"Loading best model from {save_model_path} for testing...")
    checkpoint = torch.load(save_model_path, map_location=device)
    ms_encoder.load_state_dict(checkpoint['ms_encoder'])
    proj.load_state_dict(checkpoint['proj'])
    
    test_loss, test_auc = validate_clip(
        ms_encoder=ms_encoder,
        val_loader=test_loader,
        proj=proj,
        device=device,
        loss_type = args.loss_type,
        predictor = predictor,
        temperature = args.temperature,
        use_sum = use_sum
    )
    print(f"Final Test Loss = {test_loss:.6f}")
    print(f"Final Test AUC = {test_auc:.4f}")
    # print(f"Final Test Top-10 Sim = {test_topk_sim:.4f}")
    print(f"Best Val Loss = {best_val_loss:.6f}")

    # print("\n>>> saving the best model...")
    # plot_validation_metrics(
    #     topk_sim_history, 
    #     val_loss_history, 
    #     val_auc_history, 
    #     args.epochs
    # )

if __name__ == "__main__":
    args = parse_args()
    main(args)