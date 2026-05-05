from email import parser
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import deepchem as dc
# from openpom.models.mpnn_pom import MPNNPOMModel
# from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import IterativeStratifiedSplitter

from data import ClassificationData, classification_collate, MSAugmentor
from utils_clas import Classification, FocalLoss, train_classfication, valid_classification, plot_average_roc_curve, AsymmetricMSELoss
from utils_ms import Spec2Emb, MSTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Downstream Classification")
    parser.add_argument('--data_path', type=str, default='.../multimodal_emb_clean', 
                        help='数据根目录路径')
    parser.add_argument('--label_file', type=str, default='GSLF_clas',
                        help='存储gslf是数据的文件夹')
    parser.add_argument('--label_csv', type=str, default='train_gslf.csv',
                        help='用于初始化 Chem Encoder 的 CSV 文件名')
    parser.add_argument('--ms_csv', type=str, default='train_ms.csv',
                        help='用于 CLIP 训练的 Merged CSV 文件名')
    parser.add_argument('--test_ms_df', type=str, default='test_ms.csv',
                        help='测试集的 MS CSV 文件路径')
    parser.add_argument('--test_label_df', type=str, default='test_label.csv',
                        help='测试集的标签 CSV 文件路径')
    parser.add_argument('--base_model_name', type=str, default='base_model.pt',
                        help='预训练的 MS Encoder 权重文件名')
    parser.add_argument('--save_model_name', type=str, default='gslf_cla_best.pt',
                        help='保存的最佳模型文件名')
    parser.add_argument('--mode', type = str, default = 'ms',
                        help = '使用的编码器类型')
    parser.add_argument('--fold',     type=int, required=True,  help='当前折编号 (1~5)')
    parser.add_argument('--fold_dir', type=str, required=True,  help='kfold_splits 根目录路径')
    parser.add_argument('--test_dir', type=str, required=True,  help='test 文件夹路径')
  

    # --- 训练超参数 ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='training epoch')
    parser.add_argument('--lr_enc', type=float, default=1e-3, help='Encoder learning rate')
    parser.add_argument('--lr_cla', type=float, default=1e-4, help='Projector learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--alpha', type=float, default=0.25, help='focal loss alpha')
    parser.add_argument('--gamma', type=float, default=2.0, help='focal loss gamma')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers number')
    parser.add_argument('--fractions', nargs='+', type=float,
                    default=[0.1,0.2,0.4,0.6,0.8,1.0])
    parser.add_argument('--learning_curves_filename', type=str, default='learning_curves.csv',
                        help='保存学习曲线的文件名')
    parser.add_argument('--top_k', type=int, default=5, help='k for Precision@k metric')
    parser.add_argument('--loss', type=str, default='bce', help='Loss function to use')
    parser.add_argument('--use_augmentor', action='store_true', help='Whether to use data augmentation')


    return parser.parse_args()

def main(args):
    # 1. 设备配置
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置随机种子 (可选，为了复现)
    # torch.manual_seed(args.seed)
    torch.manual_seed(42)

    # 路径拼接
    # input_gslf_label = os.path.join(args.data_path, args.label_file, args.label_csv)
    # input_gslf_ms = os.path.join(args.data_path, args.label_file, args.ms_csv)
    # ms_df = pd.read_csv(input_gslf_ms)
    # label_df = pd.read_csv(input_gslf_label)

    # input_gslf_label_test = os.path.join(args.data_path, args.label_file, args.test_label_df)
    # input_gslf_ms_test = os.path.join(args.data_path, args.label_file, args.test_ms_df)
    # ms_df_test = pd.read_csv(input_gslf_ms_test)
    # label_df_test = pd.read_csv(input_gslf_label_test)

    base_model_path = os.path.join('.../multimodal_emb_clean', args.base_model_name)
    
    save_model_path = os.path.join(args.data_path, args.save_model_name)

    fold_dir = os.path.join(args.fold_dir, f'fold_{args.fold}')

    ms_df       = pd.read_csv(os.path.join(fold_dir,      'train_ms.csv'))
    label_df    = pd.read_csv(os.path.join(fold_dir,      'train_label.csv'))
    val_ms_df   = pd.read_csv(os.path.join(fold_dir,      'valid_ms.csv'))
    val_label_df= pd.read_csv(os.path.join(fold_dir,      'valid_label.csv'))
    ms_df_test  = pd.read_csv(os.path.join(args.test_dir, 'test_ms.csv'))
    label_df_test = pd.read_csv(os.path.join(args.test_dir,'test_label.csv'))

    results = {} 
    for fraction in args.fractions:
        
        print(f'current fract: {fraction}')
        base_encoder = Spec2Emb(num_emb=1000, emb_dim=500)
        checkpoint = torch.load(base_model_path, map_location=device)

        if args.mode in ['openpom', 'molformer']:
            ms_encoder = MSTransformer(
                base_encoder=base_encoder,
                d_model=500, nhead=4, num_layers=2
            ).to(device)
            ms_encoder.load_state_dict(checkpoint['ms_encoder'])
            use_sum = False
        else:
            if 'ms_encoder' in checkpoint:
                keys = list(checkpoint['ms_encoder'].keys())
                has_transformer = any('transformer' in k for k in keys)
                if has_transformer:
                    ms_encoder = MSTransformer(
                        base_encoder=base_encoder,
                        d_model=500, nhead=4, num_layers=2
                    ).to(device)
                    ms_encoder.load_state_dict(checkpoint['ms_encoder'])
                    use_sum = False
                else:
 
                    ms_encoder = Spec2Emb(num_emb=1000, emb_dim=500).to(device)
                    ms_encoder.load_state_dict(checkpoint['ms_encoder'])
                    use_sum = True
            else:
                ms_encoder = Spec2Emb(num_emb=1000, emb_dim=500).to(device)
                ms_encoder.load_state_dict(checkpoint)
                use_sum = True


        
        OUTPUT_DIM = 138
        if args.mode == 'openpom':
            chem_emb      = pd.read_csv(os.path.join(fold_dir,      'train_pom_emb.csv'))
            val_chem_emb  = pd.read_csv(os.path.join(fold_dir,      'valid_pom_emb.csv'))
            test_chem     = pd.read_csv(os.path.join(args.test_dir, 'test_pom_emb.csv'))
            chem_emb = chem_emb.iloc[:, 2:]
            val_chem_emb = val_chem_emb.iloc[:, 2:]
            test_chem = test_chem.iloc[:, 2:]
            INPUT_DIM = 256
            classification_head = Classification(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
            optimizer = optim.Adam([
                        {'params': classification_head.parameters(), 'lr': args.lr_cla}
                        ])
            freeze = True

        elif args.mode == 'molformer':
            chem_emb      = pd.read_csv(os.path.join(fold_dir,      'train_mol_emb.csv'))
            val_chem_emb  = pd.read_csv(os.path.join(fold_dir,      'valid_mol_emb.csv'))
            test_chem     = pd.read_csv(os.path.join(args.test_dir, 'test_mol_emb.csv'))
            chem_emb = chem_emb.iloc[:, 2:]
            val_chem_emb = val_chem_emb.iloc[:, 2:]
            test_chem = test_chem.iloc[:, 2:]
            INPUT_DIM = 768
            classification_head = Classification(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
            optimizer = optim.Adam([
                    {'params': classification_head.parameters(), 'lr': args.lr_cla}
                    ])
            freeze = True
        else:
            chem_emb     = None
            val_chem_emb = None
            test_chem    = None
            # chem_emb = chem_emb.iloc[:, 2:].values
            INPUT_DIM = 500
            classification_head = Classification(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
            optimizer = optim.Adam([
                    {'params': ms_encoder.parameters(), 'lr': args.lr_enc, 'weight_decay': 0,},
                    {'params': classification_head.parameters(), 'lr': args.lr_cla, 'weight_decay': 1e-4}        
                    ])
            freeze = False
        
        # dataset = ClassificationData(ms_df, label_df, chem_emb)
        augmentor = MSAugmentor(min_mz=50, max_mz=180, 
                    jitter_prob=0, jitter_scale=0, 
                    ghost_prob=0, num_ghosts=0,
                    drop_seg_prob=0, max_drop_width=0)
        # dataset = ClassificationData(ms_df, label_df, chem_emb)
        # dataset_size = len(dataset)
        # train_size = int(0.9 * dataset_size)
        # # val_size   = int(0.1 * dataset_size)
        # val_size  = dataset_size - train_size

        # train_ds, val_ds = random_split(
        #     dataset,
        #     [train_size, val_size],
        #     generator=torch.Generator().manual_seed(args.seed)
        # )

        # best_micro_auc = 0
        # epoch_losses = []
        # n_use = max(1, int(fraction * len(train_ds)))
        # sub_indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed))[:n_use]
        # sub_train_ds = torch.utils.data.Subset(train_ds, sub_indices.tolist())
        train_ds = ClassificationData(ms_df,     label_df,     chem_emb)
        val_ds   = ClassificationData(val_ms_df, val_label_df, val_chem_emb)  

        best_micro_auc = 0
        epoch_losses = []
        n_use = max(1, int(fraction * len(train_ds)))
        sub_indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.fold))[:n_use]
        sub_train_ds = torch.utils.data.Subset(train_ds, sub_indices.tolist())

        from functools import partial
        if args.use_augmentor:
            collate_fn_train = partial(classification_collate, augmentor=augmentor)
        else:   
            collate_fn_train = partial(classification_collate, augmentor=None)

        # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
        #                             num_workers=args.num_workers, collate_fn=collate_fn_train)
        train_loader = DataLoader(sub_train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, collate_fn=collate_fn_train)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                                    num_workers=args.num_workers, collate_fn=collate_fn_train)


        test_set = ClassificationData(ms_df_test, label_df_test, test_chem)
        test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                                    num_workers=args.num_workers, collate_fn=collate_fn_train)
        
            
        scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                threshold_mode='abs',   
                threshold=0.005,      
                min_lr=1e-6
            )
        
        label_cols = label_df.columns[3:] 
        labels_np = label_df[label_cols].values
  
        total_samples = len(label_df)                  
        pos_counts = labels_np.sum(axis=0)       
        neg_counts = total_samples - pos_counts  
        


        raw_weight = neg_counts / (pos_counts + 1e-5)
        

        smoothed_weight = np.sqrt(raw_weight)
        
        pos_weight_tensor = torch.tensor(smoothed_weight, dtype=torch.float32)
        

        if args.loss == 'focal':
            criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
        elif args.loss == 'weight_bce':
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        elif args.loss == 'bce':
            criterion = nn.BCEWithLogitsLoss()


 # fraction -> list of (train_loss, val_loss)
        print("\n============ Start classification training ============\n")
            
        best_micro_auc = 0

        for epoch in range(args.epochs):
            print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

            # TRAIN
            train_loss = train_classfication(ms_encoder, classification_head, train_loader, optimizer, criterion, device, freeze, use_sum)
            print(f"[Epoch {epoch+1}] Train Loss = {train_loss:.6f}")

            # VALIDATE
            val_loss, micro_auc, weighted_auc_val, prec_k_val, val_labels_np, val_probs_np = valid_classification(ms_encoder, classification_head, val_loader, criterion, device, freeze, use_sum
            )
            print(f"[Epoch {epoch+1}] Val   Loss = {val_loss:.6f}")

            scheduler.step(val_loss)
            # epoch_losses.append((train_loss, val_loss))
            epoch_losses.append((train_loss, val_loss, micro_auc, weighted_auc_val, prec_k_val))

            results[fraction] = epoch_losses

            auc_score = plot_average_roc_curve(
                    labels_np=val_labels_np,
                    probs_np=val_probs_np,
                    title=f"Micro-Averaged ROC Curve (Epoch {epoch+1})",
                    output_path= os.path.join(args.data_path, "best_micro_roc.png") 
                )

            if micro_auc > best_micro_auc:
                best_micro_auc = micro_auc

                checkpoint = {
                    'ms_encoder': ms_encoder.state_dict(),
                    'classification': classification_head.state_dict()
                }
                torch.save(checkpoint, save_model_path)
                print(f"  ➤ Best model updated. Saved to: {save_model_path}")

        print("\n============ Training Completed ============\n")

        print(f"Loading best model from {save_model_path} for testing...")
        checkpoint = torch.load(save_model_path, map_location=device)
        ms_encoder.load_state_dict(checkpoint['ms_encoder'])
        classification_head.load_state_dict(checkpoint['classification'])

        test_loss, micro_auc_test, weighted_auc_test, prec_k_test, test_labels_np, test_probs_np = valid_classification(ms_encoder, classification_head, test_loader, criterion, device, freeze, use_sum)
        print(f"Final Test Loss = {test_loss:.6f}")
        print(f"Final Test AUC = {micro_auc_test:.6f}")
        print(f"Final Test Weighted AUC = {weighted_auc_test:.6f}")
        print(f"Final Test Precision@{args.top_k} = {prec_k_test:.6f}")
        print(f"Best Val AUC = {best_micro_auc:.6f}")
        
        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # cmap = plt.cm.viridis(__import__('numpy').linspace(0, 1, len(results)))

        # for color, (frac, losses) in zip(cmap, sorted(results.items())):
        #     train_l = [l[0] for l in losses]
        #     val_l   = [l[1] for l in losses]
        #     label   = f"{int(frac*100)}%"
        #     axes[0].plot(train_l, color=color, label=label)
        #     axes[1].plot(val_l,   color=color, label=label)

        # for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
        #     ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(title)
        #     ax.legend(title="Train %", fontsize=8); ax.grid(alpha=0.3)

        # plt.tight_layout()
        # plt.savefig(os.path.join(args.data_path, args.learning_curves_filename), dpi=150)
        rows = []
        for frac, losses in sorted(results.items()):
            for epoch_idx, (tl, vl, auc, wauc, preck) in enumerate(losses, start=1):
                rows.append({
                    'fraction': frac,
                    'epoch': epoch_idx,
                    'split': 'val',
                    'train_loss': tl,
                    'val_loss': vl,
                    'val_micro_auc': auc,
                    'val_weighted_auc': wauc,
                    'val_prec_k': preck,
                    'test_loss': None,
                    'test_micro_auc': None,
                    'test_weighted_auc': None,
                    'test_prec_k': None,
                })

        rows.append({
            'fraction': fraction,
            'epoch': -1,
            'split': 'test',
            'train_loss': None,
            'val_loss': None,
            'val_micro_auc': None,
            'val_weighted_auc': None,
            'val_prec_k': None,
            'test_loss': test_loss,
            'test_micro_auc': micro_auc_test,
            'test_weighted_auc': weighted_auc_test,
            'test_prec_k': prec_k_test,
        })

        pd.DataFrame(rows).to_csv(os.path.join(args.data_path, args.learning_curves_filename), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)