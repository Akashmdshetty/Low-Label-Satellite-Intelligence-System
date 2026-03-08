import argparse
import yaml
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_eurosat_dataset
from data.transforms import get_inference_transforms
from data.split import get_train_val_split, get_labeled_split
from models.backbone import get_backbone
from trainers.linear_probe import linear_probe
from utils.logger import Logger
from torch.utils.data import DataLoader

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    transform = get_inference_transforms(config['image_size'])
    
    # Logger
    logger = Logger(use_wandb=config.get('use_wandb', False), project_name=config.get('project_name', "linear-probe-eurosat"), config=config)
    
    # Load dataset
    full_dataset = get_eurosat_dataset(config['data_dir'], transform=transform)
    train_set, val_set = get_train_val_split(full_dataset, val_percentage=0.2)
    
    # Backbone
    backbone, _ = get_backbone(config['backbone'])
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        # BYOL saving logic in trainers/pretrain.py:
        # 'backbone': model.online_encoder[0].state_dict()
        if 'backbone' in ckpt:
            backbone.load_state_dict(ckpt['backbone'])
            print(f"Loaded backbone weights from {args.ckpt}")
        else:
            # Fallback for older formats if necessary
            state_dict = ckpt.get('model_state_dict', ckpt)
            backbone_dict = {k.replace('online_encoder.0.', ''): v for k, v in state_dict.items() if k.startswith('online_encoder.0.')}
            backbone.load_state_dict(backbone_dict)
            print(f"Loaded weights using fallback from {args.ckpt}")
    
    backbone = backbone.to(device)
    
    results = {"Label %": [], "BYOL Fine-Tune": [], "Linear Probe": [], "Supervised (Scratch)": []}
    # We will only fill "Linear Probe" for now, or match existing CSV structure
    
    for split in config.get('labeled_splits', [0.01, 0.05, 0.10]):
        print(f"\n--- Starting Linear Probing on {split*100}% labels ---")
        labeled_train_set = get_labeled_split(train_set, split)
        train_loader = DataLoader(labeled_train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
        
        # 5 epochs for benchmark consistency
        best_acc = linear_probe(
            backbone, train_loader, val_loader, 
            num_classes=10, lr=config.get('finetune_lr', 0.01), 
            epochs=5, device=device
        )
        results["Label %"].append(int(split * 100))
        results["Linear Probe"].append(round(best_acc, 1))

    # Save to CSV
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    csv_path = "results/benchmark_results.csv"
    
    # Load existing or create new
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Update Linear Probe column
        for i, split_val in enumerate(results["Label %"]):
            df.loc[df['Label %'] == split_val, 'Linear Probe'] = results["Linear Probe"][i]
    else:
        # Basic structure
        df = pd.DataFrame(results)
    
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--ckpt', type=str, help='Path to BYOL checkpoint for weights')
    args = parser.parse_args()
    main(args)
