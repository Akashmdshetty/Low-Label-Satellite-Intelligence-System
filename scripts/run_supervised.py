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
from trainers.finetune import finetune
from utils.logger import Logger
from torch.utils.data import DataLoader

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    transform = get_inference_transforms(config['image_size'])
    
    # Logger
    logger = Logger(use_wandb=config.get('use_wandb', False), project_name=config.get('project_name', "supervised-eurosat"), config=config)
    
    # Load dataset
    full_dataset = get_eurosat_dataset(config['data_dir'], transform=transform)
    train_set, val_set = get_train_val_split(full_dataset, val_percentage=0.2)
    
    use_amp = args.use_amp or config.get('use_amp', False)
    
    results = {}
    for split in config.get('labeled_splits', [0.01, 0.05, 0.10]):
        print(f"\n--- Starting Supervised Training (From Scratch) on {split*100}% labels ---")
        labeled_train_set = get_labeled_split(train_set, split)
        train_loader = DataLoader(labeled_train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
        
        # New Backbone for each split (from scratch)
        backbone, _ = get_backbone(config['backbone'])
        backbone = backbone.to(device)
        
        best_acc = finetune(
            backbone, train_loader, val_loader, 
            num_classes=10, lr=config.get('finetune_lr', 0.01), 
            epochs=config.get('finetune_epochs', 5), device=device,
            use_amp=use_amp, logger=logger
        )
        print(f"Best Val Acc for supervised {split*100}% split: {best_acc:.2f}%")
        results[split] = best_acc

    logger.finish()
    print("\nSupervised Baseline Results Summary:")
    for split, acc in results.items():
        print(f"{split*100}%: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--use_amp', action='store_true', help='Use Mixed Precision training')
    args = parser.parse_args()
    main(args)
