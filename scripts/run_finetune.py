import argparse
import yaml
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_eurosat_dataset
from data.transforms import get_inference_transforms
from data.split import get_labeled_split, get_train_val_split
from models.backbone import get_backbone
from trainers.finetune import finetune
from utils.logger import Logger
from torch.utils.data import DataLoader

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # We now use separate transforms for training (with augmentation) and validation (deterministic)
    from data.transforms import get_finetune_transforms
    train_transform = get_finetune_transforms(config['image_size'])
    val_transform = get_inference_transforms(config['image_size'])
    
    # Logger
    logger = Logger(use_wandb=config.get('use_wandb', False), project_name=config.get('project_name', "ft-eurosat"), config=config)
    
    # Load dataset twice to apply different transforms
    full_dataset_train = get_eurosat_dataset(config['data_dir'], transform=train_transform)
    full_dataset_val = get_eurosat_dataset(config['data_dir'], transform=val_transform)
    
    # Use same seed for split to ensure disjoint sets even with different objects
    train_set, _ = get_train_val_split(full_dataset_train, val_percentage=0.2, seed=config['seed'])
    _, val_set = get_train_val_split(full_dataset_val, val_percentage=0.2, seed=config['seed'])
    
    # Backbone
    backbone, _ = get_backbone(config['backbone'])
    start_epoch = 0
    classifier_state = None
    
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        # BYOL saving logic in trainers/pretrain.py:
        if 'backbone' in ckpt:
            backbone.load_state_dict(ckpt['backbone'])
            print(f"Loaded backbone weights from {args.ckpt}")
            if 'classifier' in ckpt:
                classifier_state = ckpt['classifier']
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"Detected finetune checkpoint. Resuming from epoch {start_epoch}")
        else:
            state_dict = ckpt.get('model_state_dict', ckpt)
            backbone_dict = {k.replace('online_encoder.0.', ''): v for k, v in state_dict.items() if k.startswith('online_encoder.0.')}
            backbone.load_state_dict(backbone_dict)
            print(f"Loaded weights using fallback from {args.ckpt}")
    
    backbone = backbone.to(device)
    use_amp = args.use_amp or config.get('use_amp', False)
    
    # Experiment across splits
    for split in config['labeled_splits']:
        print(f"\n--- Starting Advanced Fine-tuning on {split*100}% labels ---")
        labeled_train_set = get_labeled_split(train_set, split, seed=config['seed'])
        train_loader = DataLoader(labeled_train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
        
        best_acc = finetune(
            backbone, train_loader, val_loader, 
            num_classes=10, lr=config['finetune_lr'], 
            epochs=config['finetune_epochs'], device=device,
            use_amp=use_amp, logger=logger,
            start_epoch=start_epoch, classifier_state=classifier_state
        )
        print(f"Best Val Acc for {split*100}% split: {best_acc:.2f}%")

    logger.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--ckpt', type=str, help='Path to BYOL checkpoint for weights')
    parser.add_argument('--use_amp', action='store_true', help='Use Mixed Precision training')
    args = parser.parse_args()
    main(args)
