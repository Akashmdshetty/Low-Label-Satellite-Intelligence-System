import argparse
import yaml
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_eurosat_dataset
from data.transforms import get_byol_transforms, get_inference_transforms
from models.byol import BYOL
from trainers.pretrain import train_one_epoch, save_checkpoint, load_checkpoint
from evaluation.knn import test_knn
from utils.seed import set_seed
from utils.logger import Logger
from torch.utils.data import DataLoader

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Logger
    logger = Logger(use_wandb=config.get('use_wandb', False), project_name=config.get('project_name', "ssl-byol-eurosat"), config=config)
    
    # 1. Configuration Override for Scale-Run
    total_epochs = 20 if args.test_run else config['epochs']
    batch_size = config['batch_size']
    
class BYOLTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2
    def __call__(self, img):
        return self.transform1(img), self.transform2(img)

class DualViewDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform
    def __getitem__(self, idx):
        img, label = self.base[idx]
        v1, v2 = self.transform(img)
        return v1, v2, label
    def __len__(self):
        return len(self.base)

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Logger
    logger = Logger(use_wandb=config.get('use_wandb', False), project_name=config.get('project_name', "ssl-byol-eurosat"), config=config)
    
    # 1. Configuration Override for Scale-Run
    total_epochs = 20 if args.test_run else config['epochs']
    batch_size = config['batch_size']
    
    # Transforms
    t1, t2 = get_byol_transforms(config['image_size'])
    byol_transform = BYOLTransform(t1, t2)
    
    # Dataset
    base_dataset = get_eurosat_dataset(config['data_dir'])
    dataset = DualViewDataset(base_dataset, byol_transform)
    
    # OOM Resilient DataLoader Setup
    def get_loader(bs):
        return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=config['num_workers'], pin_memory=True)

    dataloader = get_loader(batch_size)
    
    # 1. Model & Optimizer (AdamW)
    model = BYOL(config['backbone'], config['hidden_dim'], config['projection_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # 3. Cosine Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    
    # 2. Mixed Precision (AMP)
    use_amp = args.use_amp or config.get('use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
    
    # Training Loop
    print(f"\n🚀 Starting BYOL Pretraining | Epochs: {total_epochs} | Batch Size: {batch_size}")
    if args.test_run:
        print("🛠️ TEST MODE ENABLED: Running only 20 epochs.")

    for epoch in range(start_epoch, total_epochs):
        try:
            loss = train_one_epoch(
                model, dataloader, optimizer, scaler, epoch, device, 
                config['tau_base'], total_epochs, logger=logger
            )
            
            if torch.isnan(torch.tensor(loss)):
                break
                
            scheduler.step()
            
            # 5. Logging (Every 10 epochs)
            if epoch % 10 == 0 or epoch == total_epochs - 1:
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} | Loss: {loss:.4f} | LR: {curr_lr:.6f}")
            
            # 6. Checkpointing
            if epoch % 20 == 0 or epoch == total_epochs - 1:
                save_checkpoint(model, optimizer, scheduler, epoch, f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, "last_checkpoint.pth")
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n⚠️ OOM detected at batch size {batch_size}. Retrying with smaller batch size...")
                batch_size = int(batch_size * 0.75)
                if batch_size < 1:
                    raise e
                torch.cuda.empty_cache()
                dataloader = get_loader(batch_size)
                continue
            else:
                raise e

    logger.finish()
    print("\n✅ Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--use_amp', action='store_true', help='Use Mixed Precision training')
    parser.add_argument('--test_run', action='store_true', help='Train for only 20 epochs')
    args = parser.parse_args()
    main(args)
