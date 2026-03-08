import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import math
import os

def train_one_epoch(model, dataloader, optimizer, scaler, epoch, device, base_m, total_epochs, logger=None):
    model.train()
    total_loss = 0
    
    # 4. Cosine EMA Momentum Schedule
    # m = 1 - (1 - base_m) * (1 + cos(pi * epoch / total_epochs)) / 2
    m = 1 - (1 - base_m) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (view1, view2, _) in enumerate(pbar):
        view1, view2 = view1.to(device), view2.to(device)
        
        optimizer.zero_grad()
        
        # 2. Mixed Precision (MANDATORY)
        with autocast(enabled=scaler is not None):
            loss = model(view1, view2)
            
        # 7. Safety: Detect NaN loss
        if torch.isnan(loss):
            print(f"\n[FATAL] NaN loss detected at epoch {epoch}, batch {batch_idx}. Stopping training.")
            return float('nan')

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        # 4. Update target network via EMA with dynamic momentum
        model._update_target_network(m)
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'm': f"{m:.4f}",
            'gpu_mem': f"{torch.cuda.memory_allocated(device)/1024**2:.0f}MB" if device.type == 'cuda' else 'N/A'
        })
        
        if logger:
            logger.log_metrics({
                "batch_loss": loss.item(), 
                "ema_momentum": m,
                "epoch": epoch
            }, step=epoch * len(dataloader) + batch_idx)
            
    # GPU Memory Logging
    if device.type == 'cuda':
        print(f"Propagating GPU Memory Usage: {torch.cuda.max_memory_allocated(device)/1024**2:.0f}MB")
        torch.cuda.reset_peak_memory_stats(device)

    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """6. Checkpointing Logic"""
    state = {
        'epoch': epoch,
        'backbone': model.online_encoder[0].state_dict(),
        'projector': model.online_encoder[1].state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, scheduler, path, device):
    """Resumes training from a saved checkpoint."""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0
        
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    model.online_encoder[0].load_state_dict(checkpoint['backbone'])
    model.online_encoder[1].load_state_dict(checkpoint['projector'])
    # Synchronize target network
    model.target_encoder[0].load_state_dict(checkpoint['backbone'])
    model.target_encoder[1].load_state_dict(checkpoint['projector'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    print(f"Resuming training from epoch {checkpoint['epoch']}")
    return checkpoint['epoch'] + 1
