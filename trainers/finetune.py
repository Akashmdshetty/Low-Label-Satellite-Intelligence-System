import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os

def finetune(backbone, train_loader, val_loader, num_classes, lr, epochs, device, use_amp=True, logger=None, start_epoch=0, classifier_state=None):
    """
    Advanced Fine-tuning:
    - Differential Learning Rates (Backbone vs Head)
    - AdamW Optimizer
    - Cosine Annealing Scheduler
    """
    backbone.train()
    feature_dim = 512 # resnet18 output
    # The checkpoint indicates a single-layer classifier: 1.weight torch.Size([10, 512])
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, num_classes)
    ).to(device)
    
    if classifier_state:
        classifier.load_state_dict(classifier_state)
        print("Loaded classifier head from checkpoint")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Differential Learning Rates: Backbone gets 10% of the head's LR
    optimizer = optim.AdamW([
        {'params': backbone.parameters(), 'lr': lr * 0.1},
        {'params': classifier.parameters(), 'lr': lr}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    best_acc = 0
    for epoch in range(start_epoch, epochs):
        backbone.train()
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Mixup Regularization
            alpha = 0.2
            lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
            index = torch.randperm(images.size(0)).to(device)
            mixed_images = lam * images + (1 - lam) * images[index, :]
            labels_a, labels_b = labels, labels[index]
            
            with autocast(enabled=scaler is not None):
                features = backbone(mixed_images)
                outputs = classifier(features)
                # Compute Mixup Loss
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            # Accuracy is slightly tricky with Mixup, we compute it on the dominant label
            correct += (lam * predicted.eq(labels_a).float() + (1 - lam) * predicted.eq(labels_b).float()).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
            
        train_acc = 100. * correct / total
        val_acc = validate(backbone, classifier, val_loader, device)
        
        if logger:
            logger.log_metrics({
                "ft_train_loss": train_loss/len(train_loader),
                "ft_train_acc": train_acc,
                "ft_val_acc": val_acc,
                "ft_lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'backbone': backbone.state_dict(),
                'classifier': classifier.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }, 'best_finetune_model.pth')

    return best_acc

def validate(backbone, classifier, loader, device):
    backbone.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total
