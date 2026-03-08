import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import yaml
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import get_backbone
from data.dataset import get_eurosat_dataset
from data.split import get_train_val_split

def investigate():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset to see class mapping
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = get_eurosat_dataset(config['data_dir'], transform=transform)
    print(f"Dataset Classes: {full_dataset.dataset.classes}")
    print(f"Dataset Class to Idx: {full_dataset.dataset.class_to_idx}")
    
    # Load Model
    backbone, feature_dim = get_backbone(config['backbone'])
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )
    
    model_path = 'best_finetune_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return

    checkpoint = torch.load(model_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    backbone.to(device).eval()
    classifier.to(device).eval()
    
    # Validation loop
    _, val_set = get_train_val_split(full_dataset, 0.2)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
                
    print(f"\nOverall Accuracy: {100 * correct / total:.2f}%")
    print("\nPer-class Accuracy:")
    for i in range(10):
        cls_name = full_dataset.dataset.classes[i]
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{cls_name:25s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

if __name__ == "__main__":
    investigate()
