import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def linear_probe(backbone, train_loader, val_loader, num_classes, lr, epochs, device):
    """
    Linear probing: Fix the backbone and only train a linear classifier.
    """
    backbone.eval() # Backbone in eval mode
    feature_dim = 512 
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Only optimize classifier parameters
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    best_acc = 0
    for epoch in range(epochs):
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Linear Probe Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                features = backbone(images)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        val_acc = validate_probe(backbone, classifier, val_loader, device)
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            
    return best_acc

def validate_probe(backbone, classifier, loader, device):
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
