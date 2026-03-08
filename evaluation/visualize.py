import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@torch.no_grad()
def visualize_tsne(backbone, dataloader, device, save_path, num_samples=1000):
    backbone.eval()
    features = []
    labels = []
    
    count = 0
    for images, targets in dataloader:
        images = images.to(device)
        feat = backbone(images)
        features.append(feat.cpu().numpy())
        labels.append(targets.numpy())
        count += images.size(0)
        if count >= num_samples:
            break
            
    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    df = pd.DataFrame(features_2d, columns=['x', 'y'])
    df['label'] = labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='viridis', legend='full')
    plt.title('t-SNE Visualization of BYOL Features (EuroSAT)')
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")
