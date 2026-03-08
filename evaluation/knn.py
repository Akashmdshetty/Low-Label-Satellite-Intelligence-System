import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_sigma):
    """
    k-NN monitoring during pre-training.
    """
    # Calculate cosine similarity
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_sigma).exp()

    # Counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # Weighted score
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

@torch.no_grad()
def test_knn(model, memory_data_loader, test_data_loader, device, knn_k=200, knn_sigma=0.1):
    model.eval()
    classes = 10 # EuroSAT has 10 classes
    total_top1 = 0.
    total_num = 0
    feature_bank = []
    feature_labels = []
    
    # Build feature bank
    for data, target in tqdm(memory_data_loader, desc='Feature bank'):
        feature = model(data.to(device))
        feature = F.normalize(feature, dim=1)
        feature_bank.append(feature)
        feature_labels.append(target.to(device))
        
    # [D, N]
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    # [N]
    feature_labels = torch.cat(feature_labels, dim=0)
    
    # Predict
    for data, target in tqdm(test_data_loader, desc='k-NN evaluation'):
        data, target = data.to(device), target.to(device)
        feature = model(data)
        feature = F.normalize(feature, dim=1)
        
        pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_sigma)
        total_num += data.size(0)
        total_top1 += (pred_labels[:, 0] == target).float().sum().item()
        
    return total_top1 / total_num * 100
