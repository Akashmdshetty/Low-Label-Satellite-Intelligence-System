import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def get_labeled_split(dataset, percentage, seed=42):
    """
    Returns a subset of the dataset with the specified percentage of labeled data.
    Ensures class balance using stratified split.
    """
    labels = [dataset[i][1] for i in range(len(dataset))]
    train_idx, _ = train_test_split(
        np.arange(len(labels)),
        train_size=percentage,
        stratify=labels,
        random_state=seed
    )
    return Subset(dataset, train_idx)

def get_train_val_split(dataset, val_percentage=0.1, seed=42):
    labels = [dataset[i][1] for i in range(len(dataset))]
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=val_percentage,
        stratify=labels,
        random_state=seed
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
