import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class SafeEuroSAT(Dataset):
    """
    Wrapper for EuroSAT to handle corrupted images.
    If an image fails to load, it returns the next valid image.
    """
    def __init__(self, root, transform=None, download=True):
        self.dataset = datasets.EuroSAT(root=root, transform=None, download=download)
        self.transform = transform

    def __getitem__(self, index):
        try:
            image, label = self.dataset[index]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Skipping corrupted image at index {index}: {e}")
            # Recursively try the next image
            new_index = (index + 1) % len(self.dataset)
            return self.__getitem__(new_index)

    def __len__(self):
        return len(self.dataset)

def get_eurosat_dataset(data_dir, transform=None):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    dataset = SafeEuroSAT(
        root=data_dir,
        transform=transform,
        download=True
    )
    return dataset

class TwoViewsDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2, label

    def __len__(self):
        return len(self.base_dataset)
