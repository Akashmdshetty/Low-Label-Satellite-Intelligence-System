import torch
import torchvision.transforms as T
from PIL import ImageFilter, ImageOps
import random

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR/BYOL."""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarize(object):
    """Solarize augmentation from BYOL."""
    def __call__(self, x):
        return ImageOps.solarize(x)

def get_byol_transforms(image_size=64):
    """
    BYOL data augmentations.
    Standard pipeline: RandomResizedCrop, ColorJitter, Grayscale, GaussianBlur, Solarize, Flip.
    """
    # View 1
    transform1 = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # View 2 (different probability for blur and solarize as per BYOL paper)
    transform2 = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        T.RandomApply([Solarize()], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform1, transform2

def get_inference_transforms(image_size=64):
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_finetune_transforms(image_size=64):
    """
    Standard fine-tuning augmentations for satellite imagery.
    H-Flips and V-Flips are safe and beneficial for aerial views.
    Added ColorJitter and RandomRotation for better generalization.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
