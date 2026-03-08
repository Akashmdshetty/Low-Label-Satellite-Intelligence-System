import torch.nn as nn
from torchvision import models

def get_backbone(name="resnet18", pretrained=False):
    """
    Returns the backbone model.
    BYOL typically uses the features before the final FC layer.
    """
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # Remove the last fully connected layer
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    else:
        raise ValueError(f"Backbone {name} not supported.")
