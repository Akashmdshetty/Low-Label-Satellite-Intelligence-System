import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Modular imports
from .backbone import get_backbone
from .projector import MLP
from .predictor import Predictor
from utils.ema import EMA

class BYOL(nn.Module):
    """
    Proper Research-Grade BYOL Implementation.
    
    Architecture:
    - Online Network: Backbone f_theta -> Projector g_theta -> Predictor q_theta
    - Target Network: Backbone f_xi -> Projector g_xi (EMA updated, no gradients)
    
    Why Stop-Gradient is crucial:
    Without stop-gradient on the target network, the online network would minimize 
    the loss by moving the target representations. In a Siamese setup, the easiest 
    way to minimize distance is for both networks to output a constant value. 
    Stop-gradient forces the online network to stay 'meaningful' by preventing 
    the target from taking the 'easy way out'.
    
    Collapse Prevention:
    BYOL prevents collapse through asymmetry (the predictor head is only online), 
    the EMA update of the target (momentum provides temporal smoothness), and 
    Batch Normalization within the MLP heads (implicit contrastive effect).
    """
    def __init__(self, backbone_name, hidden_dim=4096, projection_dim=256):
        super(BYOL, self).__init__()
        
        # 1. Initialize Online Network
        backbone, feature_dim = get_backbone(backbone_name)
        self.online_encoder = nn.Sequential(
            backbone,
            MLP(feature_dim, hidden_dim, projection_dim),
            Predictor(projection_dim, hidden_dim)
        )
        
        # 2. Initialize Target Network (Backbone + Projector only)
        # We extract them from the online encoder parts to ensure sync start
        target_backbone = copy.deepcopy(backbone)
        target_projector = copy.deepcopy(self.online_encoder[1])
        
        self.target_encoder = nn.Sequential(
            target_backbone,
            target_projector
        )
        
        # Freeze target network
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        # EMA Updater
        self.ema_updater = EMA(momentum=0.99)

    @torch.no_grad()
    def _update_target_network(self, tau):
        """
        Updates the target encoder (f_xi, g_xi) from the online encoder (f_theta, g_theta).
        Note: The online predictor q_theta is NOT mirrored in the target network.
        """
        self.ema_updater.m = tau
        # Update target backbone from online backbone
        self.ema_updater.update(self.target_encoder[0], self.online_encoder[0])
        # Update target projector from online projector
        self.ema_updater.update(self.target_encoder[1], self.online_encoder[1])

    def regression_loss(self, x, y):
        """
        L2-normalized Negative Cosine Similarity.
        L = 2 - 2 * <p, z> / (||p|| * ||z||)
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, x1, x2):
        """
        Symmetric forward pass:
        - View 1 through Online, View 2 through Target -> loss1
        - View 2 through Online, View 1 through Target -> loss2
        """
        # Online predictions
        p1 = self.online_encoder(x1)
        p2 = self.online_encoder(x2)
        
        # Target projections (with stop-gradient)
        with torch.no_grad():
            z1 = self.target_encoder(x1).detach()
            z2 = self.target_encoder(x2).detach()
            
        # Compute symmetric loss
        loss = self.regression_loss(p1, z2) + self.regression_loss(p2, z1)
        
        return loss.mean()
