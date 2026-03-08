import torch.nn as nn

class Predictor(nn.Module):
    """
    Dedicated 2-layer MLP predictor for Bootstrap Your Own Latent (BYOL).
    
    Architecture:
    Linear(dim, hidden_dim) -> BatchNorm1d(hidden_dim) -> ReLU -> Linear(hidden_dim, dim)
    
    Why is the predictor necessary?
    The predictor is a key component of the asymmetric architecture in BYOL. It is only 
    present in the online network, not the target network. This asymmetry, along with 
    the EMA update of the target network, is what prevents the representation from 
    collapsing to a trivial constant solution (constant output for all inputs).
    
    How does it prevent collapse?
    By making the online network 'predict' the output of the target network through 
    an additional non-linear transformation, the symmetry of the task is broken. 
    Without this asymmetry, the loss-minimizing solution would be for both networks 
    to map all images to the same point. The predictor forces the online network 
    to learn more complex features to map its representation to the target's.
    """
    def __init__(self, input_dim, hidden_dim=4096):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)
