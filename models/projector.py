import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron used for Projection and Prediction heads in BYOL.
    Structure: Linear -> BatchNorm -> ReLU -> Linear.
    Note: The prediction head in BYOL is asymmetric.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
