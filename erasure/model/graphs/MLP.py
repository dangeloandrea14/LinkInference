import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_classes=2, dropout=0.5):
        super(MLP, self).__init__()
        assert isinstance(hidden_channels, list) and len(hidden_channels) > 0, \
            "hidden_channels must be a non-empty list"
        
        layers = []
        input_dim = in_channels

        self.hidden_channels = hidden_channels

        for hidden_dim in hidden_channels:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.net(x)