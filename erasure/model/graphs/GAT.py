import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_heads=1, n_classes=2, dropout=0.5):
        super(GAT, self).__init__()
        assert isinstance(hidden_channels, list) and len(hidden_channels) > 0, \
            "hidden_channels must be a non-empty list"

        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.n_heads = n_heads

        self.convs.append(GATConv(in_channels, hidden_channels[0] // n_heads, heads=n_heads, dropout=dropout))

        for i in range(1, len(hidden_channels)):
            self.convs.append(
                GATConv(hidden_channels[i - 1], hidden_channels[i] // n_heads, heads=n_heads, dropout=dropout)
            )

        self.convs.append(GATConv(hidden_channels[-1], out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x
