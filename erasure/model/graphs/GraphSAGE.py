import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_classes=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        assert isinstance(hidden_channels, list) and len(hidden_channels) > 0, \
            "hidden_channels must be a non-empty list"

        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels[0]))

        # Intermediate layers
        for i in range(1, len(hidden_channels)):
            self.convs.append(SAGEConv(hidden_channels[i - 1], hidden_channels[i]))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels[-1], out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x
