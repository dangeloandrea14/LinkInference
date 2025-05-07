import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.typing import Adj, OptTensor

class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, alpha=0.5, n_classes=2, add_self_loops=True):
        super(SGC, self).__init__()
        self.K = K
        self.alpha = alpha
        self.add_self_loops = add_self_loops
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        # Build propagation matrix: P = D^{-alpha} A D^{-(1-alpha)}
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_left = deg.pow(-self.alpha)
        deg_inv_right = deg.pow(self.alpha - 1)
        deg_inv_left[deg_inv_left == float('inf')] = 0
        deg_inv_right[deg_inv_right == float('inf')] = 0

        edge_weight = deg_inv_left[row] * deg_inv_right[col]

        for _ in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return self.linear(x)

    def propagate(self, edge_index, x, edge_weight):
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, edge_weight.view(-1, 1) * x[col])
        return out
