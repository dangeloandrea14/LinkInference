import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree


class FeaturePropagator(nn.Module):
    """
    Feature propagation module using SGC-style K-step propagation.
    This matches the propagation behavior in the certified removal framework.
    """
    def __init__(self,  K=2, alpha=0.0, XdegNorm=False, add_self_loops=True):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.XdegNorm = XdegNorm
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index):
        self.device = x.device
        edge_index = edge_index.to(self.device)
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        row = row.to(self.device)
        col = col.to(self.device)
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_left = deg.pow(-self.alpha)
        deg_inv_right = deg.pow(self.alpha - 1)
        deg_inv_left[deg_inv_left == float('inf')] = 0
        deg_inv_right[deg_inv_right == float('inf')] = 0
        edge_weight = deg_inv_left[row] * deg_inv_right[col]

        if self.XdegNorm:
            deg_node = degree(row, x.size(0), dtype=x.dtype)
            deg_inv = deg_node.pow(-1).unsqueeze(-1)
            deg_inv[deg_inv == float('inf')] = 0
            x = deg_inv * x

        for _ in range(self.K):
            x = self.propagate(edge_index, x, edge_weight)

        return x

    def propagate(self, edge_index, x, edge_weight):
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, edge_weight.view(-1, 1) * x[col])
        return out


class SGC_CGU(nn.Module):
    """
    Modular SGC model combining feature propagation and linear classification.
    Can be used end-to-end for training, or split during unlearning.
    """
    def __init__(self, in_channels, out_channels, n_classes=2, K=2, alpha=0.0,
                 XdegNorm=False, add_self_loops=True):
        super().__init__()
        self.n_classes = n_classes
        self.feat_prop = FeaturePropagator(K=K, alpha=alpha,
                                     XdegNorm=XdegNorm, add_self_loops=add_self_loops)
        self.classifier = nn.Linear(in_channels, out_channels)
        self.hidden_channels = []


    def forward(self, x, edge_index):
        x = self.feat_prop(x, edge_index)
        return self.classifier(x)

    def propagate_feature(self, x, edge_index):
        return self.feat_prop(x, edge_index)
