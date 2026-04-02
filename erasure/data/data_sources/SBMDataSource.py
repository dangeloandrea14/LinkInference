import torch
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph

from erasure.data.data_sources.datasource import DataSource
from erasure.data.data_sources.TwitchGamersDataSource import SingleGraphDataset
from erasure.data.data_sources.EdgeFileDataSource import GeometricWrapper
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local


class SBMDataSource(DataSource):
    """
    Structure-reliant synthetic dataset based on the Stochastic Block Model.

    Node features are pure Gaussian noise — they carry zero class signal.
    Labels are community (block) memberships.  A GNN that aggregates
    neighbourhood information can solve the task easily; an MLP that ignores
    edges cannot do better than random chance.

    Config parameters:
        num_classes     (int)   : number of communities          (default: 2)
        nodes_per_class (int)   : nodes in each community        (default: 500)
        p_in            (float) : intra-community edge prob.     (default: 0.05)
        p_out           (float) : inter-community edge prob.     (default: 0.001)
        num_features    (int)   : dimensionality of node features (default: 64)
        name            (str)   : dataset identifier             (default: "SBM")
    """

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        p = self.local_config["parameters"]
        self.num_classes     = p.get("num_classes",     2)
        self.nodes_per_class = p.get("nodes_per_class", 500)
        self.p_in            = p.get("p_in",            0.05)
        self.p_out           = p.get("p_out",           0.001)
        self.num_features    = p.get("num_features",    64)
        self.name            = p.get("name",            "SBM")

    def get_name(self):
        return self.name

    def create_data(self):
        k = self.num_classes
        n = self.nodes_per_class

        block_sizes = [n] * k
        edge_probs  = [
            [self.p_in if i == j else self.p_out for j in range(k)]
            for i in range(k)
        ]

        edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs)

        # Pure noise features — zero class signal
        total_nodes = k * n
        x = torch.randn(total_nodes, self.num_features)

        # Labels = block membership
        y = torch.repeat_interleave(
            torch.arange(k, dtype=torch.long),
            torch.tensor([n] * k, dtype=torch.long)
        )

        data = Data(x=x, edge_index=edge_index, y=y)
        return GeometricWrapper(SingleGraphDataset(data), self.preprocess)

    def get_simple_wrapper(self, data):
        return GeometricWrapper(data, self.preprocess)

    def check_configuration(self):
        super().check_configuration()
