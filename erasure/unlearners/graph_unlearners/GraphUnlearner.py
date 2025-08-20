from abc import ABCMeta, abstractmethod
import copy
import torch
from erasure.core.base import Configurable
from erasure.data.datasets.Dataset import DatasetWrapper
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.unlearners.torchunlearner import TorchUnlearner


class GraphUnlearner(TorchUnlearner):

    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)  

        self.hops = len(self.predictor.model.hidden_channels) + 1
        self.removal_type = self.global_ctx.removal_type

        og_graph =  self.dataset.partitions['all'] 
        
        self.x = og_graph[0][0].x.to(self.device).float()
        self.edge_index = og_graph[0][0].edge_index.to(self.device).long()
        self.labels = og_graph[0][1].to(self.device).long()
        self.labels = torch.tensor(self.labels)
        self.forget_part = self.local.config['parameters']['forget_part']
        self.retain_part = self.local.config['parameters']['retain_part']
        self.train_part = self.local.config['parameters']['train_part']
        

    def infected_nodes(self, edges_to_forget, hops):
        import networkx as nx

        G = nx.Graph()
        all_edges = self.dataset.partitions['all'][0][0].edge_index.t().tolist()  
        G.add_edges_from(all_edges)

        edge_nodes = set()
        for u, v in edges_to_forget:
            edge_nodes.add(u)
            edge_nodes.add(v)

        infected = set()
        for node in edge_nodes:
            if node in G:
                neighbors = nx.single_source_shortest_path_length(G, node, cutoff=hops).keys()
                infected.update(neighbors)

        return list(infected)
    

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['forget_part'] = self.local.config['parameters'].get('forget_part','forget')
        self.local.config['parameters']['retain_part'] = self.local.config['parameters'].get('retain_part','retain')
        self.local.config['parameters']['train_part'] = self.local.config['parameters'].get('train_part','train')

