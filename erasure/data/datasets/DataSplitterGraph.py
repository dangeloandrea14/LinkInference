from abc import ABC, abstractmethod
import random
from torch.utils.data import Subset
from erasure.core.base import Configurable
from .Dataset import DatasetWrapper
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import hashlib
import numpy as np

class DataSplitter(ABC):
    def __init__(self, ref_data,parts_names):
        self.ref_data = ref_data
        self.parts_names = parts_names
    
    @abstractmethod
    def split_data(self, data):
        pass

    def set_source(self, datasource):
        self.source = datasource

    
class DataSplitterPercentage(DataSplitter):
    def __init__(self, percentage, parts_names, ref_data = 'all', shuffle=True, edge_removal = False):
        super().__init__(ref_data,parts_names) 
        self.percentage = percentage
        self.shuffle = shuffle
        self.edge_removal = edge_removal

    def split_data(self,partitions):

        if self.edge_removal:
            edge_index = partitions['all'].data.edge_index
            all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
            if self.ref_data != 'all':
                ref_nodes = set(partitions[self.ref_data])
                directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
            else:
                directed_edges = all_edges

            # Canonicalize to undirected edges so both directions are kept together
            indices = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        else:
            indices = partitions[self.ref_data] if self.ref_data != 'all' else list(range(len(partitions[self.ref_data].data.x)))


        self.total_size = len(indices)
        split_point = int(self.total_size * self.percentage)

        indices = self.get_indices(indices) if self.shuffle else indices

        split_indices_1 = indices[:split_point]
        split_indices_2 = indices[split_point:]

        if self.edge_removal:
            split_indices_1 = self._expand_to_directed(split_indices_1)
            split_indices_2 = self._expand_to_directed(split_indices_2)

        partitions[self.parts_names[0]] = split_indices_1
        partitions[self.parts_names[1]] = split_indices_2

        return partitions
    
    def _expand_to_directed(self, undirected_edges):
        """Expand undirected (min,max) edges back to both directions."""
        directed = []
        for u, v in undirected_edges:
            directed.append((u, v))
            if u != v:
                directed.append((v, u))
        return directed

    def get_indices(self, indices):
        seed = self.get_seed_from_name(self.parts_names[0])
        return self.shuffle_with_seed(indices, seed)

    def shuffle_with_seed(self, indices, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)

        permuted_order = torch.randperm(len(indices), generator=generator).tolist()

        shuffled_indices = [indices[i] for i in permuted_order]

        return shuffled_indices

    def get_seed_from_name(self, name):
        hashed_value = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        return hashed_value % (2**32)
    


