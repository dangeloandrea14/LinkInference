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


class DataSplitterCyclicEdges(DataSplitter):
    """Selects edges that participate in n-cycles (e.g. triangles for n=3).

    parts_names[0] receives the selected percentage of cycle-participating edges,
    sorted by ascending cycle count (fewest n-cycles first).  This ordering
    maximises coverage uniformity: the first p% of edges are each in a distinct
    set of n-cycles, so selecting p% of edges touches approximately p% of the
    distinct n-cycles.  Edges in many cycles come last — they are redundant once
    the unique-cycle edges are already selected.

    parts_names[1] receives all remaining edges (unselected cycle-edges plus
    non-cycle edges), so the two partitions always cover the full edge set.

    Cycle count per edge is [A^{n-1}]_{u,v}: the number of (n-1)-step walks
    between u and v.  This is exact for n=3 on simple undirected graphs (equals
    the number of common neighbours).  For n>3 it is an upper-bound
    approximation because it counts non-simple walks as well.

    Computation uses sparse matrix-vector products grouped by source node,
    so memory usage is O(|E| + N) regardless of n.
    """

    def __init__(self, n, parts_names, ref_data='all', percentage=1.0):
        super().__init__(ref_data, parts_names)
        self.n = n
        self.percentage = percentage

    def split_data(self, partitions):
        edge_index = partitions['all'].data.edge_index
        all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        if self.ref_data != 'all':
            ref_nodes = set(partitions[self.ref_data])
            directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
        else:
            directed_edges = all_edges

        undirected_edges = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        cycle_counts = self._compute_cycle_counts(partitions['all'].data, undirected_edges)

        # Separate cycle-edges (sorted ascending by count) from non-cycle edges
        cycle_edges = sorted(
            [(e, c) for e, c in zip(undirected_edges, cycle_counts) if c > 0],
            key=lambda x: x[1]
        )
        non_cycle_edges = [e for e, c in zip(undirected_edges, cycle_counts) if c == 0]

        split_point = int(len(cycle_edges) * self.percentage)
        selected  = [e for e, _ in cycle_edges[:split_point]]
        remaining = [e for e, _ in cycle_edges[split_point:]] + non_cycle_edges

        partitions[self.parts_names[0]] = self._expand_to_directed(selected)
        partitions[self.parts_names[1]] = self._expand_to_directed(remaining)

        return partitions

    def _compute_cycle_counts(self, data, undirected_edges):
        """Return [A^{n-1}]_{u,v} for each (u,v) via sparse MV products.

        Groups edges by source node u so each A^{n-1} e_u vector is computed
        once and reused for all edges sharing that source.
        """
        from collections import defaultdict

        N = data.x.size(0)
        edge_index = data.edge_index

        vals = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        A = torch.sparse_coo_tensor(edge_index, vals, (N, N)).coalesce()

        u_groups = defaultdict(list)
        for idx, (u, v) in enumerate(undirected_edges):
            u_groups[u].append((idx, v))

        counts = [0.0] * len(undirected_edges)
        for u, pairs in u_groups.items():
            e_u = torch.zeros(N, dtype=torch.float32, device=edge_index.device)
            e_u[u] = 1.0

            # Compute A^{n-1} e_u via n-1 sparse MV products
            vec = e_u
            for _ in range(self.n - 1):
                vec = torch.sparse.mm(A, vec.unsqueeze(1)).squeeze(1)

            # [A^{n-1}]_{u,v} = (A^{n-1} e_u)[v]  (valid because A is symmetric)
            for idx, v in pairs:
                counts[idx] = vec[v].item()

        return counts

    def _expand_to_directed(self, undirected_edges):
        directed = []
        for u, v in undirected_edges:
            directed.append((u, v))
            if u != v:
                directed.append((v, u))
        return directed


class DataSplitterEdgeDifficulty(DataSplitter):
    """Splits edges into two partitions by percentage, ordered either randomly
    (mode='simple') or by descending walk-centrality (mode='hard').

    Walk centrality measures how much of the information flowing through
    k-step walks (under the symmetrically normalised adjacency A_hat) passes
    through a given edge.  High-centrality edges influence more walks and are
    therefore harder to unlearn.

        WalkCentrality(i,j) = sum_{t=0}^{k-1}  (1^T A_hat^t)_i * (A_hat^{k-t-1} 1)_j

    Because A_hat is symmetric for undirected graphs this simplifies to:

        WalkCentrality(i,j) = sum_{t=0}^{k-1}  r_t[i] * r_{k-1-t}[j]

    where r_p = A_hat^p * 1 (row-sum vector of A_hat^p), computed via k-1
    sparse matrix-vector products.
    """

    def __init__(self, percentage, parts_names, ref_data='all', mode='hard', k=2):
        super().__init__(ref_data, parts_names)
        self.percentage = percentage
        self.mode = mode
        self.k = k

    def split_data(self, partitions):
        edge_index = partitions['all'].data.edge_index
        all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        if self.ref_data != 'all':
            ref_nodes = set(partitions[self.ref_data])
            directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
        else:
            directed_edges = all_edges

        # Canonicalize to undirected (min, max) pairs
        undirected_edges = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        if self.mode == 'simple':
            seed = self.get_seed_from_name(self.parts_names[0])
            undirected_edges = self.shuffle_with_seed(undirected_edges, seed)
        else:  # 'hard' or 'easy': sort by walk centrality
            centralities = self._compute_walk_centrality(partitions['all'].data, undirected_edges)
            descending = (self.mode != 'easy')  # hard → descending (highest first); easy → ascending (lowest first)
            undirected_edges = [e for _, e in sorted(zip(centralities, undirected_edges), reverse=descending)]

        split_point = int(len(undirected_edges) * self.percentage)
        partitions[self.parts_names[0]] = self._expand_to_directed(undirected_edges[:split_point])
        partitions[self.parts_names[1]] = self._expand_to_directed(undirected_edges[split_point:])

        return partitions

    def _compute_walk_centrality(self, data, undirected_edges):
        N = data.x.size(0)
        edge_index = data.edge_index

        # A_tilde = A + I
        self_loops = torch.arange(N, device=edge_index.device)
        loop_index = torch.stack([self_loops, self_loops])
        edge_index_tilde = torch.cat([edge_index, loop_index], dim=1)

        # Symmetric normalisation: A_hat = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
        row, col = edge_index_tilde
        deg = torch.zeros(N, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones(edge_index_tilde.size(1), device=edge_index.device))
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e9)

        weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        A_hat = torch.sparse_coo_tensor(edge_index_tilde, weights, (N, N)).coalesce()

        # r_p = A_hat^p * 1  for p = 0, ..., k-1
        r = [torch.ones(N, device=edge_index.device)]
        for _ in range(self.k - 1):
            r.append(torch.sparse.mm(A_hat, r[-1].unsqueeze(1)).squeeze(1))

        # WalkCentrality(i,j) = sum_t r_t[i] * r_{k-1-t}[j]
        centralities = [
            sum(r[t][u].item() * r[self.k - 1 - t][v].item() for t in range(self.k))
            for (u, v) in undirected_edges
        ]

        return centralities

    def _expand_to_directed(self, undirected_edges):
        directed = []
        for u, v in undirected_edges:
            directed.append((u, v))
            if u != v:
                directed.append((v, u))
        return directed

    def shuffle_with_seed(self, indices, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        permuted_order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in permuted_order]

    def get_seed_from_name(self, name):
        hashed_value = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        return hashed_value % (2**32)
