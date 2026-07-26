from abc import ABC, abstractmethod
from collections import Counter, defaultdict
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

    
class _EdgeSplitterBase(DataSplitter):
    """Shared machinery for edge-partitioning splitters.

    The four original splitters in this file each carry their own copy of
    ``_expand_to_directed`` / ``shuffle_with_seed`` / ``get_seed_from_name`` and
    inline the canonicalisation preamble.  They are deliberately left as they
    are: every published benchmark config depends on them, and
    ``rebuttal/walkcentrality_impact/measure_impact.py`` instantiates
    ``DataSplitterEdgeDifficulty`` directly.  New splitters share this base
    instead of adding a fifth copy.

    Subclasses implement ``select_forget(partitions, edges) -> list[(u,v)]``,
    receiving the canonical undirected edge list and returning the subset that
    goes into the forget partition.
    """

    def __init__(self, percentage, parts_names, ref_data='all', exclude_parts=None):
        super().__init__(ref_data, parts_names)
        self.percentage = percentage
        self.exclude_parts = exclude_parts or []

    def split_data(self, partitions):
        edges = self.canonical_edges(partitions)
        forget = self.select_forget(partitions, edges)

        forget_set = set(forget)
        retain = [e for e in edges if e not in forget_set]

        return self.emit(partitions, forget, retain, edges)

    @abstractmethod
    def select_forget(self, partitions, edges):
        pass

    def canonical_edges(self, partitions):
        """Canonical undirected (min,max) edge list, honouring ref_data/exclude_parts.

        Mirrors the preamble of ``DataSplitterEdgeDifficulty.split_data`` exactly,
        so the eligible edge pool is identical across old and new splitters.
        """
        edge_index = partitions['all'].data.edge_index
        all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        if self.ref_data != 'all':
            ref_nodes = set(partitions[self.ref_data])
            directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
        else:
            directed_edges = all_edges

        undirected_edges = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        if self.exclude_parts:
            excluded = set()
            for part in self.exclude_parts:
                for u, v in partitions.get(part, []):
                    excluded.add((min(u, v), max(u, v)))
            undirected_edges = [e for e in undirected_edges if e not in excluded]

        return undirected_edges

    def emit(self, partitions, forget, retain, edges):
        """Assign the two partitions after asserting forget and retain tile `edges`.

        Every downstream consumer relies on this: the Gold Model retrains on
        ``retain``, so a leak or a gap silently corrupts the reference baseline
        that the whole benchmark is measured against.
        """
        if len(forget) + len(retain) != len(edges):
            raise ValueError(
                f"{type(self).__name__}: forget ({len(forget)}) + retain ({len(retain)}) "
                f"!= eligible edges ({len(edges)})")
        if set(forget) & set(retain):
            raise ValueError(f"{type(self).__name__}: forget and retain overlap")

        partitions[self.parts_names[0]] = self._expand_to_directed(forget)
        partitions[self.parts_names[1]] = self._expand_to_directed(retain)
        return partitions

    def budget(self, edges):
        return int(len(edges) * self.percentage)

    def node_attr(self, data, name):
        """Fetch a per-node attribute as a 1-D tensor.

        ``partitions['all'].data`` is a PyG ``InMemoryDataset`` whose
        ``__getattr__`` forwards node-level keys off the collated ``Data``.  The
        two fallbacks cover data sources that expose the underlying ``Data``
        differently (e.g. ``SingleGraphDataset`` in TwitchGamersDataSource).
        """
        for get in (lambda: getattr(data, name),
                    lambda: getattr(data, '_data')[name],
                    lambda: data[0][name]):
            try:
                value = get()
            except (AttributeError, KeyError, TypeError, IndexError):
                continue
            if value is not None:
                return torch.as_tensor(value).reshape(-1)

        available = []
        for probe in (data, getattr(data, '_data', None)):
            if probe is not None and hasattr(probe, 'keys'):
                keys = probe.keys
                available = list(keys() if callable(keys) else keys)
                break
        raise AttributeError(
            f"{type(self).__name__}: node attribute '{name}' not found on the graph. "
            f"Available keys: {available}")

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

    def seeded_shuffle(self, items):
        """Shuffle seeded from the forget partition's name, as the other splitters do."""
        return self.shuffle_with_seed(items, self.get_seed_from_name(self.parts_names[0]))


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


class DataSplitterEdgeHoldout(DataSplitter):
    """Holds out link-prediction supervision edges: test / validation / train.

    Needed because a link-prediction model must not see the edges it is evaluated
    on in the adjacency it aggregates over.  ``parts_names`` is
    ``[test, validation, train]`` and receives, respectively, ``test_percentage``
    and ``val_percentage`` of the canonical undirected edges (seeded shuffle), with
    the remainder going to the train partition.  All three are expanded back to
    directed pairs, matching the convention of the other edge splitters.

    ``DataSplitterPercentage(edge_removal=True)`` cannot be chained to do this: its
    ``ref_data`` filter treats the referenced partition as a *node* list, so it
    cannot subdivide an edge partition.
    """

    def __init__(self, parts_names, test_percentage=0.1, val_percentage=0.05,
                 ref_data='all'):
        super().__init__(ref_data, parts_names)
        self.test_percentage = test_percentage
        self.val_percentage = val_percentage

    def split_data(self, partitions):
        edge_index = partitions['all'].data.edge_index
        all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        if self.ref_data != 'all':
            ref_nodes = set(partitions[self.ref_data])
            directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
        else:
            directed_edges = all_edges

        undirected_edges = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        seed = self.get_seed_from_name(self.parts_names[0])
        undirected_edges = self.shuffle_with_seed(undirected_edges, seed)

        total = len(undirected_edges)
        n_test = int(total * self.test_percentage)
        n_val = int(total * self.val_percentage)

        chunks = [undirected_edges[:n_test],
                  undirected_edges[n_test:n_test + n_val],
                  undirected_edges[n_test + n_val:]]

        for name, chunk in zip(self.parts_names, chunks):
            partitions[name] = self._expand_to_directed(chunk)

        return partitions

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

    def __init__(self, percentage, parts_names, ref_data='all', mode='hard', k=2,
                 exclude_parts=None):
        super().__init__(ref_data, parts_names)
        self.percentage = percentage
        self.mode = mode
        self.k = k
        # Partitions whose edges are not eligible for either output partition.
        # Used by the link-prediction arm so the forget set is drawn from the
        # training edges only, never from the held-out lp_val/lp_test supervision
        # edges.  Defaults to None, i.e. the node-classification behaviour.
        self.exclude_parts = exclude_parts or []

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

        if self.exclude_parts:
            excluded = set()
            for part in self.exclude_parts:
                for u, v in partitions.get(part, []):
                    excluded.add((min(u, v), max(u, v)))
            undirected_edges = [e for e in undirected_edges if e not in excluded]

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


class DataSplitterEdgeTemporal(_EdgeSplitterBase):
    """Forget set drawn from a contiguous slice of the graph's timeline.

    Models the time-scoped deletion requests that dominate practice: "remove my
    recent activity" (``mode='recent'``), a retention period expiring
    (``mode='oldest'``), or a bulk request tied to one period
    (``mode='window'``).

    An edge carries no timestamp of its own, so it inherits one from its
    endpoints: ``t(u,v) = max(t[u], t[v])`` with ``edge_time='max'``.  In a
    citation graph this is the edge's creation time -- the citation cannot exist
    before the later of the two papers is published.  ``edge_time='min'`` gives
    the earliest endpoint instead.  This is a proxy, not a recorded edge
    timestamp, and should be described as such.

    Timestamps are coarse (arXiv years), so the boundary of the cut holds far more
    tied edges than it needs.  The tie is broken by a seeded shuffle applied
    *before* a stable sort, which samples the boundary period uniformly at random;
    sorting by ``(t_e, u, v)`` instead would bias it towards low node ids.
    """

    def __init__(self, percentage, parts_names, ref_data='all', mode='recent',
                 time_attr='node_year', edge_time='max', window=None,
                 window_pad='nearest', missing='exclude', exclude_parts=None):
        super().__init__(percentage, parts_names, ref_data, exclude_parts)
        if mode not in ('recent', 'oldest', 'window'):
            raise ValueError(f"mode must be recent|oldest|window, got {mode!r}")
        if mode == 'window' and window is None:
            raise ValueError("mode='window' requires the `window` parameter")
        self.mode = mode
        self.time_attr = time_attr
        self.edge_time = edge_time
        self.window = window
        self.window_pad = window_pad
        self.missing = missing

    def select_forget(self, partitions, edges):
        times = self.node_attr(partitions['all'].data, self.time_attr)
        reduce = torch.maximum if self.edge_time == 'max' else torch.minimum

        src = torch.tensor([u for u, _ in edges], dtype=torch.long)
        dst = torch.tensor([v for _, v in edges], dtype=torch.long)
        t_edge = reduce(times[src], times[dst]).tolist()

        # A node with no recorded time makes its edges undatable.  They stay in
        # retain (so coverage still holds) unless imputation is requested.
        valid = [t > 0 and t == t for t in t_edge]           # t == t rejects NaN
        n_missing = len(valid) - sum(valid)
        if n_missing and self.missing == 'median':
            median = float(np.median([t for t, ok in zip(t_edge, valid) if ok]))
            t_edge = [t if ok else median for t, ok in zip(t_edge, valid)]
            valid = [True] * len(t_edge)
        if n_missing:
            print(f"[{type(self).__name__}] {n_missing} edges without a usable "
                  f"'{self.time_attr}' ({self.missing})")

        eligible = [(e, t) for e, t, ok in zip(edges, t_edge, valid) if ok]
        budget = self.budget(edges)

        if self.mode == 'window':
            return self._select_window(eligible, budget)

        # Seeded shuffle first, then a stable sort: ties inside a period end up in
        # a reproducible random order.
        ordered = sorted(self.seeded_shuffle(eligible), key=lambda pair: pair[1])
        chosen = ordered[-budget:] if self.mode == 'recent' else ordered[:budget]
        span = [t for _, t in chosen]
        print(f"[{type(self).__name__}] mode={self.mode} |E_f|={len(chosen)} "
              f"(budget {budget}) {self.time_attr} span "
              f"{min(span) if span else '-'}..{max(span) if span else '-'}")
        return [e for e, _ in chosen]

    def _select_window(self, eligible, budget):
        in_window = [(e, t) for e, t in eligible if t == self.window]

        if len(in_window) >= budget:
            forget = [e for e, _ in self.seeded_shuffle(in_window)[:budget]]
            print(f"[{type(self).__name__}] window={self.window} |E_f|={len(forget)} "
                  f"drawn from {len(in_window)} edges in that period")
            return forget

        if self.window_pad != 'nearest':
            print(f"[{type(self).__name__}] WARNING window={self.window} holds only "
                  f"{len(in_window)} edges < budget {budget}; window_pad='none' so "
                  f"|E_f| does NOT match the other settings")
            return [e for e, _ in in_window]

        # Pad outwards from the window by temporal distance, ties shuffled.
        rest = self.seeded_shuffle([(e, t) for e, t in eligible if t != self.window])
        rest.sort(key=lambda pair: abs(pair[1] - self.window))
        padded = in_window + rest[:budget - len(in_window)]
        composition = sorted(Counter(t for _, t in padded).items())
        print(f"[{type(self).__name__}] window={self.window} held {len(in_window)} edges "
              f"< budget {budget}; padded to {len(padded)} across periods {composition}")
        return [e for e, _ in padded]


class DataSplitterEdgeGroup(_EdgeSplitterBase):
    """Forget set scoped to one cohort of nodes.

    Models a group of users deleting their connections collectively, rather than a
    request scattered uniformly over the graph.  ``mode='intra'`` takes edges with
    both endpoints inside the cohort (its internal links); ``mode='incident'``
    takes every edge touching it.

    The cohort is read from a node attribute -- by default ``y``, i.e. the node
    class, which stands in for product category / research community / subject area
    depending on the dataset.  These are cohorts, not protected attributes: none of
    the benchmark datasets ships a real user attribute.

    ``group=None`` resolves the cohort automatically and prints the full per-group
    table.  Shipped configs should pin an explicit integer so a run cannot drift.
    The default ``select='nearest'`` picks the *smallest cohort that can still fund
    the budget*, which is the most defensible unpinned choice: it needs the least
    subsampling, so the forget set is closest to a *complete* cohort deletion rather
    than an arbitrary fraction of a large community's links.  Note this is nearest
    *from above*, not nearest in absolute distance -- the latter can land on a cohort
    below the budget and silently shrink the forget set (Flickr has no class whose
    internal links reach 5% until class 4 at 11%, so absolute distance would pick a
    cohort 56% short).
    """

    # A cohort within this fraction of the budget counts as funding it; the
    # residual shortfall is reported but not warned about, because a cohort that
    # naturally sits at ~5% of the graph is the realistic case we want and being a
    # handful of edges short of the nominal budget changes nothing.
    BUDGET_TOLERANCE = 0.01

    def __init__(self, percentage, parts_names, ref_data='all', mode='incident',
                 group_attr='y', group=None, select='nearest', fill='none',
                 exclude_parts=None):
        super().__init__(percentage, parts_names, ref_data, exclude_parts)
        if mode not in ('intra', 'incident'):
            raise ValueError(f"mode must be intra|incident, got {mode!r}")
        if select not in ('nearest', 'largest', 'smallest', 'median'):
            raise ValueError(f"select must be nearest|largest|smallest|median, got {select!r}")
        self.mode = mode
        self.group_attr = group_attr
        self.group = group
        self.select = select
        self.fill = fill

    def select_forget(self, partitions, edges):
        groups = self.node_attr(partitions['all'].data, self.group_attr).long().tolist()
        budget = self.budget(edges)

        wanted = self._resolve_groups(self._eligible_counts(edges, groups), budget)
        eligible = [e for e in edges if self._matches(e, groups, wanted)]

        print(f"[{type(self).__name__}] mode={self.mode} group(s)={sorted(wanted)} "
              f"eligible={len(eligible)} "
              f"({100 * len(eligible) / max(len(edges), 1):.1f}% of edges) budget={budget}")

        if len(eligible) < budget and self.fill == 'none':
            shortfall = 1 - len(eligible) / max(budget, 1)
            if shortfall > self.BUDGET_TOLERANCE:
                print(f"[{type(self).__name__}] WARNING cohort holds {len(eligible)} edges "
                      f"< budget {budget} ({shortfall:.1%} short); |E_f| does NOT match "
                      f"the other settings")
            else:
                print(f"[{type(self).__name__}] cohort deleted in full: {len(eligible)} "
                      f"edges, {shortfall:.2%} under the nominal budget")
            return eligible

        return self.seeded_shuffle(eligible)[:budget]

    def _matches(self, edge, groups, wanted):
        u, v = edge
        if self.mode == 'intra':
            return groups[u] in wanted and groups[v] in wanted
        return groups[u] in wanted or groups[v] in wanted

    def _eligible_counts(self, edges, groups):
        """Per-group eligible edge count under the active mode."""
        counts = Counter()
        for u, v in edges:
            gu, gv = groups[u], groups[v]
            if self.mode == 'intra':
                if gu == gv:
                    counts[gu] += 1
            else:
                counts[gu] += 1
                if gv != gu:
                    counts[gv] += 1
        for g in set(groups):
            counts.setdefault(g, 0)
        return counts

    def _resolve_groups(self, counts, budget):
        if self.group is not None:
            return {self.group} if isinstance(self.group, int) else set(self.group)

        table = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"[{type(self).__name__}] group_attr={self.group_attr} mode={self.mode} "
              f"eligible edges per group (budget {budget}): {table}")

        # "Viable" = able to fund the budget, allowing the same slack as select_forget so
        # a cohort sitting a handful of edges under it still counts.
        floor = budget * (1 - self.BUDGET_TOLERANCE)
        viable = [g for g, c in table if c >= floor]

        if not viable:
            # Nothing can fund the budget; the largest cohort is the least-bad choice and
            # select_forget will warn about the shortfall.
            chosen = [table[0][0]]
        elif self.select == 'nearest':
            # The smallest cohort that still funds the budget, i.e. nearest from above --
            # least subsampling, so the forget set is closest to a *complete* cohort
            # deletion.  Deliberately not the nearest in absolute distance: that can
            # select a cohort below the budget and silently shrink the forget set.
            chosen = [viable[-1]]
        elif self.select == 'largest':
            chosen = [viable[0]]
        elif self.select == 'smallest':
            chosen = [viable[-1]]
        else:
            chosen = [viable[len(viable) // 2]]

        # Merge in further groups only if a single one cannot fund the budget.
        if self.fill == 'next_group' and counts[chosen[0]] < budget:
            total = counts[chosen[0]]
            for g, c in table:
                if g in chosen:
                    continue
                chosen.append(g)
                total += c
                if total >= budget:
                    break
            print(f"[{type(self).__name__}] fill='next_group' merged {chosen}")

        return set(chosen)


class DataSplitterEdgeUserDeletion(_EdgeSplitterBase):
    """Forget set built from whole node neighbourhoods -- account deletion.

    "Delete my account" is the most common unlearning request in practice (GDPR
    Art. 17), and in a graph it removes *every* edge incident to a node rather than
    a scattered sample.  Structurally this is a different regime from the
    benchmark's edge-scoped settings, and it needs no metadata, so it applies to
    every dataset.

    Set ``group_attr``/``group`` to restrict deletion to one cohort -- a group of
    users closing their accounts together.

    Whole nodes are added until the edge budget is reached; the last node is never
    truncated, because a partially deleted account is not the scenario being
    modelled.  ``|E_f|`` therefore lands in ``[budget, budget + deg(last))`` and is
    reported.
    """

    def __init__(self, percentage, parts_names, ref_data='all', mode='random',
                 group_attr=None, group=None, exclude_parts=None):
        super().__init__(percentage, parts_names, ref_data, exclude_parts)
        if mode not in ('random', 'high_degree', 'low_degree'):
            raise ValueError(f"mode must be random|high_degree|low_degree, got {mode!r}")
        self.mode = mode
        self.group_attr = group_attr
        self.group = group

    def select_forget(self, partitions, edges):
        incident = defaultdict(list)
        for edge in edges:
            u, v = edge
            incident[u].append(edge)
            if v != u:
                incident[v].append(edge)

        candidates = self._candidates(partitions, incident)
        budget = self.budget(edges)

        forget, seen, n_accounts = [], set(), 0
        for node in candidates:
            if len(forget) >= budget:
                break
            n_accounts += 1
            for edge in incident[node]:
                if edge not in seen:
                    seen.add(edge)
                    forget.append(edge)

        print(f"[{type(self).__name__}] mode={self.mode} deleted {n_accounts} accounts "
              f"-> |E_f|={len(forget)} (budget {budget}, "
              f"{100 * len(forget) / max(len(edges), 1):.2f}% of edges)")
        if len(forget) < budget:
            print(f"[{type(self).__name__}] WARNING exhausted candidate nodes at "
                  f"{len(forget)} edges < budget {budget}")
        return forget

    def _candidates(self, partitions, incident):
        nodes = sorted(incident)

        if self.group is not None:
            if self.group_attr is None:
                raise ValueError("`group` requires `group_attr`")
            groups = self.node_attr(partitions['all'].data, self.group_attr).long().tolist()
            wanted = {self.group} if isinstance(self.group, int) else set(self.group)
            in_cohort = [n for n in nodes if groups[n] in wanted]
            print(f"[{type(self).__name__}] restricted to {self.group_attr}="
                  f"{sorted(wanted)}: {len(in_cohort)}/{len(nodes)} nodes")
            # Nodes outside the cohort are kept as a tail so the budget is still
            # reachable if the cohort alone cannot fund it; select_forget warns.
            nodes = self.seeded_shuffle(in_cohort) + self.seeded_shuffle(
                [n for n in nodes if groups[n] not in wanted])
        else:
            nodes = self.seeded_shuffle(nodes)

        if self.mode == 'random':
            return nodes
        # Stable sort over the shuffled order: ties in degree stay randomised.
        return sorted(nodes, key=lambda n: len(incident[n]),
                      reverse=(self.mode == 'high_degree'))
