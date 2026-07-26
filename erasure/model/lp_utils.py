"""Helpers shared by the link-prediction arm of the benchmark.

The node-classification pipeline treats *nodes* as samples: partitions are node-id
lists, losses index ``model(x, edge_index)[node_ids]``.  Link prediction treats
*node pairs* as samples.  These helpers do the translation in one place so the
predictor (``TorchGraphLinkModel``), the evaluation measure
(``measures.LinkPredictionGraph``) and the unlearners (``GraphUnlearner``) cannot
drift apart in how they build the message-passing graph or sample negatives.

Conventions used throughout:

* a *pair tensor* is ``LongTensor[M, 2]`` of canonical undirected pairs (u < v);
* an *edge_index* is the usual ``LongTensor[2, E]`` holding both directions,
  which is how the datasets in this benchmark store their graphs;
* held-out link-prediction supervision edges live in the ``lp_val_pos`` /
  ``lp_test_pos`` partitions and are removed from the message-passing graph, so
  the model cannot read a test edge straight off the adjacency it aggregates over.
"""

from contextlib import contextmanager

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling

# Partitions held out of the message-passing graph for link prediction.
HOLDOUT_PARTS = ('lp_val_pos', 'lp_test_pos')


@contextmanager
def seeded(seed):
    """Run a block with a fixed torch seed, restoring the global RNG afterwards.

    Used so that evaluation negatives are identical for every model in a run
    without perturbing the training RNG stream (`negative_sampling` draws from
    the global generator and takes no `generator` argument).
    """
    if seed is None:
        yield
        return
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)


def as_pair_tensor(edges, device=None):
    """Coerce a partition edge list or an edge_index into ``LongTensor[M, 2]``."""
    if torch.is_tensor(edges):
        t = edges.t() if (edges.dim() == 2 and edges.size(0) == 2 and edges.size(1) != 2) else edges
        t = t.reshape(-1, 2)
    else:
        edges = list(edges)
        if len(edges) == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=device)
        t = torch.as_tensor(edges, dtype=torch.long)
    t = t.to(dtype=torch.long)
    return t if device is None else t.to(device)


def canonical_pairs(edges, device=None):
    """Deduplicated undirected pairs (u < v) from a directed edge list/index."""
    t = as_pair_tensor(edges, device=device)
    if t.numel() == 0:
        return t.reshape(0, 2)
    lo = torch.minimum(t[:, 0], t[:, 1])
    hi = torch.maximum(t[:, 0], t[:, 1])
    pairs = torch.stack([lo, hi], dim=1)
    pairs = pairs[lo != hi]                      # drop self-loops
    return torch.unique(pairs, dim=0)


def pair_keys(pairs, num_nodes):
    """Flatten pairs to a single integer key so membership tests are vectorised."""
    pairs = pairs.reshape(-1, 2)
    return pairs[:, 0] * num_nodes + pairs[:, 1]


def keys_in(keys, test_keys):
    """Vectorised ``keys in test_keys``, without ``torch.isin``.

    ``torch.isin`` may fall back to a broadcast comparison, which allocates
    ``len(keys) * len(test_keys)`` elements -- on Flickr that is 899,756 x 134,960
    = 121 GB, and it aborts on MPS.  Sorting once and binary-searching is exact,
    allocates O(len(keys)), and is what makes the large graphs tractable here.
    """
    if test_keys.numel() == 0 or keys.numel() == 0:
        return torch.zeros_like(keys, dtype=torch.bool)
    ordered, _ = torch.sort(test_keys)
    idx = torch.searchsorted(ordered, keys).clamp(max=ordered.numel() - 1)
    return ordered[idx] == keys


def both_direction_keys(pairs, num_nodes):
    """Keys for (u,v) and (v,u) of every pair."""
    pairs = pairs.reshape(-1, 2)
    fwd = pairs[:, 0] * num_nodes + pairs[:, 1]
    bwd = pairs[:, 1] * num_nodes + pairs[:, 0]
    return torch.cat([fwd, bwd])


def mp_edge_index(edge_index, exclude, num_nodes):
    """`edge_index` with both directions of every pair in `exclude` removed.

    `exclude` may be a partition edge list, a pair tensor or an edge_index.
    """
    ex = canonical_pairs(exclude)
    if ex.numel() == 0:
        return edge_index
    device = edge_index.device
    keys = edge_index[0].to(torch.long) * num_nodes + edge_index[1].to(torch.long)
    ex_keys = both_direction_keys(ex.to(device), num_nodes)
    return edge_index[:, ~keys_in(keys, ex_keys)]


def sample_negative_pairs(num, edge_index, num_nodes, seed=None):
    """`num` node pairs that are **not** edges of `edge_index`, as ``[num, 2]``.

    `edge_index` should be the *full* observed graph (not the message-passing
    graph), so that held-out or forgotten edges are never handed out as negatives.
    """
    if num <= 0:
        return torch.zeros((0, 2), dtype=torch.long, device=edge_index.device)
    with seeded(seed):
        neg = negative_sampling(edge_index, num_nodes=num_nodes,
                                num_neg_samples=int(num), method='sparse')
    return neg.t().contiguous()


def corrupt_pairs(pos_pairs, edge_index, num_nodes, seed=None, tries=4):
    """Negatives that share a head node with `pos_pairs` (corrupted-tail sampling).

    Keeps the negatives anchored on the same nodes as the positives, which is what
    makes the loss *local* to a node subset -- the property the unlearners rely on
    when they restrict a gradient step to the nodes affected by the forget set.
    """
    pos_pairs = pos_pairs.reshape(-1, 2)
    if pos_pairs.numel() == 0:
        return pos_pairs
    device = pos_pairs.device
    existing = (edge_index[0].to(torch.long) * num_nodes
                + edge_index[1].to(torch.long)).to(device)

    heads = pos_pairs[:, 0]
    with seeded(seed):
        tails = torch.randint(0, num_nodes, (heads.numel(),), device=device)
        for _ in range(tries):
            bad = (tails == heads) | keys_in(heads * num_nodes + tails, existing)
            if not bool(bad.any()):
                break
            tails[bad] = torch.randint(0, num_nodes, (int(bad.sum()),), device=device)

    keep = ~((tails == heads) | keys_in(heads * num_nodes + tails, existing))
    return torch.stack([heads[keep], tails[keep]], dim=1)


def pairs_touching(pairs, node_subset, num_nodes):
    """Subset of `pairs` with at least one endpoint in `node_subset`."""
    pairs = pairs.reshape(-1, 2)
    if pairs.numel() == 0:
        return pairs
    mask_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=pairs.device)
    idx = torch.as_tensor(list(node_subset), dtype=torch.long, device=pairs.device)
    if idx.numel() == 0:
        return pairs.new_zeros((0, 2))
    mask_nodes[idx] = True
    return pairs[mask_nodes[pairs[:, 0]] | mask_nodes[pairs[:, 1]]]


def ranking_scores(pos_scores, neg_scores):
    """ROC-AUC and average precision of positive vs negative pair scores."""
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return float('nan'), float('nan')
    y = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    s = torch.cat([pos_scores, neg_scores])
    y = y.detach().cpu().numpy()
    s = s.detach().float().cpu().numpy()
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
