"""Adversary knowledge models for the link inference attacks.

The benchmark's default threat model gives the adversary the unlearned model plus the
full retain graph G' = (N, E \\ E_f, X).  This module lets an attack query the model on a
*weaker* or *stronger* view of the graph instead, so the same attack can be replayed
across a ladder of adversary capabilities.

Only the message-passing graph handed to the model changes.  The positive/negative
candidate edges an attack is scored on are evaluation ground truth and are built
elsewhere -- they must not be derived from the adversary's view.
"""

import torch


KNOWLEDGE_MODES = ('retain', 'partial', 'none', 'oracle')


def build_adversary_graph(retain_graph, original_graph, knowledge='retain',
                          fraction=1.0, seed=42):
    """Return the graph the adversary queries the model with.

    retain_graph:   G' = (N, E \\ E_f, X), the benchmark default.
    original_graph: G, including the forgotten edges.

    knowledge:
        'retain'  -- the full retain graph (default; identical to previous behaviour).
        'partial' -- a uniform `fraction` of the retain graph's edges.
        'none'    -- node features only, no edges at all.
        'oracle'  -- the original graph, forgotten edges included (upper bound).
    """
    if knowledge == 'retain':
        return retain_graph
    if knowledge == 'oracle':
        return original_graph
    if knowledge == 'none':
        return _with_edge_index(retain_graph, torch.empty((2, 0), dtype=torch.long))
    if knowledge == 'partial':
        return _with_edge_index(retain_graph,
                                _subsample_edges(retain_graph.edge_index, fraction, seed))
    raise ValueError(f"unknown adversary knowledge '{knowledge}', expected one of {KNOWLEDGE_MODES}")


def _subsample_edges(edge_index, fraction, seed):
    """Keep a uniform `fraction` of undirected edges, then re-expand to directed.

    Sampling happens on canonical (min,max) pairs so both directions of an edge are kept
    or dropped together -- a half-edge would be a graph the adversary could never observe.
    """
    if fraction >= 1.0:
        return edge_index
    if fraction <= 0.0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    undirected = sorted({(min(u, v), max(u, v)) for u, v in zip(src, dst)})

    generator = torch.Generator().manual_seed(seed)
    keep_count = int(round(len(undirected) * fraction))
    keep_idx = torch.randperm(len(undirected), generator=generator)[:keep_count].tolist()

    kept = []
    for i in keep_idx:
        u, v = undirected[i]
        kept.append((u, v))
        if u != v:
            kept.append((v, u))

    if not kept:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    return torch.tensor(kept, dtype=torch.long, device=edge_index.device).t().contiguous()


def _with_edge_index(graph, edge_index):
    """Shallow clone of `graph` carrying a different edge_index."""
    clone = graph.clone()
    clone.edge_index = edge_index.to(graph.edge_index.device)
    if getattr(clone, 'edge_attr', None) is not None:
        clone.edge_attr = None
    return clone


def tagged(base_key, tag):
    """Append an adversary tag to a result key, leaving untagged keys byte-identical.

    Existing configs pass no tag, so their result keys -- which downstream analysis
    scripts and the committed run JSONs both parse -- keep exactly their current form.
    """
    return base_key if not tag else f"{base_key} [{tag}]"
