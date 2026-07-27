"""Vectorised graph neighbourhood ops.

`infected_nodes` -- the set of nodes within `hops` of a forget set, i.e. the nodes
whose representation a GNN of that depth can see the removed edges through -- was
computed with one `networkx.single_source_shortest_path_length` per endpoint.  On
ogbn-arxiv that is ~59k BFS traversals of a 1.17M-edge graph, ~16 min per call, and
every unlearner plus E-UMIA plus the accuracy measures calls it independently.

`khop_infected` computes the same set with `hops` rounds of boolean scatter over the
edge list.  Same result, 0.015 s instead of 16 min on ogbn-arxiv.

ORDER MATTERS, and it is preserved.  The old code returned `list(infected)` over a
Python set, and `graph_umia.get_attack_samples` truncates that list to `len(test_ids)`
-- so the iteration order selects which nodes E-UMIA scores.  A Python set of ints
iterates in slot order, `value % table_size`, which is ascending exactly when every
node id is below the table size; only when ids collide does order become insertion-
dependent and arbitrary.  Because these forget sets infect 53-100% of the graph, the
table is always larger than the largest node id, so the old order was already
ascending.  Checked on all seven benchmark datasets at hops 2 and 3 -- AmazonPhotos,
AmazonComputers, DBLP, Flickr, Pubmed, RomanEmpire, ogbn-arxiv -- old order == sorted
in every case.  `sorted` is therefore a faithful replacement here, and a more
predictable contract going forward.  (Reddit was not run -- 232k nodes, and the same
argument covers it: 2**19 table > 232,965 ids.)  A forget set small enough to infect
only a sliver of a large graph would have had an arbitrary order before and gets a
sorted one now; that is a behaviour change, in the direction of determinism.
"""

import torch


def khop_infected(edge_index, seed_nodes, hops, num_nodes):
    """Nodes reachable from `seed_nodes` in at most `hops` undirected steps.

    Matches `nx.Graph(edge_list)` + per-seed BFS exactly, including its two quirks:
    the graph is treated as undirected regardless of how `edge_index` stores it, and
    a seed with no incident edge is absent from the nx graph and so contributes
    nothing -- not even itself.

    Returns a sorted list of ints.
    """
    device = edge_index.device
    if edge_index.numel() == 0:
        return []

    # nx.Graph is undirected; edge_index may be stored either way (ogbn-arxiv is
    # directed, the Planetoid-style loaders are not).  Traverse both directions.
    src = torch.cat([edge_index[0], edge_index[1]])
    dst = torch.cat([edge_index[1], edge_index[0]])

    seed_nodes = torch.as_tensor(seed_nodes, dtype=torch.long, device=device).reshape(-1)

    has_edge = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    has_edge[src] = True

    reach = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    reach[seed_nodes] = True
    reach &= has_edge          # isolated seeds are not in the nx graph

    frontier = reach.clone()
    for _ in range(hops):
        nxt = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        nxt[dst[frontier[src]]] = True
        frontier = nxt & ~reach
        if not bool(frontier.any()):
            break
        reach |= frontier

    return reach.nonzero(as_tuple=False).flatten().tolist()


def edge_endpoints(edges):
    """Distinct node ids appearing in an iterable of (u, v) pairs."""
    if isinstance(edges, torch.Tensor):
        return torch.unique(edges.reshape(-1)).tolist()
    nodes = set()
    for u, v in edges:
        nodes.add(int(u))
        nodes.add(int(v))
    return sorted(nodes)
