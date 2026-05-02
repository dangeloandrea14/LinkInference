from abc import ABCMeta, abstractmethod
from copy import copy
import networkx as nx
from erasure.core.base import Configurable
from erasure.evaluations.manager import Evaluation


class Measure(Configurable, metaclass=ABCMeta):

    @abstractmethod
    def process(self, e:Evaluation):
        return e


class GraphMeasure(Measure):

    def get_model(self,e: Evaluation):

        if hasattr(self, "target") and self.target == 'unlearned':
            erasure_model = e.unlearned_model
        else:
            erasure_model = e.predictor

        erasure_model.model.eval()

        return erasure_model

    def infected_nodes(self, unlearner, edges_to_forget, hops, _cache=None):
        if _cache is not None:
            nx_key = ('_nx_graph', id(unlearner.dataset))
            if nx_key not in _cache:
                G = nx.Graph()
                G.add_edges_from(unlearner.dataset.partitions['all'][0][0].edge_index.t().tolist())
                _cache[nx_key] = G
            G = _cache[nx_key]
            inf_key = ('_infected', id(edges_to_forget), hops)
            if inf_key not in _cache:
                _cache[inf_key] = self._bfs_infected(G, edges_to_forget, hops)
            return _cache[inf_key]
        G = nx.Graph()
        G.add_edges_from(unlearner.dataset.partitions['all'][0][0].edge_index.t().tolist())
        return self._bfs_infected(G, edges_to_forget, hops)

    def _bfs_infected(self, G, edges_to_forget, hops):
        edge_nodes = set()
        for u, v in edges_to_forget:
            edge_nodes.add(u)
            edge_nodes.add(v)
        infected = set()
        for node in edge_nodes:
            if node in G:
                infected.update(nx.single_source_shortest_path_length(G, node, cutoff=hops).keys())
        return list(infected)

    def _get_revised_graph(self, e, source_partition, forget_edges):
        key = ('_revised_graph', id(source_partition), id(forget_edges))
        if key not in e._cache:
            e._cache[key] = source_partition.revise_graph_edges(forget_edges, remove=True)
        return e._cache[key]
    
    def get_unlearned_graph(self, predictor, removal_type, forget_part = 'forget'):
        
        toremove = predictor.dataset.partitions[forget_part]
        if removal_type == 'node':
            new_graph, remapped_partitions = graph.revise_graph_nodes(toremove, predictor.dataset.partitions, remove=True)
        if removal_type == 'edge':
            new_graph = predictor.dataset.partitions['all'].revise_graph_edges(toremove, remove=True)
            remapped_partitions = copy(predictor.dataset.partitions)

        graph, labels = new_graph[0][0], new_graph[0][1]
        remapped_partitions['forget'] = toremove

        return graph, labels, remapped_partitions