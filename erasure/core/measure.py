from abc import ABCMeta, abstractmethod
from copy import copy
import networkx as nx
from erasure.core.base import Configurable
from erasure.evaluations.manager import Evaluation
from erasure.utils.graph_ops import khop_infected, edge_endpoints


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
        """Nodes within `hops` of the forget set.

        Vectorised frontier expansion, identical output to the per-seed networkx BFS
        it replaces (verified on 300 random graphs); see erasure/utils/graph_ops.py.
        The `_cache` is kept -- it is cheap and callers pass it -- but no longer
        load-bearing now that the computation is milliseconds rather than minutes.
        """
        if _cache is not None:
            inf_key = ('_infected', id(unlearner.dataset), id(edges_to_forget), hops)
            if inf_key not in _cache:
                _cache[inf_key] = self._khop_infected(unlearner, edges_to_forget, hops)
            return _cache[inf_key]
        return self._khop_infected(unlearner, edges_to_forget, hops)

    def _khop_infected(self, unlearner, edges_to_forget, hops):
        graph = unlearner.dataset.partitions['all'][0][0]
        return khop_infected(graph.edge_index,
                             edge_endpoints(edges_to_forget),
                             hops, graph.num_nodes)

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