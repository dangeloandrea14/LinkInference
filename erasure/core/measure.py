from abc import ABCMeta, abstractmethod
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

    def infected_nodes(self, unlearner, edges_to_forget, hops):

        G = nx.Graph()
        all_edges = unlearner.dataset.partitions['all'][0][0].edge_index.t().tolist()  

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
    
    def get_unlearned_graph(self, predictor, removal_type, forget_part = 'forget'):
        
        toremove = predictor.dataset.partitions[forget_part]
        if removal_type == 'node':
            new_graph, remapped_partitions = graph.revise_graph_nodes(toremove, predictor.dataset.partitions, remove=True)
        if removal_type == 'edge':
            new_graph = predictor.dataset.partitions['all'].revise_graph_edges(toremove, remove=True)
            remapped_partitions = predictor.dataset.partitions

        graph, labels = new_graph[0][0], new_graph[0][1]
        remapped_partitions['forget'] = toremove

        return graph, labels, remapped_partitions