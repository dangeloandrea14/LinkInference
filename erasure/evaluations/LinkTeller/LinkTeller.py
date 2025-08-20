from erasure.core.measure import GraphMeasure
from erasure.evaluations.manager import Evaluation
import torch
import random
import networkx as nx
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from sklearn import metrics
import numpy as np
from erasure.evaluations.LinkTeller.utils import construct_edge_sets, construct_edge_sets_from_random_subgraph, construct_balanced_edge_sets


class LinkTeller(GraphMeasure):
    """ LINKTELLER
       https://www.computer.org/csdl/proceedings-article/sp/2022/131600a522/1FlQypPVMis
    """

    def init(self):
        super().init()
        self.influence = self.params["influence"]
        self.approx = self.params["approx"]
        self.edge_sampler = self.params["edge_sampler"]
        self.target = self.params["target"]
        self.forget_part = self.params["forget_part"]
        self.retain_part = self.params["retain_part"]
        self.removal_type = self.global_ctx.removal_type
        

    def check_configuration(self):
        self.params["influence"] = self.params.get("influence", 0.0001)
        self.params["approx"] = self.params.get("approx", True)
        self.params["edge_sampler"] = self.params.get("edge_sampler", "balanced")
        self.params["target"] = self.params.get("target","unlearned")
        self.params["forget_part"] = self.params.get("forget_part","forget")
        self.params["retain_part"] = self.params.get("retain_part","retain")



    def process(self, e: Evaluation):

        unlearned_graph, labels, remapped_partitions = self.get_unlearned_graph(e.predictor, self.removal_type)
        
        og_graph = e.predictor.dataset.partitions['all'][0][0]

        self.features = unlearned_graph.x
        self.edge_index = unlearned_graph.edge_index
        self.n_features = len(unlearned_graph.x[0])

        self.forget = e.unlearner.dataset.partitions[self.forget_part]
        self.retain = e.unlearner.dataset.partitions[self.retain_part]

        self.model = e.unlearned_model if 'unlearn' in self.target else e.predictor

        self.model.model = self.model.model.to(self.model.device)
        self.model.model.eval()
        self.features = self.features.to(self.model.device)
        self.edge_index = self.edge_index.to(self.model.device)

        sampler = self.get_edge_sampler(self.edge_sampler)

        self.forget_edges, self.nonexist_edges = sampler(og_graph, self.forget)

        norm_exist = []
        norm_nonexist = []

        with torch.no_grad():
            for u, v in self.forget_edges:

                grad = self.get_gradient_eps(u, v) if self.approx else self.get_gradient(u, v)
                norm_exist.append(grad.norm().item())


            i = 0
            for u, v in self.nonexist_edges:

                i += 1 

                grad = self.get_gradient_eps(u, v) if self.approx else self.get_gradient(u, v)
                norm_nonexist.append(grad.norm().item())


        y = [1] * len(norm_exist) + [0] * len(norm_nonexist)
        pred = norm_exist + norm_nonexist

        print(f"[Norm Stats] Exist Edges: mean={np.mean(norm_exist):.4f}, std={np.std(norm_exist):.4f}, min={np.min(norm_exist):.4f}, max={np.max(norm_exist):.4f}")
        print(f"[Norm Stats] Non-Exist Edges: mean={np.mean(norm_nonexist):.4f}, std={np.std(norm_nonexist):.4f}, min={np.min(norm_nonexist):.4f}, max={np.max(norm_nonexist):.4f}")


        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.hist(norm_exist, bins=50, alpha=0.5, label='Exist Edges')
        plt.hist(norm_nonexist, bins=50, alpha=0.5, label='Non-Exist Edges')
        plt.xlabel("Gradient Norm")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Distribution of Gradient Norms")
        plt.savefig('test.png'
                    )
        #y - pred evaluation

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        print(f"[ROC Thresholds] Sample thresholds: {thresholds[:5]}")
        print(f"[ROC] FPR: {fpr[:5]}, TPR: {tpr[:5]}")
        auc = metrics.auc(fpr, tpr)
        print('auc =', auc)

        ap = metrics.average_precision_score(y,pred)
        print('ap =', ap)

        lt = {
                "auc": auc,
                "ap":ap
            }

        self.info(f'LinkTeller {self.target}: {lt}')
        e.add_value(f'LinkTeller {self.target}:', lt)


        return e


    def link_prediction_attack_efficient(self):
        norm_exist = []
        norm_nonexist = []

        ## should it be only for test nodes and not all nodes?

        # 2. compute influence value for all pairs of nodes
        influence_val = np.zeros((self.args.n_test, self.args.n_test))

        with torch.no_grad():

            for i in range(self.args.n_test):
                u = self.test_nodes[i]
                grad_mat = self.get_gradient_eps_mat(u)

                for j in range(self.args.n_test):
                    v = self.test_nodes[j]

                    grad_vec = grad_mat[v]

                    influence_val[i][j] = grad_vec.norm().item()

        node2ind = { node : i for i, node in enumerate(self.test_nodes) }

        for u, v in self.exist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_exist.append(influence_val[j][i])

        for u, v in self.nonexist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_nonexist.append(influence_val[j][i])

        self.compute_and_save(norm_exist, norm_nonexist)



    def get_gradient_eps(self, u, v):
        pert_1 = torch.zeros_like(self.features)

        pert_1[v] = self.features[v] * self.influence

        grad = (self.model.model(self.features + pert_1, self.edge_index).detach() - 
                self.model.model(self.features, self.edge_index).detach()) / self.influence
        

        return grad[u]
    
    def get_gradient(self, u, v):
        h = 0.0001
        ret = torch.zeros(self.n_features)
        for i in range(self.n_features):
            pert = torch.zeros_like(self.features)
            pert[v][i] = h
            with torch.no_grad():
                grad = (self.model.model(self.features + pert, self.edge_index).detach() -
                        self.model.model(self.features - pert, self.edge_index).detach()) / (2 * h)
                ret[i] = grad[u].sum()

        print(f"[get_gradient] Sum of finite diff grad vector: {ret.norm().item():.4f}")


        return ret


    def get_edge_sampler(self,name):
        func_map = {
            'balanced': self.get_edges,
            'balanced-full': construct_balanced_edge_sets,
            'unbalanced': construct_edge_sets_from_random_subgraph,
            'unbalanced-lo': construct_edge_sets_from_random_subgraph,
            'unbalanced-hi': construct_edge_sets_from_random_subgraph,
            'bfs': self.construct_edge_sets_through_bfs,
            'bfs+': self.construct_edge_sets_through_bfs_plus
        }
        return func_map.get(name)
    
    
    def get_edges(self, graph, forget_set):


        forget_edges = list(forget_set)

        existing_edges = set(map(tuple, map(sorted, graph.edge_index.t().tolist())))

        nodes = list(range(graph.num_nodes))
        all_pairs = {(u, v) for u in nodes for v in nodes if u < v}

        non_edges = list(all_pairs - existing_edges)

        non_edges = random.sample(non_edges, len(forget_edges))


        print(f"[Edge Sampling] #Exist: {len(forget_edges)} | #Non-Exist: {len(non_edges)}")
        print(f"[Sample Exist] {forget_edges[:5]}")
        print(f"[Sample Non-Exist] {non_edges[:5]}")


        return forget_edges, non_edges

    

    def get_gradient_eps_mat(self, v):
        pert_1 = torch.zeros_like(self.features)

        pert_1[v] = self.features[v] * self.influence

        grad = (self.model.model(self.features + pert_1, self.edge_index).detach() - 
                self.model.model(self.features, self.edge_index).detach()) / self.influence

        return grad
    
    def construct_edge_sets_through_bfs(self, graph, subset_nodes, max_hops=2):
        print("am here")
        """
        Generate (pos, neg) edge pairs from subset_nodes using BFS.

        Positive: existing edges within subset.
        Negative: node pairs within max_hops in BFS but not connected.

        Returns:
            - existent_edges: list of (u, v)
            - non_existent_edges: list of (u, v)
        """
        G = to_networkx(graph, to_undirected=True)
        subset_nodes = list(set(subset_nodes))

        edge_set = set()
        existent_edges = set()
        
        for u, v in G.edges():
            if u != v and u in subset_nodes and v in subset_nodes:
                edge = tuple(sorted((u, v)))
                edge_set.add(edge)
                existent_edges.add(edge)

        # Prepare candidate negatives: node pairs within max_hops that are not in edge_set
        negative_candidates = set()
        for u in subset_nodes:
            neighbors = nx.single_source_shortest_path_length(G, u, cutoff=max_hops)
            for v in neighbors:
                if u < v and v in subset_nodes and (u, v) not in edge_set:
                    negative_candidates.add((u, v))

        # Sample negatives to match the number of positives
        existent_edges = list(existent_edges)
        negative_candidates = list(negative_candidates)
        random.seed(42)

        non_existent_edges = random.sample(
            negative_candidates, min(len(existent_edges), len(negative_candidates))
        )

        return existent_edges, non_existent_edges
    


    def construct_edge_sets_through_bfs_plus(self,graph, subset_nodes, max_hops=3, min_shared_neighbors=2, max_degree_diff=2):
        """
        Generate (pos, neg) edge pairs from subset_nodes using BFS traversal,
        and filter negative candidates by structural similarity.

        Positive: existing edges within subset.
        Negative: node pairs within max_hops, not connected, but with:
            - degree difference ≤ max_degree_diff
            - at least min_shared_neighbors neighbors in common

        Returns:
            - existent_edges: list of (u, v)
            - non_existent_edges: list of (u, v)
        """
        G = to_networkx(graph, to_undirected=True)
        subset_nodes = list(set(subset_nodes))

        # collect actual existing edges in the subset
        existent_edges = set()
        for u, v in G.edges():
            if u != v and u in subset_nodes and v in subset_nodes:
                edge = tuple(sorted((u, v)))
                existent_edges.add(edge)

        # prepare hard negatives
        negative_candidates = set()
        for u in subset_nodes:
            # BFS up to max_hops
            neighbors = nx.single_source_shortest_path_length(G, u, cutoff=max_hops)
            for v in neighbors:
                if u < v and v in subset_nodes:
                    edge = (u, v)
                    if edge in existent_edges:
                        continue  # skip real edges

                    # Filter by structural similarity
                    degree_diff = abs(G.degree[u] - G.degree[v])
                    shared_nbrs = set(G.neighbors(u)).intersection(G.neighbors(v))

                    if degree_diff <= max_degree_diff and len(shared_nbrs) >= min_shared_neighbors:
                        negative_candidates.add(edge)

        # Balance number of negatives and positives
        existent_edges = list(existent_edges)
        negative_candidates = list(negative_candidates)
        random.seed(42)
        non_existent_edges = random.sample(
            negative_candidates, min(len(existent_edges), len(negative_candidates))
        )

        return existent_edges, non_existent_edges