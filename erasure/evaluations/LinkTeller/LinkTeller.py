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
        self.k_hat = self.params["k_hat"]
        

    def check_configuration(self):
        self.params["influence"] = self.params.get("influence", 0.0001)
        self.params["approx"] = self.params.get("approx", True)
        self.params["edge_sampler"] = self.params.get("edge_sampler", "bfs+")
        self.params["target"] = self.params.get("target","unlearned")
        self.params["forget_part"] = self.params.get("forget_part","forget")
        self.params["retain_part"] = self.params.get("retain_part","retain")
        self.params["k_hat"] = self.params.get("k_hat", None)



    def process(self, e: Evaluation):

        try: 
            self.max_hops = e.unlearner.hops
        except:
            self.max_hops = len(e.predictor.model.hidden_channels) + 1

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

        self.forget_edges, self.nonexist_edges = sampler(og_graph, self.forget, max_hops = self.max_hops)

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

        if self.k_hat is not None:
            n = len(set([u for u,_ in self.forget_edges] + [v for _,v in self.forget_edges]))
            m = int(self.k_hat * n * (n - 1) / 2)
            scores = np.array(pred)
            order = np.argsort(-scores)
            y_hat = np.zeros_like(scores, dtype=int)
            y_hat[order[:m]] = 1
            prec = metrics.precision_score(y, y_hat, zero_division=0)
            rec  = metrics.recall_score(y, y_hat, zero_division=0)

            print(f"[LinkTeller@k̂] precision={prec:.4f}, recall={rec:.4f}")


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

        self.info(f'LinkTeller {self.target} with sampler {self.edge_sampler}: {lt}')
        e.add_value(f'LinkTeller {self.target} with sampler {self.edge_sampler}:', lt)


        return e


    
    def get_gradient(self, u, v):
        h = 1e-4
        base = self.features
        pert_plus = torch.zeros_like(base); pert_plus[v] = base[v] * h
        pert_minus = torch.zeros_like(base); pert_minus[v] = -base[v] * h

        with torch.no_grad():
            out_plus  = self.model.model(base + pert_plus,  self.edge_index).detach()
            out_minus = self.model.model(base + pert_minus, self.edge_index).detach()

        grad_u = (out_plus[u] - out_minus[u]) / (2 * h)  
        return grad_u


    

    def get_gradient_eps_mat(self, v):
        pert_1 = torch.zeros_like(self.features)

        pert_1[v] = self.features[v] * self.influence

        grad = (self.model.model(self.features + pert_1, self.edge_index).detach() - 
                self.model.model(self.features, self.edge_index).detach()) / self.influence

        return grad
    
    def get_gradient_eps(self, u, v):
        pert = torch.zeros_like(self.features)
        pert[v] = self.features[v] * self.influence
        with torch.no_grad():
            out_plus  = self.model.model(self.features + pert, self.edge_index).detach()
            out_base  = self.model.model(self.features,          self.edge_index).detach()
        return (out_plus[u] - out_base[u]) / self.influence
        

    def get_edge_sampler(self,name):
        func_map = {
            'balanced': self.get_edges,
            'bfs+': self.construct_edge_sets_through_bfs_plus
        }
        return func_map.get(name)
    
    
    def get_edges(self, graph, forget_set, max_hops=0):


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


    def construct_edge_sets_through_bfs_plus(self, graph, forget_set, max_hops=3, min_shared_neighbors=2, max_degree_diff=2):
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
        

        # collect actual existing edges in the subset
        forget_edges = [tuple(sorted(e)) for e in forget_set]
        forget_edge_set = set(forget_edges)

        subset_nodes = {n for e in forget_edges for n in e}

        negative_candidates = set()
        for u in subset_nodes:
            neighbors = nx.single_source_shortest_path_length(G, u, cutoff=max_hops)
            for v in neighbors.keys():
                if u == v: 
                    continue
                if v not in subset_nodes:
                    continue
                edge = tuple(sorted((u, v)))
                if edge in forget_edge_set or G.has_edge(*edge):
                    continue
                degree_diff = abs(G.degree[u] - G.degree[v])
                shared_nbrs = set(G.neighbors(u)).intersection(G.neighbors(v))
                if degree_diff <= max_degree_diff and len(shared_nbrs) >= min_shared_neighbors:
                    negative_candidates.add(edge)

        negative_candidates = list(negative_candidates)
        random.seed(42)
        non_existent_edges = random.sample(
            negative_candidates, min(len(forget_edges), len(negative_candidates))
        )

        return forget_edges, non_existent_edges