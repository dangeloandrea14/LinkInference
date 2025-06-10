from erasure.core.measure import GraphMeasure
from erasure.evaluations.manager import Evaluation
import torch
import random
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from erasure.evaluations.LinkTeller.utils import construct_edge_sets, construct_edge_sets_from_random_subgraph, construct_edge_sets_through_bfs, construct_balanced_edge_sets


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
        

    def check_configuration(self):
        self.params["influence"] = self.params.get("influence", 0.0001)
        self.params["approx"] = self.params.get("approx", True)
        self.params["edge_sampler"] = self.params.get("edge_sampler", "balanced")
        self.params["target"] = self.params.get("target","unlearned")
        self.params["forget_part"] = self.params.get("forget_part","test")



    def process(self, e: Evaluation):
        
        graph = e.unlearner.dataset.partitions['all'][0][0]
        self.features = graph.x
        self.edge_index = graph.edge_index
        self.n_features = len(graph.x[0])

        self.forget = e.unlearner.dataset.partitions[self.forget_part]

        self.model = e.unlearned_model if 'unlearn' in self.target else e.predictor

        self.model.model = self.model.model.to(self.model.device)
        self.features = self.features.to(self.model.device)
        self.edge_index = self.edge_index.to(self.model.device)


        self.exist_edges, self.nonexist_edges = self.get_edges(self.forget)
        norm_exist = []
        norm_nonexist = []

        with torch.no_grad():
            for u, v in tqdm(self.exist_edges):

                grad = self.get_gradient_eps(u, v) if self.approx else self.get_gradient(u, v)
                norm_exist.append(grad.norm().item())


            i = 0
            for u, v in tqdm(self.nonexist_edges):

                i += 1 

                grad = self.get_gradient_eps(u, v) if self.approx else self.get_gradient(u, v)
                norm_nonexist.append(grad.norm().item())


        y = [1] * len(norm_exist) + [0] * len(norm_nonexist)
        pred = norm_exist + norm_nonexist


        #y - pred evaluation

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        auc = metrics.auc(fpr, tpr)
        print('auc =', auc)

        ap = metrics.average_precision_score(y,pred)
        print('ap =', ap)

        lt = {
                "auc": auc,
                "ap":ap
            }

        self.info(f'LinkTeller : {lt}')
        e.add_value('LinkTeller:', lt)


        return e


    def link_prediction_attack_efficient(self):
        norm_exist = []
        norm_nonexist = []

        ## should it be only for test nodes and not all nodes?

        # 2. compute influence value for all pairs of nodes
        influence_val = np.zeros((self.args.n_test, self.args.n_test))

        with torch.no_grad():

            for i in tqdm(range(self.args.n_test)):
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

        return ret


    def get_edge_sampler(name):
        func_map = {
            'balanced': construct_edge_sets,
            'balanced-full': construct_balanced_edge_sets,
            'unbalanced': construct_edge_sets_from_random_subgraph,
            'unbalanced-lo': construct_edge_sets_from_random_subgraph,
            'unbalanced-hi': construct_edge_sets_from_random_subgraph,
            'bfs': construct_edge_sets_through_bfs,
        }
        return func_map.get(name)
    
    
    def get_edges(self, subset_nodes):
        subset_nodes = set(subset_nodes)
        edge_set = set()

        for u, v in self.edge_index.t().tolist():
            if u != v and u in subset_nodes and v in subset_nodes:
                edge_set.add(tuple(sorted((u, v))))

        existent_edges = list(edge_set)

        subset_list = sorted(subset_nodes)
        all_possible = set((i, j) for idx, i in enumerate(subset_list) for j in subset_list[idx+1:])

        non_existent_edges = list(all_possible - edge_set)

        non_existent_edges = random.sample(non_existent_edges, len(existent_edges))

        return existent_edges, non_existent_edges

    

    def get_gradient_eps_mat(self, v):
        pert_1 = torch.zeros_like(self.features)

        pert_1[v] = self.features[v] * self.influence

        grad = (self.model.model(self.features + pert_1, self.edge_index).detach() - 
                self.model.model(self.features, self.edge_index).detach()) / self.influence

        return grad