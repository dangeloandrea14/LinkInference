from erasure.core.measure import GraphMeasure
from erasure.evaluations.manager import Evaluation
import torch
import random
from tqdm import tqdm
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax

import numpy as np
from erasure.evaluations.LinkTeller.utils import construct_edge_sets, construct_edge_sets_from_random_subgraph, construct_edge_sets_through_bfs, construct_balanced_edge_sets


class LinkStealing0(GraphMeasure):
    """ LinkStealing attack, version 0
       https://arxiv.org/pdf/2308.01469
       https://github.com/xinleihe/link_stealing_attack/blob/master/stealing_link/attack_0.py
    """

    def init(self):
        super().init()
        self.target = self.params["target"]
        self.ratio = self.params["ratio"]
        self.train_part = self.params["train_part"]
        self.test_part = self.params["test_part"]
        self.forget_part = self.params["forget_part"]

        

    def check_configuration(self):
        self.params["ratio"] = self.params.get("ratio", 0.2)
        self.params["train_part"] = self.params.get("train_part", "train")
        self.params["test_part"] = self.params.get("test_part", "test")
        self.params["forget_part"] = self.params.get("forget_part", "forget")
        self.params["target"] = self.params.get("target","unlearned")



    def process(self, e: Evaluation):
        
        graph = e.unlearner.dataset.partitions['all'][0][0]

        self.features = graph.x
        self.edge_index = graph.edge_index
        self.n_features = len(graph.x[0])

        self.forget = e.unlearner.dataset.partitions[self.forget_part]
        self.train = e.unlearner.dataset.partitions[self.train_part]
        self.test = e.unlearner.dataset.partitions[self.test_part]

        self.model = self.get_model(e)

        self.model.model = self.model.model.to(self.model.device)
        self.features = self.features.to(self.model.device)
        self.edge_index = self.edge_index.to(self.model.device)

        exist_edges, non_existent_edges = get_link_from_edge_index(graph.edge_index, graph.num_nodes)

        exist_edges, non_existent_edges = random.sample(exist_edges, k=round(len(exist_edges) * self.ratio)), random.sample(non_existent_edges, k=round(len(non_existent_edges) * self.ratio))

        with torch.no_grad():
            pred_train = self.model.model(self.features ,self.edge_index)
            probs_train = softmax(pred_train, dim=1) 

        probs_train = probs_train.detach().cpu().numpy()

        aucs1 = compute_auc(probs_train, exist_edges, non_existent_edges)

        self.info(f"Link Stealing Attack on {self.target} 0 exist/non_exist: {aucs1}")
        e.add_value(f"Link Stealing Attack {self.target} 0 exist/non_exist", aucs1)

        aucs2 = compute_auc(probs_train, self.forget, non_existent_edges)

        self.info(f"Link Stealing Attack 0 on {self.target} forget/non_exist: {aucs2}")
        e.add_value(f"Link Stealing Attack 0 {self.target} forget/non_exist", aucs2)

        return e



def compute_auc(probs, edge_set_1, edge_set_2):

    edges_with_labels = (
        [(probs[u], probs[v], 1) for (u, v) in edge_set_1] +
        [(probs[u], probs[v], 0) for (u, v) in edge_set_2]
    )

    target_posterior_list = [(p_u, p_v) for (p_u,p_v,label) in edges_with_labels]
    label_list = [label for (p_u,p_v,label) in edges_with_labels]

    sim_list_target = attack_0(target_posterior_list)
    aucs = write_auc(sim_list_target, label_list, desc="target posterior similarity")

    return aucs




## Returns existent and non-existent edges. Needs adjacency matrix and number of nodes in the graph.
def get_link(adj, node_num):
    unlink = []
    link = []
    existing_set = set([])
    rows, cols = adj.nonzero()
    print("There are %d edges in this dataset" % len(rows))
    for i in range(len(rows)):
        r_index = rows[i]
        c_index = cols[i]
        if r_index < c_index:
            link.append([r_index, c_index])
            existing_set.add(",".join([str(r_index), str(c_index)]))

    random.seed(1)
    while len(unlink) < len(link):
        if len(unlink) % 1000 == 0:
            print(len(unlink))

        row = random.randint(0, node_num - 1)
        col = random.randint(0, node_num - 1)
        if row > col:
            row, col = col, row
        edge_str = ",".join([str(row), str(col)])
        if (row != col) and (edge_str not in existing_set):
            unlink.append([row, col])
            existing_set.add(edge_str)
    return link, unlink

def get_link_from_edge_index(edge_index, node_num):
    rows = edge_index[0].tolist()
    cols = edge_index[1].tolist()

    link = []
    existing_set = set()
    for r, c in zip(rows, cols):
        if r < c:
            link.append([r, c])
            existing_set.add(f"{r},{c}")
        elif c < r:
            existing_set.add(f"{c},{r}")

    unlink = []
    random.seed(1)
    while len(unlink) < len(link):
        if len(unlink) % 1000 == 0:
            print(len(unlink))
        u = random.randint(0, node_num - 1)
        v = random.randint(0, node_num - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        key = f"{u},{v}"
        if key not in existing_set:
            unlink.append([u, v])
            existing_set.add(key)
    return link, unlink


def attack_0(target_posterior_list):
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list_target = [[] for _ in range(len(sim_metric_list))]
    for i in range(len(target_posterior_list)):
        for j in range(len(sim_metric_list)):
            # using target only
            target_sim = sim_metric_list[j](target_posterior_list[i][0],
                                            target_posterior_list[i][1])
            sim_list_target[j].append(target_sim)
    return sim_list_target


def write_auc(pred_prob_list, label, desc):
    print("Attack 0 " + desc)
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    
    results = {}
    
    for i in range(len(sim_list_str)):
        pred = np.array(pred_prob_list[i], dtype=np.float64)
        where_are_nan = np.isnan(pred)
        where_are_inf = np.isinf(pred)
        pred[where_are_nan] = 0
        pred[where_are_inf] = 0

        i_auc = roc_auc_score(label, pred)
        if i_auc < 0.5:
            i_auc = 1 - i_auc

        results[sim_list_str[i]] =  i_auc

    return results['cosine']

