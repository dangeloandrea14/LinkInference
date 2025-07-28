from erasure.core.measure import GraphMeasure
from erasure.evaluations.manager import Evaluation
import torch
import random
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from erasure.evaluations.LinkTeller.utils import construct_edge_sets, construct_edge_sets_from_random_subgraph, construct_edge_sets_through_bfs, construct_balanced_edge_sets


class LinkStealing3(GraphMeasure):
    """ LinkStealing attack, version 3
       https://arxiv.org/pdf/2308.01469
       https://github.com/xinleihe/link_stealing_attack/blob/master/stealing_link/attack_3.py
    """

    def init(self):
        super().init()
        self.influence = self.params["influence"]

        

    def check_configuration(self):
        self.params["influence"] = self.params.get("influence", 0.0001)



    def process(self, e: Evaluation):
        
        graph = e.unlearner.dataset.partitions['all'][0][0]
        self.features = graph.x
        self.edge_index = graph.edge_index
        self.n_features = len(graph.x[0])

        self.forget = e.unlearner.dataset.partitions[self.forget_part]

        self.model = self.get_model(e)

        self.model.model = self.model.model.to(self.model.device)
        self.features = self.features.to(self.model.device)
        self.edge_index = self.edge_index.to(self.model.device)


        