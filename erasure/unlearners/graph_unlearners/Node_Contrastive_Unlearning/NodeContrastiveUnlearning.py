from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction
import torch.optim as optim
from erasure.utils.config.local_ctx import Local
from copy import deepcopy
import torch
from torch_geometric.loader import NeighborLoader
from types import SimpleNamespace

import copy
import logging
from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F


class NodeContrastiveUnlearning(GraphUnlearner):
    def init(self):
        """
        Initializes the NodeContrastiveUnlearning class with global and local contexts.
        """

        super().init()

        self.unlearn_method = self.local.config['parameters']['unlearn_method']
        self.split = self.local.config['parameters']['split']
        self.num_unlearn = self.local.config['parameters']['num_unlearn']
        self.repeat = self.local.config['parameters']['repeat']

   
    def __unlearn__(self):
        """
        An implementation of the Advanced NegGrad unlearning algorithm proposed in the following paper:
        "Node-level Contrastive Unlearning on Graph Neural Networks"
        
        Codebase taken from the original implementation: https://anonymous.4open.science/r/Node-CUL-E30D/readme.md
        """

        self.info(f'Starting NodeContrastiveUnlearning')      

        graph = self.dataset.partitions['all'] 

        self.forget_set = self.dataset.partitions[self.forget_part]
        self.retain_set = self.dataset.partitions[self.retain_part]
        self.train_set = self.dataset.partitions[self.train_part]
        self.test_set = self.dataset.partitions["test"]

        if self.removal_type == 'node':
            ul_graph, remapped_partitions = graph.revise_graph_nodes(self.forget_set, self.dataset.partitions, remove=True)
        if self.removal_type == 'edge':
            self.global_ctx.logger.info("NodeContrastiveUnlearning does not work with edge-level Unlearning. Converting to nodes.")
            self.forget_set = self.infected_nodes(self.forget_set, self.hops)
            self.retain_set = self.infected_nodes(self.retain_set, self.hops)
            self.retain_set = [n for n in self.retain_set if n not in self.forget_set]

            ul_graph, remapped_partitions = graph.revise_graph_nodes(self.forget_set, self.dataset.partitions, remove=True)

        self.ul_retain_set = remapped_partitions[self.retain_part]
        self.ul_train_set = remapped_partitions[self.train_part]
        self.test_set = remapped_partitions["test"]

        graph = graph[0][0]
        ul_graph = ul_graph[0][0]


        args = SimpleNamespace(
            device=self.predictor.device,
            batch_size=self.dataset.batch_size,
            unlearn_method=self.unlearn_method,   
            depth=2,
            epochs=30,
            beta=8.0,
            repeat=self.repeat,
            use_ppr=False,                        
            temperature=0.7,
            lr=getattr(self.predictor, "lr", 1e-3),
            optim = self.predictor.optimizer
        )

        if not isinstance(args.optim, torch.optim.Optimizer):
            args.optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        model = self.predictor.model

        ####START TRAINER

        trainer = torch_UnlearnTrainer(
            args=args,
            model=model,
            graph=graph,
            forget_set=self.forget_set,
            retain_set=self.retain_set,
            labels=self.labels
        )

        for epoch in range(args.epochs):
            stats = trainer.fit(epoch)
            self.info(f"Epoch {epoch}: {stats}")


        self.predictor.model = model
        return self.predictor



    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['unlearn_method'] = self.local.config['parameters'].get('unlearn_method','contrastive')
        self.local.config['parameters']['split'] = self.local.config['parameters'].get('split','transductive')
        self.local.config['parameters']['repeat'] = self.local.config['parameters'].get('repeat',1)
        self.local.config['parameters']['num_unlearn'] = self.local.config['parameters'].get('num_unlearn', 0.1)


 

class torch_UnlearnTrainer:
    def __init__(self, args, model, graph, forget_set, retain_set, labels):
        self.args = args
        self.model = model.to(args.device)
        self.graph = graph.to(args.device)
        self.forget_set = torch.tensor(forget_set, device=args.device, dtype=torch.long)
        self.retain_set = torch.tensor(retain_set, device=args.device, dtype=torch.long)
        self.labels = labels

        self.optim = args.optim
        self.device = args.device
        self.beta = args.beta
        self.repeat = args.repeat
        self.temperature = args.temperature
        self.base_temperature = args.temperature
        self.use_ppr = args.use_ppr

    def fit(self, epoch: int):
        self.model.train()
        epoch_stats = {}

        # ---- Build UL seed view and 1-hop neighbor view ----
        ul_seed_idx = self.forget_set                     # anchors
        rt_seed_idx = self.retain_set                     # positives
        nb1_nodes = self._neighbors_of_seeds(ul_seed_idx) # negatives

        if nb1_nodes.numel() == 0:
            return {}

        ppr_vec = torch.ones(ul_seed_idx.size(0), device=self.device)

        for _repeat in range(self.repeat):
            # Forward on whole graph
            logits, feats = self._forward_pyg(self.graph)

            ul_feat = feats[ul_seed_idx]
            ul_label = self.labels[ul_seed_idx]

            rt_feat = feats[rt_seed_idx]
            rt_label = self.labels[rt_seed_idx]

            nb1_feat = feats[nb1_nodes]
            nb1_label = self.labels[nb1_nodes]

            # Loss
            self.optim.zero_grad()
            loss, loss_val = self.CT_Loss(
                ul_feat=ul_feat,
                rt_feat=rt_feat,
                nb1_feat=nb1_feat,
                ul_pred=logits[ul_seed_idx],
                rt_pred=logits[rt_seed_idx],
                ul_label=ul_label,
                rt_label=rt_label,
                nb1_label=nb1_label,
                ppr_vec=ppr_vec
            )
            loss.backward()
            self.optim.step()

            for k, v in loss_val.items():
                epoch_stats.setdefault(k, []).append(v)

        # Aggregate
        for k in epoch_stats:
            epoch_stats[k] = sum(epoch_stats[k]) / len(epoch_stats[k])
        return epoch_stats

    # ---------- helpers ----------
    def _forward_pyg(self, graph):
        out = self.model(graph.x, graph.edge_index)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, out

    def _neighbors_of_seeds(self, seed_idx):
        """1-hop neighbors in full graph.edge_index"""
        row, col = self.graph.edge_index
        seed_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        seed_mask[seed_idx] = True
        seed_edges = seed_mask[row] | seed_mask[col]
        src, dst = row[seed_edges], col[seed_edges]
        other = torch.where(seed_mask[src], dst, src)
        return other[~seed_mask[other]].unique()

    def CT_Loss(self,
                ul_feat,
                rt_feat,
                nb1_feat,
                ul_pred,
                rt_pred,
                ul_label,
                rt_label,
                nb1_label,
                ppr_vec):
        
        loss_track = dict()

        ct_loss = self._CT_Loss(ul_feat, rt_feat, nb1_feat, ul_label, rt_label, nb1_label, ppr_vec)
        ce_loss = self._CE_Loss(rt_pred, rt_label)
        loss_track["ulct_loss"] = ct_loss.item()
        loss_track["ulce_loss"] = ce_loss.item()

        return  ct_loss + self.beta * ce_loss, loss_track
    
    def _CT_Loss(self, ul_feat, rt_feat, nb1_feat, ul_label, rt_label, nb1_label, ppr_vec):
        """
        Contrastive Loss

        Anchor: unlearning nodes (ul_pred)
        Positive: (Ones who attract) retain nodes with different labels
        Negative: (Ones who repulse)neighbors
        
        """

        ul_label = ul_label.contiguous().view(-1, 1)
        rt_label  = rt_label.contiguous().view(-1, 1)
        nb1_label = nb1_label.contiguous().view(-1, 1)

        mask = torch.eq(ul_label, rt_label.T)
        n_mask = (mask).clone().float()
        p_mask = (~mask).clone().float()

        p_logits = torch.matmul(ul_feat, rt_feat.T)
        # print("p_logits_init", p_logits)
        p_logits = torch.div(p_logits, self.temperature)
        p_logits_max, _ =  torch.max(p_logits, dim=1, keepdim=True)
        p_logits -= p_logits_max.detach()

        nb_mask = torch.eq(ul_label, nb1_label.T)
        n_logits = torch.matmul(ul_feat, nb1_feat.T)
        n_logits = torch.div(n_logits, self.temperature)
        n_logits_max, _ = torch.max(n_logits, dim=1, keepdim=True)
        n_logits -= n_logits_max.detach()
        n_logits = torch.exp(n_logits) * nb_mask

        # Add same classes as n_logits here
        n_logits_2 = torch.exp(p_logits) * n_mask
        p_logits = p_logits * p_mask

        # print("n_logits_2", n_logits_2)
        # print("p_logits", p_logits)

        # n_logits = n_logits.sum(1, keepdim=True) + n_logits_2.sum(1, keepdim=True)
        # n_logits = n_logits_2.sum(1, keepdim=True)
        n_logits = n_logits.sum(1, keepdim=True)

        # Apply PPR here
        # print(n_logits.shape, ppr_vec.unsqueeze(1).shape)
        n_logits = n_logits * ppr_vec.unsqueeze(1) + 1e-20
        # print("n_logits", n_logits)

        # print("n_logits after ppr", n_logits)

        log_prob = p_logits - torch.log(n_logits)
        # print("log_prob", log_prob)
        p_mask_sum = p_mask.sum(1)
        p_mask_sum_mask = p_mask_sum < 1.0
        if sum(p_mask_sum_mask > 0):
            mean_log_prob = log_prob.sum(1)[~p_mask_sum_mask] / p_mask.sum(1)[~p_mask_sum_mask]
            print(mean_log_prob)
        else:
            mean_log_prob = log_prob.sum(1) / p_mask.sum(1)
        # print("P-mask-sum", p_mask.sum(1))
        # print("p_mask.sum(1)", p_mask.sum(1))
        # print("mean_log_prob", mean_log_prob)
        # print("temperature", self.temperature, self.temperature/self.base_temperature)

        loss = -(self.temperature/self.base_temperature) * mean_log_prob
        loss = loss.mean()
        # print("ct_loss", loss.item())

        return loss

    def _CE_Loss(self, pred, label):
        loss = F.cross_entropy(pred, label)
        return loss
