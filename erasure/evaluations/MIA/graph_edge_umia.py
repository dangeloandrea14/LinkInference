import copy
import os
import random

import sklearn
import sklearn.linear_model
import sklearn.model_selection
import torch

from erasure.core.factory_base import get_instance_config
from erasure.core.measure import GraphMeasure
from erasure.evaluations.evaluation import Evaluation
from erasure.evaluations.utils import compute_accuracy
from erasure.utils.config.local_ctx import Local


class Attack(GraphMeasure):
    """ Edge-centric U-MIA for node classification on graphs.

    Positives are forgotten edges, negatives are retained edges sampled
    from the same graph (optionally label-matched). Each edge is turned
    into features by toggling it in the unlearned graph and measuring
    endpoint-level changes in loss/margin.
    """

    def init(self):
        self.attack_in_data_cfg = self.params["attack_in_data"]
        self.attack_model_cfg = self.params["attack_model"]

        self.local_config["parameters"]["attack_in_data"]["parameters"]["DataSource"]["parameters"]["path"] += "_" + str(
            self.global_ctx.config.globals["seed"]
        )
        self.data_out_path = self.local_config["parameters"]["attack_in_data"]["parameters"]["DataSource"]["parameters"]["path"]

        self.forget_part = self.params["forget_part"]
        self.metric_name = self.params["metric_name"]
        self.match_by_label = self.params["match_by_label"]
        self.treat_as_undirected = self.params["treat_as_undirected"]
        self.n_splits = self.params["n_splits"]
        self.test_size = self.params["test_size"]
        self.max_edges = self.params["max_edges"]
        self.random_state = self.params["random_state"]

        self.params["loss_fn"]["parameters"]["reduction"] = "none"
        self.loss_fn = get_instance_config(self.params["loss_fn"])

    def check_configuration(self):
        super().check_configuration()

        self.params["forget_part"] = self.params.get("forget_part", "forget")
        self.params["metric_name"] = self.params.get("metric_name", "UMIA_edge")
        self.params["match_by_label"] = self.params.get("match_by_label", True)
        self.params["treat_as_undirected"] = self.params.get("treat_as_undirected", True)
        self.params["n_splits"] = int(self.params.get("n_splits", 3))
        self.params["test_size"] = float(self.params.get("test_size", 0.3))
        self.params["max_edges"] = self.params.get("max_edges", 200)
        self.params["random_state"] = int(self.params.get("random_state", self.global_ctx.config.globals["seed"]))

        if "attack_model" not in self.params:
            self.params["attack_model"] = None

        if "loss_fn" not in self.params:
            self.params["loss_fn"] = copy.deepcopy(self.global_ctx.config.predictor["parameters"]["loss_fn"])

    def process(self, e: Evaluation):
        target_model = e.unlearned_model
        target_model.model.eval()

        self.info("Creating edge attack dataset")
        attack_dataset = self.create_attack_dataset(target_model)

        if len(attack_dataset.partitions["all"]) == 0:
            self.global_ctx.logger.warning("Edge U-MIA not calculated: empty attack dataset")
            e.add_value(self.metric_name, -1.0)
            return e

        self.info("Creating edge attack model")
        if self.attack_model_cfg:
            current = Local(self.attack_model_cfg)
            current.dataset = attack_dataset
            attack_model = self.global_ctx.factory.get_object(current)
            test_loader, _ = attack_dataset.get_loader_for("test")
            umia_accuracy = compute_accuracy(test_loader, attack_model.model)
        else:
            attack_loader, _ = attack_dataset.get_loader_for("all")
            X, y = attack_loader.dataset[:]
            X_np = X.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            attack_model = sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=self.random_state)
            cv = sklearn.model_selection.StratifiedShuffleSplit(
                n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state
            )
            try:
                accuracies = sklearn.model_selection.cross_val_score(
                    attack_model, X_np, y_np, cv=cv, scoring="accuracy"
                )
                umia_accuracy = float(accuracies.mean())
            except ValueError as err:
                self.global_ctx.logger.warning(repr(err))
                self.global_ctx.logger.warning("Edge U-MIA not calculated")
                umia_accuracy = -1.0

        self.info(f"{self.metric_name}: {umia_accuracy}")
        e.add_value(self.metric_name, umia_accuracy)
        return e

    def create_attack_dataset(self, target_model):
        samples, labels = self.get_attack_samples(target_model)

        if len(samples) == 0:
            attack_samples = torch.empty((0, 8), dtype=torch.float32)
            attack_labels = torch.empty((0,), dtype=torch.long)
        else:
            attack_samples = torch.stack(samples)
            attack_labels = torch.tensor(labels, dtype=torch.long)

            perm_idxs = torch.randperm(len(attack_samples))
            attack_samples = attack_samples[perm_idxs]
            attack_labels = attack_labels[perm_idxs]

        attack_dataset = torch.utils.data.TensorDataset(attack_samples, attack_labels)
        attack_dataset.n_classes = 2

        os.makedirs(os.path.dirname(self.data_out_path), exist_ok=True)
        torch.save(attack_dataset, self.data_out_path)
        return self.global_ctx.factory.get_object(Local(self.attack_in_data_cfg))

    def get_attack_samples(self, model):
        graph, labels = model.dataset.partitions["all"][0][0], model.dataset.partitions["all"][0][1]
        x = graph.x.to(model.device)
        labels = labels.to(model.device)

        forget_edges = self._canonicalize_edges(model.dataset.partitions[self.forget_part])
        if len(forget_edges) == 0:
            return [], []

        edge_index = graph.edge_index.to(model.device)
        original_edge_set = self._edge_set_from_edge_index(edge_index)
        forget_edges = [e for e in forget_edges if e in original_edge_set]
        if len(forget_edges) == 0:
            return [], []

        baseline_edge_index = self._remove_edges(edge_index, set(forget_edges))
        baseline_edge_set = self._edge_set_from_edge_index(baseline_edge_index)

        if self.max_edges is not None and len(forget_edges) > self.max_edges:
            rng = random.Random(self.random_state)
            forget_edges = rng.sample(forget_edges, self.max_edges)

        neg_edges = self._sample_negative_edges(
            baseline_edge_set=baseline_edge_set,
            positive_edges=forget_edges,
            labels=labels.detach().cpu(),
        )

        n_pairs = min(len(forget_edges), len(neg_edges))
        if n_pairs == 0:
            return [], []
        forget_edges = forget_edges[:n_pairs]
        neg_edges = neg_edges[:n_pairs]

        with torch.no_grad():
            base_logits = self._forward_logits(model, x, baseline_edge_index)
            base_losses = self._per_sample_loss(base_logits, labels)
            base_margin = self._margin(base_logits)
            degrees = self._node_degrees(baseline_edge_index, x.size(0))

        samples = []
        attack_labels = []

        for edge in forget_edges:
            features = self._edge_features(
                model, x, labels, baseline_edge_index, edge, add_edge=True,
                base_losses=base_losses, base_margin=base_margin, degrees=degrees
            )
            if features is not None:
                samples.append(features)
                attack_labels.append(1)

        for edge in neg_edges:
            features = self._edge_features(
                model, x, labels, baseline_edge_index, edge, add_edge=False,
                base_losses=base_losses, base_margin=base_margin, degrees=degrees
            )
            if features is not None:
                samples.append(features)
                attack_labels.append(0)

        return samples, attack_labels

    def _edge_features(self, model, x, labels, baseline_edge_index, edge, add_edge, base_losses, base_margin, degrees):
        u, v = edge
        num_nodes = x.size(0)
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            return None

        pert_edge_index = self._toggle_edge(baseline_edge_index, edge, add=add_edge)
        with torch.no_grad():
            pert_logits = self._forward_logits(model, x, pert_edge_index)
            pert_losses = self._per_sample_loss(pert_logits, labels)
            pert_margin = self._margin(pert_logits)

        dl_u = float((pert_losses[u] - base_losses[u]).item())
        dl_v = float((pert_losses[v] - base_losses[v]).item())
        dm_u = float((pert_margin[u] - base_margin[u]).item())
        dm_v = float((pert_margin[v] - base_margin[v]).item())

        deg_u = float(torch.log1p(degrees[u]).item())
        deg_v = float(torch.log1p(degrees[v]).item())
        same_label = float((labels[u] == labels[v]).item())
        mean_abs_dl = 0.5 * (abs(dl_u) + abs(dl_v))

        return torch.tensor(
            [dl_u, dl_v, dm_u, dm_v, deg_u, deg_v, same_label, mean_abs_dl],
            dtype=torch.float32,
        )

    def _sample_negative_edges(self, baseline_edge_set, positive_edges, labels):
        rng = random.Random(self.random_state)
        candidates = [e for e in baseline_edge_set if e not in set(positive_edges)]
        if len(candidates) == 0:
            return []

        target_n = min(len(positive_edges), len(candidates))
        if not self.match_by_label:
            rng.shuffle(candidates)
            return candidates[:target_n]

        def edge_type(edge):
            a, b = int(labels[edge[0]].item()), int(labels[edge[1]].item())
            return (min(a, b), max(a, b))

        pools = {}
        for edge in candidates:
            pools.setdefault(edge_type(edge), []).append(edge)
        for edges in pools.values():
            rng.shuffle(edges)

        selected = []
        used = set()
        for edge in positive_edges:
            key = edge_type(edge)
            pool = pools.get(key, [])
            while pool and pool[-1] in used:
                pool.pop()
            if pool:
                chosen = pool.pop()
                selected.append(chosen)
                used.add(chosen)

        if len(selected) < target_n:
            remaining = [e for e in candidates if e not in used]
            rng.shuffle(remaining)
            selected.extend(remaining[: target_n - len(selected)])

        return selected[:target_n]

    def _remove_edges(self, edge_index, edges_to_remove):
        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()

        mask = [self._canonical_edge(u, v) not in edges_to_remove for u, v in zip(src, dst)]
        mask = torch.tensor(mask, dtype=torch.bool, device=edge_index.device)
        return edge_index[:, mask]

    def _toggle_edge(self, edge_index, edge, add):
        u, v = edge
        if self.treat_as_undirected and u != v:
            directed_edges = [(u, v), (v, u)]
        else:
            directed_edges = [(u, v)]

        src = edge_index[0]
        dst = edge_index[1]

        if add:
            missing = []
            for a, b in directed_edges:
                exists = bool(((src == a) & (dst == b)).any().item())
                if not exists:
                    missing.append((a, b))
            if not missing:
                return edge_index
            added = torch.tensor(missing, dtype=edge_index.dtype, device=edge_index.device).t().contiguous()
            return torch.cat([edge_index, added], dim=1)

        mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        for a, b in directed_edges:
            mask &= ~((src == a) & (dst == b))
        return edge_index[:, mask]

    def _edge_set_from_edge_index(self, edge_index):
        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()
        return list({self._canonical_edge(u, v) for u, v in zip(src, dst)})

    def _canonicalize_edges(self, edges):
        result = []
        seen = set()
        for edge in edges:
            if isinstance(edge, torch.Tensor):
                edge = edge.tolist()
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            canonical = self._canonical_edge(edge[0], edge[1])
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result

    def _canonical_edge(self, u, v):
        u, v = int(u), int(v)
        if self.treat_as_undirected and u > v:
            u, v = v, u
        return (u, v)

    def _forward_logits(self, model, x, edge_index):
        logits = model.model(x, edge_index)
        if isinstance(logits, tuple):
            logits = logits[-1]
        return logits

    def _per_sample_loss(self, logits, labels):
        losses = self.loss_fn(logits, labels)
        if losses.dim() > 1:
            losses = losses.mean(dim=1)
        return losses

    def _margin(self, logits):
        if logits.dim() == 1:
            return logits
        if logits.size(1) == 1:
            return logits[:, 0]
        topk = torch.topk(logits, k=2, dim=1).values
        return topk[:, 0] - topk[:, 1]

    def _node_degrees(self, edge_index, num_nodes):
        row = edge_index[0]
        col = edge_index[1]
        deg = torch.bincount(row, minlength=num_nodes).float() + torch.bincount(col, minlength=num_nodes).float()
        return deg
