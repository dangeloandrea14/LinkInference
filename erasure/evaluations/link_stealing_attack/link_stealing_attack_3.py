from erasure.core.measure import GraphMeasure
from erasure.evaluations.manager import Evaluation
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import softmax
import numpy as np
from erasure.evaluations.LinkTeller.utils import construct_edge_sets, construct_edge_sets_from_random_subgraph, construct_edge_sets_through_bfs, construct_balanced_edge_sets
from erasure.evaluations.adversary import build_adversary_graph, tagged
from erasure.evaluations.link_stealing_attack.link_stealing_attack_1 import MLP, edge_features


class LinkStealing3(GraphMeasure):
    """ LinkStealing attack, version 3
       https://arxiv.org/pdf/2308.01469
       https://github.com/xinleihe/link_stealing_attack/blob/master/stealing_link/attack_3.py
    """

    def init(self):
        super().init()
        self.influence = self.params["influence"]

        ##target

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


class LinkStealingSupervised(GraphMeasure):
    """Link stealing with partial-graph knowledge (He et al. attack-3).

    The adversary additionally holds `n_labelled` edges and `n_labelled` non-edges that it
    knows the ground truth for, all drawn from the *retain* graph, and trains a classifier
    on posterior-pair features instead of thresholding a fixed distance.  This is the
    strengthening direction of the knowledge ladder: attack-0 is the same adversary with
    no labelled examples at all.

    The labelled set never touches the forget set -- an adversary holding labelled
    forgotten edges would already know the answer it is being asked to infer.
    """

    def init(self):
        super().init()
        self.target = self.params["target"]
        self.n_labelled = self.params["n_labelled"]
        self.forget_part = self.params["forget_part"]
        self.graph_knowledge = self.params["graph_knowledge"]
        self.knowledge_fraction = self.params["knowledge_fraction"]
        self.knowledge_seed = self.params["knowledge_seed"]
        self.tag = self.params["tag"]
        self.metric_type = self.params["metric_type"]
        self.operator = self.params["operator"]
        self.epochs = self.params["epochs"]
        self.batch_size = self.params["batch_size"]
        self.hidden = self.params["hidden"]
        self.dropout = self.params["dropout"]
        self.lr = self.params["lr"]
        self.apply_softmax = self.params["apply_softmax"]

    def check_configuration(self):
        self.params["target"] = self.params.get("target", "unlearned")
        self.params["n_labelled"] = self.params.get("n_labelled", 1000)
        self.params["forget_part"] = self.params.get("forget_part", "forget")
        self.params["graph_knowledge"] = self.params.get("graph_knowledge", "retain")
        self.params["knowledge_fraction"] = self.params.get("knowledge_fraction", 1.0)
        self.params["knowledge_seed"] = self.params.get("knowledge_seed", 42)
        self.params["tag"] = self.params.get("tag", "")
        self.params["metric_type"] = self.params.get("metric_type", "entropy")
        self.params["operator"] = self.params.get("operator", "concate_all")
        self.params["epochs"] = self.params.get("epochs", 50)
        self.params["batch_size"] = self.params.get("batch_size", 128)
        self.params["hidden"] = self.params.get("hidden", 32)
        self.params["dropout"] = self.params.get("dropout", 0.5)
        self.params["lr"] = self.params.get("lr", 1e-3)
        # False for the link-prediction task arm, where the model output is a node
        # embedding rather than a class posterior.  See LinkStealing0.
        self.params["apply_softmax"] = self.params.get("apply_softmax", True)

    def process(self, e: Evaluation):
        graph = e.unlearner.dataset.partitions['all'][0][0]
        num_nodes = graph.num_nodes

        unlearned_graph, _, _ = self.get_unlearned_graph(e.unlearner, self.global_ctx.removal_type)
        adv_graph = build_adversary_graph(unlearned_graph, graph,
                                          knowledge=self.graph_knowledge,
                                          fraction=self.knowledge_fraction,
                                          seed=self.knowledge_seed)

        model = self.get_model(e)
        model.model = model.model.to(model.device)
        with torch.no_grad():
            logits = model.model(adv_graph.x.to(model.device),
                                 adv_graph.edge_index.to(model.device))
            probs = (softmax(logits, dim=1) if self.apply_softmax else logits).detach().cpu().numpy()

        forget = {(min(u, v), max(u, v)) for u, v in e.unlearner.dataset.partitions[self.forget_part]}
        retain = {(min(u, v), max(u, v)) for u, v in zip(unlearned_graph.edge_index[0].tolist(),
                                                         unlearned_graph.edge_index[1].tolist())}
        retain -= forget
        all_edges = retain | forget

        rng = np.random.default_rng(self.knowledge_seed)

        # Adversary's labelled training pairs: known edges and known non-edges, retain only.
        retain_list = sorted(retain)
        n_train = min(self.n_labelled, len(retain_list))
        train_pos = [retain_list[i] for i in rng.choice(len(retain_list), size=n_train, replace=False)]
        train_neg = self._sample_non_edges(n_train, all_edges, num_nodes, rng)

        # Test pairs: the forgotten edges against a fresh balanced non-edge set.
        test_pos = sorted(forget)
        excluded = all_edges | set(train_neg)
        test_neg = self._sample_non_edges(len(test_pos), excluded, num_nodes, rng)

        # The adversary must not be handed the answer it is being asked to infer.
        assert not (set(train_pos) & forget), "labelled positives leak forgotten edges"
        assert not (set(train_neg) & set(test_neg)), "labelled negatives overlap the test set"

        X_train, y_train = self._featurise(probs, train_pos, train_neg)
        X_test, y_test = self._featurise(probs, test_pos, test_neg)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = self._train_classifier(X_train, y_train, X_train.shape[1])
        with torch.no_grad():
            scores = torch.softmax(clf(torch.from_numpy(X_test.astype(np.float32))), dim=1).numpy()[:, 1]
        auc = roc_auc_score(y_test, scores)

        key = tagged(f"Link Stealing Attack 3 {self.target} forget/non_exist", self.tag)
        self.info(f"{key} (n_labelled={n_train}): {auc}")
        e.add_value(key, auc)

        return e

    def _sample_non_edges(self, count, excluded, num_nodes, rng):
        """Uniformly sample `count` node pairs that are not edges of the original graph."""
        seen, out = set(), []
        while len(out) < count:
            u, v = int(rng.integers(0, num_nodes)), int(rng.integers(0, num_nodes))
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            if pair in excluded or pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
        return out

    def _featurise(self, probs, pos, neg):
        X = [edge_features(probs[u], probs[v], self.metric_type, self.operator) for u, v in pos]
        X += [edge_features(probs[u], probs[v], self.metric_type, self.operator) for u, v in neg]
        y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
        return np.vstack(X).astype(np.float32), y

    def _train_classifier(self, X_train, y_train, in_dim):
        clf = MLP(in_dim, num_classes=2, hidden=self.hidden, dropout=self.dropout)
        clf.train()
        opt = optim.Adam(clf.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        idx = np.arange(X_train.shape[0])
        X_train = X_train.astype(np.float32)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                batch = idx[start:start + self.batch_size]
                opt.zero_grad()
                loss = criterion(clf(torch.from_numpy(X_train[batch])),
                                 torch.from_numpy(y_train[batch]))
                loss.backward()
                opt.step()
        return clf.eval()
