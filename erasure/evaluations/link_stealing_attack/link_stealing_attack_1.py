import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev, braycurtis, canberra, cityblock, sqeuclidean

from erasure.core.measure import GraphMeasure
from erasure.evaluations.manager import Evaluation


def attack0_similarities(p_u, p_v):
    """Return 8 similarities/distances as a numpy array for the pair of vectors."""
    sims = [
        cosine(p_u, p_v),
        euclidean(p_u, p_v),
        correlation(p_u, p_v),
        chebyshev(p_u, p_v),
        braycurtis(p_u, p_v),
        canberra(p_u, p_v),
        cityblock(p_u, p_v),
        sqeuclidean(p_u, p_v)
    ]
    return np.nan_to_num(np.array(sims, dtype=np.float32))


def _safe_prob(x):
    x = np.clip(x, 1e-12, 1.0)
    x = x / x.sum()
    return x

def kl_divergence(p, q):
    p = _safe_prob(np.asarray(p, dtype=np.float64))
    q = _safe_prob(np.asarray(q, dtype=np.float64))
    return float(np.sum(p * (np.log(p) - np.log(q))))

def js_divergence(p, q):
    p = _safe_prob(np.asarray(p, dtype=np.float64))
    q = _safe_prob(np.asarray(q, dtype=np.float64))
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def entropy(p):
    p = _safe_prob(np.asarray(p, dtype=np.float64))
    return float(-np.sum(p * np.log(p)))

def average(a, b):     return (a + b) / 2.0
def hadamard(a, b):    return a * b
def weighted_l1(a, b): return np.abs(a - b)
def weighted_l2(a, b): return np.abs((a - b) * (a - b))

def concate_all(a, b):
    return np.concatenate((average(a, b),
                           hadamard(a, b),
                           weighted_l1(a, b),
                           weighted_l2(a, b)), axis=-1)

OPERATORS = {
    "average": average,
    "hadamard": hadamard,
    "weighted_l1": weighted_l1,
    "weighted_l2": weighted_l2,
    "concate_all": concate_all
}

def metric_block(p_u, p_v, metric_type="entropy", operator="concate_all"):
    """
    Build divergence/entropy feature block for a pair of posteriors.
    Returns a 1D numpy array.
    """
    op = OPERATORS[operator]
    if metric_type == "kl_divergence":
        s1 = np.array([kl_divergence(p_u, p_v)], dtype=np.float32)
        s2 = np.array([kl_divergence(p_v, p_u)], dtype=np.float32)
    elif metric_type == "js_divergence":
        s1 = np.array([js_divergence(p_u, p_v)], dtype=np.float32)
        s2 = np.array([js_divergence(p_v, p_u)], dtype=np.float32)
    elif metric_type == "entropy":
        s1 = np.array([entropy(p_u)], dtype=np.float32)
        s2 = np.array([entropy(p_v)], dtype=np.float32)
    else:
        raise ValueError("Unknown metric_type: {}".format(metric_type))
    block = op(s1, s2)  # shape depends on operator
    return np.nan_to_num(block.astype(np.float32))

def edge_features(p_u, p_v, metric_type="entropy", operator="concate_all"):
    sims = attack0_similarities(p_u, p_v)           # 8 dims
    met  = metric_block(p_u, p_v, metric_type, operator)  # k dims
    return np.concatenate([sims, met], axis=-1)      # 8 + k


def get_link_from_edge_index(edge_index, node_num):
    rows = edge_index[0].tolist()
    cols = edge_index[1].tolist()
    link = []
    existing_set = set()
    for r, c in zip(rows, cols):
        if r < c:
            link.append((r, c))
            existing_set.add((r, c))
        elif c < r:
            existing_set.add((c, r))
    unlink = []
    rng = np.random.default_rng(1)
    while len(unlink) < len(link):
        u = int(rng.integers(0, node_num))
        v = int(rng.integers(0, node_num))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in existing_set:
            unlink.append((u, v))
            existing_set.add((u, v))
    return link, unlink



### SIMPLE MLP TO TRAIN 
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=32, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)


class LinkStealing1(GraphMeasure):
    """
    Attack 1: train a classifier on shadow posteriors (edges vs non-edges),
    then test on target posteriors, including 'forget' vs non-edges.
    """

    def init(self):
        super().init()
        self.target = self.params["target"]
        self.shadow = self.params["shadow"]              
        self.ratio = self.params["ratio"]
        self.train_part = self.params["train_part"]
        self.test_part = self.params["test_part"]
        self.forget_part = self.params["forget_part"]
        self.metric_type = self.params["metric_type"]
        self.operator = self.params["operator"]
        self.epochs = self.params["epochs"]
        self.batch_size = self.params["batch_size"]
        self.hidden = self.params["hidden"]
        self.dropout = self.params["dropout"]
        self.lr = self.params["lr"]

    def check_configuration(self):
        # Default config
        self.params["ratio"] = self.params.get("ratio", 0.5)
        self.params["train_part"] = self.params.get("train_part", "train")
        self.params["test_part"] = self.params.get("test_part", "test")
        self.params["forget_part"] = self.params.get("forget_part", "forget")
        self.params["target"] = self.params.get("target", "unlearned")
        self.params["shadow"] = self.params.get("shadow", None)  
        self.params["metric_type"] = self.params.get("metric_type", "entropy")
        self.params["operator"] = self.params.get("operator", "concate_all")
        self.params["epochs"] = self.params.get("epochs", 50)
        self.params["batch_size"] = self.params.get("batch_size", 128)
        self.params["hidden"] = self.params.get("hidden", 32)
        self.params["dropout"] = self.params.get("dropout", 0.5)
        self.params["lr"] = self.params.get("lr", 1e-3)

    def _get_model_probs(self, model, features, edge_index):
        model.model = model.model.to(model.device)
        features = features.to(model.device)
        edge_index = edge_index.to(model.device)
        with torch.no_grad():
            logits = model.model(features, edge_index)
            probs  = softmax(logits, dim=1).detach().cpu().numpy()
        return probs

    def _build_dataset(self, probs, pos_edges, neg_edges):

        n_pos = int(round(len(pos_edges) * self.ratio))
        n_neg = int(round(len(neg_edges) * self.ratio))
        rng = np.random.default_rng(0)
        pos_sample = rng.choice(len(pos_edges), size=n_pos, replace=False)
        neg_sample = rng.choice(len(neg_edges), size=n_neg, replace=False)

        X, y = [], []
        for idx in pos_sample:
            u, v = pos_edges[idx]
            X.append(edge_features(probs[u], probs[v], self.metric_type, self.operator))
            y.append(1)
        for idx in neg_sample:
            u, v = neg_edges[idx]
            X.append(edge_features(probs[u], probs[v], self.metric_type, self.operator))
            y.append(0)
        X = np.vstack(X).astype(np.float32)
        y = np.array(y, dtype=np.int64)
        return X, y

    def _train_classifier(self, X_train, y_train, in_dim):
        model = MLP(in_dim, num_classes=2, hidden=self.hidden, dropout=self.dropout)
        model.train()
        opt = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # Simple mini-batch loop
        N = X_train.shape[0]
        idx = np.arange(N)
        for epoch in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch = idx[start:end]
                xb = torch.from_numpy(X_train[batch])
                yb = torch.from_numpy(y_train[batch])
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
        return model.eval()

    def _eval_classifier(self, clf, X, y):
        with torch.no_grad():
            logits = clf(torch.from_numpy(X))
            probs = torch.softmax(logits, dim=1).numpy()
        y_pred = (probs[:, 1] >= 0.5).astype(np.int64)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, probs[:, 1])
        return {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}

    def process(self, e: Evaluation):

        graph = e.unlearner.dataset.partitions['all'][0][0]
        features = graph.x
        edge_index = graph.edge_index
        node_num = graph.num_nodes

        exist_edges, non_existent_edges = get_link_from_edge_index(edge_index, node_num)

        target_model = self.get_model(e, which=self.target) 
        target_probs = self._get_model_probs(target_model, features, edge_index)

        # Shadow model: if provided, use it; else fall back to target for training data (weaker transfer, but works)
        shadow_key = getattr(self, "shadow", None)
        if shadow_key is not None:
            shadow_model = self.get_model(e, which=shadow_key)
            shadow_probs = self._get_model_probs(shadow_model, features, edge_index)
        else:
            shadow_probs = target_probs  # fallback

        # ----- Build datasets -----
        # Train on (shadow) exist vs non-exist
        X_train, y_train = self._build_dataset(shadow_probs, exist_edges, non_existent_edges)

        # Test 1: target exist vs non-exist
        X_test_exist, y_test_exist = self._build_dataset(target_probs, exist_edges, non_existent_edges)

        # Test 2: target forget vs non-exist 
        forget_edges = e.unlearner.dataset.partitions[self.forget_part]
        X_test_forget, y_test_forget = self._build_dataset(target_probs, forget_edges, non_existent_edges)

        # ----- Scale (fit on train only) -----
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_exist = scaler.transform(X_test_exist)
        X_test_forget = scaler.transform(X_test_forget)

        # ----- Train classifier -----
        in_dim = X_train.shape[1]
        clf = self._train_classifier(X_train, y_train, in_dim=in_dim)

        # ----- Evaluate -----
        res_exist = self._eval_classifier(clf, X_test_exist, y_test_exist)
        res_forget = self._eval_classifier(clf, X_test_forget, y_test_forget)

        self.info(f"Attack1 ({self.metric_type}/{self.operator}) on {self.target} EXIST vs NON-EXIST: {res_exist}")
        self.info(f"Attack1 ({self.metric_type}/{self.operator}) on {self.target} FORGET vs NON-EXIST: {res_forget}")

        e.add_value(f"Link Stealing Attack1 {self.target} exist/non_exist", res_exist)
        e.add_value(f"Link Stealing Attack1 {self.target} forget/non_exist", res_forget)

        return e
