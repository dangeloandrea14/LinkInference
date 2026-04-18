import copy
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from erasure.unlearners.torchunlearner import TorchUnlearner


class ScaleGUN(TorchUnlearner):
    """
    Certified edge unlearning via objective perturbation and Hessian-based updates.
    Adapted from: https://github.com/ludan3134/ScaleGUN
    """

    def check_configuration(self):
        super().check_configuration()
        self.params.setdefault('std', 1e-2)
        self.params.setdefault('lam', 1e-2)
        self.params.setdefault('eps', 1.0)
        self.params.setdefault('delta', 1e-4)
        self.params.setdefault('prop_step', 3)
        self.params.setdefault('num_steps_optimizer', 100)
        self.params.setdefault('lr', 0.01)
        self.params.setdefault('ref_data_retain', 'retain')
        self.params.setdefault('ref_data_forget', 'forget')
        self.params.setdefault('train_mode', 'ovr')

    def init(self):
        super().init()
        self.std = self.params['std']
        self.lam = self.params['lam']
        self.eps = self.params['eps']
        self.delta = self.params['delta']
        self.prop_step = self.params['prop_step']
        self.num_steps_optimizer = self.params['num_steps_optimizer']
        self.lr = self.params['lr']
        self.ref_data_retain = self.params['ref_data_retain']
        self.ref_data_forget = self.params['ref_data_forget']
        self.train_mode = self.params['train_mode']

    def __unlearn__(self):
        self.info('Starting ScaleGUN')

        data = self.dataset.partitions['all'].data[0]
        num_nodes = self.dataset.partitions['all'].num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[self.dataset.partitions['train']] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[self.dataset.partitions['validation']] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[self.dataset.partitions['test']] = True

        labels = self.dataset.partitions['all'][0][1]
        edge_index = self.dataset.partitions['all'][0][0].edge_index

        y_train = (F.one_hot(labels[train_mask]) * 2 - 1).float().to(self.device)
        y_val = labels[val_mask].to(self.device)
        y_test = labels[test_mask].to(self.device)
        num_classes = y_train.size(1)

        Propagation = self.predictor.model.feat_prop.to(self.device)
        X_raw = data.x.float()
        X_scaled_copy = X_raw.clone().detach()

        if self.prop_step > 0:
            X = Propagation(X_raw.to(self.device), edge_index.to(self.device)).float()
        else:
            X = X_raw.to(self.device)

        X_train = X[train_mask].to(self.device)
        X_val = X[val_mask].to(self.device)
        X_test = X[test_mask].to(self.device)

        feat_dim = X_train.size(1)

        # Initial training with objective perturbation
        b = self.std * torch.randn(feat_dim, num_classes).float().to(self.device)
        w = self._ovr_lr_optimize(X_train, y_train, self.lam, b=b, lr=self.lr)

        self.info(f'Initial val acc: {self._ovr_lr_eval(w, X_val, y_val):.4f}, '
                  f'test acc: {self._ovr_lr_eval(w, X_test, y_test):.4f}')

        c_val = math.sqrt(2 * math.log(1.5 / self.delta))
        budget = (self.std * self.eps / c_val) * num_classes
        gamma = 1.0 / 4.0
        self.info(f'Budget: {budget:.6f}')

        # Deduplicate forget edges to unique undirected pairs
        forget = self.dataset.partitions[self.ref_data_forget]
        seen = set()
        unique_forget = []
        for u, v in forget:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                unique_forget.append((u, v))

        edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        w_approx = w.clone().detach()
        X_old = X_train.clone().detach()
        accum_grad_norm = 0.0
        num_retrain = 0

        for i, (u, v) in enumerate(unique_forget):
            # Remove both directions of the edge
            fwd = torch.logical_and(edge_index[0] == u, edge_index[1] == v)
            rev = torch.logical_and(edge_index[0] == v, edge_index[1] == u)
            edge_mask[fwd] = False
            edge_mask[rev] = False

            if self.prop_step > 0:
                X_new = Propagation(
                    X_scaled_copy.to(self.device), edge_index[:, edge_mask]
                ).float()
            else:
                X_new = X_scaled_copy.float().to(self.device)

            X_train_new = X_new[train_mask].to(self.device)
            X_val_new = X_new[val_mask].to(self.device)
            X_test_new = X_new[test_mask].to(self.device)

            K = X_train_new.t().mm(X_train_new)
            spec_norm = self._sqrt_spectral_norm(K)

            step_grad_norm = 0.0
            for k in range(num_classes):
                y_k = y_train[:, k]
                H_inv = self._lr_hessian_inv(w_approx[:, k], X_train_new, y_k, self.lam)
                grad_old = self._lr_grad(w_approx[:, k], X_old, y_k, self.lam)
                grad_new = self._lr_grad(w_approx[:, k], X_train_new, y_k, self.lam)
                Delta = H_inv.mv(grad_old - grad_new)
                Delta_p = X_train_new.mv(Delta)
                w_approx[:, k] = w_approx[:, k] + Delta
                step_grad_norm += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu().item()

            accum_grad_norm += step_grad_norm

            if accum_grad_norm > budget:
                accum_grad_norm = 0.0
                b = self.std * torch.randn(feat_dim, num_classes).float().to(self.device)
                w_approx = self._ovr_lr_optimize(X_train_new, y_train, self.lam, b=b, lr=self.lr)
                num_retrain += 1
                self.info(f'Retrain at removal {i} (total: {num_retrain})')

            self.info(f'Removal {i}: val={self._ovr_lr_eval(w_approx, X_val_new, y_val):.4f}, '
                      f'test={self._ovr_lr_eval(w_approx, X_test_new, y_test):.4f}')

            X_old = X_train_new.clone().detach()

        self.info(f'ScaleGUN done. Total retrains: {num_retrain}')

        with torch.no_grad():
            self.predictor.model.classifier.weight.copy_(w_approx.T)
            if self.predictor.model.classifier.bias is not None:
                self.predictor.model.classifier.bias.zero_()

        og_graph = self.dataset.partitions['all']
        new_graph = og_graph.revise_graph_edges(forget, remove=True)
        remapped_partitions = copy.deepcopy(self.dataset.partitions)

        self.predictor.dataset.partitions = {}
        self.predictor.dataset.partitions['all'] = new_graph
        self.predictor.dataset.partitions['train'] = remapped_partitions['train']
        self.predictor.dataset.partitions['test'] = remapped_partitions['test']
        self.predictor.dataset.partitions['forget'] = remapped_partitions['forget']

        return self.predictor

    def _lr_hessian_inv(self, w, X, y, lam, batch_size=50000):
        z = torch.sigmoid(y * X.mv(w))
        D = z * (1 - z)
        H = None
        for i in range(math.ceil(X.size(0) / batch_size)):
            lo = i * batch_size
            hi = min(lo + batch_size, X.size(0))
            Xi = X[lo:hi]
            block = Xi.t().mm(D[lo:hi].unsqueeze(1) * Xi)
            H = block if H is None else H + block
        return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(self.device)).inverse()

    def _lr_grad(self, w, X, y, lam):
        z = torch.sigmoid(y * X.mv(w))
        return X.t().mv((z - 1) * y) + lam * X.size(0) * w

    def _sqrt_spectral_norm(self, A, num_iters=100):
        x = torch.randn(A.size(0)).float().to(A.device)
        for _ in range(num_iters):
            x = A.mv(x)
            norm = x.norm()
            if norm == 0:
                break
            x = x / norm
        max_lam = torch.dot(x, A.mv(x)) / torch.dot(x, x)
        return math.sqrt(max_lam.item())

    def _ovr_lr_optimize(self, X, y, lam, b=None, lr=0.01):
        w_init = torch.zeros(b.size()).float() if b is not None else torch.zeros(X.size(1), y.size(1)).float()
        w = torch.autograd.Variable(w_init.to(self.device), requires_grad=True)

        def closure():
            loss = self._ovr_lr_loss(w, X, y, lam)
            if b is not None:
                loss = loss + (b * w).sum() / X.size(0)
            return loss

        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=1e-32, tolerance_change=1e-32)
        for _ in range(self.num_steps_optimizer):
            optimizer.zero_grad()
            loss = closure()
            loss.backward()
            optimizer.step(closure)

        return w.clone().detach()

    def _ovr_lr_loss(self, w, X, y, lam):
        z = X.mm(w) * y
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2

    def _ovr_lr_eval(self, w, X, y):
        return X.mm(w).max(1)[1].eq(y).float().mean()
