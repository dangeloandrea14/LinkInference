import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler

from erasure.core.trainable_base import Trainable
from erasure.utils.cfg_utils import init_dflts_to_of
from erasure.core.factory_base import get_instance_kvargs

from torch_geometric.loader import NeighborLoader

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class LogSigmoidBinaryLoss(nn.Module):
    def __init__(self, lam=1e-2):
        super().__init__()
        self.lam = lam
    def forward(self, logits, targets, weights=None):
        loss = -F.logsigmoid(targets * logits).mean()
        if weights is not None:
            loss = loss + self.lam * weights.pow(2).sum() / 2
        return loss


class TorchGraphModelBatched(Trainable):

    def init(self):
        self.epochs = self.local_config['parameters']['epochs']

        self.model = get_instance_kvargs(
            self.local_config['parameters']['model']['class'],
            self.local_config['parameters']['model']['parameters']
        )
        self.model.apply(init_weights)

        self.optimizer = get_instance_kvargs(
            self.local_config['parameters']['optimizer']['class'],
            {'params': self.model.parameters(),
             **self.local_config['parameters']['optimizer']['parameters']}
        )

        self.loss_fn = get_instance_kvargs(
            self.local_config['parameters']['loss_fn']['class'],
            self.local_config['parameters']['loss_fn']['parameters']
        )

        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        self.lr_scheduler = lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs
        )

        self.training_set = self.local_config['parameters'].get('training_set', 'train')

        params = self.local_config['parameters']
        self.batch_size = params.get('batch_size', 1024)                
        self.num_neighbors = params.get('num_neighbors', [15, 10])      
        self.shuffle = params.get('shuffle', True)
        self.num_workers = params.get('num_workers', 0)
        self.drop_last = params.get('drop_last', False)

        # device
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
        self.model.to(self.device)
        self.model.device = self.device

        self.patience = 0
        self.fit()

    # ---------------- core training loop (mini-batch) ----------------
    def real_fit(self):
        """
        Mini-batch training using neighborhood sampling. Each step:
        - sample a *subgraph* around a set of seed (target) nodes
        - forward on the subgraph
        - compute loss on the *seed nodes only* (first batch.batch_size nodes)
        """
        # Access the big graph and labels the same way you do in full-batch
        graph, labels = self.dataset.partitions['all'][0][0], self.dataset.partitions['all'][0][1]
        num_nodes = self.dataset.partitions['all'].num_nodes

        # index lists from your partitions; used as seed pools
        train_idx = torch.as_tensor(self.dataset.partitions[self.training_set], dtype=torch.long)
        val_idx = torch.as_tensor(self.dataset.partitions.get('val', []), dtype=torch.long) \
                  if 'val' in self.dataset.partitions else None
        test_idx = torch.as_tensor(self.dataset.partitions.get('test', []), dtype=torch.long) \
                   if 'test' in self.dataset.partitions else None

        # Build loaders
        train_loader = NeighborLoader(
            graph,
            num_neighbors=self.num_neighbors,
            input_nodes=train_idx,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last
        )

        # Inference loaders (layer-wise, covers all nodes in manageable chunks)
        # num_neighbors=[-1] means gather full neighborhood per layer, but processed in batches
        infer_loader = NeighborLoader(
            graph,
            num_neighbors=[-1] * max(1, len(self.num_neighbors)),
            input_nodes=None,
            batch_size=max(self.batch_size, 4096),
            shuffle=False,
            num_workers=self.num_workers
        )

        labels = labels.to(self.device)

        # ----- training epochs -----
        for epoch in range(self.epochs):
            self.model.train()
            batch_losses = []
            batch_accs = []

            for batch in train_loader:
                # batch is a subgraph with .n_id mapping to global ids
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward on the subgraph
                pred = self.model(batch.x, batch.edge_index)

                # Labels for nodes in this subgraph
                batch_labels = labels[batch.n_id]

                # First batch.batch_size nodes are the *seed targets* for loss
                target_idx = torch.arange(batch.batch_size, device=self.device)

                loss = self.loss_fn(pred[target_idx], batch_labels[target_idx])
                loss.backward()
                self.optimizer.step()

                batch_losses.append(float(loss.detach().cpu()))
                # match your accuracy style (argmax over probs/logits)
                acc = self.accuracy(
                    batch_labels[target_idx].detach().cpu().numpy(),
                    pred[target_idx].detach().cpu().numpy()
                )
                batch_accs.append(acc)

            # Step LR scheduler once per epoch (keeps parity with your original code)
            self.lr_scheduler.step()

            mean_loss = float(np.mean(batch_losses)) if batch_losses else float('nan')
            mean_acc = float(np.mean(batch_accs)) if batch_accs else float('nan')
            self.global_ctx.logger.info(
                f'epoch = {epoch} ---> loss = {mean_loss:.4f}\t accuracy = {mean_acc:.4f}'
            )

            # ----- optional early stopping on validation -----
            if self.early_stopping_threshold is not None and val_idx is not None and len(val_idx) > 0:
                val_acc = self._evaluate_node_split(infer_loader, labels, val_idx)
                self.global_ctx.logger.info(f'epoch = {epoch} ---> val_accuracy = {val_acc:.4f}')
                # simplistic early stopping heuristic
                if val_acc >= self.early_stopping_threshold:
                    self.global_ctx.logger.info('Early stopping threshold reached.')
                    break

        # Optionally evaluate on test at the end
        if test_idx is not None and len(test_idx) > 0:
            test_acc = self._evaluate_node_split(infer_loader, labels, test_idx)
            self.global_ctx.logger.info(f'Test accuracy = {test_acc:.4f}')

    # ---------------- utility: full-graph inference in batches ----------------
    @torch.no_grad()
    def _predict_all_nodes(self, infer_loader):
        """
        Layer-wise batched inference over all nodes. Returns a tensor of shape [num_nodes, n_classes].
        """
        self.model.eval()
        # total number of nodes from the data object inside the loader
        total_nodes = infer_loader.data.num_nodes
        out = None

        for batch in infer_loader:
            batch = batch.to(self.device)
            pred = self.model(batch.x, batch.edge_index)  # logits for nodes in this subgraph
            # Map back to global indices
            global_nid = batch.n_id
            if out is None:
                out = pred.new_zeros((total_nodes, pred.size(-1)))
            out[global_nid] = pred

        return out

    @torch.no_grad()
    def _evaluate_node_split(self, infer_loader, labels, split_idx_tensor):
        """
        Compute accuracy on a given split (e.g., val or test) using batched full-graph inference.
        """
        logits_all = self._predict_all_nodes(infer_loader)
        split_idx_tensor = split_idx_tensor.to(self.device)
        y_true = labels[split_idx_tensor].detach().cpu().numpy()
        y_pred = logits_all[split_idx_tensor].detach().cpu().numpy()
        return self.accuracy(y_true, y_pred)

    # ---------------- config/compat layer (mirrors your original) ----------------
    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config

        # defaults matching your original
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 50)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 1024)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)

        # optimizer/loss defaults
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam', lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')  # (note: see accuracy discussion below)

        # pass dataset-dependent params into model
        local_config['parameters']['model']['parameters']['n_classes'] = self.dataset.n_classes

        # alias & partition selection
        local_config['parameters']['alias'] = local_config['parameters']['model']['class']
        local_config['parameters']['training_set'] = local_config['parameters'].get('training_set', 'train')

        # new mini-batch parameters (safe defaults)
        params = local_config['parameters']
        params['num_neighbors'] = params.get('num_neighbors', [15, 10])  # per-layer fanouts
        params['shuffle'] = params.get('shuffle', True)
        params['num_workers'] = params.get('num_workers', 0)
        params['drop_last'] = params.get('drop_last', False)

    def accuracy(self, testy, probs):
        # same metric as your original (argmax over last dim)
        acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc
