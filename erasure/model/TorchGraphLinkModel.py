import copy

import torch
import torch.optim.lr_scheduler as lr_scheduler

from erasure.core.factory_base import get_instance_kvargs
from erasure.core.trainable_base import Trainable
from erasure.model.TorchGraphModel import init_weights
from erasure.model.lp_utils import (HOLDOUT_PARTS, canonical_pairs, corrupt_pairs,
                                    mp_edge_index, pairs_touching, ranking_scores,
                                    sample_negative_pairs)
from erasure.utils.cfg_utils import init_dflts_to_of


class TorchGraphLinkModel(Trainable):
    """Link-prediction predictor: sibling of ``TorchGraphModel`` for the LP task arm.

    Same lifecycle as the node-classification predictor (``init`` ends in
    ``self.fit()``, early stopping on validation loss), but the samples are node
    *pairs* rather than nodes:

    * **message-passing graph** -- the observed graph minus the held-out
      ``lp_val_pos`` / ``lp_test_pos`` partitions, so a validation/test edge is
      never visible in the adjacency the model aggregates over;
    * **supervision positives** -- the edges of that message-passing graph;
    * **negatives** -- pairs that are not edges of the *observed* graph, resampled
      every epoch;
    * **loss** -- ``BCEWithLogitsLoss`` over ``model.decode(z, pairs)``.

    Supervision is derived from ``partitions['all']`` rather than from a partition
    list, which is what lets ``GoldModelGraph`` work unchanged: it hands the
    predictor a data manager whose graph already has the forget edges stripped, so
    they disappear from both message passing and supervision automatically.  The
    ``training_set`` parameter is therefore accepted (the Gold Model sets it) but
    carries no meaning here.
    """

    def init(self):
        self.epochs = self.local_config['parameters']['epochs']

        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                        self.local_config['parameters']['model']['parameters'])
        self.model.apply(init_weights)

        self.optimizer = get_instance_kvargs(
            self.local_config['parameters']['optimizer']['class'],
            {'params': self.model.parameters(),
             **self.local_config['parameters']['optimizer']['parameters']})

        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                          self.local_config['parameters']['loss_fn']['parameters'])

        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        self.lr_scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0,
                                                  end_factor=0.5, total_iters=self.epochs)

        self.training_set = self.local_config['parameters'].get('training_set', 'train')

        es_cfg = self.local_config['parameters'].get('early_stopping', {})
        self.es_patience = int(es_cfg.get('patience', 10))
        self.es_min_delta = float(es_cfg.get('min_delta', 1e-2))

        self.grad_clip = self.local_config['parameters']['grad_clip']
        self.neg_ratio = float(self.local_config['parameters']['neg_ratio'])
        self.eval_seed = int(self.local_config['parameters']['eval_seed'])
        self.holdout_parts = tuple(self.local_config['parameters']['holdout_parts'])
        self.max_supervision_edges = self.local_config['parameters']['max_supervision_edges']

        self.device = self.local_config['parameters'].get('device') or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.model.device = self.device
        self.loss_fn.to(self.device)

        self.patience = 0
        self.fit()

    # -- link-prediction context ---------------------------------------------

    def lp_context(self):
        """``(x, observed_edge_index, mp_edge_index, num_nodes)`` on the model device.

        Only the message-passing edge_index is memoised -- it costs a membership test
        over every edge to build, and the data manager handed to a predictor never
        changes after construction (the Gold Model builds a *new* manager rather than
        mutating one).  ``x`` and the observed edge_index are taken from the graph on
        each call so that ``Saveable`` does not pickle a second copy of them; the
        node-classification predictor fetches them per epoch in the same way.
        """
        graph = self.dataset.partitions['all'][0][0]
        num_nodes = self.dataset.partitions['all'].num_nodes

        x = graph.x.to(self.device).float()
        observed = graph.edge_index.to(self.device).long()

        mp_ei = getattr(self, '_lp_mp_ei', None)
        if mp_ei is None or mp_ei.device.type != observed.device.type:
            holdout = []
            for part in self.holdout_parts:
                holdout.extend(self.dataset.partitions.get(part, []))
            mp_ei = mp_edge_index(observed, holdout, num_nodes)
            self._lp_mp_ei = mp_ei

        return x, observed, mp_ei, num_nodes

    def supervision_pairs(self):
        """Canonical undirected positives the model is trained on."""
        cached = getattr(self, '_supervision_pairs_cache', None)
        if cached is not None:
            return cached
        _, _, mp_ei, _ = self.lp_context()
        pairs = canonical_pairs(mp_ei)
        if self.max_supervision_edges and pairs.size(0) > self.max_supervision_edges:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.eval_seed)
                keep = torch.randperm(pairs.size(0))[:self.max_supervision_edges]
            pairs = pairs[keep.to(pairs.device)]
        self._supervision_pairs_cache = pairs
        return pairs

    def _scores_and_labels(self, z, pos, neg):
        pairs = torch.cat([pos, neg], dim=0)
        scores = self.model.decode(z, pairs)
        labels = torch.cat([torch.ones(pos.size(0), device=scores.device),
                            torch.zeros(neg.size(0), device=scores.device)])
        return scores, labels

    # -- the hook the unlearners call ----------------------------------------

    def task_loss(self, node_subset=None, edge_subset=None, negatives=None, seed=None):
        """Link-prediction loss, optionally restricted to a node or edge subset.

        * ``edge_subset`` -- score exactly those edges as positives.  Used by the
          gradient-ascent methods, which want to push the forgotten edges *down*;
          negatives are excluded by default so the ascent term is purely about the
          forgotten links.
        * ``node_subset`` -- supervision edges with at least one endpoint in the
          subset, plus corrupted-tail negatives anchored on the same nodes.  This is
          the pair-level analogue of the node-classification loss over the nodes
          infected by the forget set.
        """
        if negatives is None:
            negatives = edge_subset is None

        x, observed, mp_ei, num_nodes = self.lp_context()

        if edge_subset is not None:
            pos = canonical_pairs(edge_subset).to(self.device)
        else:
            pos = self.supervision_pairs().to(self.device)
            if node_subset is not None:
                pos = pairs_touching(pos, node_subset, num_nodes)

        z = self.model(x, mp_ei)

        if pos.numel() == 0:
            return z.sum() * 0.0

        if negatives:
            neg = corrupt_pairs(pos, observed, num_nodes, seed=seed)
        else:
            neg = pos.new_zeros((0, 2))

        scores, labels = self._scores_and_labels(z, pos, neg)
        return self.loss_fn(scores, labels)

    # -- training ------------------------------------------------------------

    def real_fit(self):
        x, observed, mp_ei, num_nodes = self.lp_context()
        pos = self.supervision_pairs().to(self.device)

        val_pos = canonical_pairs(
            self.dataset.partitions.get('lp_val_pos', [])).to(self.device)
        val_neg = sample_negative_pairs(val_pos.size(0), observed, num_nodes,
                                        seed=self.eval_seed).to(self.device)
        has_val = val_pos.numel() > 0 and val_neg.numel() > 0

        self.global_ctx.logger.info(
            f"[DBG] LP graph: nodes={num_nodes}, observed_edges={observed.size(1)}, "
            f"mp_edges={mp_ei.size(1)} | supervision_pairs={pos.size(0)} | "
            f"val_pairs={val_pos.size(0)} | decoder={getattr(self.model, 'decoder_type', '?')}"
        )

        best_val_loss = float('inf')
        best_state = None
        no_improve_epochs = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            n_neg = int(round(pos.size(0) * self.neg_ratio))
            neg = sample_negative_pairs(n_neg, observed, num_nodes).to(self.device)

            z = self.model(x, mp_ei)
            scores, labels = self._scores_and_labels(z, pos, neg)
            train_loss = self.loss_fn(scores, labels)

            train_loss.backward()
            # The pair-level BCE over ~100K pairs produces occasional very large
            # gradients early on, which sends the loss into the tens before it
            # recovers.  Clipping keeps every model in a run -- including the Gold
            # Model, which is the reference the whole table is read against --
            # on the same stable trajectory.
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.lr_scheduler.step()

            train_loss_val = float(train_loss.detach().cpu().item())
            with torch.no_grad():
                train_auc = self.accuracy(labels.detach().cpu().numpy(),
                                          scores.detach().cpu().numpy())

            if not has_val:
                self.global_ctx.logger.info(
                    f'epoch = {epoch} ---> train_loss = {train_loss_val:.4f}\t '
                    f'train_auc = {train_auc:.4f} (no validation edges)')
                continue

            self.model.eval()
            with torch.no_grad():
                z_val = self.model(x, mp_ei)
                val_scores, val_labels = self._scores_and_labels(z_val, val_pos, val_neg)
                val_loss_val = float(self.loss_fn(val_scores, val_labels).detach().cpu().item())
                val_auc, _ = ranking_scores(val_scores[:val_pos.size(0)],
                                            val_scores[val_pos.size(0):])

            self.global_ctx.logger.info(
                f'epoch = {epoch} ---> train_loss = {train_loss_val:.4f}\t '
                f'val_loss = {val_loss_val:.4f}\t train_auc = {train_auc:.4f}\t '
                f'val_auc = {val_auc:.4f}')

            if val_loss_val < (best_val_loss - self.es_min_delta):
                best_val_loss = val_loss_val
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= self.es_patience:
                self.global_ctx.logger.info(
                    f"[EARLY-STOP] Patience reached at epoch {epoch}. "
                    f"Restoring best weights (best val_loss={best_val_loss:.6f}).")
                if best_state is not None:
                    self.model.load_state_dict(best_state)
                break

    def accuracy(self, testy, probs):
        """ROC-AUC of the pair scores (the LP counterpart of test accuracy)."""
        pos = torch.as_tensor(probs)[torch.as_tensor(testy) > 0.5]
        neg = torch.as_tensor(probs)[torch.as_tensor(testy) <= 0.5]
        auc, _ = ranking_scores(pos, neg)
        return auc

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        params = local_config['parameters']

        params['epochs'] = params.get('epochs', 100)
        params['batch_size'] = params.get('batch_size', 4)
        params['early_stopping_threshold'] = params.get('early_stopping_threshold', 0.01)
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam', lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCEWithLogitsLoss')

        params['grad_clip'] = params.get('grad_clip', 1.0)
        params['neg_ratio'] = params.get('neg_ratio', 1.0)
        params['eval_seed'] = params.get('eval_seed', 12345)
        params['holdout_parts'] = params.get('holdout_parts', list(HOLDOUT_PARTS))
        params['max_supervision_edges'] = params.get('max_supervision_edges', None)

        # Deliberately NOT injecting n_classes: `out_channels` is the embedding
        # width here, not a number of classes.
        params['alias'] = params.get('alias', params['model']['class'])
        params['training_set'] = params.get('training_set', 'train')
