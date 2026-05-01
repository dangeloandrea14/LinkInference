import numpy as np
import random
import torch
import copy
from erasure.core.trainable_base import Trainable
from erasure.utils.cfg_utils import init_dflts_to_of
from erasure.core.factory_base import get_instance_kvargs, get_instance
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
#from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from fractions import Fraction


class TorchGraphModel(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']

        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])
        
        self.model.apply(init_weights)

        self.global_ctx.logger.info(
            f"[DBG] model cfg: epochs={self.epochs} | hidden={getattr(self.model,'hidden_channels','?')} "
            f"| out={getattr(self.model,'out_channels','?')}"
        )
                
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        loss_params = self.local_config['parameters']['loss_fn']['parameters']
        if 'weight' in loss_params and isinstance(loss_params['weight'], list):
            loss_params = {**loss_params, 'weight': torch.tensor(loss_params['weight'], dtype=torch.float)}
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'], loss_params)
        
        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        self.training_set = self.local_config['parameters'].get('training_set','train')

        es_cfg = self.local_config['parameters'].get('early_stopping', {})
        self.es_patience = int(es_cfg.get('patience', 10))
        self.es_min_delta = float(es_cfg.get('min_delta', 1e-2))

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
                if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.model.device = self.device
        self.loss_fn.to(self.device)
        
        self.patience = 0     
        self.fit() 

    
    def real_fit(self):
        
        #
        #Instead of loading the train loader, we transform the indices in the train set into a boolean mask
        #That is compatible with how usually node/edge classification models expect their input to be.
        #

        graph,labels = self.dataset.partitions['all'][0][0],self.dataset.partitions['all'][0][1]
         
        num_nodes = self.dataset.partitions['all'].num_nodes


        g = self.dataset.partitions['all'][0][0]
        train_idx = self.dataset.partitions[self.training_set]
        val_idx = list(self.dataset.partitions.get('validation', []))
        test_idx  = self.dataset.partitions.get('test', [])

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_idx] = True
        train_mask[train_idx] = True
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        

        self.global_ctx.logger.info(
            f"[DBG] graph: nodes={g.num_nodes}, edges={g.edge_index.size(1)} | "
            f"train_len={len(train_idx)} | test_len={len(test_idx)} | training_set='{self.training_set}'"
        )
                                
        best_val_loss = float('inf')
        best_state = None
        no_improve_epochs = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            X, edge_index = graph.x.to(self.device), graph.edge_index.to(self.device)
            y = labels.to(self.device)

            pred = self.model(X, edge_index)

            self.global_ctx.logger.info(
                f"[DBG] devices: model={next(self.model.parameters()).device} "
                f"| x={graph.x.device} | ei={graph.edge_index.device} | y={labels.device}"
            )

            train_loss = self.loss_fn(pred[train_mask], y[train_mask])
            train_loss_val = float(train_loss.detach().cpu().item())

            train_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()


            with torch.no_grad():
                train_preds_np = pred[train_mask].detach().cpu().numpy()
                train_labels_np = y[train_mask].detach().cpu().numpy()
                train_acc = self.accuracy(train_labels_np, train_preds_np)


            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X, edge_index)  
                val_loss = self.loss_fn(val_logits[val_mask], y[val_mask])
                val_loss_val = float(val_loss.detach().cpu().item())

            self.global_ctx.logger.info(
                f'epoch = {epoch} ---> train_loss = {train_loss_val:.4f}\t val_loss = {val_loss_val:.4f}\t train_acc = {train_acc:.4f}'
            )

            if val_loss_val < (best_val_loss - self.es_min_delta):
                best_val_loss = val_loss_val
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= self.es_patience:
                self.global_ctx.logger.info(
                    f"[EARLY-STOP] Patience reached at epoch {epoch}. "
                    f"Restoring best weights (best val_loss={best_val_loss:.6f})."
                )
                if best_state is not None:
                    self.model.load_state_dict(best_state)
                break
            
                
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults

        ###LOCAL_CONFIG MUST REFERENCE CONTEXT LOCAL
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 50)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', 0.01)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')

        local_config['parameters']['model']['parameters']['n_classes'] = self.dataset.n_classes

        local_config['parameters']['alias'] = local_config['parameters'].get('alias', local_config['parameters']['model']['class'])
        local_config['parameters']['training_set'] = local_config['parameters'].get("training_set", "train")

    '''
    def rebuild_optimizer_and_scheduler(self):
        self.optimizer = get_instance_kvargs(
            self.local_config['parameters']['optimizer']['class'],
            {
                'params': self.model.parameters(),
                **self.local_config['parameters']['optimizer']['parameters']
            }
        )

        if 'lr_scheduler' in self.local_config['parameters']:
            self.lr_scheduler = get_instance_kvargs(
                self.local_config['parameters']['lr_scheduler']['class'],
                {
                    'optimizer': self.optimizer,
                    **self.local_config['parameters']['lr_scheduler']['parameters']
                }
            )
        else:
            self.lr_scheduler = lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs
            )
    '''
        
    def accuracy(self, testy, probs):
        acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc
'''
    def read(self):
        super().read()
        if isinstance(self.model, list):
            for mod in self.model:
                mod.to(self.device)
        else:
            self.model.to(self.device)
            
    def to(self, device):
        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        elif isinstance(self.model, list):
            for model in self.model:
                if isinstance(model, torch.nn.Module):
                    model.to(self.device)
                    '''

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:  # Check that bias exists
            m.bias.data.fill_(0.01)
            
class LogSigmoidBinaryLoss(nn.Module):
    def __init__(self, lam=1e-2):
        """
        Logistic regression loss with L2 regularization.
        Labels must be in {-1, +1}.
        """
        super().__init__()
        self.lam = lam

    def forward(self, logits, targets, weights=None):

        loss = -F.logsigmoid(targets * logits).mean()

        if weights is not None:
            loss = loss + self.lam * weights.pow(2).sum() / 2

        return loss