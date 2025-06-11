import numpy as np
import random
import torch

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
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        self.training_set = self.local.config['parameters'].get('training_set','train')



        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
                if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device) 
        self.model.device = self.device
        
        self.patience = 0     
        self.fit() 

    
    def real_fit(self):
        
        #
        #Instead of loading the train loader, we transform the indices in the train set into a boolean mask
        #That is compatible with how usually node/edge classification models expect their input to be.
        #

        graph,labels = self.dataset.partitions['all'][0][0],self.dataset.partitions['all'][0][1]
         
        num_nodes = self.dataset.partitions['all'].num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[self.dataset.partitions[self.training_set]] = True

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[self.dataset.partitions['test']] = True
                         
        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.model.train()

            self.optimizer.zero_grad()

            X,edge_index= graph.x,graph.edge_index

            X,edge_index,labels = X.to(self.device),edge_index.to(self.device), labels.to(self.device)
                
            pred = self.model(X,edge_index)

            loss = self.loss_fn(pred[train_mask], labels[train_mask])
                
            losses.append(loss.to('cpu').detach().numpy())
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
                
            preds = pred[train_mask].detach().cpu().numpy()
            labels_list = labels[train_mask].cpu().numpy()
            accuracy = self.accuracy(labels_list, preds)
               
            self.global_ctx.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            
            
                
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults

        ###LOCAL_CONFIG MUST REFERENCE CONTEXT LOCAL
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 50)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')

        local_config['parameters']['model']['parameters']['n_classes'] = self.dataset.n_classes

        local_config['parameters']['alias'] = local_config['parameters']['model']['class']
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