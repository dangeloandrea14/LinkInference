from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from erasure.core.factory_base import get_instance_kvargs

class FisherForgetting(GraphUnlearner):
    def init(self):
        """
        Initializes the Fisher Forgetting class with global and local contexts.
        """
        super().init()

        self.ref_data_retain = self.local.config['parameters']['ref_data']
        self.alpha = self.local.config['parameters'].get('alpha', 1e-6)
        self.num_classes = self.dataset.n_classes

    def compute_fisher_information(self, retain_set):
        """
        Computes the Fisher Information Matrix for each parameter using the retain set.
        Now generalized for any type of data without class-specific computation.
        """
        self.predictor.model.eval()

        num_nodes = self.x.size(0)

        if self.removal_type == 'edge':
            retain_set = self.infected_nodes(retain_set, self.hops)

        # Initialize gradient accumulators
        for p in self.predictor.model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        

        for node_idx in tqdm(retain_set, desc="Computing Fisher Information per node"):
            node_idx = int(node_idx)

            output = self.predictor.model(self.x, self.edge_index)[node_idx].unsqueeze(0)  # shape: [1, num_classes]
            orig_target = self.labels[node_idx].unsqueeze(0).to(self.device)

            prob = F.softmax(output, dim=-1).detach()

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = self.predictor.loss_fn(output, target)
                self.predictor.model.zero_grad()

                loss.backward(retain_graph=True)
                for p in self.predictor.model.parameters():
                    if p.requires_grad:
                        if p.grad is None:
                            continue
                        
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        num_retain = len(retain_set)
        for p in self.predictor.model.parameters():
            p.grad_acc /= num_retain
            p.grad2_acc /= num_retain           
        

    def get_mean_var(self, p, is_base_dist=False, alpha=3e-6):
        var = copy.deepcopy(1./(p.grad2_acc+1e-8))
        if isinstance(var, float):  
            var = torch.tensor(var) 
        var = var.clamp(max=1e3)
        if p.size(0) == self.num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var
        
        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = copy.deepcopy(p.data0.clone())
        else:
            mu = copy.deepcopy(p.data0.clone())
        if p.size(0) == self.num_classes:
            # Last layer
            var *= 10
        elif p.ndim == 1:
            # BatchNorm
            var *= 10
    #         var*=1
        return mu, var

    def apply_fisher_noise(self):
        """
        Applies Fisher noise to model parameters for selective forgetting.
        """        
        for p in self.predictor.model.parameters():
            
            if not isinstance(p, torch.Tensor):
                continue

            if not hasattr(p, 'data0'):
                p.data0 = copy.deepcopy(p.data.clone())

            if not isinstance(p.grad2_acc, torch.Tensor):
                    continue
            
            mu, var = self.get_mean_var(p)
            p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()

    def __unlearn__(self):
        """
        An implementation of the Fisher Forgetting unlearning algorithm proposed in the following paper:
        "Golatkar, Aditya, Alessandro Achille, and Stefano Soatto. "Eternal sunshine of the spotless net: Selective forgetting in deep networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020."
        
        Codebase taken and rearranged from the original implementation: https://github.com/AdityaGolatkar/SelectiveForgetting/tree/master
        """
        self.info(f'Starting Fisher Forgetting')

        retain_set = self.dataset.partitions[self.ref_data_retain]

        # Compute Fisher Information using retain set
        self.info('Computing Fisher Information Matrix')
        self.compute_fisher_information(retain_set)

        # Apply Fisher noise for selective forgetting
        self.info('Applying Fisher noise for selective forgetting')
        self.apply_fisher_noise()

        return self.predictor

    def check_configuration(self):
        """
        Checks and sets default configuration parameters.
        """
        super().check_configuration()

        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'retain')
        self.local.config['parameters']['alpha'] = self.local.config['parameters'].get("alpha", 1e-6)
