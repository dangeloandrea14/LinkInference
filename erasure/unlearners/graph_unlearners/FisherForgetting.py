from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
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
        Uses C batched backward passes (one per class) instead of N*C per-node passes.
        """
        self.predictor.model.eval()

        if self.removal_type == 'edge':
            retain_set = self.infected_nodes(retain_set, self.hops)

        for p in self.predictor.model.parameters():
            p.grad_acc = torch.zeros_like(p.data)
            p.grad2_acc = torch.zeros_like(p.data)

        if len(retain_set) == 0:
            return

        retain_idx = torch.tensor(retain_set, dtype=torch.long, device=self.device)
        orig_targets = self.labels[retain_idx].to(self.device)

        # Forward pass with grad enabled so backward can propagate through params
        output_grad = self.predictor.model(self.x, self.edge_index)[retain_idx]
        prob = F.softmax(output_grad.detach(), dim=-1)  # [n_retain, C], constant weights

        n_retain = len(retain_set)
        n_classes = output_grad.shape[1]

        for y in range(n_classes):
            target_y = torch.full((n_retain,), y, dtype=torch.long, device=self.device)
            # Weighted loss: sum_n prob[n,y] * CE(output_n, y)
            per_sample_loss = F.cross_entropy(output_grad, target_y, reduction='none')
            weighted_loss = (prob[:, y] * per_sample_loss).sum()

            self.predictor.model.zero_grad()
            weighted_loss.backward(retain_graph=(y < n_classes - 1))

            is_true_class = (orig_targets == y).float().mean()
            for p in self.predictor.model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad_acc += is_true_class * p.grad.data
                    p.grad2_acc += p.grad.data.pow(2)

        for p in self.predictor.model.parameters():
            p.grad_acc /= n_retain
            p.grad2_acc /= n_retain
        

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
