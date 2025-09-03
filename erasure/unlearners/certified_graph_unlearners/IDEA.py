from erasure.unlearners.torchunlearner import TorchUnlearner
from fractions import Fraction
import torch.optim as optim
import numpy as np
import torch
import time
from collections import Counter
from torch import Tensor
import copy
from typing import Tuple
import math
from functools import reduce
from scipy.optimize import fmin_cg, fmin_ncg
from torch.autograd import grad
from erasure.model.graphs.SGC import SGC
from sklearn import preprocessing
from numpy.linalg import norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from typing import Optional
import torch.nn.functional as F
from erasure.core.factory_base import get_instance_kvargs
from torch_geometric.utils import degree

class IDEA(TorchUnlearner):
    def init(self):
        """
        Initializes the IDEA class with global and local contexts.
        """

        super().init()
        self.iteration = self.local.config['parameters']['iteration']
        self.scale = self.local.config['parameters']['scale']
        self.gaussian_std = self.local.config['parameters']['gaussian_std']
        self.gaussian_mean = self.local.config['parameters']['gaussian_mean']
        self.l = self.local.config['parameters']['l']
        self.lambda_param = self.local.config['parameters']['lambda']
        self.c = self.local.config['parameters']['c']
        self.lambda_edge_unlearn = self.local.config['parameters']['lambda_edge_unlearn']
        self.gamma_2 = self.local.config['parameters']['gamma_2']
        self.M = self.local.config['parameters']['M']
        self.c1 = self.local.config['parameters']['c1']
        self.damp = self.local.config['parameters']['damp']
        self.training_set = self.local.config['parameters']['training_set']
        self.hops = len(self.predictor.model.hidden_channels) + 1
        self.deleted_nodes = np.array([])     
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([]) 


    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['iteration'] = self.local.config['parameters'].get("iteration", 100)
        self.local.config['parameters']['scale'] = self.local.config['parameters'].get("scale", 500)
        self.local.config['parameters']['gaussian_std'] = self.local.config['parameters'].get("gaussian_std", 0.6)
        self.local.config['parameters']['gaussian_mean'] = self.local.config['parameters'].get("gaussian_mean", 0)
        self.local.config['parameters']['l'] = self.local.config['parameters'].get("l", 0.25)
        self.local.config['parameters']['lambda'] = self.local.config['parameters'].get("lambda", 1)
        self.local.config['parameters']['c'] = self.local.config['parameters'].get("c", 0.5)
        self.local.config['parameters']['lambda_edge_unlearn'] = self.local.config['parameters'].get("lambda_edge_unlearn", 1)
        self.local.config['parameters']['gamma_2'] = self.local.config['parameters'].get("gamma_2", 1)
        self.local.config['parameters']['M'] = self.local.config['parameters'].get("M", 0.25)
        self.local.config['parameters']['c1'] = self.local.config['parameters'].get("c1", 1)
        self.local.config['parameters']['damp'] = self.local.config['parameters'].get("damp", 0.01)
        self.local.config['parameters']['training_set'] = self.local.config['parameters'].get("training_set", "train")




    def __unlearn__(self):
        """
        An implementation of the IDEA unlearning algorithm.
                
        Codebase taken from the original implementation: https://github.com/yushundong/IDEA/tree/main
        """

        self.info(f'Starting IDEA')

        self.hidden = self.predictor.model.hidden_channels

        ## ERASURE: get train and test masks
        num_nodes = self.dataset.partitions['all'].num_nodes
        self.num_train_nodes = len(self.dataset.partitions['train'])
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask[self.dataset.partitions['test']] = True
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.train_mask[self.dataset.partitions['train']] = True
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask[self.dataset.partitions['validation']] = True


        self.labels = self.dataset.partitions['all'][0][1]
        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels, device=self.device)
        else:
            self.labels = self.labels.to(self.device)
            
        og_graph =  self.dataset.partitions['all'] 
        edges_to_forget = self.dataset.partitions['forget']
        

        if self.removal_type == 'node':
            gold_training_set = self.dataset.partitions[self.training_set]
            new_graph, remapped_partitions = og_graph.revise_graph_nodes(gold_training_set, self.dataset.partitions)
        if self.removal_type == 'edge':
            new_graph = og_graph.revise_graph_edges(edges_to_forget, remove=True)
            remapped_partitions = copy.deepcopy(self.dataset.partitions)


        """Construct the result_tuple like they do in their code"""

        #out1 = self.model.forward_once(self.data, self.edge_weight)
        #out2 = self.model.forward_once_unlearn(self.data, self.edge_weight_unlearn)
        #out1 is a forward pass of the model on all data
        #out2 is the forward pass of the model if the data was removed beforehand

        self.x = og_graph[0][0].x.to(self.device)
        self.edge_index = og_graph[0][0].edge_index.to(self.device)
        self.x_unlearned = new_graph[0][0].x.to(self.device)
        self.edge_index_unlearned = new_graph[0][0].edge_index.to(self.device)


        self.find_k_hops(self.dataset.partitions['train'], self.edge_index)

        self.originally_trained_model_params = [p for p in self.predictor.model.parameters()]


        out1 = self.predictor.model(self.x, self.edge_index)
        out2 = self.predictor.model(self.x_unlearned, self.edge_index_unlearned)

        y = self.labels[self.train_mask]

        unlearn_info = (self.deleted_nodes, self.feature_nodes, self.influence_nodes)

        if self.removal_type == "edge":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[2]] = True
            mask2 = mask1
        if self.removal_type == "node":
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[0]] = True
            mask1[unlearn_info[2]] = True
            mask2 = np.array([False] * out2.shape[0])
            mask2[unlearn_info[2]] = True
        if self.removal_type in ['feature', 'partial_feature']:
            mask1 = np.array([False] * out1.shape[0])
            mask1[unlearn_info[1]] = True
            mask1[unlearn_info[2]] = True
            mask2 = mask1

        loss = F.cross_entropy(out1[self.train_mask], self.labels[self.train_mask], reduction='sum')
        loss1 = F.cross_entropy(out1[mask1], self.labels[mask1], reduction='sum')
        loss2 = F.cross_entropy(out2[mask2], self.labels[mask2], reduction='sum')
        model_params = [p for p in self.predictor.model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        self.loss_all = loss

        
        result_tuple = (grad_all, grad1, grad2)
        params_change = self.approxi(result_tuple)

        my_bound, certified_edge_bound, certified_edge_worst_bound, actual_diff = self.alpha_computation(params_change)

        """Use influence to update the model"""

        parameters = [p for p in self.predictor.model.parameters() if p.requires_grad]

        with torch.no_grad():
            delta = [p + infl for p, infl in zip(parameters, params_change)]
            for i, p in enumerate(parameters):
                p.copy_(delta[i])
    
        ## Remove edges from the graph associated with the predictor
        og_graph =  self.dataset.partitions['all'] 


        new_graph = og_graph.revise_graph_edges(edges_to_forget, remove=True)
        remapped_partitions = copy.deepcopy(self.dataset.partitions)
        
        self.predictor.dataset.partitions = {}
        self.predictor.dataset.partitions['all'] = new_graph

        if self.removal_type == 'node':
            self.predictor.dataset.partitions['train'] = list(range(new_graph.num_nodes))
        if self.removal_type == 'edge':
            self.predictor.dataset.partitions['train'] = remapped_partitions['train']

        self.predictor.dataset.partitions['test'] = remapped_partitions['test']
        self.predictor.dataset.partitions['forget'] = remapped_partitions['forget']
        
        return self.predictor

    def find_k_hops(self, unique_nodes, edge_index):
        edge_src = edge_index[0].cpu().numpy()
        edge_dst = edge_index[1].cpu().numpy()

        influenced_nodes = np.array(unique_nodes, dtype=np.int64)

        for _ in range(self.hops):
            target_nodes_location = np.isin(edge_src, influenced_nodes)
            neighbor_nodes = edge_dst[target_nodes_location]
            influenced_nodes = np.unique(np.concatenate((influenced_nodes, neighbor_nodes)))

        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)

        if self.removal_type in ['feature', 'partial_feature']:
            self.feature_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        elif self.removal_type == 'node':
            self.deleted_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        elif self.removal_type == 'edge':
            self.influence_nodes = influenced_nodes



    def approxi(self, res_tuple):
        '''
        res_tuple == (grad_all, grad1, grad2)
        '''
        v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))

        v_norms = [torch.norm(v_).item() for v_ in v]
        
        for _ in range(self.iteration):

            model_params  = [p for p in self.predictor.model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)

            
            with torch.no_grad():
                h_estimate    = [ v1 + (1-self.damp)*h_estimate1 - hv1/self.scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

        params_change = [h_est / self.scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        self.params_esti = params_esti

        # add Gaussian Noise
        gaussian_noise = [(torch.randn(item.size()) * self.gaussian_std + self.gaussian_mean).to(item.device) for item in params_esti]
        params_esti = [item1 + item2 for item1, item2 in zip(gaussian_noise, params_esti)]

        return params_change
    
    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)
        
        return_grads = grad(element_product,model_params,create_graph=True)

        return return_grads
    
    def alpha_computation(self, params_change):

        # bound given by alpha 1 + alpha 2
        m = float(int(len(self.dataset.partitions['forget'])))
        t = self.influence_nodes.shape[0]

        #self.certification_alpha1 = (m * self.args['l'] + (m ** 2 + self.args['l'] ** 2 + 4 * self.args['lambda'] * len(self.train_indices) * t * self.args['c']) ** 0.5) / (self.args['lambda'] * len(self.train_indices))
        self.certification_alpha1 = (m * self.l + (m ** 2 + self.l ** 2 + 4 * self.lambda_param * len(self.dataset.partitions['train']) * t * self.c) ** 0.5) / (self.lambda_param * len(self.dataset.partitions['train']))
        params_change_flatten = [item.flatten() for item in params_change]
        self.certification_alpha2 = torch.norm(torch.cat(params_change_flatten), 2)
        self.info("Certification related stats:  ")
        self.info("certification_alpha1 (bound): %s" % self.certification_alpha1)
        self.info("certification_alpha2 (l2 of modification): %s" % self.certification_alpha2)
        total_bound = self.certification_alpha1 + self.certification_alpha2
        self.info("total bound given by alpha1 + alpha2: %s" % total_bound)

        # bound given by certified edge
        certified_edge_bound = self.certification_alpha2 ** 2 * self.M / self.l
        self.info("data-dependent bound given by certified edge: %s" % certified_edge_bound)

        # worset bound given by certified edge  
        certified_edge_worst_bound = self.M * (self.gamma_2 ** 2) * (self.c1 ** 2) * (t ** 2) / ((self.lambda_edge_unlearn ** 4) * len(self.dataset.partitions['train']))
        self.info("worst bound given by certified edge: %s" % certified_edge_worst_bound)

        # recover the originally trained model
        idx = 0
        for p in self.predictor.model.parameters():
            p.data = self.originally_trained_model_params[idx].clone()
            idx = idx + 1

        # continue optimizing the model with data already updated
        self.train_model_continue((self.deleted_nodes, self.feature_nodes, self.influence_nodes))


        # actual difference
        original_params = [p.flatten() for p in self.params_esti]
        retraining_model_params = [p.flatten() for p in self.predictor.model.parameters() if p.requires_grad]
        actual_param_difference = torch.norm((torch.cat(original_params) - torch.cat(retraining_model_params)), 2).detach()
        self.info("actual params difference: %s" % actual_param_difference)


        return total_bound.cpu().numpy(), certified_edge_bound.cpu().numpy(), certified_edge_worst_bound, actual_param_difference.cpu().numpy()


    def train_model_continue(self, unlearn_info=None):
        self.info("training model continue")
        self.predictor.model.train()


        optimizer = self.predictor.optimizer
        lr = optimizer.param_groups[0]['lr']
 
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 1e2


        #optimizer = torch.optim.Adam(self.predictor.model.parameters(), lr=(self.predictor.lr / 1e2), weight_decay=self.decay) 
        
        training_mask = self.train_mask
        if unlearn_info[0] is not np.array([]):
            training_mask[unlearn_info[0]] = False



        for epoch in range(int(self.predictor.epochs * 0.1)):
            optimizer.zero_grad()
            out = self.predictor.model(self.x_unlearned, self.edge_index_unlearned)

            loss = F.nll_loss(out[training_mask], self.labels[training_mask])
            loss.backward()
            optimizer.step()