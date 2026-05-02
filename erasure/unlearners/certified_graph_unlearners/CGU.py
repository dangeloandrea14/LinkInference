from erasure.unlearners.torchunlearner import TorchUnlearner
from fractions import Fraction
import torch.optim as optim
import numpy as np
import torch
import time
from collections import Counter
from torch import Tensor
import copy
import math
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

class CGU_edge(TorchUnlearner):
    def init(self):
        """
        Initializes the CGU_edge class with global and local contexts.
        """

        super().init()

        self.std = self.local.config['parameters']['std']
        self.prop_step = self.local_config['parameters']['prop_step']
        self.compare_retrain = self.local_config['parameters']['compare_retrain']
        self.ref_data_retain = self.local.config['parameters']["ref_data_retain"]
        self.ref_data_forget = self.local.config['parameters']["ref_data_forget"]
        self.delta = self.local.config['parameters']['delta']
        self.eps = self.local.config['parameters']['eps']
        self.train_mode = self.local.config['parameters']['train_mode']
        self.y_binary = self.local.config['parameters']['y_binary']
        self.lam = self.local.config['parameters']['lam']
        self.num_steps_optimizer = self.local.config['parameters']['num_steps_optimizer']
        self.optimizer = self.predictor.optimizer
        self.lr = self.predictor.optimizer.param_groups[0]['lr']



    def __unlearn__(self):
        """
        An implementation of the CGU unlearning algorithm proposed in the following paper:
        "Chien, Pan, Milenkovic (2022): Certified Graph Unlearning
                
        Codebase taken from the original implementation: https://github.com/thupchnsky/sgc_unlearn
        """

        self.info(f'Starting CGU_edge')

        data = self.dataset.partitions['all'].data[0]

        if not hasattr(self.predictor.model, 'feat_prop'):
            raise NotImplementedError(
                f"CGU_edge requires a model with a 'feat_prop' propagation layer (e.g. SGC_CGU). "
                f"Got: {type(self.predictor.model).__name__}"
            )
        Propagation = self.predictor.model.feat_prop.to(self.device) ## takes the feat_prop module from the model
        #Propagation = MyGraphConv(K=self.prop_step, add_self_loops=True, device=self.device,
                             #alpha=0, XdegNorm=False, GPR=False).to(self.device)

        ## ERASURE: get train and test masks
        num_nodes = self.dataset.partitions['all'].num_nodes
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[self.dataset.partitions['test']] = True
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[self.dataset.partitions['train']] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[self.dataset.partitions['validation']] = True


        labels = self.dataset.partitions['all'][0][1]

        """
        LABEL HANDLING
        This method notably only works on binary classifiers, so in the case of multiclass classification we need to use 
        k binary - classifiers
        """

        if self.train_mode == 'ovr':
            # multiclass classification
            val_mask = val_mask
            test_mask = test_mask
            y_train = F.one_hot(labels[train_mask]) * 2 - 1
            y_train = y_train.float()
            y_val = labels[val_mask]
            y_test = labels[test_mask]

            y_train, y_val, y_test = y_train.to(self.device), y_val.to(self.device), y_test.to(self.device)

        # save the degree of each node for later use
        edge_index = self.dataset.partitions['all'][0][0].edge_index
        row = edge_index[0]
        deg = degree(row)

        X = data.x.to(self.device)


        #X = self.preprocess_data(data.x).to(self.device)

        # save a copy of X for removal
        X_scaled_copy_guo = X.clone().detach().float()

        """
        INITIAL PROPAGATION
        """

        if self.prop_step > 0:
            X = Propagation(X, edge_index)

        X = X.float()
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]

        X_train, X_val, X_test = X_train.to(self.device), X_val.to(self.device), X_test.to(self.device)


        #X is now the tensor of propagated features.

        forget = self.dataset.partitions['forget']
        retain = self.dataset.partitions['retain']

        """
        Initial training (required as this method needs k binary classifiers for multiclass classification)
        """

        weight = None
        # in our case weight should always be None
        assert weight is None
        opt_grad_norm = 0

        if self.train_mode == 'ovr':
            b = self.std * torch.randn(X_train.size(1), y_train.size(1)).float().to(self.device)

            # train K binary LR models jointly
            w = self.ovr_lr_optimize(X_train, y_train, self.lam, weight, b=b, num_steps=self.num_steps_optimizer, verbose=False,
                                 lr=1.0, wd=5e-4)
            
            # record the opt_grad_norm
            for k in range(y_train.size(1)):
                opt_grad_norm += self.lr_grad(w[:, k], X_train, y_train[:, k], self.lam).norm().cpu()

        if self.train_mode == 'ovr':
            print('Val accuracy = %.4f' % self.ovr_lr_eval(w, X_val, y_val))
            print('Test accuracy = %.4f' % self.ovr_lr_eval(w, X_test, y_test))


        pred = X_val.mm(w).max(1)[1]


        """
        BUDGET COMPUTATION
        """

        # budget for removal
        c_val = self.get_c(self.delta)

        # b_std in the original code always defaults to std, implementation-wise
        # In OVR mode the accumulator sums over all K classes, so the budget must be scaled by K
        budget = self.get_budget(self.std, self.eps, c_val) * y_train.size(1)


        gamma = 1/4  # pre-computed for -logsigmoid loss

        self.info(f'Budget for removal is: {budget}')


        """
        REMOVAL
        """

        ##########
        # removal
        # grad_norm_approx is the data dependent upper bound of residual gradient norm

        grad_norm_approx = torch.zeros((len(forget), 1)).float()

        # get a random permutation for edge indices for each trail
        perm = torch.from_numpy(np.random.permutation(edge_index.shape[1]))
        # Note that all edges are used in training, so we just need to decide the order to remove edges
        # the number of training samples will always be m
        edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)

        X_scaled_copy = X_scaled_copy_guo.clone().detach().float()
        w_approx = w.clone().detach()
        X_old = X.clone().detach().to(self.device)

        #get weights from the model
        #w = self.predictor.model.classifier 

        """
        EDGE-BY-EDGE REMOVAL PROCESS
        """

        acc_removal = torch.zeros((2, len(forget), 1)).float()  # record the acc after removal, 0 for val, 1 for test

        num_retrain = 0 
        grad_norm_approx_sum = 0
        perm_idx = 0

        # start the removal process
        for i in range(len(forget)):
            while (edge_index[0, perm[perm_idx]] == edge_index[1, perm[perm_idx]]) or (not edge_mask[perm[perm_idx]]):
                perm_idx += 1

            edge_mask[perm[perm_idx]] = False
            source_idx = edge_index[0, perm[perm_idx]]
            dst_idx = edge_index[1, perm[perm_idx]]
            # find the other undirected edge
            rev_edge_idx = torch.logical_and(edge_index[0] == dst_idx,
                                             edge_index[1] == source_idx).nonzero().squeeze(-1)
            if rev_edge_idx.size(0) > 0:
                edge_mask[rev_edge_idx] = False

            perm_idx += 1
            # Get propagated features
            if self.prop_step > 0:
                X_new = Propagation(X_scaled_copy, edge_index[:, edge_mask]).to(self.device)
            else:
                X_new = X_scaled_copy
  

            X_test_new = X_new[test_mask]
            X_val_new = X_new[val_mask]

            K = self.get_K_matrix(X_new[train_mask]).to(self.device)
            spec_norm = self.sqrt_spectral_norm(K)


            if self.train_mode == 'ovr':
                # removal from all one-vs-rest models
                X_rem = X_new[train_mask]
                for k in range(y_train.size(1)):
                    assert weight is None
                    y_rem = y_train[:, k]
                    H_inv = self.lr_hessian_inv(w_approx[:, k], X_rem, y_rem, self.lam)
                    # grad_i is the difference
                    grad_old = self.lr_grad(w_approx[:, k], X_old[train_mask], y_rem, self.lam)
                    grad_new = self.lr_grad(w_approx[:, k], X_rem, y_rem, self.lam)
                    grad_i = grad_old - grad_new
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    # update w here. If beta exceed the budget, w_approx will be retrained
                    w_approx[:, k] += Delta
                    grad_norm_approx[i, 0] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
   
                # decide after all classes
                if grad_norm_approx_sum + grad_norm_approx[i, 0] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = self.std * torch.randn(X_train.size(1), y_train.size(1)).float().to(self.device)
                    w_approx = self.ovr_lr_optimize(X_rem, y_train, self.lam, weight, b=b, num_steps=self.num_steps_optimizer, verbose=False,
                                                lr=1.0, wd=5e-4)
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, 0]
                # record acc each round
                acc_removal[0, i, 0] = self.ovr_lr_eval(w_approx, X_val_new, y_val)
                acc_removal[1, i, 0] = self.ovr_lr_eval(w_approx, X_test_new, y_test)

                X_old = X_new.clone().detach()
                
                print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i, 0], acc_removal[1, i, 0]))

        with torch.no_grad():
            self.predictor.model.classifier.weight.copy_(w_approx.T)


        ## Remove edges from the graph associated with the predictor
        og_graph =  self.dataset.partitions['all'] 

        if self.removal_type == 'node':
            new_graph, remapped_partitions = og_graph.revise_graph_nodes(forget, self.dataset.partitions, remove=True)
        if self.removal_type == 'edge':
            new_graph = og_graph.revise_graph_edges(forget, remove=True)
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

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['std'] = self.local.config['parameters'].get("std", 1e-2)
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain') 
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  
        self.local.config['parameters']['prop_step'] = self.local.config['parameters'].get("prop_step", 2)
        self.local.config['parameters']['compare_retrain'] = self.local.config['parameters'].get("compare_retrain", False)
        self.local.config['parameters']['delta'] = self.local.config['parameters'].get("delta", 1e-4) 
        self.local.config['parameters']['eps'] = self.local.config['parameters'].get("eps", 1.0) 
        self.local.config['parameters']['train_mode'] = self.local.config['parameters'].get("train_mode","ovr")
        self.local.config['parameters']['y_binary'] = self.local.config['parameters'].get("y_binary",1)
        self.local.config['parameters']['lam'] = self.local.config['parameters'].get("lam",1e-2)
        self.local.config['parameters']['num_steps_optimizer'] = self.local.config['parameters'].get('num_steps_optimizer',100)


    # hessian of loss wrt w for binary classification
    def lr_hessian_inv(self, w, X, y, lam, batch_size=50000):
        '''
        The hessian here is computed wrt sum.
        input:
            w: (d,)
            X: (n,d)
            y: (n,)
            lambda: scalar
            batch_size: int
        return:
            hessian: (d,d)
        '''
        z = torch.sigmoid(y * X.mv(w))
        D = z * (1 - z)
        H = None
        num_batch = int(math.ceil(X.size(0) / batch_size))
        for i in range(num_batch):
            lower = i * batch_size
            upper = min((i + 1) * batch_size, X.size(0))
            X_i = X[lower:upper]
            if H is None:
                H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
            else:
                H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(self.device)).inverse()






    ### UTILS



    def get_c(self,delta):
        return np.sqrt(2*np.log(1.5/delta))


    def get_budget(self,std, eps, c):
        return std * eps / c


    # K = X^T * X for fast computation of spectral norm
    def get_K_matrix(self,X):
        K = X.t().mm(X)
        return K

    def sqrt_spectral_norm(self,A, num_iters=100):
        '''
        return:
            sqrt of maximum eigenvalue/spectral norm
        '''
        x = torch.randn(A.size(0)).float().to(self.device)
        for i in range(num_iters):
            x = A.mv(x)
            x_norm = x.norm()
            x /= x_norm
        max_lam = torch.dot(x, A.mv(x)) / torch.dot(x, x)
        return math.sqrt(max_lam)
    
    def cgu_ovr_lr_loss(self, w, X, y, lam, weight=None):
        """
        CGU-safe one-vs-rest logistic regression loss.

        Args:
            w: (d, c) weight matrix
            X: (n, d) input features
            y: (n, c) labels, in {-1, 1} (NOT one-hot with 0/1)
            lam: float, L2 regularization coefficient
            weight: optional class weights, shape (c,)

        Returns:
            scalar loss (float)
        """
        # Compute margin: (n, c)
        margin = X @ w  # logits
        z = y * margin  # signed margins

        # Compute binary logistic loss: log(1 + exp(-y * xw))
        loss_per_sample = F.softplus(-z)  # shape (n, c)

        if weight is not None:
            loss_per_sample = loss_per_sample * weight  # broadcasted multiply

        # Mean over all samples and classes
        loss = loss_per_sample.mean()

        # Add regularization
        reg = lam * w.pow(2).sum() / 2
        return loss + reg

    def old_ovr_lr_loss(self,w, X, y, lam, weight=None):
        '''
        input:
            w: (d,c)
            X: (n,d)
            y: (n,c), one-hot
            lambda: scalar
            weight: (c,) / None
        return:
            loss: scalar
        '''
        z = self.batch_multiply(X, w) * y
        if weight is None:
            return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
        else:
            return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2


    def batch_multiply(self,A, B, batch_size=500000):
        B = B.to(self.device)
        if A.is_cuda:
            if len(B.size()) == 1:
                return A.mv(B)
            else:
                return A.mm(B)
        else:
            out = []
            num_batch = int(math.ceil(A.size(0) / float(batch_size)))
            #with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i+1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
            return torch.cat(out, dim=0)
        
    def ovr_lr_optimize(self,X, y, lam, weight=None, b=None, num_steps=100, tol=1e-32, verbose=True, lr=0.5, wd=0, X_val=None, y_val=None):
        '''
        y: (n_train, c). one-hot
        y_val: (n_val,) NOT one-hot
        '''
        # We use random initialization as in common DL literature.
        # w = torch.zeros(X.size(1), y.size(1)).float()
        # init.kaiming_uniform_(w, a=math.sqrt(5))
        # w = torch.autograd.Variable(w.to(device), requires_grad=True)
        # zero initialization
        w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(self.device), requires_grad=True)
        

        def closure():
            if b is None:
                return self.old_ovr_lr_loss(w, X, y, lam, weight)
            else:
                return self.old_ovr_lr_loss(w, X, y, lam, weight) + (b * w).sum() / X.size(0)
        
        '''
        if opt_choice == 'LBFGS':
            optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
        elif opt_choice == 'Adam':
            optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
        else:
            raise("Error: Not supported optimizer.")
        '''

        # CGU trains a convex logistic regression — LBFGS converges in far fewer steps than Adam
        optimizer = torch.optim.LBFGS([w], lr=lr, tolerance_grad=1e-32, tolerance_change=1e-32)
        
        best_val_acc = 0
        w_best = None
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = self.old_ovr_lr_loss(w, X, y, lam, weight)
            if b is not None:
                if weight is None:
                    loss += (b * w).sum() / X.size(0)
                else:
                    loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
            loss.backward()

            '''    
            if opt_choice == 'LBFGS':
                optimizer.step(closure)
            elif opt_choice == 'Adam':
                optimizer.step()
            else:
                raise("Error: Not supported optimizer.")
            '''   
            
            if verbose:
                print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
            
            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(closure)
            else:
                # closure()  # compute gradients but they don't use it in the original implementation
                optimizer.step()

            
            if X_val is not None:
                val_acc = self.ovr_lr_eval(w, X_val, y_val)
                if verbose:
                    print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    w_best = w.clone().detach()
            else:
                w_best = w.clone().detach()

        if w_best is None:
            raise("Error: Training procedure failed")
        return w_best

    # gradient of loss wrt w for binary classification
    def lr_grad(self,w, X, y, lam):
        '''
        The gradient here is computed wrt sum.
        input:
            w: (d,)
            X: (n,d)
            y: (n,)
            lambda: scalar
        return:
            gradient: (d,)
        '''
        z = torch.sigmoid(y * X.mv(w))
        return X.t().mv((z-1) * y) + lam * X.size(0) * w

    def ovr_lr_eval(self,w, X, y):
        '''
        input:
            w: (d,c)
            X: (n,d)
            y: (n,), NOT one-hot
        return:
            loss: scalar
        '''
        pred = X.mm(w).max(1)[1]
        return pred.eq(y).float().mean()

    # evaluate function for binary classification
    def lr_eval(self,w, X, y):
        '''
        input:
            w: (d,)
            X: (n,d)
            y: (n,)
        return:
            prediction accuracy
        '''
        return X.mv(w).sign().eq(y).float().mean()

    def preprocess_data(self,X):
        X_np = X.cpu().numpy()
        X_scaled = X_np  
        return torch.from_numpy(X_scaled).float()




class MyGraphConv(MessagePassing):
    """
    Use customized propagation matrix. Just PX (or PD^{-1}X), no linear layer yet.
    """
    _cached_x: Optional[Tensor]

    def __init__(self, K: int = 1,
                add_self_loops: bool = True, device='cuda',
                alpha=0.5, XdegNorm=False, GPR=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.K = K
        self.add_self_loops = add_self_loops
        self.alpha = alpha
        self.XdegNorm = XdegNorm
        self.GPR = GPR
        self._cached_x = None # Not used
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        self._cached_x = None # Not used

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = self.get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)
        elif isinstance(edge_index, SparseTensor):
            edge_index = self.get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)
        
        if self.XdegNorm:
            # X <-- D^{-1}X, our degree normalization trick
            num_nodes = maybe_num_nodes(edge_index, None)
            row, col = edge_index[0], edge_index[1]
            deg = degree(row).unsqueeze(-1)
            
            deg_inv = deg.pow(-1)
            deg_inv = deg_inv.masked_fill_(deg_inv == float('inf'), 0) 
        
        if self.GPR:
            xs = []
            xs.append(x)
            if self.XdegNorm:
                x = deg_inv * x # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                xs.append(x)
            return torch.cat(xs, dim=1) / (self.K + 1)
        else:
            if self.XdegNorm:
                x = deg_inv * x # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')
    

    def get_propagation(self, edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None, alpha=0.5):
        """
        return:
            P = D^{-\alpha}AD^{-(1-alpha)}.
        """
        fill_value = 2. if improved else 1.
        assert (0 <= alpha) and (alpha <= 1)
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=self.device)
        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_left = deg.pow(-alpha)
        deg_inv_right = deg.pow(alpha-1)
        deg_inv_left.masked_fill_(deg_inv_left == float('inf'), 0)
        deg_inv_right.masked_fill_(deg_inv_right == float('inf'), 0)

        return edge_index, deg_inv_left[row] * edge_weight * deg_inv_right[col]

