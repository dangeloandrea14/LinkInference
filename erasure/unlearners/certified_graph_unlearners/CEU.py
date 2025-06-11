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

class CEU(TorchUnlearner):
    def init(self):
        """
        Initializes the CEU class with global and local contexts.
        """

        super().init()
        self.cg_approx = self.local.config['parameters']['cg_approx']
        self.transductive_edge = self.local.config['parameters']['transductive_edge']
        self.lam = self.local.config['parameters']['lam']
        self.damping = self.local.config['parameters']['damping']


    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['cg_approx'] = self.local.config['parameters'].get("cg_approx", True)
        self.local.config['parameters']['transductive_edge'] = self.local.config['parameters'].get("transductive_edge", True)
        self.local.config['parameters']['lam'] = self.local.config['parameters'].get("lam", 1e-4) #same as CGU
        self.local.config['parameters']['damping'] = self.local.config['parameters'].get("damping", 0) 


    def __unlearn__(self):
        """
        An implementation of the CEU unlearning algorithm proposed in the following paper:
        "Wu et. al, Certified Edge Unlearning for Graph Neural Networks"
                
        Codebase taken from the original implementation: https://github.com/kunwu522/certified_edge_unlearning
        """

        self.info(f'Starting CEU (Certified Edge Unlearning)')

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
        self.labels = torch.tensor(self.labels).to(self.device)
        og_graph =  self.dataset.partitions['all'] 

        # Retraining and unlearning
        edges_to_forget = self.dataset.partitions['forget']

        data_prime = og_graph.revise_graph_edges(edges_to_forget, remove=True)

        influence = self.unlearn_ceu(self.predictor.model, og_graph, data_prime, edges_to_forget)

        """Use influence to update the model"""


        parameters = [p for p in self.predictor.model.parameters() if p.requires_grad]

        with torch.no_grad():
            delta = [p + infl for p, infl in zip(parameters, influence)]
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
        
        return self.predictor

    def unlearn_ceu(self, model, data, data_prime, edges_to_forget, return_bound=False):
        _model = copy.deepcopy(model)
        # parameters = [p for p in _model.parameters() if p.requires_grad]
        infected_nodes = self.infected_nodes(edges_to_forget, len(self.hidden) + 1)
        infected_nodes = torch.tensor(infected_nodes, device=self.device)

        
        infected_labels = self.labels[infected_nodes]
        #infected_labels = torch.tensor(data.labels_of_nodes(infected_nodes.cpu().tolist()), device=self.device)
        # print('Number of nodes:', len(infected_nodes))

        t0 = time.time()
        if self.cg_approx:
            infl, bound = self.influence(_model, data, data_prime, infected_nodes, infected_labels, device=self.device, use_torch=True, return_norm=return_bound)
            # print('------------------------------------------')
            # print('infl norm:', [torch.norm(i) for i in infl])
            # print('------------------------------------------')
            # print('infl:', infl)
            # train_loader = DataLoader(data.train_set, shuffle=True, batch_size=args.batch)
            # cg_pert_influence = CGPertuabtionInfluence(
            #     model=model,
            #     objective=EdgeInfluence(data.edges, data_prime.edges, device),
            #     train_loader=train_loader,
            #     device=device,
            #     damp=0.01,
            #     maxiter=200
            # )
            # I = cg_pert_influence.influences(infected_indices)
            # infl = [- (1 / data.num_train_nodes) * i for i in I]
        
            
        else:
            """ Reproducing this code is impossible given the authors' code, and as such it is not available. """
            print("Not available.")
            #edge_index = torch.tensor(data.edges, device=self.device).t()
            #edge_index_prime = torch.tensor(data_prime.edges, device=self.device).t()
            #infl = s_infected_nodes(model, infected_nodes, infected_labels,
            #                        edge_index, edge_index_prime, data.train_set, self.device, recusion_depth=4000)

        # print('!!!!!!', torch.linalg.norm(torch.cat([ii.view(-1) for ii in infl])))
        return infl


    def infected_nodes(self, edges_to_forget, hops):
        import networkx as nx

        G = nx.Graph()
        all_edges = self.dataset.partitions['all'][0][0].edge_index.t().tolist()  
        G.add_edges_from(all_edges)

        edge_nodes = set()
        for u, v in edges_to_forget:
            edge_nodes.add(u)
            edge_nodes.add(v)

        infected = set()
        for node in edge_nodes:
            if node in G:
                neighbors = nx.single_source_shortest_path_length(G, node, cutoff=hops).keys()
                infected.update(neighbors)

        return list(infected)
        
    def influence(self, model, data, data_prime,
                infected_nodes, infected_labels,
                use_torch=True, device=torch.device('cpu'), return_norm=False):
        parameters = [p for p in model.parameters() if p.requires_grad]

        self.x = data[0][0].x.to(self.device)
        all_edges = torch.tensor( data[0][0].edge_index.t().tolist() )
        unlearned_edges = data_prime[0][0].edge_index.t().tolist() 

        if self.transductive_edge:
            edge_index = torch.tensor(all_edges, device=device).t().to(self.device)
            edge_index_prime = torch.tensor(unlearned_edges, device=device).t().to(self.device)

        p = 1 / (self.num_train_nodes)

        # t1 = time.time()
        model = model.to(self.device)
        model.eval()

        y_hat = model(self.x, edge_index_prime)[infected_nodes]
        #y_hat = model(infected_nodes, edge_index_prime)
        loss1 = self.loss_sum(y_hat, infected_labels)
        g1 = grad(loss1, parameters)
        # print(f'CEU, grad new duration: {(time.time() - t1):.4f}.')

        # t1 = time.time()
        y_hat = model(self.x, edge_index)[infected_nodes]
        loss2 = self.loss_sum(y_hat, infected_labels)
        g2 = grad(loss2, parameters)
        # print(f'CEU, grad old duration: {(time.time() - t1):.4f}.')

        v = [gg1 - gg2 for gg1, gg2 in zip(g1, g2)]
        # ihvp = inverse_hvp(data, model, edge_index, v, args.damping, device)
        # ihvp, (cg_grad, status) = inverse_hvp_cg_sep(data, model, edge_index, v, args.damping, device, use_torch)


        # t1 = time.time()
        if isinstance(model, SGC):
            ihvp, (cg_grad, status) = self.inverse_hvp_cg_sgc(data, model, edge_index, v, self.lam, device)
        else:
            if len(self.hidden) == 0:
                ihvp, (cg_grad, status) = self.inverse_hvp_cg(data, model, edge_index, v, self.damping, device, use_torch)
            else:
                ihvp, (cg_grad, status) = self.inverse_hvp_cg_sep(data, model, edge_index, v, self.damping, device, use_torch)
        # print(f'CEU, hessian inverse duration: {(time.time() - t1):.4f}.')
        
        I = [- p * i for i in ihvp]

        # print('-------------------------')
        # print('Norm:', [torch.norm(ii) for ii in I])
        # print('v norm:', [torch.norm(vv) for vv in v])
        # print('CG gradient:', cg_grad)
        # print('status:', status)
        # print('-------------------------')

        return I, (torch.norm(torch.cat([i.view(-1) for i in ihvp])) ** 2) * (p ** 2)

        # return inverse_hvps, loss, status

    def inverse_hvp_cg(self,data, model, edge_index, vs, damping, device, use_torch=True):
        x_train = torch.tensor(self.dataset.partitions['train'], device=device)
        y_train = torch.tensor( self.labels[self.dataset.partitions['train']], device=device)

        #x_train = torch.tensor(data.train_set.nodes, device=device)
        #y_train = torch.tensor(data.train_set.labels, device=device)
        inverse_hvp = []
        status = []
        cg_grad = []
        # for i, (v, p) in enumerate(zip(vs, model.parameters())):
        sizes = [p.size() for p in model.parameters() if p.requires_grad]
        # v = to_vector(vs)
        v = torch.cat([vv.view(-1) for vv in vs])
        i = None
        fmin_loss_fn = self._get_fmin_loss_fn(v, model=model,
                                        x_train=x_train, y_train=y_train,
                                        edge_index=edge_index, damping=damping,
                                        sizes=sizes, p_idx=i, device=device,
                                        use_torch=use_torch)
        fmin_grad_fn = self._get_fmin_grad_fn(v, model=model,
                                        x_train=x_train, y_train=y_train,
                                        edge_index=edge_index, damping=damping,
                                        sizes=sizes, p_idx=i, device=device,
                                        use_torch=use_torch)
        fmin_hvp_fn = self._get_fmin_hvp_fn(v, model=model,
                                    x_train=x_train, y_train=y_train,
                                    edge_index=edge_index, damping=damping,
                                    sizes=sizes, p_idx=i, device=device,
                                    use_torch=use_torch)
        cg_callback = self._get_cg_callback(v, model=model,
                                    x_train=x_train, y_train=y_train,
                                    edge_index=edge_index, damping=damping,
                                    sizes=sizes, p_idx=i, device=device,
                                    use_torch=use_torch)

        # res = minimize(fmin_loss_fn, v.view(-1), method='cg', max_iter=100)
        if use_torch:
            res = fmin_cg(
                f=fmin_loss_fn,
                x0=self.to_vector(vs),
                fprime=fmin_grad_fn,
                gtol=1E-4,
                # norm='fro',
                # callback=cg_callback,
                disp=False,
                full_output=True,
                maxiter=100,
            )
            # inverse_hvp.append(to_list(res[0], sizes, device)[0])
            inverse_hvp = self.to_list(torch.from_numpy(res[0]), sizes, device)
            cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
            status = res[4]
            # print('-----------------------------------')
            # cg_grad.append(np.linalg.norm(fmin_grad_fn(res[0]), ord=np.inf))

        else:
            res = fmin_ncg(
                f=fmin_loss_fn,
                x0=self.to_vector(vs),
                fprime=fmin_grad_fn,
                fhess_p=fmin_hvp_fn,
                # callback=cg_callback,
                avextol=1e-5,
                disp=False,
                full_output=True,
                maxiter=100)
            # inverse_hvp.append(to_list(res[0], sizes, device)[0])
            inverse_hvp = self.to_list(torch.from_numpy(res[0]), sizes, device)
            # print('-----------------------------------')
            status = res[5]
            cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))

        #     x, _err, d = fmin_l_bfgs_b(
        #         func=fmin_loss_fn,
        #         x0=to_vector(v),
        #         fprime=fmin_grad_fn,
        #         iprint=0,
        #     )
        #     inverse_hvp.append(to_list(x, sizes, device)[0])
        #     status.append(d['warnflag'])
        #     err += _err.item()
        # print('error:', err, status)
        return inverse_hvp, (cg_grad, status)

    def inverse_hvp_cg_sgc(self,data, model, edge_index, vs, lam, device):
        w = [p for p in model.parameters() if p.requires_grad][0]

        x_train = torch.tensor(self.dataset.partitions['train'], device=device)
        y_train = torch.tensor( self.labels[self.dataset.partitions['train']], device=device)
        inverse_hvp = []
        status = []
        cg_grad = []
        sizes = [p.size() for p in model.parameters() if p.requires_grad]
        v = torch.cat([vv.view(-1) for vv in vs])
        i = None
        fmin_loss_fn = self._get_fmin_loss_fn_sgc(v, model=model, w=w, lam=lam,
                                            nodes=x_train, labels=y_train,
                                            edge_index=edge_index, device=device)
        fmin_grad_fn = self._get_fmin_grad_fn_sgc(v, model=model, w=w, lam=lam,
                                            nodes=x_train, labels=y_train,
                                            edge_index=edge_index, device=device)
        fmin_hvp_fn = self._get_fmin_hvp_fn_sgc(v, model=model, w=w, lam=lam,
                                            nodes=x_train, labels=y_train,
                                            edge_index=edge_index, device=device)
        
        res = fmin_ncg(
            f=fmin_loss_fn,
            x0=self.to_vector(vs),
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp_fn,
            # fhess=fmin_fhess_fn,
            # callback=cg_callback,
            avextol=1e-5,
            disp=False,
            full_output=True,
            maxiter=100)
        # inverse_hvp.append(to_list(res[0], sizes, device)[0])
        inverse_hvp = self.to_list(torch.from_numpy(res[0]), sizes, device)
        # print('-----------------------------------')
        status = res[5]
        cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
        return inverse_hvp, (cg_grad, status)

        
    def inverse_hvp_cg_sep(self, data, model, edge_index, vs, damping, device, use_torch=True):
        x_train = torch.tensor(self.dataset.partitions['train'], device=device)
        y_train = torch.tensor( self.labels[self.dataset.partitions['train']], device=device)
        inverse_hvp = []
        status = []
        cg_grad = []
        
        parameters = [p for p in model.parameters() if p.requires_grad]
        for i, (v, p) in enumerate(zip(vs, parameters)):
            sizes = [p.size()]
            v = v.view(-1)
            fmin_loss_fn = self._get_fmin_loss_fn(v, model=model,
                                            x_train=x_train, y_train=y_train,
                                            edge_index=edge_index, damping=damping,
                                            sizes=sizes, p_idx=i, device=device,
                                            use_torch=use_torch)
            fmin_grad_fn = self._get_fmin_grad_fn(v, model=model,
                                            x_train=x_train, y_train=y_train,
                                            edge_index=edge_index, damping=damping,
                                            sizes=sizes, p_idx=i, device=device,
                                            use_torch=use_torch)
            fmin_hvp_fn = self._get_fmin_hvp_fn(v, model=model,
                                        x_train=x_train, y_train=y_train,
                                        edge_index=edge_index, damping=damping,
                                        sizes=sizes, p_idx=i, device=device,
                                        use_torch=use_torch)
            cg_callback = self._get_cg_callback(v, model=model,
                                        x_train=x_train, y_train=y_train,
                                        edge_index=edge_index, damping=damping,
                                        sizes=sizes, p_idx=i, device=device,
                                        use_torch=use_torch)
            if use_torch:
                res = fmin_cg(
                    f=fmin_loss_fn,
                    x0=self.to_vector(v),
                    fprime=fmin_grad_fn,
                    gtol=1E-4,
                    # norm='fro',
                    # callback=cg_callback,
                    disp=False,
                    full_output=True,
                    maxiter=100,
                )
                inverse_hvp.append(self.to_list(torch.from_numpy(res[0]), sizes, device)[0])
                # inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
                # cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
                # status = res[4]
                # print('-----------------------------------')
                # cg_grad.append(np.linalg.norm(fmin_grad_fn(res[0]), ord=np.inf))

            else:
                res = fmin_ncg(
                    f=fmin_loss_fn,
                    x0=self.to_vector(v),
                    fprime=fmin_grad_fn,
                    fhess_p=fmin_hvp_fn,
                    # callback=cg_callback,
                    avextol=1e-5,
                    disp=False,
                    full_output=True,
                    maxiter=100)
                inverse_hvp.append(self.to_list(torch.from_numpy(res[0]), sizes, device)[0])
                # inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)

            #     x, _err, d = fmin_l_bfgs_b(
            #         func=fmin_loss_fn,
            #         x0=to_vector(v),
            #         fprime=fmin_grad_fn,
            #         iprint=0,
            #     )
            #     inverse_hvp.append(to_list(x, sizes, device)[0])
            #     status.append(d['warnflag'])
            #     err += _err.item()
            # print('error:', err, status)
        return inverse_hvp, (cg_grad, status)
        


    def _get_fmin_loss_fn(self, v, **kwargs):
        device = kwargs['device']

        def get_fmin_loss(x):
            x = torch.tensor(x, dtype=torch.float, device=device)
            hvp = self._mini_batch_hvp(x, **kwargs)
            obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
            return obj.detach().cpu().numpy()

        return get_fmin_loss


    def _get_fmin_grad_fn(self, v, **kwargs):
        device = kwargs['device']

        def get_fmin_grad(x):
            x = torch.tensor(x, dtype=torch.float, device=device)
            hvp = self._mini_batch_hvp(x, **kwargs)
            # return to_vector(hvp - v.view(-1))
            return (torch.cat(hvp, dim=0) - v).cpu().numpy()

        return get_fmin_grad


    def _get_fmin_hvp_fn(self, v, **kwargs):
        device = kwargs['device']

        def get_fmin_hvp(self, x, p):
            p = torch.tensor(p, dtype=torch.float, device=device)
            hvp = self._mini_batch_hvp(p, **kwargs)
            return self.to_vector(hvp)
        return get_fmin_hvp


    def _mini_batch_hvp(self, x, **kwargs):
        model = kwargs['model']
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        edge_index = kwargs['edge_index']
        damping = kwargs['damping']
        device = kwargs['device']
        sizes = kwargs['sizes']
        p_idx = kwargs['p_idx']
        use_torch = kwargs['use_torch']

        x = self.to_list(x, sizes, device)
        if use_torch:
            _hvp,_ = self.hessian_vector_product(model, edge_index, x_train, y_train, x, device, p_idx)
        else:
            model.eval()
            y_hat = model(self.x, edge_index)[self.dataset.partitions['train']]
            loss = model.loss(y_hat, y_train)
            params = [p for p in model.parameters() if p.requires_grad]
            if p_idx is not None:
                params = params[p_idx:p_idx + 1]
            _hvp = self.hvp(loss, params, x)
        # return _hvp[0].view(-1) + damping * x
        return [(a + damping * b).view(-1) for a, b in zip(_hvp, x)]


    def to_list(self, v, sizes, device):
        _v = v
        result = []
        for size in sizes:
            total = reduce(lambda a, b: a * b, size)
            result.append(_v[:total].reshape(size).float().to(device))
            _v = _v[total:]
        return tuple(result)



    def hessian_vector_product(self, model, edge_index, x, y, v, device, p_idx=None):
        parameters = [p for p in model.parameters() if p.requires_grad]
        if p_idx is not None:
            parameters = parameters[p_idx:p_idx+1]

        y_hat = model(self.x, edge_index)[self.dataset.partitions['train']]
        # train_loss = model.loss(y_hat, y)
        train_loss = self.loss_sum(y_hat, y)

        _, train_loss = self._as_tuple(train_loss, "outputs of the user-provided function", "hvp")

        with torch.enable_grad():
            jac = self._autograd_grad(train_loss, parameters, create_graph=True)
            grad_jac = tuple(
                torch.zeros_like(p, requires_grad=True, device=device) for p in parameters
            )
            double_back = self._autograd_grad(jac, parameters, grad_jac, create_graph=True)

        grad_res = self._autograd_grad(double_back, grad_jac, v, create_graph=False)
        hvp = self._fill_in_zeros(grad_res, parameters, False, False, "double_back_trick")
        hvp = self._grad_postprocess(hvp, False)
        return hvp, train_loss[0]
    

    def _autograd_grad(self,outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None):
        # Version of autograd.grad that accepts `None` in outputs and do not compute gradients for them.
        # This has the extra constraint that inputs has to be a tuple
        assert isinstance(outputs, tuple)
        if grad_outputs is None:
            grad_outputs = (None,) * len(outputs)
        assert isinstance(grad_outputs, tuple)
        assert len(outputs) == len(grad_outputs)

        new_outputs: Tuple[torch.Tensor, ...] = tuple()
        new_grad_outputs: Tuple[torch.Tensor, ...] = tuple()
        for out, grad_out in zip(outputs, grad_outputs):
            if out is not None and out.requires_grad:
                new_outputs += (out,)
                new_grad_outputs += (grad_out,)

        if len(new_outputs) == 0:
            # No differentiable output, we don't need to call the autograd engine
            return (None,) * len(inputs)
        else:
            return torch.autograd.grad(new_outputs, inputs, new_grad_outputs, allow_unused=True,
                                    create_graph=create_graph, retain_graph=retain_graph)


    def to_vector(self,v):
        if isinstance(v, tuple) or isinstance(v, list):
            # return v.cpu().numpy().reshape(-1)
            return np.concatenate([vv.cpu().numpy().reshape(-1) for vv in v])
        else:
            return v.cpu().numpy().reshape(-1)

    def _get_fmin_loss_fn_sgc(self,v, **kwargs):
        model = kwargs['model']
        edge_index = kwargs['edge_index']
        w = kwargs['w']
        lam = kwargs['lam']
        nodes = kwargs['nodes']
        labels = kwargs['labels']
        device = kwargs['device']
        y = F.one_hot(labels)

        H = self._hessian_sgc(model, edge_index, w, nodes, y, lam, device)
        # print('H:', H.size())

        def get_fmin_loss(x):
            x = torch.tensor(x, dtype=torch.float, device=device)
            # print(x.size)
            hvp = H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
            obj = 0.5 * torch.dot(hvp, x) - torch.dot(v, x)
            return obj.detach().cpu().numpy()

        return get_fmin_loss

    def _get_fmin_grad_fn_sgc(self,v, **kwargs):
        model = kwargs['model']
        edge_index = kwargs['edge_index']
        w = kwargs['w']
        nodes = kwargs['nodes']
        labels = kwargs['labels']
        device = kwargs['device']
        lam = kwargs['lam']
        y = F.one_hot(labels)

        H = self._hessian_sgc(model, edge_index, w, nodes, y, lam, device)
        def get_fmin_grad(x):
            x = torch.tensor(x, dtype=torch.float, device=device)
            # hvp = _mini_batch_hvp(x, **kwargs)
            # hvp = H.mv(x).view(-1, w.size(0)).t()
            # print(H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).size())
            hvp = H.view(w.size(0), w.size(1), -1).bmm(x.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
            # return to_vector(hvp - v.view(-1))
            return (hvp - v).cpu().numpy()

        return get_fmin_grad


    def _get_fmin_hvp_fn_sgc(self,v, **kwargs):
        model = kwargs['model']
        edge_index = kwargs['edge_index']
        w = kwargs['w']
        nodes = kwargs['nodes']
        labels = kwargs['labels']
        device = kwargs['device']
        lam = kwargs['lam']
        y = F.one_hot(labels)

        H = self._hessian_sgc(model, edge_index, w, nodes, y, lam, device)
        def get_fmin_hvp(x, p):
            p = torch.tensor(p, dtype=torch.float, device=device)
            # hvp = _mini_batch_hvp(p, **kwargs)
            with torch.no_grad():
                hvp = H.view(w.size(0), w.size(1), -1).bmm(p.view(-1, w.size(0)).t().unsqueeze(2)).squeeze().t().flatten()
            return hvp.cpu().numpy()
        return get_fmin_hvp

    def _hessian_sgc(self, model, edge_index, w, nodes, y, lam, device):
        model.eval()
        with torch.no_grad():
            X = model.propagate(edge_index, self.x)
            #slice it to get the propagated features of the train_nodes only
            X = X[nodes]

        with torch.no_grad():
            z = torch.sigmoid(y * X.mm(w.t()))
            D = z * (1 - z)
            H = []
            for k in range(w.size(0)):
                h = X.t().mm(D[:, k].unsqueeze(1) * X)
                h += lam * X.size(0) * torch.eye(X.size(1)).float().to(device)
                H.append(h)
            
        return torch.cat(H).to(device)


    def _get_fmin_loss_fn(self, v, **kwargs):
        device = kwargs['device']

        def get_fmin_loss(x):
            x = torch.tensor(x, dtype=torch.float, device=device)
            hvp = self._mini_batch_hvp(x, **kwargs)
            obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
            return obj.detach().cpu().numpy()

        return get_fmin_loss


    def _get_fmin_grad_fn(self,v, **kwargs):
        device = kwargs['device']

        def get_fmin_grad(x):
            x = torch.tensor(x, dtype=torch.float, device=device)
            hvp = self._mini_batch_hvp(x, **kwargs)
            # return to_vector(hvp - v.view(-1))
            return (torch.cat(hvp, dim=0) - v).cpu().numpy()

        return get_fmin_grad
    
    def _get_cg_callback(self,v, **kwargs):
        device = kwargs['device']

        def cg_callback(x):
            x = torch.tensor(x, dtype=torch.float, device=device)
            hvp = self._mini_batch_hvp(x, **kwargs)
            obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
            # obj = 0.5 * torch.dot(hvp, x) - torch.dot(v.view(-1), x)
            # g = to_vector(hvp - v.view(-1))
            g = (torch.cat(hvp, dim=0) - v).cpu().numpy()
            print(f'loss: {obj:.4f}, grad: {np.linalg.norm(g):.8f}')
        return cg_callback

    def hvp(self, loss, params, vec):
        """
        Compute Hessian-vector product for a scalar loss.

        Args:
            loss (Tensor): scalar loss tensor
            params (list of Tensors): parameters w.r.t. which Hessian is computed
            vec (list of Tensors): vector to multiply with the Hessian (same shapes as params)

        Returns:
            list of Tensors: Hessian-vector product with the same shapes as `params`
        """
        grad1 = torch.autograd.grad(loss, params, create_graph=True)
        grad_dot_vec = sum(torch.sum(g * v) for g, v in zip(grad1, vec))
        hvp = torch.autograd.grad(grad_dot_vec, params, retain_graph=True)
        return hvp
    
    def _as_tuple(self,inp, arg_name=None, fn_name=None):
        # Ensures that inp is a tuple of Tensors
        # Returns whether or not the original inp was a tuple and the tupled version of the input
        if arg_name is None and fn_name is None:
            return self._as_tuple_nocheck(inp)

        is_inp_tuple = True
        if not isinstance(inp, tuple):
            inp = (inp,)
            is_inp_tuple = False

        for i, el in enumerate(inp):
            if not isinstance(el, torch.Tensor):
                if is_inp_tuple:
                    raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                                    " value at index {} has type {}.".format(arg_name, fn_name, i, type(el)))
                else:
                    raise TypeError("The {} given to {} must be either a Tensor or a tuple of Tensors but the"
                                    " given {} has type {}.".format(arg_name, fn_name, arg_name, type(el)))

        return is_inp_tuple, inp
    

    def _as_tuple_nocheck(self,x):
        if isinstance(x, tuple):
            return x
        elif isinstance(x, list):
            return tuple(x)
        else:
            return x,


    def _fill_in_zeros(self,grads, refs, strict, create_graph, stage):
        # Used to detect None in the grads and depending on the flags, either replace them
        # with Tensors full of 0s of the appropriate size based on the refs or raise an error.
        # strict and create graph allow us to detect when it is appropriate to raise an error
        # stage gives us information of which backward call we consider to give good error message
        if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
            raise RuntimeError("Invalid stage argument '{}' to _fill_in_zeros".format(stage))

        res = tuple()
        for i, grads_i in enumerate(grads):
            if grads_i is None:
                if strict:
                    if stage == "back":
                        raise RuntimeError("The output of the user-provided function is independent of "
                                        "input {}. This is not allowed in strict mode.".format(i))
                    elif stage == "back_trick":
                        raise RuntimeError("The gradient with respect to the input is independent of entry {}"
                                        " in the grad_outputs when using the double backward trick to compute"
                                        " forward mode gradients. This is not allowed in strict mode.".format(i))
                    elif stage == "double_back":
                        raise RuntimeError("The jacobian of the user-provided function is independent of "
                                        "input {}. This is not allowed in strict mode.".format(i))
                    else:
                        raise RuntimeError("The hessian of the user-provided function is independent of "
                                        "entry {} in the grad_jacobian. This is not allowed in strict "
                                        "mode as it prevents from using the double backward trick to "
                                        "replace forward mode AD.".format(i))

                grads_i = torch.zeros_like(refs[i])
            else:
                if strict and create_graph and not grads_i.requires_grad:
                    if "double" not in stage:
                        raise RuntimeError("The jacobian of the user-provided function is independent of "
                                        "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
                    else:
                        raise RuntimeError("The hessian of the user-provided function is independent of "
                                        "input {}. This is not allowed in strict mode when create_graph=True.".format(i))

            res += (grads_i,)

        return res
    
    def _grad_postprocess(self,inputs, create_graph):
        # Postprocess the generated Tensors to avoid returning Tensors with history when the user did not
        # request it.
        if isinstance(inputs[0], torch.Tensor):
            if not create_graph:
                return tuple(inp.detach() for inp in inputs)
            else:
                return inputs
        else:
            return tuple(self._grad_postprocess(inp, create_graph) for inp in inputs)


    def loss_sum(self, y_hat, y_true):
        return F.cross_entropy(y_hat, y_true, reduction='sum')