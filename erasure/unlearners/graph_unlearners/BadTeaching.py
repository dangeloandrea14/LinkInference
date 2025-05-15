from erasure.unlearners.torchunlearner import TorchUnlearner

import torch
import torch.nn.functional as F

import numpy as np
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data
from erasure.utils.config.local_ctx import Local
from erasure.core.factory_base import get_instance_kvargs

class BadTeaching(TorchUnlearner):
    def init(self):
        """
        Initializes the Bad Teaching class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs']
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget']
        self.transform = self.local.config['parameters']['transform']
        self.batch_size = self.dataset.batch_size
        self.KL_temperature = self.local.config['parameters']['KL_temperature']
        self.hops = len(self.predictor.model.hidden_channels) + 1
        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})

        self.cfg_bad_teacher = self.local.config['parameters']['bad_teacher']
        self.current_bt = Local(self.cfg_bad_teacher)
        self.current_bt.dataset = self.dataset

    def UnlearnerLoss(self, output, labels, gt_logits, bt_logits, KL_temperature):
        labels = torch.unsqueeze(labels, dim = 1)
        
        gt_output = F.softmax(gt_logits / KL_temperature, dim=1)
        bt_output = F.softmax(bt_logits / KL_temperature, dim=1)

        # label 1 means forget sample
        # label 0 means retain sample
        overall_teacher_out = labels * bt_output + (1-labels)*gt_output
        overall_teacher_out = F.normalize(overall_teacher_out, p=1, dim=1)
        student_out = F.log_softmax(output / KL_temperature, dim=1)


        return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

    def unlearning_step(self, model, good_teacher, bad_teacher, graph, optimizer, 
                device, KL_temperature):
        losses = []


        x = graph[0][0].x
        edge_index = graph[0][0].edge_index


        with torch.no_grad():
            gt_logits = good_teacher(x, edge_index)[self.mask]
            bt_logits = bad_teacher(x, edge_index)[self.mask]

        output = self.predictor.model(x, edge_index)[self.mask]

        optimizer.zero_grad()

        loss = self.UnlearnerLoss(output = output, labels=self.labels, gt_logits=gt_logits,
                        bt_logits=bt_logits, KL_temperature=KL_temperature)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        
        return np.mean(losses)

    def __unlearn__(self):
        """
        An implementation of the Bad Teaching unlearning algorithm proposed in the following paper:
        "Chundawat, V.S., Tarun, A.K., Mandal, M. and Kankanhalli, M., 2023, June. Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7210-7217)."
        
        Codebase taken from the original implementation: https://github.com/vikram2000b/bad-teaching-unlearning
        """

        self.info(f'Starting BadTeaching with {self.epochs} epochs')

        self.bad_teacher = self.global_ctx.factory.get_object(self.current_bt)

        og_graph =  self.dataset.partitions['all'] 

        self.x = og_graph[0][0].x
        self.edge_index = og_graph[0][0].edge_index
        self.labels = self.dataset.partitions['all'][0][1]
        self.labels = torch.tensor(self.labels)

        self.retain_set = self.dataset.partitions[self.ref_data_retain]
        self.forget_set = self.dataset.partitions[self.ref_data_forget]

        num_nodes = self.x.size(0)
        all_nodes = torch.arange(num_nodes)

        if self.removal_type == 'edge':
            self.forget_set = self.infected_nodes(self.forget_set, self.hops)
            self.retain_set = [node for node in all_nodes if node not in self.forget_set]

        self.mask = torch.tensor(self.forget_set + self.retain_set, dtype=torch.long)

        # Binary labels: 1 for forget, 0 for retain
        self.labels = torch.cat([
            torch.ones(len(self.forget_set), dtype=torch.float32),
            torch.zeros(len(self.retain_set), dtype=torch.float32)])

        good_teacher = copy.deepcopy(self.predictor.model)

        good_teacher.eval()
        self.bad_teacher.model.eval()        

        for epoch in range(self.epochs):
            loss = self.unlearning_step(model = self.predictor.model, good_teacher= good_teacher, 
                            bad_teacher=self.bad_teacher.model, graph=og_graph, 
                            optimizer=self.predictor.optimizer, device=self.device, KL_temperature=self.KL_temperature)
            self.info(f'Epoch {epoch} Unlearning Loss {loss}')
            
        return self.predictor

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
    

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget
        self.local.config['parameters']['transform'] = self.local.config['parameters'].get("transform", None) # Default transformation applied to the data is None
        self.local.config['parameters']['KL_temperature'] = self.local.config['parameters'].get("KL_temperature", 1.0) # Default KL temperature is 1.0
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam

        if 'bad_teacher' not in self.local.config['parameters']: 
            self.local.config['parameters']['bad_teacher'] = copy.deepcopy(self.global_ctx.config.predictor) # Default bad teacher has the same configuration of the original predictor trained for 0 epochs
            self.local.config['parameters']['bad_teacher']['parameters']['cached'] = False
            self.local.config['parameters']['bad_teacher']['parameters']['epochs'] = 0
            self.local.config['parameters']['bad_teacher']['parameters']['training_set'] = self.local.config['parameters']['ref_data_retain']
        else: 
            self.local.config['parameters']['predictor']['parameters']['training_set'] = self.local.config['parameters']['predictor']['parameters'].get('training_set',self.local.config['parameters']['ref_data_retain'])

        self.local.config['parameters']['bad_teacher']['parameters']['cached'] = self.local.config['parameters']['bad_teacher']['parameters'].get('cached',False)

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data, transform):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.transform = transform
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            if isinstance(self.forget_data[index], dict):
                x = self.forget_data[index].values()
                x = self.transform(list(x)) if self.transform else list(x)
                x = torch.tensor(x)
            else:
                x = self.transform(self.forget_data[index][0]) if self.transform else self.forget_data[index][0]
            y = 1
            return x,y
        else:
            if isinstance(self.retain_data[index - self.forget_len], dict):
                x = self.retain_data[index - self.forget_len].values()
                x = self.transform(list(x)) if self.transform else list(x)
                print(x)
                x = torch.tensor(x)
            else:
                x = self.transform(self.retain_data[index - self.forget_len][0]) if self.transform else self.retain_data[index - self.forget_len][0]
            y = 0
            return x,y