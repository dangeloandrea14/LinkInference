from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner

import torch
import torch.nn.functional as F

import numpy as np
import copy
from erasure.utils.config.local_ctx import Local
from erasure.core.factory_base import get_instance_kvargs

class BadTeaching(GraphUnlearner):
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
        labels = labels.to(self.device)
        bt_output = bt_output.to(self.device)
        gt_output = gt_output.to(self.device)

        overall_teacher_out = labels * bt_output + (1-labels)*gt_output
        overall_teacher_out = F.normalize(overall_teacher_out, p=1, dim=1)
        student_out = F.log_softmax(output / KL_temperature, dim=1)


        return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

    def unlearning_step(self, model, good_teacher, bad_teacher, graph, optimizer, 
                device, KL_temperature):
        losses = []


        x = graph[0][0].x.to(self.device)
        edge_index = graph[0][0].edge_index.to(self.device)


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

        self.retain_set = self.dataset.partitions[self.ref_data_retain]
        self.forget_set = self.dataset.partitions[self.ref_data_forget]

        num_nodes = self.x.size(0)

        if self.removal_type == 'edge':
            self.forget_set = self.infected_nodes(self.forget_set, self.hops)
            forget_set_s = set(self.forget_set)
            self.retain_set = [n for n in range(num_nodes) if n not in forget_set_s]

        self.mask = torch.tensor(self.forget_set + self.retain_set, dtype=torch.long)

        # Binary labels: 1 for forget, 0 for retain
        self.labels = torch.cat([
            torch.ones(len(self.forget_set), dtype=torch.float32),
            torch.zeros(len(self.retain_set), dtype=torch.float32)])

        good_teacher = copy.deepcopy(self.predictor.model).to(self.device)

        good_teacher.eval()
        self.bad_teacher.model.eval()        

        for epoch in range(self.epochs):
            loss = self.unlearning_step(model = self.predictor.model, good_teacher= good_teacher, 
                            bad_teacher=self.bad_teacher.model, graph=og_graph, 
                            optimizer=self.predictor.optimizer, device=self.device, KL_temperature=self.KL_temperature)
            self.info(f'Epoch {epoch} Unlearning Loss {loss}')
            
        return self.predictor

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
