from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import nn

from copy import deepcopy

from erasure.core.factory_base import get_instance_kvargs

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class Scrub(GraphUnlearner):

    def init(self):
        """
        Initializes the scrub class with global and local contexts.
        """
        super().init()
        
        self.epochs = self.local.config['parameters']['epochs']  
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain'] 
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget']
        self.T = self.local.config['parameters']['T']  

        self.criterion_div = DistillKL(self.T)

        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})


    def __unlearn__(self):
        """
        An implementation of the SCRUB unlearning algorithm proposed in the following paper:
        "Kurmanji, M., Triantafillou, P., Hayes, J. and Triantafillou, E., 2024. Towards unbounded machine unlearning. Advances in neural information processing systems, 36."
        
        Codebase taken from this implementation: https://github.com/ndb796/MachineUnlearning
        """


        self.info(f'Starting scrub with {self.epochs} epochs')

        self.retain_set = self.dataset.partitions[self.ref_data_retain]
        self.forget_set = self.dataset.partitions[self.ref_data_forget]

        self.teacher = deepcopy(self.predictor.model)

        num_nodes = self.x.size(0)

        if self.removal_type == 'edge':
            self.forget_set = self.infected_nodes(self.forget_set, self.hops)
            forget_set_s = set(self.forget_set)
            self.retain_set = [n for n in range(num_nodes) if n not in forget_set_s]

        total_loss_retain = 0
        total_loss_forget = 0

        for epoch in range(self.epochs):
            self.predictor.model.train()
            self.teacher.eval()

            #Training with retain data.
            self.predictor.optimizer.zero_grad()

            #Forward pass: Student
            outputs_retain_student = self.predictor.model(self.x, self.edge_index)[self.retain_set]
            labels_retain = self.labels[self.retain_set]

            #Forward pass: Teacher
            with torch.no_grad():
                outputs_retain_teacher = self.teacher(self.x,self.edge_index)[self.retain_set]

            #Loss computation
            loss_cls = self.predictor.loss_fn(outputs_retain_student, labels_retain)
            loss_div_retain = self.criterion_div(outputs_retain_student, outputs_retain_teacher)

            loss = loss_cls + loss_div_retain

            # Update total loss and accuracy for retain data.
            total_loss_retain += loss.item()

            # Backward pass
            loss.backward()

            self.predictor.optimizer.step()

            # Training with forget data.


            self.predictor.optimizer.zero_grad()

            #Forward pass: student
            outputs_forget_student = self.predictor.model(self.x,self.edge_index)[self.forget_set]

            #Forward pass: Teacher
            with torch.no_grad():
                outputs_forget_teacher =self.teacher(self.x,self.edge_index)[self.forget_set]

            #We want to maximize the divergence for the forget data.
            loss_div_forget = -self.criterion_div(outputs_forget_student, outputs_forget_teacher)

            total_loss_forget += loss_div_forget.item()

            #Backward pass
            loss_div_forget.backward()
            self.predictor.optimizer.step()

            avg_loss_retain = total_loss_retain / len(self.retain_set)

            avg_loss_forget = total_loss_forget / len(self.forget_set)

            self.info(f'scrub - epoch = {epoch} ---> loss_retain = {avg_loss_retain:.4f} - loss_forget = {avg_loss_forget:.4f}')

                
        return self.predictor
    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 1)  # Default 1 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget
        self.local.config['parameters']['T'] = self.local.config['parameters'].get("T", 4.0)  # Default temperature is 4.0
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam