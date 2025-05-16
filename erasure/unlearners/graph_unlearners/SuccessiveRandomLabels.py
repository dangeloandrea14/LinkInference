from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from erasure.core.factory_base import get_instance_kvargs

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class SuccessiveRandomLabels(GraphUnlearner):
    def init(self):
        """
        Initializes the SuccessiveRandomLabels class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs']
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']  
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget'] 

        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})

        self.n_classes = self.dataset.n_classes

    def __unlearn__(self):
        """
        Fine-tunes the model with both the retain set and forget set. The labels for the forget set are randomly assigned and different from the original ones.
        """

        self.info(f'Starting SRL with {self.epochs} epochs')

        retain_set = self.dataset.partitions[self.ref_data_retain]
        forget_set = self.dataset.partitions[self.ref_data_forget]

        num_nodes = self.x.size(0)
        all_nodes = torch.arange(num_nodes)

        if self.removal_type == 'edge':
            forget_set = self.infected_nodes(forget_set, self.hops)
            retain_set = [node for node in all_nodes if node not in forget_set]
            
        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            random_labels = []
            for label in self.labels[forget_set]:
                random_label = np.random.choice([c for c in range(self.n_classes) if c != label.item()])
                random_labels.append(random_label)

            random_labels = torch.tensor(random_labels, device=self.device)

            self.predictor.optimizer.zero_grad()
            out = self.predictor.model(self.x, self.edge_index)[forget_set]
            loss = self.predictor.loss_fn(out, random_labels)


            losses.append(loss.to('cpu').detach().numpy())

            loss.backward()
            self.predictor.optimizer.step()

                ####

            self.predictor.optimizer.zero_grad() 

            out = self.predictor.model(self.x, self.edge_index)[retain_set]
            loss = self.predictor.loss_fn(out, self.labels[retain_set])

            losses.append(loss.to('cpu').detach().numpy())

            loss.backward()
            self.predictor.optimizer.step()

            epoch_loss = sum(losses) / len(losses)
            self.info(f'SRL - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam
