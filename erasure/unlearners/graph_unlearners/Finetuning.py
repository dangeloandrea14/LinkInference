from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction
import torch
from erasure.core.factory_base import get_instance_kvargs


class Finetuning(GraphUnlearner):
    def init(self):
        """
        Initializes the Finetuning class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs'] 
        self.ref_data = self.local.config['parameters']['ref_data'] 
        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})


    def __unlearn__(self):
        """
        Fine-tunes the model with a specific (sub)set of the full dataset (usually retain set)
        """

        self.info(f'Starting Finetuning with {self.epochs} epochs')

        self.retain = self.dataset.partitions[self.ref_data]

        num_nodes = self.x.size(0)

        if self.removal_type == 'edge':
            self.retain = self.infected_nodes(self.retain, self.hops)
        
        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            self.predictor.optimizer.zero_grad()

            pred = self.predictor.model(self.x, self.edge_index)[self.retain]

            loss = self.predictor.loss_fn(pred,self.labels[self.retain])
            losses.append(loss.to('cpu').detach().numpy())

            loss.backward()

            self.predictor.optimizer.step()
            
            epoch_loss = sum(losses) / len(losses)
            self.info(f'Finetuning - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor
    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'retain')  # Default reference data is retain
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam