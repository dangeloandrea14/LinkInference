from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction
import torch.optim as optim
from erasure.utils.config.local_ctx import Local
from copy import deepcopy
import torch

from erasure.core.factory_base import get_instance_kvargs

class AdvancedNegGrad(GraphUnlearner):
    def init(self):
        """
        Initializes the AdvancedNegGrad class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs']
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']  
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget'] 
        self.training_set = self.local.config['parameters']['training_set']
        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})

    def __unlearn__(self):
        """
        An implementation of the Advanced NegGrad unlearning algorithm proposed in the following paper:
        "Choi, D. and Na, D., 2023. Towards machine unlearning benchmarks: Forgetting the personal identities in facial recognition systems. arXiv preprint arXiv:2311.02240."
        
        Codebase taken from the original implementation: https://github.com/ndb796/MachineUnlearning
        """

        self.info(f'Starting AdvancedNegGrad with {self.epochs} epochs')      


        retain_set = self.dataset.partitions[self.ref_data_retain]
        forget_edges = self.dataset.partitions[self.ref_data_forget]
        forget_set = forget_edges

        num_nodes = self.x.size(0)

        if self.removal_type == 'edge':
            forget_set = self.infected_nodes(forget_edges, self.hops)
            forget_set_s = set(forget_set)
            retain_set = [n for n in range(num_nodes) if n not in forget_set_s]
        else:
            forget_edges = None


        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            # Order matters: retain first, then forget, so the dropout masks drawn by
            # the two forward passes match the original inline implementation.
            loss_retain = self.task_loss(node_subset=retain_set)
            loss_ascent_forget = -self.task_loss(node_subset=forget_set, edge_subset=forget_edges)

            # Overall loss
            joint_loss = loss_ascent_forget + loss_retain

            losses.append(joint_loss.to('cpu').detach().numpy())

            joint_loss.backward()
            self.predictor.optimizer.step()

            
            epoch_loss = sum(losses) / len(losses)
            self.info(f'AdvancedNegGrad - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor
    

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam
        self.local.config['parameters']['training_set'] = self.local.config['parameters'].get("training_set", 'train')