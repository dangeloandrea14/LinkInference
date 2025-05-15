from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction
import torch
from erasure.core.factory_base import get_instance_kvargs

class NegGrad(GraphUnlearner):
    def init(self):
        """
        Initializes the NegGrad class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs']  
        self.ref_data = self.local.config['parameters']['ref_data'] 
        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})


    def __unlearn__(self):
        """
        An implementation of the Negative Gradient unlearning algorithm proposed in the following paper:
        "Golatkar, A., Achille, A. and Soatto, S., 2019. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In 2020 IEEE. In CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 9301-9309)."
        
        Codebase taken from this implementation: https://github.com/ndb796/MachineUnlearning
        """

        self.info(f'Starting NegGrad with {self.epochs} epochs')

        forget_set = self.dataset.partitions[self.ref_data]


        if self.removal_type == 'edge':
            forget_set = self.infected_nodes(forget_set, self.hops)
            

        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.predictor.model.train()


            self.predictor.optimizer.zero_grad() 

            pred = self.predictor.model(self.x,self.edge_index)[forget_set]

            loss = -self.predictor.loss_fn(pred, self.labels[forget_set])

            losses.append(loss.to('cpu').detach().numpy())

            loss.backward()

            labels_list += list(self.labels.squeeze().long().detach().to('cpu').numpy())
            preds += list(pred.squeeze().detach().to('cpu').numpy())

            self.predictor.optimizer.step()

            epoch_loss = sum(losses) / len(losses)
            self.info(f'NegGrad - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor
    

    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'forget')  # Default reference data is forget
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam