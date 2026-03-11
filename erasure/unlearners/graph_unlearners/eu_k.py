from erasure.unlearners.graph_unlearners.GraphUnlearner import GraphUnlearner
from fractions import Fraction

from erasure.core.factory_base import get_instance_kvargs


class eu_k(GraphUnlearner):
    def init(self):
        """
        Initializes the eu-k class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs'] 
        self.ref_data = self.local.config['parameters']['ref_data'] 

        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})


    def __unlearn__(self):
        """
        Freeze all the model layers except the last k layers. Then the trainable layers are reset and finetuned with the reference data.
        """

        self.info(f"Starting eu-{self.local.config['parameters']['last_trainable_layers']} with {self.epochs} epochs")

        self.retain = self.dataset.partitions[self.ref_data]

        if self.removal_type == 'edge':
            forget_edges = self.dataset.partitions[self.forget_part]
            self.retain = self.infected_nodes(forget_edges, self.hops)


        for i, layer in enumerate(list(self.predictor.model.children())):
                if i >= len(list(self.predictor.model.children())) - self.local.config['parameters']['last_trainable_layers']:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                        self.info(f'Layer {i} is reset')
        

        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            self.predictor.optimizer.zero_grad() 

            pred = self.predictor.model(self.x, self.edge_index)[self.retain]

            loss = self.predictor.loss_fn(pred, self.labels[self.retain])
            losses.append(loss.to('cpu').detach().numpy())

            loss.backward()

            self.predictor.optimizer.step()

            epoch_loss = sum(losses) / len(losses)
            self.info(f"eu-{self.local.config['parameters']['last_trainable_layers']} - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}")

            self.predictor.lr_scheduler.step()
        
        return self.predictor
    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'retain')  # Default reference data is retain
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam
        self.local.config['parameters']['last_trainable_layers'] = self.local.config['parameters'].get('last_trainable_layers', 1)  # Default last_trainable_layers is 1