import copy
from erasure.core.unlearner import Unlearner
from erasure.utils.config.local_ctx import Local
from copy import deepcopy

class GoldModel(Unlearner):
    def init(self):
        """
        Initializes the GoldModel class with global and local contexts.
        """
        super().init()

        #Create Dataset
        if self.local.config['parameters']['data'] == 'global':
            data_manager = self.global_ctx.dataset
        else:
            data_manager = self.global_ctx.factory.get_object(Local(self.local.config['parameters']['data']))
    
        #Create Predictor
        self.current = Local(self.local.config['parameters']['predictor'])
        self.current.dataset = self.global_ctx.dataset
    
    
    def __unlearn__(self):
        """
        Retrain the model from scratch with a specific (sub)set of the full dataset (usually retain set to evaluate the performance of the model after unlearning)
        """

        predictor = self.global_ctx.factory.get_object(self.current)
            
        return predictor
    
    def check_configuration(self):
        super().check_configuration()

        if 'data' not in self.local.config['parameters']:
            self.local.config['parameters']['data'] = 'global'#copy.deepcopy(self.global_ctx.dataset.local_config)

        self.local.config['parameters']['training_set'] = self.local.config['parameters'].get("training_set", 'retain')  # Default train data is retain
        self.local.config['parameters']['cached'] = self.local.config['parameters'].get("cached", False)  # Default cached to False

        if 'predictor' not in self.local.config['parameters']:
            self.local.config['parameters']['predictor'] = copy.deepcopy(self.global_ctx.predictor.local_config)
            self.local.config['parameters']['predictor']['parameters']['cached'] = self.local.config['parameters']['cached']
            self.local.config['parameters']['predictor']['parameters']['training_set'] = self.local.config['parameters']['training_set']
        else:
            self.local.config['parameters']['predictor']['parameters']['training_set'] = self.local.config['parameters']['predictor']['parameters'].get('training_set',self.local.config['parameters']['training_set'])

        self.local.config['parameters']['predictor']['parameters']['cached'] = self.local.config['parameters']['predictor']['parameters'].get('cached',self.local.config['parameters']['cached'])

        
    

class GoldModelGraph(Unlearner):
    def init(self):
        super().init()

        self.training_set = self.local.config['parameters']['training_set']

        #Create Dataset
        if self.local.config['parameters']['data'] == 'global':
            data_manager = deepcopy(self.global_ctx.dataset) 
        else:
            data_manager = self.global_ctx.factory.get_object(Local(self.local.config['parameters']['data']))
            
        og_graph =  data_manager.partitions['all'] 
        gold_training_set = self.local.config['parameters']['training_set']
        gold_training_set = data_manager.partitions[gold_training_set]

        if self.removal_type == 'node':
            new_graph, remapped_partitions = og_graph.revise_graph_nodes(gold_training_set, data_manager.partitions)
        if self.removal_type == 'edge':
            new_graph = og_graph.revise_graph_edges(gold_training_set)
            remapped_partitions = copy.deepcopy(data_manager.partitions)
        
        data_manager.partitions = remapped_partitions
        data_manager.partitions['all'] = new_graph

        if self.removal_type == 'node':
            data_manager.partitions[self.training_set] = list(range(new_graph.num_nodes))
        if self.removal_type == 'edge':
            data_manager.partitions[self.training_set] = remapped_partitions['train']

        self.data_manager = data_manager

    

    def __unlearn__(self):

        #Create Predictor
        self.current = Local(self.local.config['parameters']['predictor'])
        self.current.dataset = self.data_manager
        

        predictor = self.global_ctx.factory.get_object(self.current)
        self.hops = len(predictor.model.hidden_channels) + 1
            
        return predictor
    
    def check_configuration(self):
        super().check_configuration()

        if 'data' not in self.local.config['parameters']:
            self.local.config['parameters']['data'] = 'global' 

        self.local.config['parameters']['training_set'] = self.local.config['parameters'].get("training_set", 'retain')  # Default train data is retain
        self.local.config['parameters']['cached'] = self.local.config['parameters'].get("cached", False)  # Default cached to False

        if 'predictor' not in self.local.config['parameters']:
            self.local.config['parameters']['predictor'] = copy.deepcopy(self.global_ctx.predictor.local_config)
            self.local.config['parameters']['predictor']['parameters']['cached'] = self.local.config['parameters']['cached']
            self.local.config['parameters']['predictor']['parameters']['training_set'] = self.local.config['parameters']['training_set']
        else:
            self.local.config['parameters']['predictor']['parameters']['training_set'] = self.local.config['parameters']['predictor']['parameters'].get('training_set',self.local.config['parameters']['training_set'])

        self.local.config['parameters']['predictor']['parameters']['cached'] = self.local.config['parameters']['predictor']['parameters'].get('cached',self.local.config['parameters']['cached'])

        
    