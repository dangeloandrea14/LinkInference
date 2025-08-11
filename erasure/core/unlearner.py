from abc import ABCMeta, abstractmethod
import copy
from erasure.core.base import Configurable
from erasure.data.datasets.Dataset import DatasetWrapper
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local

class Unlearner(Configurable, metaclass=ABCMeta):

    def __init__(self, global_ctx: Global, local_ctx):
        if hasattr(local_ctx,'dataset'):
            self.dataset = local_ctx.dataset
        if hasattr(local_ctx,'predictor'):
            self.predictor = local_ctx.predictor
            self.device = self.predictor.device
        else:
            self.predictor = 'global'

        super().__init__(global_ctx, local_ctx)  
        

    def unlearn(self):
        self.__preprocess__()
        self.info('Unlearning copied predictor: '+str(self.predictor))
        new_model = self.__unlearn__()
        self.__postprocess__()
        return new_model

    def init(self):
        self.removal_type = self.global_ctx.removal_type

    def __preprocess__(self):
        pass

    @abstractmethod
    def __unlearn__(self):
        pass

    def __postprocess__(self):
        pass
        

def model_weight_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        total_norm += param.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
