from enum import Enum
from .FullyConnected import *
from .Conv import *


class base_class:
    def __init__(self):
        self.regularizer=None
        self.phase=None


    def add_regularizer(self,regularizer):
        self.regularizer=regularizer


    def calculate_loss(self,layer):
        loss=0
        if isinstance(layers, FullyConnected):
            loss=self.regularizer.norm(np.delete(layer.weights,layer.weights.shape[1]-1,axis=1))
        if isinstance(layers,Conv):
            loss=self.regularizer.norm(layer.weights)
        return loss




class Phase(Enum):
    train = 1
    test = 2
    validation = 3