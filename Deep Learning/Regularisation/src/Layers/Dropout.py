import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from .Base import *

class Dropout(base_class):

    def __init__(self,probability):
        super().__init__()
        self.probability = probability
        self.a=None
        self.phase = Phase.train

    def forward(self,input_tensor):
        ph = self.phase

        if (self.phase == 1):
            ph = Phase.train
        elif (self.phase == 2):
            ph = Phase.test

        if ph == Phase.train:
            self.a = np.random.choice([0, 1], size=input_tensor.shape, p=[1 - self.probability, self.probability])

            return (input_tensor * self.a) / self.probability

        else:

            return input_tensor


    def backward(self,error_tensor):
        return error_tensor * self.a

