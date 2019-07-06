import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve

class ReLU:


    def __init__(self):
        self.input_tensor=None


    def forward(self,input_tensor):

        self.input_tensor = input_tensor


        input_tensor_j=np.maximum(self.input_tensor, 0.)


        return input_tensor_j

    def backward(self,error_tensor):




        return error_tensor*(self.input_tensor>0).astype(float)



