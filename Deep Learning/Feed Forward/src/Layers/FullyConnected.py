import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve
import random
from random import *


class FullyConnected:
    output = None

    def __init__(self,input_size,output_size):

        self.input_size=input_size

        self.output_size=output_size



        self.input_tensor=None
        self.error_tensor_tr=None
        self.weights = np.random.rand(input_size+1,output_size)
        #self.weight_tr = np.delete(np.transpose(self.weight), -1, axis=1)

        self.batch_size=None
        self.delta=1



    def forward(self,input_tensor):

        batch_size=input_tensor.shape[0]
        one_size=np.ones((batch_size,1))
        self.input_tensor = np.hstack((input_tensor,one_size))
        output_tensor=np.dot(self.input_tensor,self.weights)
        return output_tensor

    def backward(self, error_tensor):

        grad = np.delete(np.dot(error_tensor,np.transpose(self.weights)),-1,axis=1)
        self.error_tensor=error_tensor

        self.weights= np.subtract(self.weights,(self.delta*self.get_gradient_weights()))
        return grad

    def get_gradient_weights(self, ):


        return np.dot(np.transpose(self.input_tensor),self.error_tensor)













