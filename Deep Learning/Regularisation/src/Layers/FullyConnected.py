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
import copy


class FullyConnected:
    output = None


    def __init__(self,input_size,output_size,delta=1):

        self.input_size=input_size

        self.output_size=output_size

        self.error_tensor = None
        self.input_tensor=None
        self.error_tensor_tr=None
        self.weights = np.random.rand(input_size+1,output_size)
        #self.weight_tr = np.delete(np.transpose(self.weight), -1, axis=1)

        self.batch_size=None
        self.delta=delta
        self.optimizer=None

    def initialize(self,weights_initializer,bias_initializer):

        weights = weights_initializer.initialize(np.shape(self.weights[:-1, :]), np.shape(self.weights[:-1, :])[0],
                                                 np.shape(self.weights[:-1, :])[1])
        bias = np.expand_dims(self.weights[-1,:], axis = 0)
        bias = bias_initializer.initialize(bias.shape, bias.shape[0], bias.shape[1])
        self.weights = np.concatenate((weights, bias), axis=0)


        #self.weights = initializer.initialize(weights_initializer)
        #self.bias = bias_initializer

    def forward(self,input_tensor):

        batch_size=input_tensor.shape[0]
        one_size=np.ones((batch_size,1))
        self.input_tensor = np.hstack((input_tensor,one_size))

        output_tensor=np.dot(self.input_tensor,self.weights)
        return output_tensor

    def backward(self, error_tensor):

        grad = np.delete(np.dot(error_tensor,np.transpose(self.weights)),-1,axis=1)
        self.error_tensor=error_tensor
        gradient = self.get_gradient_weights()
        if (self.optimizer != None):
            #self.weights= np.subtract(self.weights,(self.delta*self.get_gradient_weights()))
            self.weights = self.optimizer.calculate_update(self.delta, self.weights, gradient)

        return grad

    def set_optimizer(self,optimizer):

        self.optimizer= copy.deepcopy(optimizer)

    def get_gradient_weights(self, ):

        self.gradient_weights=np.dot(np.transpose(self.input_tensor),self.error_tensor)

        return self.gradient_weights













