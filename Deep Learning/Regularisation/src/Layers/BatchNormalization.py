import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
import copy
from .Base import *

class BatchNormalization(base_class):

    def __init__(self,channels=0):
        super().__init__()
        self.delta=1
        self.mean = None
        self.var=None
        self.channels = channels
        self.phase = Phase.train
        self.weights=None
        self.bias=None
        self.weight_optimizer=None
        self.bias_optimizer=None
        self.meannew=0
        self.varnew=0

    def forward(self,input_tensor):
        batch=input_tensor.shape[0]
        vol=np.prod(input_tensor.shape[1:])
        self.input_tensor = input_tensor
        self.input_tensor = self.input_tensor.reshape(batch, vol)



        if (self.channels==0):
            self.batch_size, self.input_size = self.input_tensor.shape
            return self.normalisation(self.input_tensor)

        else:
            self.row = input_tensor.shape[2]
            self.col = input_tensor.shape[3]
            self.spatial_size = int(vol / self.channels)
            self.input_tensor = np.reshape(self.input_tensor,(self.spatial_size * batch, self.channels))
            self.batch_size, self.input_size = self.input_tensor.shape

            y_hat = self.normalisation(self.input_tensor)

            return y_hat.reshape(batch,self.channels,self.row,self.col)


    def normalisation(self,input_tensor):

        self.epsilon = 1e-8

        ph = self.phase
        if (self.phase == 1):
            ph = Phase.train
        elif (self.phase == 2):
            ph = Phase.test

        self.normalized_x = np.zeros_like(input_tensor)
        alpha = 0.8
        y_hat = np.zeros_like(input_tensor)

        if self.weights is None:
            self.weights = np.ones(self.input_size)
        if self.bias is None:
            self.bias = np.zeros(self.input_size)

        if ph == Phase.train:
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)
            self.normalized_x = np.divide(input_tensor - self.mean,
                                          np.sqrt(self.var + self.epsilon))
            y_hat = self.weights * self.normalized_x + self.bias

            self.meannew=(1-alpha)*self.mean+alpha*self.meannew
            std_d=(1-alpha)*np.sqrt(self.var)+alpha*np.sqrt(self.varnew)
            self.varnew=std_d**2


        if ph == Phase.test:


            self.normalized_x = np.divide(input_tensor - self.meannew,
                                          np.sqrt(self.varnew + self.epsilon))
            y_hat = self.weights * self.normalized_x + self.bias

        return y_hat







    def backward(self,error_tensor):

        batch=error_tensor.shape[0]

        vol= np.prod(error_tensor.shape[1:])
        error_tensor = error_tensor.reshape(batch, vol)

        if self.channels == 0:

            return self.backprop_and_update(error_tensor)

        # Convolution

        else:
            self.spatial_size = math.floor(vol / self.channels)

            Conv_error_tensor = np.reshape(error_tensor, (self.spatial_size * batch, self.channels))

            backprop_error_tensor = self.backprop_and_update(Conv_error_tensor)
            #print(backprop_error_tensor.shape)

            return backprop_error_tensor.reshape(batch,self.channels,self.row,self.col)





    def backprop_and_update(self, error_tensor):
        # calculate gradients
        self._gradient_weights = np.zeros(self.input_size)
        self._gradient_bias = np.zeros(self.input_size)

        # gradient wrt bias and weights
        self._gradient_weights = np.sum(error_tensor * self.normalized_x, axis=0)
        self._gradient_bias = np.sum(error_tensor, axis=0)

        # gradient wrt input
        var_expression1 = (-0.5 * ((self.var + self.epsilon) ** (-3 / 2)))
        var_expression2 = (1 / np.sqrt(self.var + self.epsilon))

        gradient_normalized_x = error_tensor * self.weights
        gradient_variance = np.sum(gradient_normalized_x * (self.input_tensor - self.mean) * var_expression1,
                                   axis=0)

        const_grad_mean1 = np.sum(gradient_normalized_x * (-var_expression2), axis=0)
        const_grad_mean2 = np.sum(-2 * (self.input_tensor - self.mean), axis=0) / self.batch_size

        gradient_mean = const_grad_mean1 + gradient_variance * const_grad_mean2

        gradient_input = gradient_normalized_x * var_expression2 \
                         + gradient_variance * 2 * (self.input_tensor - self.mean) / self.batch_size \
                         + gradient_mean * (1 / self.batch_size)

        # update parameter gamma and beta
        if self.weight_optimizer is not None:
            self.weights = self.weight_optimizer.calculate_update(self.delta, self.weights, self._gradient_weights)

        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.delta, self.bias, self._gradient_bias)

        return gradient_input


    def get_gradient_weights(self):
        return self._gradient_weights

    def get_gradient_bias(self):
        return self._gradient_bias

    def set_optimizer(self, optimizer):
        self.weight_optimizer = optimizer
        self.bias_optimizer = copy.deepcopy(optimizer)

    def initialize(self,__,_):
        pass