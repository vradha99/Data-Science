import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve
import math
import copy

class Conv:

    delta = 1

    def __init__(self, stride_shape, convolution_shape, num_kernels, learning_rate=delta):

        self.stride = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.delta = learning_rate

        if (len(self.stride) != 1):
            fan_in = convolution_shape[0] * convolution_shape[1] * convolution_shape[2]
            fan_out = num_kernels * convolution_shape[1] * convolution_shape[2]
            self.back_kernels = np.zeros((self.convolution_shape[0], self.num_kernels, self.convolution_shape[1],
                                          self.convolution_shape[2]))
        else:
            fan_in = convolution_shape[0] * convolution_shape[1]
            fan_out = num_kernels * convolution_shape[1]
            self.back_kernels = np.zeros((self.convolution_shape[0], self.num_kernels, self.convolution_shape[1]))

        self.weights = np.random.uniform(0, 1, (self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, (self.num_kernels))
        self.gradient_bias = None
        self.gradient_weights = None
        self.gradient_weights = np.zeros((self.weights.shape))
        self.gradient_bias = np.zeros((self.bias.shape))
        self.weights_optimizer = None
        self.bias_optimizer = None
        self.output_tensor=None

    def initialize(self, weights_initializer, bias_initializer):
        if (len(self.convolution_shape) == 3):
            self.fan_in = self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2]
            self.fan_out = self.num_kernels * self.convolution_shape[1]*self.convolution_shape[2]
        else:

            self.fan_in = self.convolution_shape[0] * self.convolution_shape[1]
            self.fan_out = self.num_kernels * self.convolution_shape[1]

        shape=self.weights.shape

        self.weights = weights_initializer.initialize(shape,self.fan_in,self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape,self.fan_in,self.fan_out)

    def forward(self, input_tensor):


        self.input_tensor = input_tensor.copy()

        input_tmp = input_tensor[0]

        if (len(self.stride) == 1):
            input_downsize = input_tmp[:, ::self.stride[0]]
            next_input_tensor = np.zeros((self.input_tensor.shape[0], self.num_kernels, input_downsize.shape[1]))
        else:
            input_downsize = input_tmp[:, ::self.stride[0], ::self.stride[1]]
            next_input_tensor = np.zeros((self.input_tensor.shape[0], self.num_kernels, input_downsize.shape[1], input_downsize.shape[2]))

        for i in range(self.input_tensor.shape[0]):
            for j in range(self.num_kernels):
                corr = scipy.signal.correlate(self.input_tensor[i], self.weights[j], 'same')
                if (len(self.stride) == 1):

                    corr_downsize = corr[int((np.floor(corr.shape[0] / 2))),
                                    ::self.stride[0]]
                    corr_downsize += self.bias[j]
                    next_input_tensor[i, j, :] = corr_downsize
                else:

                    corr_downsize = corr[int((np.floor(corr.shape[0] / 2))),
                                    ::self.stride[0], ::self.stride[1]]
                    corr_downsize += self.bias[j]
                    next_input_tensor[i, j, :, :] = corr_downsize


        return next_input_tensor



    def backward(self,error_tensor):

        #self.error_tensor=error_tensor.copy()
        self.gradient_weights = np.zeros((self.weights.shape))
        result = np.zeros((self.input_tensor.shape))
        ip=self.padding(self.convolution_shape, self.input_tensor)

        self.error_tensor=np.zeros(self.input_tensor.shape)
        convolved=np.zeros_like((self.gradient_weights.shape))



        if(len(error_tensor.shape)==4):





            for i in range(self.back_kernels.shape[0]):
                for j in range(self.back_kernels.shape[1]):
                    self.back_kernels[i,j,:,:]=self.weights[j,i,:,:]

            self.back_kernels=np.flip(self.back_kernels,1)



            for i in range(error_tensor.shape[0]):
                for j in range(self.back_kernels.shape[0]):

                    self.upsample_error = np.zeros((self.input_tensor.shape[0], self.num_kernels,
                                                    self.input_tensor.shape[2], self.input_tensor.shape[3]))
                    self.upsample_error[:, :, ::self.stride[0], ::self.stride[1]] = error_tensor.copy()

                    result= scipy.signal.convolve(self.upsample_error[i], self.back_kernels[j, :, :, :], 'same')
                    result = result[int((np.floor(result.shape[0] / 2))), :, :]
                    self.error_tensor[i, j, :, :] = result



            for i in range(self.input_tensor.shape[0]):


                for j in range(self.num_kernels):

                    gradient_weights = np.zeros_like(self.gradient_weights)
                    for k in range(ip.shape[1]):
                        single_error = np.expand_dims(self.upsample_error[i, j, :, :], axis=0)
                        gradient_weights[j,:,:,:] = scipy.signal.correlate(ip[i,:,:,:], single_error, 'valid')


                    self.gradient_weights+= gradient_weights


            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))



        else:


            for i in range(self.back_kernels.shape[0]):
                for j in range(self.back_kernels.shape[1]):
                    self.back_kernels[i,j,:]=self.weights[j,i,:]


            for i in range(error_tensor.shape[0]):
                for j in range(self.back_kernels.shape[0]):

                    #for k in range(error_tensor.shape[1]):
                    self.upsample_error = np.zeros((self.input_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2]))
                    self.upsample_error[:, :, ::self.stride[0]] = error_tensor.copy()
                    result = scipy.signal.convolve(self.upsample_error[i],self.back_kernels[j, :, :],'same')
                    result = result[int((np.floor(result.shape[0] / 2))), :]
                    self.error_tensor[i, j, :] = result



            for i in range(ip.shape[0]):
                for j in range(error_tensor.shape[1]):
                    gradient_weights = np.zeros_like((self.weights[0, :, :]))
                    for k in range(ip.shape[1]):
                        convolved = scipy.signal.correlate(ip[i,k,:], self.upsample_error[i, j, :], 'valid')
                        gradient_weights[k, :] = convolved

                    self.gradient_weights[j] = gradient_weights



            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        if (self.weights_optimizer):

            self.weights = self.weights_optimizer.calculate_update(self.delta, self.weights, self.gradient_weights)
            self.bias = self.bias_optimizer.calculate_update(self.delta, self.bias, self.gradient_bias)

        return self.error_tensor

    def padding(self, convolution_shape, input_tensor):

        shape = convolution_shape
        tensor = input_tensor

        if(len(shape)>2):
            r = math.floor((shape[2] - 1) / 2.0)
            l = math.ceil((shape[2] - 1) / 2.0)
            u = math.ceil((shape[1] - 1) / 2.0)
            d = math.floor((shape[1] - 1) / 2.0)
            padding = np.zeros((tensor.shape[0], tensor.shape[1], u + d + tensor.shape[2], l + r + tensor.shape[3]))
            if (l == 0 and u != 0):
                padding[:, :, u:-d, :] = tensor
            elif (u == 0 and l != 0):
                padding[:, :, :, r:-l] = tensor
            elif ((u == 0) and (l == 0)):
                padding = tensor.copy()
            else:
                padding[:, :, u:-d, r:-l] = tensor

        else:

            r = math.floor((shape[1] - 1) / 2.0)
            l = math.ceil((shape[1] - 1) / 2.0)
            padding = np.zeros((tensor.shape[0], tensor.shape[1], l + r + tensor.shape[2]))
            if (l == 0 ):
                padding[:, :, :] = tensor
            else:

                padding[:, :, r:-l] = tensor


        return padding


    def get_gradient_weights(self):



        return self.gradient_weights


    def get_gradient_bias(self):


        return self.gradient_bias


    def set_optimizer(self,optimizer):
        self.weights_optimizer =copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)


