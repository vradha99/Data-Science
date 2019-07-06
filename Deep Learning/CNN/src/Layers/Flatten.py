import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve

class Flatten:


    def __init__(self):

        self.input_tensor=None
        self.error_tensor=None
        self.ishape=None

        #self.input_im = None

    def forward(self,input_tensor):

        #self.input_tensor = input_tensor.shape[1:]
        #return input_tensor.reshape(-1,np.prod(self.input_tensor))

        self.input_tensor=input_tensor
        self.ishape=self.input_tensor.shape
        self.input_tensor=self.input_tensor.ravel()
        y=np.prod(self.ishape)/self.ishape[0]
        return self.input_tensor.reshape(self.ishape[0],int(y))



    def backward(self,error_tensor):

        self.error_tensor=error_tensor
        self.error_tensor=self.error_tensor.ravel()
        return self.error_tensor.reshape(self.ishape)

        #return error_tensor.reshape(-1,*self.input_tensor)


