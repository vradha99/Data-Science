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

      #  error_tensor_i=error_tensor_j
       # error_tensor_i[self.input_tensor_i<=0]=0


        return error_tensor*(self.input_tensor>0).astype(float)




"""""
>>> import numpy as np
>>> a = np.arange(10)-5#i/p tensor
>>> a
array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])
>>> b = np.arange(10)#error tensor
>>> b
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b[a<0] = 0
>>> b#updated error tensor
array([0, 0, 0, 0, 0, 5, 6, 7, 8, 9])
>>>
"""""