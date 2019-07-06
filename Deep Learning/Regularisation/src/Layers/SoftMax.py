import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve


class SoftMax:
    output = None

    def __init__(self):

        self.labels_est_tensor=None


    def forward(self,input_tensor,label_tensor):
        input_tensor_i_st=input_tensor-np.max(input_tensor)

        self.labels_est_tensor  = self.predict(input_tensor_i_st)
        loss= np.sum(-np.log(self.labels_est_tensor)*label_tensor)

        return loss

    def predict(self, input_tensor):

        self.labels_est_tensor = np.exp(input_tensor) / np.expand_dims(np.sum(np.exp(input_tensor),axis=1),axis=1)

        return self.labels_est_tensor

    def backward(self,label_tensor):

        error_tensor=self.labels_est_tensor

        error_tensor= np.subtract(error_tensor, label_tensor)

        return error_tensor

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


