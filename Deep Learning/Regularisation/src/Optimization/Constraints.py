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

class L2_Regularizer:
    def __init__(self, alpha):  # alpha represents the regularization weight.
        self.alpha = alpha

    def norm(self, weights):
        return self.alpha * math.sqrt(np.sum(weights ** 2))

    def calculate(self, weights):  # Performs a (sub-)gradient update on the weights.
        return self.alpha * weights



class L1_Regularizer:
    def __init__(self, alpha):  # alpha represents the regularization weight.
        self.alpha = alpha

    def norm(self, weights):
        return  self.alpha * np.sum(np.abs(weights))

    def calculate(self, weights):  # Performs a (sub-)gradient update on the weights.
        return self.alpha * np.sign(weights)




