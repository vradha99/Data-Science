import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve



class Constant:
    def __init__(self, const):

        if const is None:
            self.constant = 0.1   # default value is 0.1
        else:
            self.constant=const    # Given weight

    def initialize(self, weights_shape,fan_in, fan_out):
        weights = np.full((weights_shape), self.constant)
        return weights


class UniformRandom:
    def initialize(self, weights_shape,fan_in, fan_out):
        weights = np.random.uniform(0,1,np.prod(weights_shape))  # random values between (0,1)
        weights=weights.reshape(weights_shape)

        return weights


class Xavier:
    def initialize(self, weights_shape,fan_in,fan_out):

        sig = np.sqrt(2 / (fan_in + fan_out))
        xavr_wts = np.random.normal(0, sig, np.prod(weights_shape))   # random values between (0,sig)
        xavr_wts = xavr_wts.reshape(weights_shape)
        return xavr_wts


class He:
    def initialize(self,weights_shape,fan_in,fan_out):
        sig = np.sqrt(2 / fan_in)
        he_wts = np.random.normal(0, sig,np.prod(weights_shape))  # random values between (0,sig)
        he_wts=he_wts.reshape(weights_shape)
        return he_wts
