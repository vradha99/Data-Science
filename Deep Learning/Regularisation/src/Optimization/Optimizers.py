import numpy as np
from Layers import *
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve
import math



class Sgd(Base.base_class):
    def __init__(self,global_learning_rate):
        super().__init__()

        self.global_learning_rate=global_learning_rate




    def calculate_update(self, individual_delta,weight_tensor,gradient_tensor):


        learning_rate = individual_delta*self.global_learning_rate

        if self.regularizer is not None:
            weight_tensor -= learning_rate * self.regularizer.calculate(weight_tensor)


        weight_tensor=weight_tensor-(learning_rate*gradient_tensor)

        return weight_tensor

class SgdWithMomentum(Base.base_class):

    def __init__(self,global_learning_rate,mu):
        super().__init__()

        self.global_learning_rate=global_learning_rate
        self.mu=mu
        self.v=0



    def calculate_update(self, individual_delta,weight_tensor,gradient_tensor):

        learning_rate = individual_delta*self.global_learning_rate

        self.v=(self.v*self.mu)-(learning_rate*gradient_tensor)
        if self.regularizer is not None:
            weight_tensor -= learning_rate * self.regularizer.calculate(weight_tensor)
        weight_tensor=weight_tensor+self.v

        return weight_tensor


class Adam(Base.base_class):
    def __init__(self,global_learning_rate,mu,ro):
        super().__init__()

        self.global_learning_rate=global_learning_rate
        self.u=mu
        self.p=ro
        self.v=0
        self.r=0
        self.k=0
        self.eps=10e-9

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        self.k = self.k + 1
        self.v = self.u * self.v + (1 - self.u) * gradient_tensor
        self.r = self.p * self.r + (1 - self.p) * gradient_tensor * gradient_tensor
        bias_correct_v = self.v / (1 - np.power(self.u, self.k))
        bias_correct_r = self.r / (1 - np.power(self.p, self.k))
        learning_rate = individual_delta * self.global_learning_rate

        if self.regularizer is not None:
            weight_tensor -= learning_rate * self.regularizer.calculate(weight_tensor)

        weight_next = weight_tensor - (learning_rate) * (
                (bias_correct_v + self.eps) / (np.sqrt(bias_correct_r) + self.eps))
        return weight_next








