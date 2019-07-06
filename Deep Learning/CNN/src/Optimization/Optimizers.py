import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve
import math


class Sgd:
    def __init__(self,global_learning_rate):

        self.global_learning_rate=global_learning_rate




    def calculate_update(self, individual_delta,weight_tensor,gradient_tensor):

        learning_rate = individual_delta*self.global_learning_rate

        weight_tensor=weight_tensor-(learning_rate*gradient_tensor)

        return weight_tensor

class SgdWithMomentum:

    def __init__(self,global_learning_rate,mu):

        self.global_learning_rate=global_learning_rate
        self.mu=mu
        self.v=0



    def calculate_update(self, individual_delta,weight_tensor,gradient_tensor):

        learning_rate = individual_delta*self.global_learning_rate

        self.v=(self.v*self.mu)-(learning_rate*gradient_tensor)

        weight_tensor=weight_tensor+self.v

        return weight_tensor


class Adam:
    def __init__(self,global_learning_rate,mu,ro):

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
        weight_next = weight_tensor - (self.global_learning_rate * individual_delta) * (
                (bias_correct_v + self.eps) / (np.sqrt(bias_correct_r) + self.eps))
        return weight_next








