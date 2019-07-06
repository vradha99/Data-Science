import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve

class Circle:
    output = None

    def __init__(self, resol, rad, posx,posy):
        self.resol=resol
        self.rad=rad
        self.posx=posx
        self.posy=posy

    def draw(self):
        background = np.zeros((self.resol, self.resol))

        y, x = np.ogrid[0:self.resol,0:self.resol]

        mask = (x-self.posx) ** 2 + (y-self.posy) ** 2 <= self.rad ** 2


        background[mask]=1

        print(background[mask])

        self.output = background


    def show(self):


        plt.figure()
        plt.imshow(self.output,cmap='gray')

        plt.show()

