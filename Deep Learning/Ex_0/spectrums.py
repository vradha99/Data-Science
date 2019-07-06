import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches


class Spectrum:
    output = None

    def __init__(self, resol):
        self.resol=resol



    def draw(self):
        x=np.linspace(255, 0, self.resol)

        y1=np.expand_dims(np.tile(x, (self.resol,1)),axis=2)
        #print(y1.shape)
        y2=np.fliplr(y1)
        #print(y2.shape)
        y3=np.expand_dims(np.transpose(np.tile(x, (self.resol,1))),axis=2)
        #print(y3)

        f=np.concatenate((y2,y1, y3), axis=2)/255

        self.output=f


    def show(self):
        plt.figure()
        plt.imshow(self.output)
        plt.show()

