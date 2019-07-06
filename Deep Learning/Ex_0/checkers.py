import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches


class Checker:
    output = None

    def __init__(self, resol, tilsiz):
        self.resol=resol
        self.tilsiz=tilsiz


    def draw(self):
        z=np.zeros((self.tilsiz,self.tilsiz))
        o=np.ones((self.tilsiz,self.tilsiz))

        a=np.hstack((z, o))
        c=np.hstack((o, z))
        x=np.vstack((a, c))

        y=int(self.resol/(2*self.tilsiz))
        b=np.tile(x, (y,y))


        self.output = b


    def show(self):
        plt.figure()
        plt.imshow(self.output , cmap='gray')
        plt.show()
