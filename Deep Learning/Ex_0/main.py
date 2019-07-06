import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from checkers import Checker
from spectrums import Spectrum
from circle import Circle
from imagegenerator import Imagegenerator


def main():
    #checker = Checker(16,4)
    #checker.draw()
    #checker.show()

    #spectrum = Spectrum(500)
    #spectrum.draw()
    #spectrum.show()

    circle = Circle(500, 70, 200, 200)
    circle.draw()
    circle.show()

    #imagegenerator = Imagegenerator(3,4,9,"/home/srijeet/PycharmProjects/Deep Learning/data",rotate=True)
    #imagegenerator.next()
    #imagegenerator.class_name()
    #imagegenerator.show()


if __name__== "__main__":
    main()