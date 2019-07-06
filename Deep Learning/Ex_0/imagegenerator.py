import random
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import math

"""
* class: imagegenerator
        * next: gets the next set of "batch size" images to be displayed. order is already randomized
        * class_name: displays labels for images corresponding to the images given by the next() fn
        * show(): takes the fetched random set of images and displays them along with their tags
"""

class Imagegenerator:

    output = None

    def __init__(self, siz, row, batchsize,folder,mirror=False,rotate=False,shuffle=False):
        self.row = row
        self.siz = siz
        self.batchsize = batchsize
        self.col = math.ceil(self.batchsize / self.row)
        self.folder=folder
        self.mirror=mirror
        self.rotate=rotate
        self.shuffle=shuffle
        self.imagelist = glob.glob(self.folder+"/exercise_data/*.npy")
        random.shuffle(self.imagelist)

    def next(self):
        PATH = os.path.abspath(self.folder)
        os.chdir(PATH)
        with open("Labels.json", 'r') as f:
            self.data = json.load(f)
        x = np.array_split(self.imagelist, self.batchsize)

        namelist = []
        self.labellist = []
        for i in x[1]:
                name = os.path.splitext(os.path.basename(i))[0]
                namelist.append(name)
                self.labellist.append(self.data[name])

        self.output=namelist

    def class_name(self):
        print('+--------------------------------------------------------+')
        print('printing labels corresponding to output')
        print('+--------------------------------------------------------+')
        print(self.labellist)
        print('+--------------------------------------------------------+')


    def show(self):

        ref_path = self.folder+"/exercise_data/"
        default_ext = ".npy"
        n = 1
        print('+--------------------------------------------------------+')
        print('printing selected files')
        print('+--------------------------------------------------------+')
        print(self.output)
        print('+--------------------------------------------------------+')
        if self.shuffle:
            random.shuffle(self.output)
            random.shuffle(self.labellist)
            print('+--------------------------------------------------------+')
            print('printing shuffled files')
            print('+--------------------------------------------------------+')
            print(self.output)
            print('+--------------------------------------------------------+')
            print('printing shuffled labels corresponding to output')
            print('+--------------------------------------------------------+')
            print(self.labellist)
            print('+--------------------------------------------------------+')

        for str1 in self.output:
            img_array = np.load(ref_path+str1+default_ext)
            if self.rotate: img_array=np.rot90(img_array)
            if self.mirror: img_array=np.fliplr(img_array)

            """
            * switching off the axis labels for better visibility of image labels
            """
            plt.xticks([])
            plt.yticks([])
            plt.subplot(self.row,self.col,n)
            plt.imshow(img_array)
            plt.title("Label: "+ str(self.labellist[n-1]))
            n+=1
        plt.xticks([])
        plt.yticks([])
        plt.show()