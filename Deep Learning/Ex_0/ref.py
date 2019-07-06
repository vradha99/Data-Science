import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
import cv2
import os
import glob
import pandas as pd
from scipy.ndimage.filters import convolve
import json
from sklearn.model_selection import train_test_split
import PIL





class Imagegenerator:

    output = None

    def __init__(self, siz, row, col):
        self.row = row
        self.col = col
        self.siz = siz
        self.imagelist = glob.glob('/home/srijeet/PycharmProjects/Deep Learning/data/exercise_data/*.npy')
        self.filename = str.replace(str(self.imagelist), "'/home/srijeet/PycharmProjects/Deep Learning/data/exercise_data/",
                               "")

    def next(self):
        PATH = os.path.abspath("/home/srijeet/PycharmProjects/Deep Learning/data")
        os.chdir(PATH)
        with open("Labels.json", 'r') as f:
            data = json.load(f)
            namelist=list(data)
            #print(namelist)

        d = self.row * self.col


            #img_array = np.load(str(x)+".npy")
            #plt.imshow(img_array)


        #for i in range(0,10,1):
        #final = (str.replace(str(self.filename), ".npy'", ""))
        #print(final)
        #os.chdir('/home/brian/Desktop/cats/')
        #files = os.listdir('/home/brian/Desktop/cats/')
        fig, axes = plt.subplots(nrows=self.row, ncols=self.col, figsize=(self.siz, self.siz))
        for ax in axes.flatten():
            ax.axis('off')
        y = np.array_split(namelist, 10)
        y1 = np.array_split(y[0], 10)
        #image_id_list = ['{}_{}'.format(i, j) for i in range(2011, 2016) for j in range(1, 3)]
        print(image_id_list)
        x=np.array_split(self.imagelist, 10)
        for i, image_id in enumerate(y1):
            #raw_image_path = './Images/' + image_id + 'jpg'
            raw_image = Image.open(x[1])
            axes[0, i].imshow(raw_image)











        #for i in
        #plt.imshow(x[1])
        #fig.add_subplot(self.row, self.col, i)
        #print(f)
        print(x[1])


        #print(y1)

        # print(final)
        # print(self.imagelist)
        # print(PATH)
        # abspath::Return a normalized absolutized version of the pathname path.
        # join::Join one or more path components intelligently.
        # SOURCE_IMAGES = os.path.join(PATH, "sample", "images")
        # images = glob(os.path.join(SOURCE_IMAGES, "*.npy"))
        # labels = pd.read_json('/PycharmProjects/Deep Learning/data/Labels.json')
        # images[0:99]
        # for i in range(0, len(self.imagelist), 10):
        # Create an index range for l of n items:
        # x=list(self.imagelist[i:i + 10])
        # a = np.linspace(0, 99, 100)
        # b = a[0:d]
        # np.random.shuffle(b)
        # os.chdir("/home/srijeet/PycharmProjects/Deep Learning/data/exercise_data")
        #for i in range(0, d):
            #b = a[0:d]
            #f2= data[str.replace(str(b[i]), ".0", "")]
        #print(f2)

        plt.show()

######################################################################################
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
import cv2
import os
import glob
import pandas as pd
from scipy.ndimage.filters import convolve
import json
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image


class Imagegenerator:
    output = None

    def __init__(self, siz, row, col):
        self.row = row
        self.col = col
        self.siz = siz
        self.image_list = (glob.glob('/home/srijeet/PycharmProjects/Deep Learning/data/exercise_data/*.npy')

    def next(self):
        PATH = os.path.abspath("/home/srijeet/PycharmProjects/Deep Learning/data")
        os.chdir(PATH)
        with open("Labels.json", 'r') as f:
            data = json.load(f)
        namelist = list(data)
        del namelist[101:1001]
        name_sub_list = np.array_split(namelist, 10)
        print(name_sub_list)
        img = cv2.imread("/home/srijeet/PycharmProjects/Deep Learning/data/exercise_data/61.npy",CV_LOAD_IMAGE_GRAYSCALE )
        #for i in range(0, len(self.imagelist), 10):
         #   x = list(self.imagelist[i:i + 10])
        #print(x)
        self.output=img

    def show(self):
        fig, axes = plt.subplots(nrows=self.row, ncols=self.col, figsize=(self.siz, self.siz))

        cv2.imshow(self.output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ######################################################################################################
        #Final Impleamentation(Wenyu)###########################################################################
        ###################################################################
        import random
        import numpy as np
        import h5py
        import matplotlib.pyplot as plt
        import scipy.io
        from scipy import signal
        from matplotlib import patches
        import cv2
        import os
        import glob
        import pandas as pd
        from scipy.ndimage.filters import convolve
        import json
        from sklearn.model_selection import train_test_split
        import PIL
        import operator
        import math

        class Imagegenerator:

            output = None
            """
            * get the list of the images using glob
            """

            def __init__(self, siz, row, batchsize):
                self.row = row

                self.siz = siz
                self.batchsize = batchsize
                self.col = math.ceil(self.batchsize / self.row)
                print(self.col)
                self.imagelist = glob.glob('/home/srijeet/PycharmProjects/Deep Learning/data/exercise_data/*.npy')
                random.shuffle(self.imagelist)
                # self.filename =str.replace(str(self.imagelist), "'/home/srijeet/PycharmProjects/Deep Learning/data/exercise_data/",
                # "")

            def next(self):
                PATH = os.path.abspath("/home/srijeet/PycharmProjects/Deep Learning/data")
                os.chdir(PATH)
                with open("Labels.json", 'r') as f:
                    self.data = json.load(f)
                    """
                    * delete the excess data
                    """
                    namelist = list(self.data)
                    namelist = list(self.data)
                    # del namelist[101:1001]
                    """
                    * creating a batch of 10 serial indexes
                    """
                    # name_sub_list = np.array_split(namelist, 10)
                    # print(self.filename)
                    # refname=list(str.replace(str(self.filename),"npy'",""))
                    x = np.array_split(self.imagelist, self.batchsize)
                    namelist = []
                    self.labellist = []
                    for i in x[0]:
                        name = os.path.splitext(os.path.basename(i))[0]
                        namelist.append(name)
                        self.labellist.append(self.data[name])
                    print(self.labellist)
                    # refname=os.path.splitext(self.filename)
                    print(namelist)
                    # ref=refname[:20]

                    # y = np.array_split(namelist, self.batchsize)
                    # print(y)
                    # print(self.imagelist)
                    # f=x[0]
                    # while '.' in ref: ref.remove('.')
                    # while ',' in ref: ref.remove(',')
                    # while ' ' in ref: ref.remove(' ')
                    # while '[' in ref: ref.remove('[')
                    # ref=[x for x in ref if type(x) == str]

                    # img_to_display = ["67","89","45","67"]
                    # print(ref)

                    self.output = namelist

                    """
                    * refname: contains the relevant indexes of the images to be displayed
                    * imagelist: contains list of images along with their path
                    """
                    """
                    d = self.row * self.col
                    #fig, axes = plt.subplots(nrows=self.row, ncols=self.col, figsize=(self.siz, self.siz))
                    y = np.array_split(namelist, 10)
                    y1 = np.array_split(y[0], 10)


                    #x=np.array_split(self.imagelist, 10)
                    print(x)
                    plt.show()
                    """

            def show(self):
                ref_path = "/home/srijeet/PycharmProjects/Deep Learning/data/exercise_data/"

                default_ext = ".npy"

                """
                * fix subplot size (3, 4) or (4, 3)
                * fetch filenames as per batch size in img_to_display
                * generate an indexing logic for adding implots in axarr. Will not work without this
                """

                # f, axarr = plt.subplots(nrows=self.row, ncols=self.col, figsize=(self.siz, self.siz))
                it = 0
                cnt = 2
                n = 1
                print(self.output)
                for str1 in self.output:
                    img_array = np.load(ref_path + str1 + default_ext)
                    # img_array=np.rot90(img_array)
                    # img_array=np.fliplr(img_array)
                    # axarr[2,2,n].imshow(img_array)
                    plt.subplot(self.row, self.col, n)
                    plt.imshow(img_array)
                    # plt.title()
                    # axarr[int(it/cnt), int(it % cnt)].imshow(img_array)
                    it += 1
                    n += 1

                """
                * re-generate the index of the image to cross-ref with data to extract tag

                for i in range(self.row): #rows
                    for j in range(self.col): #cols
                        #axarr[i,j].plot()

                        #axarr[i, j].set_title(self.data[self.output[cnt * i + j]])

                """
                plt.show()
                ####################################################################################################
                #####################################################################################################
                #####final Srijeet#######################################################################
                #####################################################################################

                import random
                import numpy as np
                import matplotlib.pyplot as plt
                import os
                import glob
                import json
                import math

                class Imagegenerator:

                    output = None

                    def __init__(self, siz, row, batchsize, folder, mirror=False, rotate=False, shuffle=True):
                        self.row = row
                        self.siz = siz
                        self.batchsize = batchsize
                        self.col = math.ceil(self.batchsize / self.row)
                        self.folder = folder
                        self.mirror = mirror
                        self.rotate = rotate
                        self.shuffle = shuffle
                        self.imagelist = glob.glob(self.folder + "/exercise_data/*.npy")
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

                        self.output = namelist

                    def class_name(self):
                        print(self.labellist)

                    def show(self):
                        ref_path = self.folder + "/exercise_data/"
                        default_ext = ".npy"
                        n = 1
                        if self.shuffle:
                            random.shuffle(self.output)
                            random.shuffle(self.labellist)
                        for str1 in self.output:
                            img_array = np.load(ref_path + str1 + default_ext)

                            if self.rotate: img_array = np.rot90(img_array)
                            if self.mirror: img_array = np.fliplr(img_array)

                            plt.subplot(self.row, self.col, n)
                            plt.imshow(img_array)
                            plt.title(str(self.labellist[n - 1]))
                            n += 1

                        plt.show()






















