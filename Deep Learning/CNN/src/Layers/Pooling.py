import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from numpy import pi
import turtle
from scipy.ndimage.filters import convolve


class Pooling:
    def __init__(self,stride_shape, pooling_shape):

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape



    def forward(self, input_tensor):


        self.input_tensor = input_tensor
        self.image_spatial_shape = self.input_tensor.shape[2:]
        self.depth = self.input_tensor.shape[1]
        self.input_image_shape = self.input_tensor[1:]
        self.out_image_shape = []
        self.out_image_shape = []


        #Reduced Dimension
        for input_dim, stride, k in zip(self.input_tensor.shape[2:], self.stride_shape, self.pooling_shape):
            temp = np.int(np.floor((np.float(input_dim) - np.float(k)) / np.float(stride)) + 1)
            self.out_image_shape.append(temp)



        #batch = input_tensor.shape[0]
        out_spatial_img = np.zeros((self.input_tensor.shape[1], *self.out_image_shape))
        #out_spatial_size = np.prod(out_spatial_img.shape)


        output_tensor = np.zeros((input_tensor.shape[0], *out_spatial_img.shape))
        # address stores y and x position of max value
        self.maxaddress = np.zeros((input_tensor.shape[0], self.input_tensor.shape[1], np.prod(self.out_image_shape), 2), dtype=np.int)

        for i in range(input_tensor.shape[0]):
            image = input_tensor[i, :].reshape(self.input_tensor.shape[1:])
            #print(self.input_tensor.shape[1])

            for j, channel in enumerate(image):
                y_count = 0
                for y in range(0, self.input_tensor.shape[2], self.stride_shape[0]):
                    x_count = 0
                    for x in range(0, self.input_tensor.shape[3], self.stride_shape[1]):

                        # max_val = this_channel[y, x] #set the first val in filter shaped image  as max

                        if y_count < self.out_image_shape[0] and x_count < self.out_image_shape[1]:
                            # 2    #make sure within output shape
                            slicey = slice(y, y + self.pooling_shape[0])
                            slicex = slice(x, x + self.pooling_shape[1])
                            try:
                                maxval = np.max(channel[slicey, slicex])
                                self.maxaddress[i, j, (y_count * self.out_image_shape[1] + x_count)] = np.where(
                                    channel == maxval)
                                out_spatial_img[j, y_count, x_count] = maxval
                                x_count += 1
                            except ValueError:  # raised if `y` is empty.
                                pass

                    y_count += 1

            output_tensor[i] = out_spatial_img

        return output_tensor

    def backward(self, error_tensor):

        #batch = error_tensor.shape[0]

        back_error_tensor = np.zeros(self.input_tensor.shape)

        for i in range(error_tensor.shape[0]):
            # original shape of error before pooling
            back_error = np.zeros(self.input_tensor.shape[1:])
            # shape after pooling
            error = error_tensor[i, :].reshape(self.input_tensor.shape[1], *reversed(self.out_image_shape))
            error_address = self.maxaddress[i]

            for j, error_channel in enumerate(error):
                error_channel = error_channel.ravel()
                address_layer = error_address[j]

                for coordinate, value in zip(address_layer, error_channel):
                    y = coordinate[0]
                    x = coordinate[1]
                    back_error[j, y, x] += value

            back_error_tensor[i,:,:,:] = back_error


        return back_error_tensor