import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib import patches
from Layers import *



class NeuralNetwork:

    def __init__(self,):

        self.loss=[] #update the loss value after each iteration calling train
        self.layers=[]#holds architecture
        self.data_layer=None#input tensor and data tensor
        self.loss_layer=None#loss and prediction
        self.label_tensor=None



    def forward(self,):

        self.input_tensor, self.label_tensor = self.data_layer.forward()#input from data layer
        output = self.input_tensor

        for l in self.layers:#pass the input through all layers of network
            output=l.forward(output)
            #self.input_tensor=self.output

        loss=self.loss_layer.forward(output,self.label_tensor)

        return loss


    def backward(self,):

       error_tensor = self.loss_layer.backward(self.label_tensor)

       for l in reversed(self.layers):
           error_tensor = l.backward(error_tensor)


    def train(self,iteration):

     #update the loss value after each iteration calling train

     for i in range(iteration):
         # Get the loss from the op
         loss = self.forward()
         self.loss.append(loss)  # Accumulate the loss\
         self.backward()

    def test(self, input_tensor):

        activation_tensor=input_tensor
        for l in self.layers:  # pass the input through all layers of network
            activation_tensor = l.forward(activation_tensor)
        prediction = self.loss_layer.predict(activation_tensor)
        return prediction



if __name__== "__main__":
    main()
