import numpy as np


class ReLU:

    def __init__(self, input_size, batch_size):

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((input_size, batch_size))  # "pre-allocation"

    def forward(self, input_tensor):

        # store the activations from the input_tensor
        # TODO
        self.input_tensor = input_tensor

          # the output is max(0, activation)
        layer_output = np.maximum(self.input_tensor, 0.)  # TODO
        return layer_output

    def backward(self, error_tensor):

        # the gradient is zero whenever the activation is negative
        return error_tensor*(self.input_tensor>0).astype(float)  # TODO
