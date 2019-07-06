import numpy as np


class SoftMax:

    def __init__(self, categories, batch_size):

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((categories, batch_size))  # "pre-allocation"
        
    def forward(self, input_tensor):
        
        
        # store the activations from the input_tensor
        self.activations = np.copy(input_tensor)
        
        
        #self.activations = np.exp(e_x) / np.expand_dims(np.sum(np.exp(e_x),axis=1),axis=1)
        self.activations=self.predict(input_tensor)
        

        # apply SoftMax to the scores: e(x_i) / sum(e(x))
        # TODO
        # ...

        return self.activations
    
    def predict(self, input_tensor):
        
        N=np.exp(input_tensor)
        D=np.sum(N,axis=0)
        return (N/D)
        

    def backward(self, label_tensor):

        error_tensor = np.copy(self.activations)

        #  Given:
        #  - the labels are one-hot vectors
        #  - the loss is cross-entropy (as implemented below)
        # Idea:
        # - decrease the output everywhere except at the position where the label is correct
        # - implemented by increasing the output at the position of the correct label
        # Hint:
        # - do not let yourself get confused by the terms 'increase/decrease'
        # - instead consider the effect of the loss and the signs used for the backward pass

        # TODO
        # ...
        
        error_tensor= np.subtract(self.activations,label_tensor)
        
        

        return error_tensor

    def loss(self, label_tensor):

       
        
        loss= np.sum(-label_tensor*np.log(self.activations))
        
                # iterate over all elements of the batch and sum the loss
        # TODO
        # ... # loss is the negative log of the activation of the correct position

        return loss
