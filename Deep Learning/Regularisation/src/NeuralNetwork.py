from Layers import *
import copy
import pickle




class NeuralNetwork(Base.base_class):

    def __init__(self,optimizer,weights_initializer,bias_initializer):
        super().__init__()

        self.loss=[] #update the loss value after each iteration calling train
        self.layers=[]#holds architecture
        self.data_layer=None#input tensor and data tensor
        self.loss_layer=None#loss and prediction
        self.label_tensor=None
        self.weights_initializer=weights_initializer
        self.bias_initializer=bias_initializer
        self.optimizer=optimizer

    def append_trainable_layer(self,layer):
        layer.set_optimizer(copy.deepcopy(self.optimizer))
        layer.initialize(self.weights_initializer,self.bias_initializer)
        self.layers.append(layer)


    def forward(self,):

        self.input_tensor, self.label_tensor = self.data_layer.forward()#input from data layer
        output = self.input_tensor
        regLoss=0

        for l in self.layers:#pass the input through all layers of network
            output=l.forward(output)
            #self.input_tensor=self.output
            if self.regularizer is not None:
                regLoss += self.calculate_regularization_loss(l)

        loss=self.loss_layer.forward(output,self.label_tensor)



        return loss,regLoss


    def backward(self,):

       error_tensor = self.loss_layer.backward(self.label_tensor)

       for l in reversed(self.layers):
           error_tensor = l.backward(error_tensor)


    def train(self,iteration):

     #update the loss value after each iteration calling train
         self.set_phase(1)
         for i in range(iteration):
             # Get the loss from the op
             loss = self.forward()
             self.loss.append(loss)  # Accumulate the loss\
             self.backward()

    def test(self, input_tensor):

        self.set_phase(2)
        activation_tensor=input_tensor
        for l in self.layers:  # pass the input through all layers of network
            activation_tensor = l.forward(activation_tensor)
        prediction = self.loss_layer.predict(activation_tensor)
        return prediction

    def set_phase(self,phase):
        self.phase = phase
        for l in self.layers:
            l.phase = phase

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data_layer']
        return state

    def __setstate__(self, state):
        # The setstate(state) method should initialize the dropped members with None.
        self.__dict__.update(state)
        self.data_layer = None


def save(filename, net):
    # Save a dictionary into a pickle file.
    temp=net.data_layer
    net.data_layer=None
    pickle.dump(net, open(filename, "wb"))
    net.data_layer=temp



def load(filename, data_layer):
    # Load the dictionary back from the pickle file.
    fileObject = open(filename, "rb")
    net = pickle.load(fileObject)
    net.data_layer = data_layer
    return net





