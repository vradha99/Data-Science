import sys
import NeuralNetwork
from Layers import *
from Optimization import *


def build():




    optimizer = Optimizers.Adam(pow(5, -4), 0.999, 0.9)
    regularizer = Constraints.L2_Regularizer(4*(10**-4))
    optimizer.add_regularizer(regularizer)



    net = NeuralNetwork.NeuralNetwork(optimizer,
                                      Initializers.He(),
                                      Initializers.Constant(0.1))

    C1 = Conv.Conv((1, 1), (1, 5, 5), 6)
    net.append_trainable_layer(C1)


    S1 = Pooling.Pooling((2, 2), (2, 2))
    net.layers.append(S1)

    net.layers.append(ReLU.ReLU())

    C2 = Conv.Conv((1, 1), (6, 5, 5), 16)
    net.append_trainable_layer(C2)


    S2 = Pooling.Pooling((2, 2), (2, 2))
    net.layers.append(S2)

    net.layers.append(ReLU.ReLU())

    S3=Flatten.Flatten()
    net.layers.append(S3)


    fc1 = FullyConnected.FullyConnected(784, 120)
    net.append_trainable_layer(fc1)
    net.layers.append(ReLU.ReLU())

    fc2 = FullyConnected.FullyConnected(120, 84)
    net.append_trainable_layer(fc2)
    net.layers.append(ReLU.ReLU())

    fc3 = FullyConnected.FullyConnected(84, 10)
    net.append_trainable_layer(fc3)
    net.layers.append(ReLU.ReLU())

    net.loss_layer = SoftMax.SoftMax()

    return net
