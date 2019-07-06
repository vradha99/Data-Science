import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

def conv2d(x, channels, filter_size, stride):
    x = tf.nn.conv2d(x, channels, filter_size, stride, padding='SAME')
    x = tf.contrib.layers.batch_norm(x)
    x = tf.nn.relu(x)
    return x

def maxpool2d(x, pool_size=3, stride=2):
    return tf.nn.max_pool(x, psize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME')


def create(x, num_outputs, dropout_rate = 0.5):
    '''
        args:
            x               network input
            num_outputs     number of logits
            dropout         dropout rate during training
    '''
    self.x = x
    self.num_outputs = num_outputs
    self.dropout_rate = dropout_rate

    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)

    # TODO
    weights={
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([2, num_outputs]))
    }
    # Convolution layer
    conv1 = conv2d(x, channels=96, filter_size=11, stride=4,)
    conv1 = maxpool2d(conv1, pool_size=3, stride=2)
    conv2 = conv2d(conv1, channels=192, filter_size=5, stride=1)
    conv2 = maxpool2d(conv2, pool_size=3, stride=2)
    conv3 = conv2d(conv2, channels=384, filter_size=3, stride=1)
    conv4 = conv2d(conv3, channels=256, filter_size=3, stride=1)
    conv5 = conv2d(conv4, channels=256, filter_size=3, stride=1)
    conv5 = maxpool2d(conv5, pool_size=3, stride=2)
    # Dropout
    conv5 = tf.nn.dropout(conv5, dropout_rate)
    # Fully connected layer
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, weights['wd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout_rate)
    fc2 = tf.matmul(fc1, weights['wd2'])
    fc2 = tf.nn.relu(fc2)
    # output
    out = tf.add(tf.matmul(fc2, weights['out']))
    return out

