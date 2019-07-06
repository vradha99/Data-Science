import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

def conv2d(x, channels, filter_size, stride):
    x = tf.nn.conv2d(x, channels, filter_size, stride, padding='SAME')
    x = tf.contrib.layers.batch_norm(x)
    x = tf.nn.relu(x)
    return x


def maxpool2d(x, pool_size=3, stride=2):
    return tf.nn.max_pool(x, psize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1],padding='SAME')


def _residual_block_first(self, x, channels, stride):
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == channels:
                if stride == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, channels, stride)
            # Residual
            x = conv2d(x, channels, 3, stride)
            x = self._bn(x)
            x = self._relu(x)
            x = conv2d(x, channels, 3, 1)
            x = self._bn(x)
            # Merge
            x = x + shortcut
            x = self._relu(x)
            return x

def _bn(self, x):
        x = utils._bn(x, self.is_train, self._global_step)
    # f = 8 * self._get_data_size(x)
    # w = 4 * x.get_shape().as_list()[-1]
    # scope_name = tf.get_variable_scope().name + "/" + name
    # self._add_flops_weights(scope_name, f, w)
        return x

def _relu(self, x):
    x = utils._relu(x, 0.0)
    # f = self._get_data_size(x)
    # scope_name = tf.get_variable_scope().name + "/" + name
    # self._add_flops_weights(scope_name, f, 0)
    return x


def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''

    self.x = x
    self.num_outputs = num_outputs
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)
    
    # TODO
    weights={

        'out': tf.Variable(tf.random_normal([2, num_outputs]))
    }
    x_out = conv2d(x, channels=34, filter_size=7, stride=2, )
    x_out = maxpool2d(x_out, pool_size=3, stride=2)
    x_out = self._residual_block_first(x_out, 64,1)
    x_out = self._residual_block_first(x_out, 128, 2)
    x_out = self._residual_block_first(x_out, 256, 2)
    x_out = self._residual_block_first(x_out, 512, 2)
    x_out = tf.reduce_mean(x_out, [1, 2])
    fc1 = tf.reshape(x_out, [-1, weights['out'].get_shape().as_list()[0]])
    out = tf.matmul(fc1, weights['out'])

    return out


    pass

