import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class Trainer:

    def __init__(self, loss, predictions, optimizer, ds_train, ds_validation, stop_patience, evaluation, inputs, labels):
        '''
            Initialize the trainer

            Args:
                loss        	an operation that computes the loss
                predictions     an operation that computes the predictions for the current
                optimizer       optimizer to use
                ds_train        instance of Dataset that holds the training data
                ds_validation   instance of Dataset that holds the validation data
                stop_patience   the training stops if the validation loss does not decrease for this number of epochs
                evaluation      instance of Evaluation
                inputs          placeholder for model inputs
                labels          placeholder for model labels
        '''

        self._train_op = optimizer.minimize(loss)

        self._loss = loss
        self._predictions = predictions
        self._ds_train = ds_train
        self._ds_validation = ds_validation
        self._stop_patience = stop_patience
        self._evaluation = evaluation
        self._validation_losses = []
        self._model_inputs = inputs
        self._model_labels = labels

        with tf.variable_scope('model', reuse = True):
            self._model_is_training = tf.get_variable('is_training', dtype = tf.bool)

    def _train_epoch(self, sess):
        '''
            trains for one epoch and prints the mean training loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''

        # TODO

        pass

    def _valid_step(self, sess):
        '''
            run the validation and print evalution + mean validation loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''

        # TODO

        pass

    def _should_stop(self):
        '''
            determine if training should stop according to stop_patience
        '''

        # TODO

        pass

    def run(self, sess, num_epochs = -1):
        '''
            run the training until num_epochs exceeds or the validation loss did not decrease
            for stop_patience epochs

            args:
                sess        the tensorflow session that should be used
                num_epochs  limit to the number of epochs, -1 means not limit
        '''

        # initial validation step
        self._valid_step(sess)

        i = 0

        # training loop
        while i < num_epochs or num_epochs == -1:
            self._train_epoch(sess)
            self._valid_step(sess)
            i += 1

            if self._should_stop():
                break

