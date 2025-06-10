##
## EPITECH PROJECT, 2025
## Robotutur
## File description:
## Robotutur
##

import numpy as np
import tensorflow as tf

class CNN:

    #conv params should be an array of dict with num_filters and filter_size as key
    def __init__(self, input_shape, conv_params, output_size, alpha):
        self._alpha = alpha
        self._input_shape = input_shape
        self._output_size = output_size
        self._conv_layers = []

        channels = input_shape[2]
        height, width = input_shape[0], input_shape[1]

        for params in conv_params:
            num_filters = params["num_filters"]
            filter_size = params["filter_size"]

            filter_shape = [filter_size, filter_size, channels, num_filters]
            filters = tf.Variable(tf.random.normal(filter_shape) * tf.sqrt(2.0 / (filter_size * filter_size * channels)))
            bias = tf.Variable(tf.zeros([num_filters]))

            self._conv_layers.append({"filters": filters, "bias": bias})

            height = (height - filter_size + 1) / 2
            width = (width - filter_size + 1) / 2
            channels = num_filters

        fc_input_size = height * width * channels
        self._W_fc = tf.Variable(tf.random.normal([fc_input_size, output_size]) * tf.sqrt(2.0 / fc_input_size))
        self._b_fc = tf.Variable(tf.zeros([output_size]))

    def relu(self, x):
        return tf.maximum(0.0, x)

    def derivateRelu(self, x):
        return tf.cast(x > 0, tf.float32)

    def convolve(self, X, filters, bias):
        return tf.nn.conv2d(X, filters, strides=1, padding='SAME') + bias

    def max_pool(self, X):
        return tf.nn.max_pool2d(X, ksize=2, strides=2, padding='SAME')

    def propagateForward(self, X_value):
        self._X = tf.cast(X_value, tf.float32)
        out = self._X
        self._conv_outs = []

        for layer in self._conv_layers:
            conv = self.convolve(out, layer["filters"], layer["bias"])
            act = self.relu(conv)
            pooled = self.max_pool(act)
            self._conv_outs.append((out, conv, act, pooled))
            out = pooled

        self._flatten = tf.reshape(out, [tf.shape(self._X)[0], -1])

        z = tf.matmul(self._flatten, self._W_fc) + self._b_fc
        output = tf.nn.softmax(z)
        self._z_fc = z
        self._output = output
        return output

    def propagateBackward(self, y_value, output):
        return

    def computeParams():
        return

    def computeLoss(self, y_pred, y_true):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=self._z_fc))

    def train(self, X_values, y_values, epochs, size):
        X_values = tf.cast(X_values, tf.float32)
        y_values = tf.cast(y_values, tf.float32)
        number_values = tf.shape(X_values)[0]

        for epoch in range(epochs):
            indices = tf.random.shuffle(tf.range(number_values))
            X_shuffled = tf.gather(X_values, indices)
            y_shuffled = tf.gather(y_values, indices)

            for start_idx in range(0, number_values, size):
                end_idx = min(start_idx + size, number_values)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                output = self.propagateForward(X_batch)
                loss = self.computeLoss(output, y_batch)
                self.computeBackward(y_batch, output);
                self.computeParams

