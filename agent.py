#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import random
import numpy as np
from collections import deque

# Hyperparameters

class Agent(object):

    def __init__(self, gamma=0.99, observe=10000, intial_epsilon=0.0001,final_epsilon=0.0001,
                 explore_frames=200000, replay_memory=50000, batch_size=32, save_freq):
        self.GAMMA = 0.99  # decay rate of past observations
        self.OBSERVE = 100000.  # to fill the replay memory

        self.INITIAL_EPSILON = 0.0001  # starting value of epsilon
        self.FINAL_EPSILON = 0.0001  # final value of epsilon
        self.EXPLORE_FRAMES = 2000000.  # frames over which to anneal epsilon

        self.REPLAY_MEMORY = 50000  # number of previous transitions to remember
        self.BATCH_SIZE = 32  # size of minibatch
        self.FRAME_PER_ACTION = 1

        self.ACTIONS = 2

        def build_network(self):
            # define the cnn
            W_conv1 = weight_variable([8, 8, 4, 32])
            b_conv1 = bias_variable([32])

            W_conv2 = weight_variable([4, 4, 32, 64])
            b_conv2 = bias_variable([64])

            W_conv3 = weight_variable([3, 3, 64, 64])
            b_conv3 = bias_variable([64])

            W_fc1 = weight_variable([1600, 512])
            b_fc1 = bias_variable([512])

            W_fc2 = weight_variable([512, self.ACTIONS])
            b_fc2 = bias_variable([self.ACTIONS])

            # hidden layers
            h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)



        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W, stride):
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

