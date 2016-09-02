#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import random
import numpy as np
from collections import deque


class Agent(object):

    def __init__(self, actions=2, gamma=0.99, observe=50000., initial_epsilon=0.0001, final_epsilon=0.0001,
                 explore_frames=2000000., memory_frames=50000, batch_size=32, save_freq=10000):
        self.GAMMA = gamma  # decay rate of past observations
        self.OBSERVE = observe  # to fill the replay memory

        self.INITIAL_EPSILON = initial_epsilon  # starting value of epsilon
        self.FINAL_EPSILON = final_epsilon  # final value of epsilon
        self.EXPLORE_FRAMES = explore_frames  # frames over which to anneal epsilon

        self.MEMORY_FRAMES = memory_frames  # number of previous transitions to remember
        self.BATCH_SIZE = batch_size  # size of minibatch
        self.FRAMES_PER_ACTION = 1

        self.ACTIONS = actions
        self.SAVE_FREQ = save_freq
        self.STATE = 'OBSERVE'
        self.GAME_NAME = 'FLAPPY'

        self.itr_num = -1
        self.epsilon = self.INITIAL_EPSILON
        self.prev_state = None
        self.prev_action = np.zeros(self.ACTIONS)  # no action
        self.prev_action[0] = 1

        self.memory = deque()
        self.session = tf.InteractiveSession()
        self.build_model()
        self.saver = self._init_saver()

    def _init_saver(self, path="saved_networks"):
        # loading weights
        saver = tf.train.Saver()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        return saver

    def build_model(self):
        # define the cnn
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))

        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))

        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
        b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))

        W_fc1 = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.01))
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))

        W_fc2 = tf.Variable(tf.truncated_normal([512, self.ACTIONS], stddev=0.01))
        b_fc2 = tf.Variable(tf.constant(0.01, shape=[self.ACTIONS]))

        # input layer
        s = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self._conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3, 1) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # output layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.model = (s, readout, h_fc1)  # inputlayer, readout, output layer

        a = tf.placeholder("float", [None, self.ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        self.train_model = (a, y, train_step)

    def observe(self, state, reward, terminal):

        # if initialize the first state
        if self.itr_num < 0:
            self.prev_state = state
            self.itr_num = self.itr_num + 1
            return

        # store the transition states in memory
        self.memory.append((self.prev_state, self.prev_action, reward, state, terminal))
        if len(self.memory) > self.MEMORY_FRAMES:
            self.memory.popleft()

        # train after done observing
        if self.itr_num > self.OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(self.memory, self.BATCH_SIZE)

            # get the batch variables
            state_batch = [d[0] for d in minibatch]
            action_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            next_state_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_batch = self.model[1].eval(feed_dict={self.model[0]: next_state_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + self.GAMMA * np.max(readout_batch[i]))

            # perform gradient step
            self.train_model[2].run(feed_dict={self.train_model[1]: y_batch, self.train_model[0]: action_batch,
                                               self.model[0]: state_batch})
        self.prev_state = state
        self.itr_num = self.itr_num + 1

        # save progress every 10000 iterations
        if self.itr_num % self.SAVE_FREQ == 0:
            self.saver.save(self.session, 'saved_networks/' + self.GAME_NAME + '-dqn', global_step=t)

        # print info
        self.STATE = ""
        if self.itr_num <= self.OBSERVE:
            self.STATE = "observe"
        elif self.itr_num > self.OBSERVE and self.itr_num <= self.OBSERVE + self.EXPLORE_FRAMES:
            self.STATE = "explore"
        else:
            self.STATE = "train"

        print("Iteration", self.itr_num, "/ State", self.STATE,
              "/ Epsilon", self.epsilon, "/ Action", self.prev_action, "/ Reward", reward)

    def act(self):
        if self.itr_num < 0:
            return self.prev_action

        # evaluvate the network
        readout_t = self.model[1].eval(feed_dict={self.model[0]: [self.prev_state]})[0]

        a_t = np.zeros([self.ACTIONS])
        if self.itr_num % self.FRAMES_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                a_t[random.randrange(self.ACTIONS)] = 1
            else:
                a_t[np.argmax(readout_t)] = 1
        else:
            a_t[0] = 1

        self.prev_action = a_t

        # vary epsilon
        if self.epsilon > self.FINAL_EPSILON and self.itr_num > self.OBSERVE:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE_FRAMES

        return self.prev_action


    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

