# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:55:45 2018

@author: Shashwat Pathak
"""

# =============================================================================
# Chapter 0: Import modules
# =============================================================================
import tensorflow as tf
import logging
tf.set_random_seed(42)


# =============================================================================
# Chapter 1: Build neural network architecture
# =============================================================================
class Artemis():
    '''
    Class to build, train and score Neural Network
    '''
    def __init__(self, n_dim):
        '''
        Initialize
        '''
        # Number of predictors
        self.n_dim = n_dim

        # Placeholder for X
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_dim))
        # Placeholder for y
        self.y = tf.placeholder(tf.float32, shape=(None,))

        # Initialize hidden layer weights
        self.ih1 = tf.Variable(tf.truncated_normal([self.n_dim, 500],
                                                   stddev=1e-5))
#        self.h1h2 = tf.Variable(tf.truncated_normal([5, 5],
#                                                    stddev=1e-5))
#        self.h2h3 = tf.Variable(tf.truncated_normal([5, 5],
#                                                    stddev=1e-5))
        self.h3o = tf.Variable(tf.truncated_normal([500, 1],
                                                   stddev=1e-5))
#        self.io = tf.Variable(tf.truncated_normal([self.n_dim, 1],
#                                                  stddev=1e-5))

        # Training parameters
        self.lr = 1e-4
        self.epochs = 500
        self.l2_alpha = 0.2

        # Initialize network
        self.optimizer, self.cost, self.acc, self.out_layer = self._build()

        # Initialize tensorflow
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _build(self):
        '''
        Build architecture
        '''
        with tf.variable_scope('model_weights'):
            # First hidden layer
            h1_layer = tf.nn.relu(tf.matmul(self.X, self.ih1))
            # Second hidden layer
#            h2_layer = tf.nn.relu(tf.matmul(h1_layer, self.h1h2))
#             Third hidden layer
#            h3_layer = tf.nn.relu(tf.matmul(h2_layer, self.h2h3))
            # Out layer
            out_layer = tf.nn.relu(tf.matmul(h1_layer, self.h3o))

        # Calculate cost
        cost = tf.reduce_mean(tf.square(out_layer - tf.expand_dims(self.y, 1)))

        # Add regularization
        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='model_weights')
        for var in model_vars:
            cost += self.l2_alpha*tf.nn.l2_loss(var)

        # Minimize weights
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)

        # Calculate accuracy
        acc = tf.square(out_layer - tf.expand_dims(self.y, 1))
        acc = tf.reduce_mean(acc)

        return optimizer, cost, acc, out_layer

    def train(self, X, y):
        '''
        Train the neural network
        '''
        # Train epoch
        for epoch in range(self.epochs):
            _, c = self.sess.run([self.optimizer, self.cost],
                                 feed_dict={self.X: X, self.y: y})

            # Print results
            logging.info('Epoch: {}, cost= {:.4f}'.format(epoch+1, c))

        logging.info('Training complete')
        logging.info('Accuracy= {:.4f}'.format(self.acc.eval({self.X: X,
                                                              self.y: y},
                     session=self.sess)))

    def predict(self, X):
        '''
        Predict results from the neural network
        '''
        return self.out_layer.eval({self.X: X}, session=self.sess)
