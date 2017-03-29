"""Model definitions for MNIST"""
# pylint: disable=C0301,C0103

import tensorflow as tf

def vae_gen(num_images):
    """Definition of the generator"""

    n_z = 20
    n_hidden_gener_1 = 500
    n_hidden_gener_2 = 500
    n_input = 28 * 28
    z = tf.Variable(tf.random_normal((num_images, n_z)), name='z')

    with tf.variable_scope('generator'):
        weights1 = tf.get_variable('w1', shape=[n_z, n_hidden_gener_1])
        bias1 = tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32), name='b1')
        hidden1 = tf.nn.softplus(tf.matmul(z, weights1) + bias1, name='h1')

        weights2 = tf.get_variable('w2', shape=[n_hidden_gener_1, n_hidden_gener_2])
        bias2 = tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32), name='b2')
        hidden2 = tf.nn.softplus(tf.matmul(hidden1, weights2) + bias2, name='h2')

        w_out = tf.get_variable('w_out', shape=[n_hidden_gener_2, n_input])
        b_out = tf.Variable(tf.zeros([n_input], dtype=tf.float32), name='b_out')
        x_hat = tf.nn.sigmoid(tf.matmul(hidden2, w_out) + b_out, name='x_hat')

    restore_path = '../models/mnist/model2.ckpt'
    restore_dict = {'Variable_7': weights1,
                    'Variable_8': weights2,
                    'Variable_9': w_out,
                    'Variable_11': bias1,
                    'Variable_12': bias2,
                    'Variable_13': b_out}

    return z, x_hat, restore_path, restore_dict
