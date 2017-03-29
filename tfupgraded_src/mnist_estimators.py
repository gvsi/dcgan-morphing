"""Estimators for compressed sensing"""
# pylint: disable=C0301,C0103

from sklearn.linear_model import Lasso
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
import tensorflow as tf
import mnist_model_def
import utils


def lasso_estimator(hparams):
    """LASSO estimator"""
    lasso_est = Lasso(alpha=hparams.lmbd)
    def estimator(A_val, y_val, hparams):
        lasso_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
        x_hat = lasso_est.coef_
        x_hat = np.reshape(x_hat, [1, -1])
        x_hat = np.maximum(np.minimum(x_hat, 1), 0)
        return x_hat
    return estimator


def omp_estimator(hparams):  #pylint: disable=unused-argument
    """Orthogonal Matching Pursuit"""
    omp_est = OrthogonalMatchingPursuit()
    def estimator(A_val, y_val, hparams):
        omp_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
        x_hat = omp_est.coef_
        x_hat = np.reshape(x_hat, (1, hparams.n_input))
        x_hat = np.maximum(np.minimum(x_hat, 1), 0)
        return x_hat
    return estimator


def vae_gen_estimator(hparams):

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y = tf.placeholder(tf.float32, shape=(1, hparams.num_measurements), name='y')

    # Create the generator
    z, x_hat, restore_path, restore_dict = mnist_model_def.vae_gen(1)
    z_likelihood_loss = tf.reduce_sum(z ** 2)

    # measure the generator output
    y_hat = tf.matmul(x_hat, A, name='y_hat')
    measurement_loss = tf.reduce_mean((y - y_hat) ** 2)

    # define total loss
    loss = tf.add(measurement_loss/(hparams.noise_std**2), 10*z_likelihood_loss, name='loss')

    # Set up gradient descent wrt to z
    hparams.learning_rate = hparams.learning_rate * (hparams.noise_std**2)
    opt = utils.get_optimizer(hparams)
    update_op = opt.minimize(loss, var_list=[z], name='update_op')

    # Get a session
    sess = tf.Session()

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""
        measurement_loss_best = 1e10
        for _ in range(hparams.num_random_restarts):
            sess.run([z.initializer])
            for _ in range(hparams.max_update_iter):
                feed_dict = {A: A_val, y: y_val}
                _, measurement_loss_val = sess.run([update_op, measurement_loss], feed_dict=feed_dict)
            if measurement_loss_val < measurement_loss_best:
                measurement_loss_best = measurement_loss_val
                x_hat_best_val = sess.run(x_hat)
        return x_hat_best_val

    return estimator
