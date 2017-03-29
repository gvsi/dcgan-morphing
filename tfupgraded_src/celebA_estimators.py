"""Estimators for compressed sensing"""
# pylint: disable=C0301,C0103

import tensorflow as tf
import numpy as np
import utils
import scipy.fftpack as fftpack
from sklearn.linear_model import Lasso

import dcgan_model
import dcgan_ops


tf.app.flags.DEFINE_integer("m", 100, "Measurements [100]")
tf.app.flags.DEFINE_integer("nIter", 100, "Update steps[100]")
tf.app.flags.DEFINE_float("snr", 0.01, "Noise energy[0.01]")
tf.app.flags.DEFINE_float("lam", None, "Regularisation[None]")
tf.app.flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
tf.app.flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
tf.app.flags.DEFINE_integer("image_size", 108,
                            "The size of image to use (will be center cropped) [108]")
tf.app.flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
tf.app.flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
tf.app.flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
tf.app.flags.DEFINE_string("checkpoint_dir", "../models/",
                           "Directory name to save the checkpoints [checkpoint]")
tf.app.flags.DEFINE_string("sample_dir", "samples",
                           "Directory name to save the image samples [samples]")
tf.app.flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
tf.app.flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
FLAGS = tf.app.flags.FLAGS


def dcgan_discrim(x_hat, sess):
    dcgan = dcgan_model.DCGAN(sess,
                              image_size=FLAGS.image_size,
                              batch_size=FLAGS.batch_size,
                              output_size=FLAGS.output_size,
                              c_dim=FLAGS.c_dim,
                              dataset_name=FLAGS.dataset,
                              is_crop=FLAGS.is_crop,
                              checkpoint_dir=FLAGS.checkpoint_dir,
                              sample_dir=FLAGS.sample_dir)

    x_hat_image = tf.reshape(x_hat, [1, 64, 64, 3])
    all_zeros = tf.zeros([64, 64, 64, 3])
    discrim_input = all_zeros + x_hat_image
    prob, _ = dcgan.discriminator(discrim_input, is_train=False)
    d_loss = - tf.log(prob[0])

    restore_vars = ['d_bn1/beta',
                    'd_bn1/gamma',
                    'd_bn1/moving_mean',
                    'd_bn1/moving_variance',
                    'd_bn2/beta',
                    'd_bn2/gamma',
                    'd_bn2/moving_mean',
                    'd_bn2/moving_variance',
                    'd_bn3/beta',
                    'd_bn3/gamma',
                    'd_bn3/moving_mean',
                    'd_bn3/moving_variance',
                    'd_h0_conv/biases',
                    'd_h0_conv/w',
                    'd_h1_conv/biases',
                    'd_h1_conv/w',
                    'd_h2_conv/biases',
                    'd_h2_conv/w',
                    'd_h3_conv/biases',
                    'd_h3_conv/w',
                    'd_h3_lin/Matrix',
                    'd_h3_lin/bias']

    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint('../models/celebA_64_64/')

    return d_loss, restore_dict, restore_path


def dcgan_discrim_estimator(hparams):

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.n_input, 3), name='A')
    y = tf.placeholder(tf.float32, shape=(1, hparams.num_measurements), name='y')

    # Create the discriminator
    x_hat = tf.Variable(tf.random_uniform([1, 64*64*3], minval=-1, maxval=1), name='x_hat')
    d_loss, restore_dict, restore_path = dcgan_discrim(x_hat, sess)

    # measure the estimate
    y_hat = tf.matmul(x_hat, A, name='y_hat')
    measurement_loss = tf.reduce_mean((y - y_hat) ** 2)

    # define loss
    loss = tf.add(measurement_loss/(hparams.noise_std**2), 20*d_loss, name='loss')

    # Set up gradient descent wrt to x_hat
    hparams.learning_rate = hparams.learning_rate * (hparams.noise_std**2)
    opt = utils.get_optimizer(hparams)
    train_op = opt.minimize(loss, var_list=[x_hat], name='train_op')
    with tf.control_dependencies([train_op]):
        project_op = tf.assign(x_hat, tf.maximum(tf.minimum(x_hat, 1), -1), name='project_op')
        update_op = tf.group(train_op, project_op, name='update_op')

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    # Function that returns the estimated image
    def estimator(A_val, y_val, hparams):
        measurement_loss_best = 1e10
        for _ in range(hparams.num_random_restarts):
            sess.run([x_hat.initializer])
            for _ in range(hparams.max_update_iter):
                feed_dict = {A: A_val, y: y_val}
                _, measurement_loss_val = sess.run([update_op, measurement_loss], feed_dict=feed_dict)
                print 'd_loss = {0}, m_loss = {1}'.format(sess.run(d_loss), measurement_loss_val)
            if measurement_loss_val < measurement_loss_best:
                measurement_loss_best = measurement_loss_val
                x_hat_best_val = sess.run(x_hat)
        return x_hat_best_val

    return estimator



def dcgan_gen(z, sess):

    dcgan = dcgan_model.DCGAN(sess,
                              image_size=FLAGS.image_size,
                              batch_size=FLAGS.batch_size,
                              output_size=FLAGS.output_size,
                              c_dim=FLAGS.c_dim,
                              dataset_name=FLAGS.dataset,
                              is_crop=FLAGS.is_crop,
                              checkpoint_dir=FLAGS.checkpoint_dir,
                              sample_dir=FLAGS.sample_dir)

    tf.get_variable_scope().reuse_variables()

    s = dcgan.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    # project `z` and reshape
    h0 = tf.reshape(dcgan_ops.linear(z, dcgan.gf_dim*8*s16*s16, 'g_h0_lin'),
                    [-1, s16, s16, dcgan.gf_dim * 8])
    h0 = tf.nn.relu(dcgan.g_bn0(h0, train=False))

    h1 = dcgan_ops.deconv2d(h0, [dcgan.batch_size, s8, s8, dcgan.gf_dim*4], name='g_h1')
    h1 = tf.nn.relu(dcgan.g_bn1(h1, train=False))

    h2 = dcgan_ops.deconv2d(h1, [dcgan.batch_size, s4, s4, dcgan.gf_dim*2], name='g_h2')
    h2 = tf.nn.relu(dcgan.g_bn2(h2, train=False))

    h3 = dcgan_ops.deconv2d(h2, [dcgan.batch_size, s2, s2, dcgan.gf_dim*1], name='g_h3')
    h3 = tf.nn.relu(dcgan.g_bn3(h3, train=False))

    h4 = dcgan_ops.deconv2d(h3, [dcgan.batch_size, s, s, dcgan.c_dim], name='g_h4')

    x_hat = tf.nn.tanh(h4)

    restore_vars = ['g_bn0/beta',
                    'g_bn0/gamma',
                    'g_bn0/moving_mean',
                    'g_bn0/moving_variance',
                    'g_bn1/beta',
                    'g_bn1/gamma',
                    'g_bn1/moving_mean',
                    'g_bn1/moving_variance',
                    'g_bn2/beta',
                    'g_bn2/gamma',
                    'g_bn2/moving_mean',
                    'g_bn2/moving_variance',
                    'g_bn3/beta',
                    'g_bn3/gamma',
                    'g_bn3/moving_mean',
                    'g_bn3/moving_variance',
                    'g_h0_lin/Matrix',
                    'g_h0_lin/bias',
                    'g_h1/biases',
                    'g_h1/w',
                    'g_h2/biases',
                    'g_h2/w',
                    'g_h3/biases',
                    'g_h3/w',
                    'g_h4/biases',
                    'g_h4/w']

    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint('../models/celebA_64_64/')

    return x_hat, restore_dict, restore_path


def dcgan_gen_estimator(hparams):

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y = tf.placeholder(tf.float32, shape=(1, hparams.num_measurements), name='y')

    # Create the generator
    z = tf.Variable(tf.random_normal([64, 100]))
    x_hat, restore_dict, restore_path = dcgan_gen(z, sess)
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



def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm = 'ortho').T, norm = 'ortho')


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm = 'ortho').T, norm = 'ortho')


def vec(channels):
    image = np.zeros((64, 64, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vec):
    image = np.reshape(vec, [64, 64, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels


def lasso_estimator(hparams):
    """LASSO estimator"""
    lasso_est = Lasso(alpha=hparams.lmbd)
    def estimator(A_val, y_val, hparams):
        # One can prove that taking 2D DCT of each row of A,
        # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
        for i in range(A_val.shape[1]):
            A_val[:, i] = vec([dct2(channel) for channel in devec(A_val[:, i])])
        lasso_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
        z_hat = lasso_est.coef_
        x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
        x_hat = np.maximum(np.minimum(x_hat, 1), -1)
        return x_hat
    return estimator
