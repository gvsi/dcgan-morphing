"""Some common utils"""
# pylint: disable=C0301,C0103,C0111

import os
import pickle
import shutil
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

import mnist_estimators
import celebA_estimators


def get_l2_loss(image1, image2):
    """Get L2 loss between the two images"""
    return np.mean((image1 - image2)**2)


def get_measurement_loss(x_hat, A, y):
    """Get measurement loss of the estimated image"""
    y_hat = np.matmul(x_hat, A)
    return np.mean((y - y_hat) ** 2)


def save_to_pickle(data, pkl_filepath):
    """Save the data to a pickle file"""
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data


def get_estimator(hparams, model_type):
    if hparams.dataset == 'mnist':
        if model_type == 'vae-gen':
            estimator = mnist_estimators.vae_gen_estimator(hparams)
        elif model_type == 'lasso':
            estimator = mnist_estimators.lasso_estimator(hparams)
        elif model_type == 'omp':
            estimator = mnist_estimators.omp_estimator(hparams)
        else:
            raise NotImplementedError
    elif hparams.dataset == 'celebA':
        if model_type == 'dcgan-discrim':
            estimator = celebA_estimators.dcgan_discrim_estimator(hparams)
        elif model_type == 'lasso':
            estimator = celebA_estimators.lasso_estimator(hparams)
        else:
            raise NotImplementedError
    return estimator


def get_estimators(hparams):
    estimators = {model_type: get_estimator(hparams, model_type) for model_type in hparams.model_types}
    return estimators


def setup_checkpointing(hparams):
    # Set up checkpoint directories
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        set_up_dir(checkpoint_dir)


def save_images(est_images, save_image, hparams):
    """Save a batch of images to png files"""
    for model_type in hparams.model_types:
        for image_num, image in est_images[model_type].iteritems():
            save_path = get_save_paths(hparams, image_num)[model_type]
            image = image.reshape(hparams.image_shape)
            save_image(image, save_path)


def checkpoint(est_images, measurement_losses, l2_losses, save_image, hparams):
    """Save images, measurement losses and L2 losses for a batch"""
    if hparams.save_images:
        save_images(est_images, save_image, hparams)

    for model_type in hparams.model_types:
        m_losses_filepath, l2_losses_filepath = get_pkl_filepaths(hparams, model_type)
        save_to_pickle(measurement_losses[model_type], m_losses_filepath)
        save_to_pickle(l2_losses[model_type], l2_losses_filepath)


def load_checkpoints(hparams):
    measurement_losses, l2_losses = {}, {}
    if hparams.save_images:
        # Load pickled loss dictionaries
        for model_type in hparams.model_types:
            m_losses_filepath, l2_losses_filepath = get_pkl_filepaths(hparams, model_type)
            measurement_losses[model_type] = load_if_pickled(m_losses_filepath)
            l2_losses[model_type] = load_if_pickled(l2_losses_filepath)
    else:
        for model_type in hparams.model_types:
            measurement_losses[model_type] = {}
            l2_losses[model_type] = {}
    return measurement_losses, l2_losses


def image_matrix(images, est_images, view_image, hparams):
    """Display images"""

    plt.figure(figsize=[hparams.num_input_images, 1 + len(hparams.model_types)])
    for i, image in images.iteritems():
        plt.subplot(1 + len(hparams.model_types), hparams.num_input_images, i+1)
        view_image(image, hparams)

    for j, model_type in enumerate(hparams.model_types):
        for i, image in est_images[model_type].iteritems():
            plt.subplot(1 + len(hparams.model_types), hparams.num_input_images, (j+1)*hparams.num_input_images + i + 1)
            view_image(image, hparams)

    if hparams.image_matrix >= 2:
        save_path = get_matrix_save_path(hparams)
        plt.savefig(save_path)

    if hparams.image_matrix in [1, 3]:
        plt.show()


def get_checkpoint_dir(hparams, model_type):
    base_dir = '../estimated/{0}/{1}/{2}/{3}/{4}/'.format(
        hparams.dataset,
        hparams.input_type,
        hparams.noise_std,
        hparams.num_measurements,
        model_type
    )

    if model_type == 'vae-gen':
        dir_name = '{0}_{1}_{2}_{3}_{4}'.format(
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    elif model_type == 'lasso':
        dir_name = '{0}'.format(
            hparams.lmbd,
        )
    elif model_type == 'omp':
        dir_name = '{0}'.format(
            hparams.omp_k,
        )
    elif model_type == 'dcgan-discrim':
        dir_name = '{0}_{1}_{2}_{3}_{4}'.format(
            hparams.optimizer_type,
            hparams.learning_rate,
            hparams.momentum,
            hparams.max_update_iter,
            hparams.num_random_restarts,
        )
    else:
        raise NotImplementedError

    ckpt_dir = base_dir + dir_name + '/'

    return ckpt_dir


def get_pkl_filepaths(hparams, model_type):
    """Return paths for the pickle files"""
    checkpoint_dir = get_checkpoint_dir(hparams, model_type)
    m_losses_filepath = checkpoint_dir + 'measurement_losses.pkl'
    l2_losses_filepath = checkpoint_dir + 'l2_losses.pkl'
    return m_losses_filepath, l2_losses_filepath


def get_save_paths(hparams, image_num):
    save_paths = {}
    for model_type in hparams.model_types:
        checkpoint_dir = get_checkpoint_dir(hparams, model_type)
        save_paths[model_type] = checkpoint_dir + '{0}.png'.format(image_num)
    return save_paths


def get_matrix_save_path(hparams):
    save_path = '../estimated/{0}/{1}/{2}/{3}/matrix_{4}.png'.format(
        hparams.dataset,
        hparams.input_type,
        hparams.noise_std,
        hparams.num_measurements,
        '_'.join(hparams.model_types)
    )
    return save_path


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)


def print_hparams(hparams):
    print ''
    for temp in dir(hparams):
        if temp[:1] != '_':
            print '{0} = {1}'.format(temp, getattr(hparams, temp))
    print ''


def get_optimizer(hparams):
    if hparams.optimizer_type == 'sgd':
        return tf.train.GradientDescentOptimizer(hparams.learning_rate)
    if hparams.optimizer_type == 'momentum':
        return tf.train.MomentumOptimizer(hparams.learning_rate, hparams.momentum)
    elif hparams.optimizer_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(hparams.learning_rate)
    elif hparams.optimizer_type == 'adam':
        return tf.train.AdamOptimizer(hparams.learning_rate)
    else:
        raise Exception('Optimizer ' + hparams.optimizer_type + ' not supported')
