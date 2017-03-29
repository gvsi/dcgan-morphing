"""Compressed sensing main script"""

# pylint: disable=C0301,C0103,C0111

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils


def main(hparams):
    hparams.n_input = np.prod(hparams.image_shape)
    images = model_input(hparams)

    estimators = utils.get_estimators(hparams)
    utils.setup_checkpointing(hparams)
    measurement_losses, l2_losses = utils.load_checkpoints(hparams)

    est_images = {model_type : {} for model_type in hparams.model_types}
    for i, image in images.iteritems():

        if not hparams.not_lazy:
            # If lazy, first check if the image has already been
            # saved before by *all* estimators. If yes, then skip this image.
            save_paths = utils.get_save_paths(hparams, i)
            if all([os.path.isfile(save_path) for save_path in save_paths.values()]):
                continue

        # Reshape input
        x_val = image.reshape(1, hparams.n_input)

        # Construct noise and measurements
        noise_val = hparams.noise_std * np.random.randn(1, hparams.num_measurements)
        A_val = np.random.randn(hparams.n_input, hparams.num_measurements)
        y_val = np.matmul(x_val, A_val) + noise_val

        # Construct estimates using each estimator
        print 'Processing image {0}'.format(i)
        for model_type in hparams.model_types:
            estimator = estimators[model_type]
            est_image = estimator(A_val, y_val, hparams)
            est_images[model_type][i] = est_image
            # Compute and store measurement and l2 loss
            measurement_losses[model_type][i] = utils.get_measurement_loss(est_image, A_val, y_val)
            l2_losses[model_type][i] = utils.get_l2_loss(est_image, image)

        # Checkpointing
        if (hparams.save_images) and ((i+1) % 100 == 0):
            utils.checkpoint(est_images, measurement_losses, l2_losses, save_image, hparams)
            est_images = {model_type : {} for model_type in hparams.model_types}
            print '\nProcessed and saved first ', i+1, 'images\n'

    # Final checkpoint
    if hparams.save_images:
        utils.checkpoint(est_images, measurement_losses, l2_losses, save_image, hparams)
        print '\nProcessed and saved all {0} images\n'.format(len(images))

    if hparams.print_stats:
        for model_type in hparams.model_types:
            print model_type
            mean_m_loss = np.mean(measurement_losses[model_type].values())
            mean_l2_loss = np.mean(l2_losses[model_type].values())
            print 'mean measurement loss = {0}'.format(mean_m_loss)
            print 'mean l2 loss = {0}'.format(mean_l2_loss)

    if hparams.image_matrix > 0:
        utils.image_matrix(images, est_images, view_image, hparams)


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='random_test', help='Where to take input from')
    PARSER.add_argument('--num-input-images', type=int, default=10, help='number of input images')

    # Problem definition
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of measurements')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+', default=None,
                        help='model used for estimation. Currently supports nn only. lasso planned in near future.')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='momentum', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=100, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=10, help='number of random restarts')

    # LASSO specific hparams
    PARSER.add_argument('--lmbd', type=float, default=0.1, help='lambda : regularization parameter for LASSO')

    # OMP specific hparams
    PARSER.add_argument('--omp-k', type=int, default=80, help='number of non zero entries allowed in OMP')

    # Output
    PARSER.add_argument('--not-lazy', action='store_false', help='whether the evaluation is lazy')
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--image-matrix', type=int, default=0,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                       )

    HPARAMS = PARSER.parse_args()

    if HPARAMS.dataset == 'mnist':
        HPARAMS.image_shape = (28, 28)
        from mnist_input import model_input
        from mnist_utils import view_image, save_image
    elif HPARAMS.dataset == 'celebA':
        HPARAMS.image_shape = (64, 64, 3)
        from celebA_input import model_input
        from celebA_utils import view_image, save_image
    else:
        raise NotImplementedError

    utils.print_hparams(HPARAMS)
    main(HPARAMS)
