"""Inputs for celebA dataset"""

import glob
import os
import numpy as np
import dcgan_utils


def model_input(hparams):
    """Create input tensors"""

    image_paths = glob.glob(os.path.join("../data", 'celebA', "*.jpg"))
    if hparams.input_type == 'full-input':
        image_paths.sort()
        image_paths = image_paths[:hparams.num_input_images]
    elif hparams.input_type == 'random-test':
        idxs = np.random.choice(len(image_paths), hparams.num_input_images)
        image_paths = [image_paths[idx] for idx in idxs]
    else:
        raise NotImplementedError
    image_size = 108
    images = [dcgan_utils.get_image(image_path, image_size) for image_path in image_paths]
    images = {i: image.reshape((1, 64*64*3)) for (i, image) in enumerate(images)}
    return images
