"""Some utils for MNIST dataset"""
# pylint: disable=C0301,C0103

import matplotlib.pyplot as plt
import png


def view_image(image, hparams):
    """Show the image"""
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(image.reshape(hparams.image_shape), cmap='Greys')


def save_image(image, path):
    """Save an image as a png file"""
    png_writer = png.Writer(28, 28, greyscale=True)
    with open(path, 'wb') as outfile:
        png_writer.write(outfile, 255*image)
