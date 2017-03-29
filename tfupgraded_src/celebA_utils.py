"""Some utils for celebA dataset"""

import png
import matplotlib.pyplot as plt
import dcgan_utils


def view_image(image, hparams):
    """Show the image"""
    image = dcgan_utils.inverse_transform(image)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame = frame.imshow(image.reshape(hparams.image_shape))


def save_image(image, path):
    """Save an image as a png file"""
    image = dcgan_utils.inverse_transform(image)
    png_writer = png.Writer(64, 64)
    with open(path, 'wb') as outfile:
        png_writer.write(outfile, 255*image.reshape([64,-1]))
