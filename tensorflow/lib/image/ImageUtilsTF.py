import tensorflow as tf
import numpy as np

from .ImageUtils import ImageUtils
from constants.Constants import Constants as constants


class ImageUtilsTF(ImageUtils):
    """
    Image Utils for tensorflow pipeline, TF operations
    """
    @classmethod
    def preprocessing(cls, image, width, height, preprocessingType, meanRGB=None):

        with tf.device('/cpu:0'):

            # We use the passed width and height (not forced to use imageParams width and height)
            image = tf.image.resize_images(image, size=(width, height))

            if preprocessingType == constants.PreprocessingType.vgg:
                image = cls.preprocessVgg(image, meanRGB)
            elif preprocessingType == constants.PreprocessingType.inception:
                image = cls.preprocessInception(image)
            else:
                raise Exception("Preprocessing type " + preprocessingType + " not supported")

            return image

    @classmethod
    def preprocessVgg(cls, image, meanRGB):
        """
        VGG Processing (equal to AlexNet, SqueezeNet, GoogleNet, ...). We support only 3 channels models.
        :param image float32 HWC-RGB Tensor in range [0.0, 255.0]
        :param meanRGB float list, Mean values RGB
        :return: processed image in format float32 HWC-RGB Tensor in range [0.0, 255.0]
        subtracted mean (we can obtain negative values)
        """

        assert (len(image.shape) == 3), "VGG preprocessing supports only 3 channels images"

        # Subtract mean
        meanRGBtf = tf.reshape(np.array(meanRGB, dtype=np.float32), shape=(1, 1, 3))
        image = tf.subtract(image, meanRGBtf)
        return image

    @classmethod
    def preprocessInception(cls, image):
        """
        Inception and MobileNet preprocessing
        :param image float32 HWC-RGB Tensor in range [0.0, 255.0]
        :return: float32 HWC-RGB Tensor in range [-1.0, 1.0]
        """
        image = tf.multiply(image, 1.0/255.0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image
