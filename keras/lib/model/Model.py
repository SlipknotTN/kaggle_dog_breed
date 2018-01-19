from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dropout
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.applications.mobilenet import MobileNet


class Model(object):

    @classmethod
    def smallCustom(cls, inputShape, numClasses):

        model = Sequential()

        # Convolutions with stride 2 are like convolution + pooling
        # 2x smaller on width and height
        model.add(Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=64,
                         input_shape=inputShape,
                         padding='same', activation='relu', name="conv1"))

        # 2x smaller on width and height
        model.add(
            Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=128, padding='same', activation='relu', name="conv2"))

        # 2x smaller on width and height
        model.add(
            Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=256, padding='same', activation='relu', name="conv3"))

        # 2x smaller on width and height
        model.add(
            Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=512, padding='same', activation='relu', name="conv4"))

        # 2x smaller on width and height
        model.add(Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=numClasses))

        model.add(GlobalAveragePooling2D())

        model.add(Activation('softmax', name="softmax"))

        return model

    @classmethod
    def mobilenet(cls, inputShape, numClasses, alpha, retrainAll=False):

        # Load MobileNet Full, with output shape of (None, 7, 7, 1024)
        baseModel = MobileNet(input_shape=inputShape, alpha=alpha,
                              depth_multiplier=1, dropout=1e-3, include_top=False,
                              weights='imagenet', input_tensor=None, pooling=None)

        fineTunedModel = Sequential()

        fineTunedModel.add(baseModel)

        # Global average output shape (None, 1, 1, 1024).
        # Global pooling with AveragePooling2D to have a 4D Tensor and apply Conv2D.
        fineTunedModel.add(AveragePooling2D(pool_size=(baseModel.output_shape[1], baseModel.output_shape[2]),
                                            strides=(1, 1), padding='valid', name="global_pooling"))

        fineTunedModel.add(Dropout(rate=0.5))

        # Convolution layer that acts like fully connected, with 2 classes, output shape (None, 1, 1, 2)
        fineTunedModel.add(Conv2D(filters=numClasses, kernel_size=(1, 1), name="fc_conv"))

        # Reshape to (None, 2) to match the one hot encoding target and final softmax
        fineTunedModel.add(Flatten())

        # Final sofmax for deploy stage
        fineTunedModel.add(Activation('softmax'))

        # Freeze the base model layers, train only the last convolution
        if retrainAll is False:
            for layer in fineTunedModel.layers[0].layers:
                layer.trainable = False

        return fineTunedModel
