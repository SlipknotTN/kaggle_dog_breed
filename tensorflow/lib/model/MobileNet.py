import tensorflow as tf

from .ClassificationModel import ClassificationModel


class MobileNet(ClassificationModel):

    def __init__(self, configParams, model, dataProvider, trainDevice):

        ClassificationModel.__init__(self, configParams, model, dataProvider, trainDevice)

        ## Tensorflow Placeholders
        with tf.device(trainDevice):
            # Image placeholder, shape NHWC, you need to provide BGR images
            self.x = model.getGraph().get_tensor_by_name(configParams.inputName + ":0")
            # Ground truth placeholder (one-hot encoding)
            self.y = tf.placeholder(dtype=tf.int32, shape=[None, self.dataProvider.datasetMetadata.numClasses], name="y")

        # Custom fine tuning definition, train only the last classifier
        with tf.device(trainDevice):
            # Loaded model is frozen in test mode
            inputLayerTrainedFromScratch = model.getGraph().get_tensor_by_name(
                configParams.lastFrozenLayerName + ":0")
            layersTrainedFromScratchNames = ["Conv2d_1c_1x1_fn"]

            # Layers trained from scratch
            lastConv = tf.layers.conv2d(
                inputLayerTrainedFromScratch,
                filters=dataProvider.datasetMetadata.numClasses,
                kernel_size=[1,1],
                strides=(1, 1),
                padding='same',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=None,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=layersTrainedFromScratchNames[0],
                reuse=None
            )

            # Reshape to 2D array (batchSize x numClasses), otherwise convolution output is 4D
            self.logits = tf.reshape(lastConv, [-1, self.dataProvider.datasetMetadata.numClasses])

        self.setTrainableVariables(layersTrainedFromScratchNames)
        self.defineTrainingOperations()
