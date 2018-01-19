import tensorflow as tf

from .ClassificationModel import ClassificationModel


class SqueezeNetV1_1(ClassificationModel):

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
            layersTrainedFromScratchNames = ["conv10_fn"]

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

            # Global Average Pooling (map size is 14x14 at this level), output is a 4D tensor
            lastPooling = tf.layers.average_pooling2d(
                inputs=lastConv,
                pool_size=(lastConv.shape[1], lastConv.shape[2]),
                strides=(1,1),
                padding='valid',
                data_format='channels_last',
                name="pool10_fn"
            )

            # Convert 4D Tensor to 2D (you can also reshape)
            self.logits = tf.squeeze(input=lastPooling, axis=[1,2])

        self.setTrainableVariables(layersTrainedFromScratchNames)
        self.defineTrainingOperations()
