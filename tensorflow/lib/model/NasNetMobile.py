import tensorflow as tf

from .ClassificationModel import ClassificationModel


class NasNetMobile(ClassificationModel):

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
            layersTrainedFromScratchNames = ["FC_fn"]

            # Layers trained from scratch (the last frozen layer is already flattened
            lastFC = tf.layers.dense(
                inputLayerTrainedFromScratch,
                units=dataProvider.datasetMetadata.numClasses,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=layersTrainedFromScratchNames[0],
                reuse=None)

            # Shape is already 2D (batchSize x numClasses)
            self.logits = lastFC

        self.setTrainableVariables(layersTrainedFromScratchNames)
        self.defineTrainingOperations()
