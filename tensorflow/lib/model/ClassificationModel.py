import tensorflow as tf

from config.LearningRateFactory import LearningRateFactory
from config.TrainOptimizerFactory import TrainOptimizerFactory


class ClassificationModel(object):
    """
    Common base class for all classification models
    """
    def __init__(self, configParams, model, dataProvider, trainDevice):
        self.configParams = configParams
        self.model = model
        self.dataProvider = dataProvider
        self.trainDevice = trainDevice
        self.x = None
        self.y = None
        self.logits = None
        self.cost = None
        self.output = None
        self.varList = None
        self.learningRate = None
        self.optimizer = None
        self.accuracy = None
        self.correctPred = None
        self.globalStep = None
        self.lrateSummary = None
        self.scalarInputForSummary = None
        self.costSummary = None
        self.accuracySummary = None

    def setTrainableVariables(self, layersTrainedFromScratchNames):
        """
        Default trainable variables set (valid for simple classification fine tuning of the last layer)
        """
        with tf.device(self.trainDevice):
            # In fine tuning we train only the new layers
            self.varList = [v for v in tf.trainable_variables()
                            if v.name.split('/')[0] in layersTrainedFromScratchNames]

    def defineTrainingOperations(self):
        with tf.device(self.trainDevice):
            # Add softmax for deploy stage
            self.output = tf.nn.softmax(self.logits, name=self.configParams.outputName)

        with tf.device('/cpu:0'):
            self.globalStep = tf.Variable(0, trainable=False)

        with tf.device(self.trainDevice):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        with tf.device(self.trainDevice):
            self.learningRate = LearningRateFactory.createLearningRate(optimizerParams=self.configParams.optimizer,
                                                                       trainBatches=self.dataProvider.getTrainBatchesNumber(),
                                                                       globalStepVar=self.globalStep)

        with tf.device('/cpu:0'):
            self.lrateSummary = tf.summary.scalar("learning rate", self.learningRate)

        with tf.device(self.trainDevice):
            self.optimizer = TrainOptimizerFactory.createOptimizer(learningRate=self.learningRate,
                                                                   optimizerParams=self.configParams.optimizer)
            self.optimizer = self.optimizer.minimize(self.cost, global_step=self.globalStep, var_list=self.varList)

        with tf.device('/cpu:0'):
            self.correctPred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))

        with tf.device(self.trainDevice):
            self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32), name='accuracy')
            self.scalarInputForSummary = tf.placeholder(dtype=tf.float32, name="scalar_input_summary")

        with tf.device('/cpu:0'):
            self.costSummary = tf.summary.scalar("Training loss", self.scalarInputForSummary)
            self.accuracySummary = tf.summary.scalar("validation accuracy", self.scalarInputForSummary)

    def getSession(self):
        return self.model.getSession()

    def getGraph(self):
        return self.model.getGraph()
