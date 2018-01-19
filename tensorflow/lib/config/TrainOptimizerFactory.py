import tensorflow as tf

from constants.Constants import Constants as constants


class TrainOptimizerFactory(object):

    @classmethod
    def createOptimizer(cls, learningRate, optimizerParams):

        optimizerType = type(optimizerParams).__name__

        if optimizerType == constants.TrainConfig.adadelta:
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=learningRate,
                rho=optimizerParams.rho,
                epsilon=optimizerParams.epsilon)

        elif optimizerType == constants.TrainConfig.adagrad:
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=learningRate,
                initial_accumulator_value=optimizerParams.initial_accumulator_value)

        elif optimizerType == constants.TrainConfig.adam:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learningRate,
                beta1=optimizerParams.beta1,
                beta2=optimizerParams.beta2,
                epsilon=optimizerParams.epsilon)

        elif optimizerType == constants.TrainConfig.ftlr:
            optimizer = tf.train.FtrlOptimizer(
                learning_rate=learningRate,
                learning_rate_power=optimizerParams.learning_rate_power,
                initial_accumulator_value=optimizerParams.initial_accumulator_value,
                l1_regularization_strength=optimizerParams.l1,
                l2_regularization_strength=optimizerParams.l2)

        elif optimizerType == constants.TrainConfig.momentumSection:
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learningRate,
                momentum=optimizerParams.momentum)

        elif optimizerType == constants.TrainConfig.sgd:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)

        elif optimizerType == constants.TrainConfig.rms:
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=learningRate,
                decay=optimizerParams.decay,
                momentum=optimizerParams.momentum,
                epsilon=optimizerParams.epsilon)

        else:
            raise Exception('Optimizer ' + optimizerType + 'not supported')

        return optimizer