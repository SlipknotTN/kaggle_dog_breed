import tensorflow as tf

from constants.Constants import Constants as constants


class LearningRateFactory(object):

    @classmethod
    def createLearningRate(cls, optimizerParams, trainBatches, globalStepVar):

        if optimizerParams.lr_params.decayPolicy == constants.LRPolicy.fixed:
            learningRate = optimizerParams.learning_rate

        elif optimizerParams.lr_params.decayPolicy == constants.LRPolicy.exponential:
            # Convert decay step from epochs (as written in config) to batches number
            learningRateDecayStep = optimizerParams.lr_params.lrDecayStep * trainBatches
            learningRate = tf.train.exponential_decay(learning_rate=optimizerParams.learning_rate,
                                                      global_step=globalStepVar,
                                                      decay_steps=learningRateDecayStep,
                                                      decay_rate=optimizerParams.lr_params.lrDecayRate,
                                                      staircase=True)

        else:
            raise Exception('LR Policy ' + optimizerParams.lr_params.decayPolicy + 'not supported')

        return learningRate