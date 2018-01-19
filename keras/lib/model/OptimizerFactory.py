from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from constants.Constants import Constants as constants


class OptimizerFactory(object):

    @classmethod
    def create(cls, optimizerParams):

        optimizerType = type(optimizerParams).__name__

        if optimizerType == constants.ConfigSection.sgd:

            return SGD(lr=optimizerParams.learningRate, momentum=optimizerParams.momentum, decay=optimizerParams.decay,
                       nesterov=optimizerParams.nesterov)

        elif optimizerType == constants.ConfigSection.adam:

            return Adam(lr=optimizerParams.learningRate, beta_1=optimizerParams.beta1, beta_2=optimizerParams.beta2,
                        epsilon=optimizerParams.epsilon, decay=optimizerParams.decay)

        elif optimizerType == constants.ConfigSection.adadelta:

            return Adadelta(lr=optimizerType.learningRate, rho=optimizerType.rho, epsilon=optimizerType.epsilon,
                            decay=optimizerType.decay)

        elif optimizerType == constants.ConfigSection.rmsprop:

            return RMSprop(lr=optimizerType.learningRate, rho=optimizerType.rho, epsilon=optimizerType.epsilon,
                            decay=optimizerType.decay)

        else:

            raise Exception("Optimizer " + optimizerType + " not supported")