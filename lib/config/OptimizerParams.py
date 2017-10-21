from constants.Constants import Constants as constants


class OptimizerParams(object):

    def __init__(self, learningRate):

        self.learningRate = learningRate


# CLASS NAMES MUST MATCH CONFIG SECTIONS NAMES !!!
class ADADELTA(OptimizerParams):

    def __init__(self, config):
        learningRate = config.getfloat(constants.ConfigSection.adadelta, constants.TrainConfig.learningRate, fallback=1.0)
        OptimizerParams.__init__(self, learningRate=learningRate)
        self.rho = config.getfloat(constants.ConfigSection.adadelta, constants.TrainConfig.rho, fallback=0.95)
        self.epsilon = config.getfloat(constants.ConfigSection.adadelta, constants.TrainConfig.epsilon, fallback=1e-8)
        self.decay = config.getfloat(constants.ConfigSection.adadelta, constants.TrainConfig.decay, fallback=.0)


class ADAM(OptimizerParams):

    def __init__(self, config):
        learningRate = config.getfloat(constants.ConfigSection.adam, constants.TrainConfig.learningRate, fallback=0.001)
        OptimizerParams.__init__(self, learningRate=learningRate)
        self.beta1 = config.getfloat(constants.ConfigSection.adam, constants.TrainConfig.beta1, fallback=0.9)
        self.beta2 = config.getfloat(constants.ConfigSection.adam, constants.TrainConfig.beta2, fallback=0.999)
        self.epsilon = config.getfloat(constants.ConfigSection.adam, constants.TrainConfig.epsilon, fallback=1e-8)
        self.decay = config.getfloat(constants.ConfigSection.adam, constants.TrainConfig.decay, fallback=.0)


class SGD(OptimizerParams):

    def __init__(self, config):
        learningRate = config.getfloat(constants.ConfigSection.sgd, constants.TrainConfig.learningRate, fallback=0.01)
        OptimizerParams.__init__(self, learningRate=learningRate)
        self.momentum = config.getfloat(constants.ConfigSection.sgd, constants.TrainConfig.momentum, fallback=0.0)
        self.decay = config.getfloat(constants.ConfigSection.sgd, constants.TrainConfig.decay, fallback=0.0)
        self.nesterov = config.getboolean(constants.ConfigSection.sgd, constants.TrainConfig.nesterov, fallback=False)


class RMSPROP(OptimizerParams):

    def __init__(self, config):
        learningRate = config.getfloat(constants.ConfigSection.rmsprop, constants.TrainConfig.learningRate, fallback=0.001)
        OptimizerParams.__init__(self, learningRate=learningRate)
        self.rho = config.getfloat(constants.ConfigSection.rmsprop, constants.TrainConfig.rho, fallback=0.9)
        self.epsilon = config.getfloat(constants.ConfigSection.rmsprop, constants.TrainConfig.epsilon, fallback=1e-8)
        self.decay = config.getfloat(constants.ConfigSection.rmsprop, constants.TrainConfig.decay, fallback=.0)