from constants.Constants import Constants as constants


class OptimizerParams(object):

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate
        self.lr_params = None

    def setLrParams(self, lr_params):
        self.lr_params = lr_params


# CLASS NAMES MUST MATCH CONFIG SECTIONS NAMES !!!
class ADADELTA(OptimizerParams):

    def __init__(self, config):
        learning_rate = config.getfloat(constants.TrainConfig.adadelta, constants.TrainConfig.starterLearningRate)
        OptimizerParams.__init__(self, learning_rate=learning_rate)
        self.rho = config.getfloat(constants.TrainConfig.adadelta, constants.TrainConfig.rho)
        self.epsilon = config.getfloat(constants.TrainConfig.adadelta, constants.TrainConfig.epsilon)


class ADAGRAD(OptimizerParams):

    def __init__(self, config):
        learning_rate = config.getfloat(constants.TrainConfig.adagrad, constants.TrainConfig.starterLearningRate)
        OptimizerParams.__init__(self, learning_rate=learning_rate)
        self.initial_accumulator_value = config.getfloat(constants.TrainConfig.adagrad, constants.TrainConfig.initialAccumulatorValue)


class ADAM(OptimizerParams):

    def __init__(self, config):
        learning_rate = config.getfloat(constants.TrainConfig.adam, constants.TrainConfig.starterLearningRate)
        OptimizerParams.__init__(self, learning_rate=learning_rate)
        self.beta1 = config.getfloat(constants.TrainConfig.adam, constants.TrainConfig.beta1)
        self.beta2 = config.getfloat(constants.TrainConfig.adam, constants.TrainConfig.beta2)
        self.epsilon = config.getfloat(constants.TrainConfig.adam, constants.TrainConfig.epsilon)


class FTLR(OptimizerParams):

    def __init__(self, config):
        learning_rate = config.getfloat(constants.TrainConfig.ftlr, constants.TrainConfig.starterLearningRate)
        OptimizerParams.__init__(self, learning_rate=learning_rate)
        self.learning_rate_power = config.getfloat(constants.TrainConfig.ftlr, constants.TrainConfig.learningRatePower)
        self.initial_accumulator_value = config.getfloat(constants.TrainConfig.ftlr, constants.TrainConfig.initialAccumulatorValue)
        self.l1 = config.getfloat(constants.TrainConfig.ftlr, constants.TrainConfig.l1RegularizationStrength)
        self.l2 = config.getfloat(constants.TrainConfig.ftlr, constants.TrainConfig.l2RegularizationStrength)


class MOMENTUM(OptimizerParams):

    def __init__(self, config):
        learning_rate = config.getfloat(constants.TrainConfig.momentumSection, constants.TrainConfig.starterLearningRate)
        OptimizerParams.__init__(self, learning_rate=learning_rate)
        self.momentum = config.getfloat(constants.TrainConfig.momentumSection, constants.TrainConfig.momentum)


class SGD(OptimizerParams):

    def __init__(self, config):
        learning_rate = config.getfloat(constants.TrainConfig.sgd, constants.TrainConfig.starterLearningRate)
        OptimizerParams.__init__(self, learning_rate=learning_rate)


class RMSPROP(OptimizerParams):

    def __init__(self, config):

        learning_rate = config.getfloat(constants.TrainConfig.rms, constants.TrainConfig.starterLearningRate)
        OptimizerParams.__init__(self, learning_rate=learning_rate)
        self.decay = config.getfloat(constants.TrainConfig.rms, constants.TrainConfig.decay)
        self.momentum = config.getfloat(constants.TrainConfig.rms, constants.TrainConfig.momentum)
        self.epsilon = config.getfloat(constants.TrainConfig.rms, constants.TrainConfig.epsilon)