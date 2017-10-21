import configparser

from constants.Constants import Constants as const
from .OptimizerParamsFactory import OptimizerParamsFactory
from model.OptimizerFactory import OptimizerFactory


class ConfigParams(object):

    def __init__(self, file):

        config = configparser.ConfigParser()
        config.read_file(open(file))

        # Model
        self.architecture = config.get(const.ConfigSection.model, "architecture")
        # Valid only for mobilenet
        if self.architecture == "mobilenet":
            self.mobilenetAlpha = config.getfloat(const.ConfigSection.model, "mobilenetAlpha", fallback=1.0)
        self.inputSize = config.getint(const.ConfigSection.model, "inputSize", fallback=224)
        self.inputChannels = config.getint(const.ConfigSection.model, "inputChannels", fallback=3)
        self.preprocessType = config.get(const.ConfigSection.model, "preprocessType", fallback="dummy")

        # HyperParameters
        self.epochs = config.getint(const.ConfigSection.hyperparameters, "epochs")
        self.batchSize = config.getint(const.ConfigSection.hyperparameters, "batchSize")
        self.patience = config.getint(const.ConfigSection.hyperparameters, "patience")
        optimizerType = config.get(const.ConfigSection.hyperparameters, "optimizer")
        optimizerParams = OptimizerParamsFactory.createOptimizerParams(optimizerType, config)
        self.optimizer = OptimizerFactory.create(optimizerParams)
