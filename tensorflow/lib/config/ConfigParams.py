import configparser
import json

from constants.Constants import Constants as const
from .OptimizerParamsFactory import OptimizerParamsFactory
from .LRPolicyParams import LRPolicyParams


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
        self.inputFormat = config.get(const.ConfigSection.model, "inputFormat", fallback="RGB")
        self.preprocessType = config.get(const.ConfigSection.model, "preprocessType", fallback="dummy")
        self.meanRGB = None
        if self.preprocessType == "vgg":
            self.meanRGB = json.loads(config.get(const.ConfigSection.model, "meanRGB", fallback="[0.0, 0.0, 0.0]"))
        self.inputName = config.get(const.ConfigSection.model, "inputName")
        self.outputName = config.get(const.ConfigSection.model, "outputName")
        self.lastFrozenLayerName = config.get(const.ConfigSection.model, "lastFrozenLayerName")

        # HyperParameters
        self.epochs = config.getint(const.ConfigSection.hyperparameters, "epochs")
        self.batchSize = config.getint(const.ConfigSection.hyperparameters, "batchSize")

        # Load the optimizer params
        optimizerType = str(config.get(const.ConfigSection.hyperparameters, const.TrainConfig.optimizer)).upper()
        self.optimizer = OptimizerParamsFactory.createOptimizerParams(optimizerType=optimizerType, config=config)

        # Load learning rate policy
        self.optimizer.setLrParams(LRPolicyParams(optimizerType=optimizerType, config=config))

        self.saveBestEpoch = config.getboolean(const.ConfigSection.hyperparameters, "saveBestEpoch")

        #Dataset creation params (image size = model size for simplicity)
        self.validationPercentage = config.getint(const.ConfigSection.datasetParameters,
                                                  const.DatasetParams.validationPercentage)
        self.imageEncoding = config.get(const.ConfigSection.datasetParameters, const.DatasetParams.imageEncoding,
                                        fallback=None)
