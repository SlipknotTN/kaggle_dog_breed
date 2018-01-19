import configparser

from constants.Constants import Constants as constants


class LRPolicyParams(object):

    def __init__(self, optimizerType, config):

        self.decayPolicy = constants.LRPolicy.fixed

        try:
            self.decayPolicy = str(config.get(optimizerType, constants.TrainConfig.decayPolicy)).lower()
        except configparser.Error:
            return

        try:
            self.lrDecayStep = config.getfloat(optimizerType, constants.TrainConfig.lrDecayStep)
            self.lrDecayRate = config.getfloat(optimizerType, constants.TrainConfig.lrDecayRate)
        except configparser.Error:
            return