import sys

# Import OptimizerParams for instantiation by name with reflection
from .OptimizerParams import *


class OptimizerParamsFactory(object):

    @classmethod
    def createOptimizerParams(cls, optimizerType, config):
        currentModule = sys.modules[__name__]
        optimizerClass = getattr(currentModule, optimizerType)
        optimizerInstance = optimizerClass(config)
        return optimizerInstance
