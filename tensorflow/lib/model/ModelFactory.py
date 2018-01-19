from .MobileNet import MobileNet
from .NasNetMobile import NasNetMobile
from .SqueezeNetV1_1 import SqueezeNetV1_1

class ModelFactory(object):

    @classmethod
    def create(cls, config, tfmodel, dataProvider, trainDevice):

        # Choose model network and build trainable layers
        if config.architecture.lower() == "squeezenet_v1.1":
            return SqueezeNetV1_1(configParams=config, model=tfmodel, dataProvider=dataProvider, trainDevice=trainDevice)
        elif config.architecture.lower() == "nasnet_mobile":
            return NasNetMobile(configParams=config, model=tfmodel, dataProvider=dataProvider, trainDevice=trainDevice)
        elif config.architecture.lower() == "mobilenet":
            return MobileNet(configParams=config, model=tfmodel, dataProvider=dataProvider, trainDevice=trainDevice)
        else:
            raise Exception('Architecture ' + config.model + 'not supported')
