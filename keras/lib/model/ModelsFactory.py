from .Model import Model


class ModelsFactory(object):

    @classmethod
    def create(cls, config, numClasses):

        inputShape = (config.inputSize, config.inputSize, config.inputChannels)

        if config.architecture == "mobilenet":

            return Model.mobilenet(inputShape=inputShape, numClasses=numClasses, alpha=config.mobilenetAlpha,
                                   retrainAll=False)

        else:

            raise Exception("Model architecture " + config.architecture + " not supported")