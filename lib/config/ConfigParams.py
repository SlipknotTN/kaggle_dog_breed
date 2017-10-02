import configparser


class ConfigParams(object):

    def __init__(self, file):

        config = configparser.ConfigParser()
        config.read_file(open(file))

        # Model
        self.architecture = config.get("MODEL", "architecture")
        if self.architecture != "mobilenet":
            raise Exception("Only mobilenet architecture supported")
        self.mobilenetAlpha = config.getfloat("MODEL", "mobilenetAlpha", fallback=1.0)
        self.inputSize = config.getint("MODEL", "inputSize", fallback=224)

        # HyperParameters
        self.epochs = config.getint("HYPERPARAMETERS", "epochs")
        self.batchSize = config.getint("HYPERPARAMETERS", "batchSize")
        self.patience = config.getint("HYPERPARAMETERS", "patience")
        self.learningRate = config.getfloat("HYPERPARAMETERS", "learningRate")
        self.optimizer = config.get("HYPERPARAMETERS", "optimizer")
        if self.optimizer != "SGD":
            raise Exception("Only SGD optimizer supported")
        self.momentum = config.getfloat("HYPERPARAMETERS", "momentum")
