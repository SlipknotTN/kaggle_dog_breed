class Constants(object):

    class ConfigSection:

        hyperparameters = "HYPERPARAMETERS"
        model = "MODEL"
        datasetParameters = "DATASET_PARAMETERS"

    class DatasetParams:

        validationPercentage = "validationPercentage"
        imageEncoding = "imageEncoding"

    class PreprocessingType:

        vgg = "vgg"
        inception = "inception"

    class FileFormats:

        jpeg = "jpeg"
        png = "png"

    class Subsets:

        training = "Training"
        validation = "Validation"

    class DatasetFeatures:

        label = "label"
        image = "image"

    class TrainConfig:

        # Sections
        adadelta = "ADADELTA"
        adagrad = "ADAGRAD"
        sgd = "SGD"
        adam = "ADAM"
        ftlr = "FTLR"
        momentumSection = "MOMENTUM"
        rms = "RMSPROP"
        lrPolicy = "LEARNING_RATE_POLICY"

        # Keys
        numEpochs = 'numEpochs'
        batchSize = 'batchSize'
        optimizer = 'optimizer'

        starterLearningRate = 'starterLearningRate'
        rho = 'rho'
        epsilon = 'epsilon'
        initialAccumulatorValue = 'initialAccumulatorValue'
        beta1 = "beta1"
        beta2 = "beta2"
        learningRatePower = "learningRatePower"
        l1RegularizationStrength = "l1RegularizationStrength"
        l2RegularizationStrength = "l2RegularizationStrength"
        momentum = "momentum"
        decay = "decay"

        decayPolicy = "decayPolicy"
        lrDecayStep = "lrDecayStep"
        lrDecayRate = "lrDecayRate"

        type = "type"

    class LRPolicy:

        fixed = "fixed"
        exponential = "exponential"
