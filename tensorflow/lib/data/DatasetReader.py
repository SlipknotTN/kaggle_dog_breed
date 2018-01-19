class DatasetReader(object):

    def __init__(self, datasetDir, datasetMetadata, configParams):

        self.datasetDir = datasetDir
        self.datasetMetadata = datasetMetadata
        self.configParams = configParams

    def getTrainBatchesNumber(self):

        return self.datasetMetadata.trainingSamplesNumber // self.configParams.batchSize

    def getValidationBatchesNumber(self):

        return self.datasetMetadata.validationSamplesNumber // self.configParams.batchSize