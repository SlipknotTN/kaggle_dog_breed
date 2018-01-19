import json

class DatasetMetadata(object):

    def __init__(self, trainingSamplesNumber=None, validationSamplesNumber=None, numClasses=None):

        self.trainingSamplesNumber = trainingSamplesNumber
        self.validationSamplesNumber = validationSamplesNumber
        self.numClasses = numClasses

    def initFromJson(self, jsonFile):

        with open(jsonFile, 'r') as jsonDatasetMetadata:
            datasetMetadataDict = json.load(jsonDatasetMetadata)
            self.trainingSamplesNumber = int(datasetMetadataDict["trainingSamplesNumber"])
            self.validationSamplesNumber = int(datasetMetadataDict["validationSamplesNumber"])
            self.numClasses = int(datasetMetadataDict["numClasses"])
        return self
