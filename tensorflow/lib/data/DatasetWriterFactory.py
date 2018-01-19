from .DatasetTFWriter import DatasetTFWriter


class DatasetWriterFactory(object):

    @classmethod
    def createDatasetWriter(cls, datasetParams, scriptArgs):

        # Init Dataset TF Writer
        dataset = DatasetTFWriter()
        dataset.setTrainValSamplesList(imagesDir=scriptArgs.imagesDir,
                                       validationPercentage=datasetParams.validationPercentage)

        return dataset
