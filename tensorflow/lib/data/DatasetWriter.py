import glob
import os
import random


class DatasetWriter(object):

    def __init__(self):

        self.numClasses = None
        self.trainingSamplesList = None
        self.validationSamplesList = None

    def setTrainValSamplesList(self, imagesDir, validationPercentage):
        """
        Set images file list separating training from validation.
        Each directory must contains a single class images.
        Use this function when images are mixed and you need to split training and validation.
        :param imagesDir:
        :param validationPercentage:
        :return: list of tuples filename, classIndex
        """
        tuplesList = []

        classNames = [f for f in os.listdir(imagesDir) if os.path.isdir(os.path.join(imagesDir, f))]
        classNames = sorted(classNames)
        for classIndex, className in enumerate(classNames):
            print(classIndex, className)
            files = sorted(glob.glob(os.path.join(imagesDir, className) + "/*.*"), key=lambda s: s.lower())
            for file in files:
                tuplesList.append((file, classIndex))

        random.shuffle(tuplesList)
        self.numClasses = len(classNames)
        splitIndex = int((len(tuplesList) * (100 - validationPercentage)) / 100)
        self.trainingSamplesList = tuplesList[:splitIndex]
        self.validationSamplesList = tuplesList[splitIndex:]

    def getTrainingSamplesNumber(self):

        return len(self.trainingSamplesList)

    def getValidationSamplesNumber(self):

        return len(self.validationSamplesList)
