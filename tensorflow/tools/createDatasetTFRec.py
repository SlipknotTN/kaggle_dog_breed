import _init_paths
import argparse
import json
import os
import tensorflow as tf

from config.ConfigParams import ConfigParams
from data.DatasetWriterFactory import DatasetWriterFactory
from data.DatasetMetadata import DatasetMetadata


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script for tfrec crestion for classification task')
    parser.add_argument('--imagesDir', required=True,
                        help='Root folder containing images (single directory that has to be split')
    parser.add_argument('--configFile', required=True, type=str, help='Config file for dataset creation')
    parser.add_argument('--outputDir', required=True, help='TFRecords destination directory, use a clean directory')
    return parser.parse_args()


def main():

    args = do_parsing()
    print(args)

    # Read dataset configuration (config file is in common for dataset creation and training hyperparameters)
    datasetParams = ConfigParams(args.configFile)

    # Get dataset writer with training and validation splits
    dataset = DatasetWriterFactory.createDatasetWriter(datasetParams=datasetParams, scriptArgs=args)

    if os.path.exists(args.outputDir) is False:
        os.makedirs(args.outputDir)

    trainingOutputFile = os.path.join(args.outputDir, "data_train.tfrecords")
    validationOutputFile = os.path.join(args.outputDir, "data_val.tfrecords")
    jsonFilePath = os.path.join(args.outputDir, "metadata.json")

    # Export Train Samples
    with tf.python_io.TFRecordWriter(trainingOutputFile) as tfrecWriter:
        print("TRAINING")
        dataset.saveTFExamplesTraining(datasetParams=datasetParams, writer=tfrecWriter)
        print("Saving file...")

    # Export Validation Samples
    with tf.python_io.TFRecordWriter(validationOutputFile) as tfrecWriter:
        print("VALIDATION")
        dataset.saveTFExamplesValidation(datasetParams=datasetParams, writer=tfrecWriter)
        print("Saving file...")

    # Export metadata to JSON
    trainingSamplesNumber = dataset.getTrainingSamplesNumber()
    validationSamplesNumber = dataset.getValidationSamplesNumber()
    datasetMetadata = DatasetMetadata(trainingSamplesNumber, validationSamplesNumber, dataset.numClasses)

    with open(jsonFilePath, 'w') as jsonOutFile:
        json.dump(datasetMetadata, jsonOutFile, default=lambda o: o.__dict__, indent=4)

    print("Dataset successfully created in " + args.outputDir)


if __name__ == '__main__':
    main()