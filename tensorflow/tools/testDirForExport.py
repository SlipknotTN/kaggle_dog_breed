import _init_paths
import argparse
import glob
import os

import tensorflow as tf
import numpy as np

from config.ConfigParams import ConfigParams
from image.ImageUtils import ImageUtils
from model.TensorflowModel import TensorflowModel
from kaggle.export import exportResults

classes = ['cat', 'dog']


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Keras test script")
    parser.add_argument("--datasetTestDir", required=True, type=str, help="Dataset test directory")
    parser.add_argument("--configFile", required=True, type=str, help="Model config file")
    parser.add_argument("--modelPath", required=False, type=str, default="./export/graph.pb",
                        help="Filepath with trained model")
    parser.add_argument("--kaggleExportFile", required=False, type=str, default=None,
                        help="CSV file in kaggle format for challenge upload")
    args = parser.parse_args()
    return args


def main():
    """
    Script to export results for Kaggle, Images are read one by one
    """
    args = doParsing()
    print(args)

    # Load config (it includes preprocessing type)
    config = ConfigParams(args.configFile)

    # Load model
    model = TensorflowModel(args.modelPath)

    print("Loaded model from " + args.modelPath)

    # Dogs and cats test dataset has 12500 samples

    results = []

    inputPlaceholder = model.getGraph().get_tensor_by_name(config.inputName + ":0")
    outputTensor = model.getGraph().get_tensor_by_name(config.outputName + ":0")

    # One by one image prediction forcing CPU usage
    with model.getSession() as sess:

        with tf.device("/cpu:0"):

            for file in sorted(glob.glob(args.datasetTestDir + "/*.jpg")):

                image = ImageUtils.loadImage(file)
                # Resize image and preprocess (inception or vgg preprocessing based on config)
                processedImage = ImageUtils.preprocessing(image=image, width=config.inputSize, height=config.inputSize,
                                                          preprocessingType=config.preprocessType,
                                                          meanRGB=config.meanRGB)

                # Convert colorspace
                processedImage = ImageUtils.convertImageFormat(processedImage, format=config.inputFormat)

                # Add 1st dimension for image index in batch
                processedImage = np.expand_dims(processedImage, axis=0)

                # Get and print TOP1 class
                result = sess.run(outputTensor, feed_dict={inputPlaceholder: processedImage})
                print(os.path.basename(file) + " -> " + classes[int(np.argmax(result[0]))])

                # Get and save dog probability
                results.append((os.path.basename(file)[:os.path.basename(file).rfind('.')], result[0][classes.index("dog")]))

    print("Test finished")

    if args.kaggleExportFile is not None:
        exportResults(results, args.kaggleExportFile)


if __name__ == '__main__':
    main()
