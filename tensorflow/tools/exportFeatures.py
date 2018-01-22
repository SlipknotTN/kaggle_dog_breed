import _init_paths
import argparse
import glob
import os

from tqdm import tqdm
import tensorflow as tf
import numpy as np

from config.ConfigParams import ConfigParams
from image.ImageUtils import ImageUtils
from model.TensorflowModel import TensorflowModel


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Tensorflow features export script")
    parser.add_argument("--datasetDir", required=True, type=str, help="Root directory with classes subdirs")
    parser.add_argument("--configFile", required=True, type=str, help="Model config file")
    parser.add_argument("--modelPath", required=False, type=str, help="Filepath with model (base model is enough)")
    parser.add_argument("--outputDir", required=True, type=str, help="Root output directory")
    args = parser.parse_args()
    return args


def main():
    """
    Script to export bottleneck features, Images are read one by one.
    """
    args = doParsing()
    print(args)

    # Load config (it includes preprocessing type)
    config = ConfigParams(args.configFile)

    # Load model
    model = TensorflowModel(args.modelPath)

    print("Loaded model from " + args.modelPath)

    inputPlaceholder = model.getGraph().get_tensor_by_name(config.inputName + ":0")
    outputTensor = model.getGraph().get_tensor_by_name(config.lastFrozenLayerName + ":0")

    # Evaluate if the dataset directory has all files or classes subdirectories
    dirs = [os.path.join(args.datasetDir, d) for d in os.listdir(args.datasetDir)
               if os.path.isdir(os.path.join(args.datasetDir, d))]

    # Create output directories (one per class)
    for dir in dirs:
        outputDir = os.path.join(args.outputDir, os.path.basename(dir))
        if os.path.exists(outputDir) is False:
            os.makedirs(outputDir)

    # Case with one directory with all files
    singleDir = False
    if dirs == []:
        singleDir = True
        dirs = [args.datasetDir]
        if os.path.exists(args.outputDir) is False:
            os.makedirs(args.outputDir)

    # One by one image prediction forcing CPU usage
    with model.getSession() as sess:

        with tf.device("/cpu:0"):

            for srcDir in tqdm(sorted(dirs), unit="directory"):

                for file in tqdm(sorted(glob.glob(srcDir + "/*.jpg")), unit="image"):

                    if singleDir:
                        outputDir = args.outputDir
                    else:
                        outputDir = os.path.join(args.outputDir, os.path.basename(srcDir))

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

                    # Save npy file (features have 1D shape, eg. mobilenet has 1024 elements, nasnet mobile 1056)
                    outputFileName = os.path.basename(file)
                    outputFileName = outputFileName[:outputFileName.rfind(".")]
                    np.save(os.path.join(outputDir, outputFileName + ".npy"), result.reshape(-1))

    print("Export features finished")


if __name__ == '__main__':
    main()
