import _init_paths
import argparse
import glob
import os

import numpy as np
from scipy.misc import imread, imresize
from tqdm import tqdm

from kaggle.export import exportResults
from keras.models import load_model
from keras.applications import mobilenet


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Keras test script")
    parser.add_argument("--datasetTestDir", required=True, type=str, help="Dataset test directory")
    parser.add_argument("--labelsFile", required=False, type=str, default="./dataset/labels.csv", help="Labels file")
    parser.add_argument("--modelPath", required=False, type=str, default="./export/mobilenet_fn.h5",
                        help="Filepath with trained model")
    parser.add_argument("--kaggleExportFile", required=False, type=str, default=None,
                        help="CSV file in kaggle format for challenge upload")
    args = parser.parse_args()
    return args


def main():
    """
    Example of predict_generator usage for images without labels, images are read one by one and you can export results
    """
    args = doParsing()
    print(args)

    model = load_model(args.modelPath, custom_objects={
                       'relu6': mobilenet.relu6,
                       'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

    print("Loaded model from " + args.modelPath)

    print(model.summary())

    results = []

    for file in tqdm(sorted(glob.glob(args.datasetTestDir + "/*.jpg"))):

        # One by one image prediction

        # Image processing (resize and inception like preprocessing to have [-1.0, 1.0] input range)
        image = imread(file)
        image = imresize(image, size=model.input_shape[1:3])
        image = image.astype(np.float32)
        processedImage = mobilenet.preprocess_input(image)

        # Add 1st dimension for image index in batch
        processedImage = np.expand_dims(processedImage, axis=0)

        # Get and save dog probability
        resultProba = model.predict_proba(x=processedImage, batch_size=1, verbose=False)
        results.append((os.path.basename(file)[:os.path.basename(file).rfind('.')], resultProba[0]))

    print("Test finished")

    if args.kaggleExportFile is not None:
        exportResults(results, args.labelsFile, args.kaggleExportFile)
        print("Results for kaggle saved in " + args.kaggleExportFile)


if __name__ == '__main__':
    main()
