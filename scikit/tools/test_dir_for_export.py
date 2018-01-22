import _init_paths
import argparse
import glob
import os

import numpy as np
from scipy.misc import imread, imresize
from tqdm import tqdm

from kaggle.export import exportResults
from sklearn.externals import joblib


# TODO: Read features

def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Keras test script")
    parser.add_argument("--datasetTestDirs", required=True, type=str, nargs='*',
                        help="Dataset test directories with features already exported")
    parser.add_argument("--labelsFile", required=False, type=str, default="../dataset/labels.csv", help="Labels file")
    parser.add_argument("--modelPath", required=False, type=str, default="./export/logreg.pkl",
                        help="Filepath with trained model")
    parser.add_argument("--kaggleExportFile", required=False, type=str, default=None,
                        help="CSV file in kaggle format for challenge upload")
    args = parser.parse_args()
    return args


def main():
    """
    Example of predict_generator usage for images without labels, images are read one by one and you can export results
    """

    # TODO: Add a generic load model

    args = doParsing()
    print(args)

    model = joblib.load(args.modelPath)

    print("Loaded model from " + args.modelPath)

    results = []

    for file in tqdm(sorted(glob.glob(args.datasetTestDirs[0] + "/*.npy"))):

        # One by one image prediction
        features = []
        features.append(np.load(file))

        if len(args.datasetTestDirs) > 1:
            for datasetTestDir in args.datasetTestDirs[1:]:
                features.append(np.load(os.path.join(datasetTestDir, os.path.basename(file))))

        # Get probabilities
        resultProba = model.predict_proba(np.concatenate(features).reshape(1, -1))
        results.append((os.path.basename(file)[:os.path.basename(file).rfind('.')], resultProba[0]))

    print("Test finished")

    if args.kaggleExportFile is not None:
        exportResults(results, args.labelsFile, args.kaggleExportFile)
        print("Results for kaggle saved in " + args.kaggleExportFile)


if __name__ == '__main__':
    main()
