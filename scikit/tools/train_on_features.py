import argparse
import glob
import os

from tqdm import tqdm
import numpy as np

#from config.ConfigParams import ConfigParams
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Tensorflow features export script")
    parser.add_argument("--configFile", required=False, default=None, type=str, help="Model config file")
    parser.add_argument("--datasetDirs", required=True, type=str, nargs='*',
                        help="Root directories with classes subdirs")
    parser.add_argument("--modelOutputFile", required=False, default="./export/logreg.pkl",
                        help='Output path where to save the model')
    args = parser.parse_args()
    return args


def main():
    """
    Script to train a scikit model on features arrays.
    """
    args = doParsing()
    print(args)

    SEED = 27

    # Load config (default params at the moment)
    #config = ConfigParams(args.configFile)

    # Retrieve directory from the first dataset
    subDirs = sorted([d for d in os.listdir(args.datasetDirs[0]) if os.path.isdir(os.path.join(args.datasetDirs[0], d))])

    trainX = []
    trainY = []

    # Read training samples
    for subDir in tqdm(sorted(subDirs), unit="directory"):

        # Retrieve target from first dataset and then read from the others
        for file in sorted(glob.glob(os.path.join(args.datasetDirs[0], subDir) + "/*.npy")):

            trainInputs = []
            trainInputs.append(np.load(file))

            if len(args.datasetDirs) > 1:
                for datasetDir in args.datasetDirs[1:]:
                    trainInputs.append(np.load(os.path.join(datasetDir, subDir, os.path.basename(file))))

            fullTrainInput = np.concatenate(trainInputs)
            trainX.append(fullTrainInput)

            # Target index
            trainY.append(subDirs.index(subDir))

    # TODO: Split in train + validation and predict on validation
    print("Init model")
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=SEED, verbose=1)
    # Input shape (n_samples, n_features)
    # Target shape (n_samples)
    print("Fitting model")
    model.fit(np.stack(trainX), np.array(trainY))

    print("Train finished")

    # Save model
    joblib.dump(model, args.modelOutputFile)
    print("Saved model in " + args.modelOutputFile)


if __name__ == '__main__':
    main()
