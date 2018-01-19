import argparse
import os
import glob
from tqdm import tqdm


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainDatasetDir", required=True, type=str, help="Train dataset directory")
    parser.add_argument("--valSplit", required=False, type=int, default=20, help="Validation split")
    parser.add_argument("--outputDatasetDir", required=True, type=str, help="Output directory with train and val subdir")
    args = parser.parse_args()
    return args


def main():

    args = doParsing()
    print(args)

    if os.path.exists(args.trainDatasetDir) is False:
        raise Exception("Train Dataset dir " + args.trainDatasetDir + " not found")

    if os.path.exists(args.outputDatasetDir) is False:
        os.makedirs(args.outputDatasetDir)
    if os.path.exists(os.path.join(args.outputDatasetDir, "train")) is False:
        os.makedirs(os.path.join(args.outputDatasetDir, "train"))
    if os.path.exists(os.path.join(args.outputDatasetDir, "val")) is False:
        os.makedirs(os.path.join(args.outputDatasetDir, "val"))

    # Iterate over subclasses
    subdirs = next(os.walk(args.trainDatasetDir))[1]
    for directory in subdirs:

        if os.path.exists(os.path.join(args.outputDatasetDir, "train", directory)) is False:
            os.makedirs(os.path.join(args.outputDatasetDir, "train", directory))

        if os.path.exists(os.path.join(args.outputDatasetDir, "val", directory)) is False:
            os.makedirs(os.path.join(args.outputDatasetDir, "val", directory))

        files = glob.glob(os.path.join(args.trainDatasetDir, directory) + "/*.jpg")
        trainFiles = files[:int(len(files) * (100 - args.valSplit) / 100)]
        valFiles = files[int(len(files) * (100 - args.valSplit) / 100):]

        for source in tqdm(trainFiles):
            try:
                os.symlink(source, os.path.join(args.outputDatasetDir, "train", directory, os.path.basename(source)))
            except FileExistsError:
                pass

        for source in tqdm(valFiles):
            try:
                os.symlink(source, os.path.join(args.outputDatasetDir, "val", directory, os.path.basename(source)))
            except FileExistsError:
                pass

    print("Success")


if __name__ == '__main__':
    main()
