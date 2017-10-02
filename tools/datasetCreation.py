import argparse
import os
import glob
import csv
from tqdm import tqdm


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainDatasetDir", required=True, type=str, help="Directory with train images (no symlink!)")
    parser.add_argument("--labelsFile", required=True, type=str, help="Labels file")
    parser.add_argument("--outputDatasetDir", required=True, type=str, help="Output directory with subdirs (no symlink!)")
    args = parser.parse_args()
    return args


def main():
    """
    Split images train dataset, create directory for each class containing symlinks to original files
    """

    args = doParsing()
    print(args)

    if os.path.exists(args.trainDatasetDir) is False:
        raise Exception("Train Dataset dir " + args.trainDatasetDir + " not found")

    if os.path.exists(args.outputDatasetDir) is False:
        os.makedirs(args.outputDatasetDir)

    samples = dict()
    with open(args.labelsFile, 'r') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        next(reader)
        for row in reader:
            samples[row[0]] = row[1]

    # Iterate over subclasses
    imageFiles = glob.glob(args.trainDatasetDir + "/*.jpg")
    for imageFile in tqdm(imageFiles):

        filenameNoExt = os.path.basename(imageFile)[:os.path.basename(imageFile).rfind('.')]
        className = samples[filenameNoExt]

        destinationDir = os.path.join(args.outputDatasetDir, className)
        if os.path.exists(destinationDir) is False:
            os.makedirs(destinationDir)

        # Create symlink to file
        try:
            os.symlink(imageFile, os.path.join(destinationDir, os.path.basename(imageFile)))
        except FileExistsError:
            pass

    print("Success")


if __name__ == '__main__':
    main()
