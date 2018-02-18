import pandas as pd
import argparse


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Clipping predictions script")
    parser.add_argument("--srcFile", required=True, type=str, help="Source CSV file path")
    parser.add_argument("--outputFile", required=True, type=str, help="Output CSV file path with clipped predictions")
    parser.add_argument("--clip", required=False, type=float, default=0.9, help="Upperbound clipping value")
    args = parser.parse_args()
    return args


def main():
    """
    Clip predictions to improve kaggle score, e.g. https://www.kaggle.com/sbrugman/tricks-for-the-kaggle-leaderboard
    """

    args = doParsing()
    print(args)

    # Read CSV using first column (image name) as row index, so the actual row has only float values
    df = pd.read_csv(args.srcFile, index_col=0)

    # Simple analysis of average max confidence to choose the clip value
    maxs = df.max(axis=0)
    avgConfidence = maxs.mean()
    print("Average max confidence: " + str(avgConfidence))

    # Clipping
    classes = df.shape[1]
    df = df.clip(lower=(1 - args.clip)/(classes - 1), upper=args.clip)

    # Normalize the values to 1 (divide by the sum on each
    df = df.div(df.sum(axis=1), axis=0)

    # Simple analysis of average max confidence to choose the clip value
    maxs = df.max(axis=0)
    avgConfidence = maxs.mean()
    print("Average max confidence after: " + str(avgConfidence))

    df.to_csv(args.outputFile)
    print("Output file saved in " + args.outputFile)


if __name__ == "__main__":
    main()