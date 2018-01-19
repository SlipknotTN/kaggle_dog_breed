import argparse

import tensorflow as tf
from tensorflow.python.platform import gfile


def doParsing():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Script to load graph file to tensorboard')
    parser.add_argument('--modelPath', required=True, help="Model pb file path")
    parser.add_argument('--tensorboardOutputDir', required=False, default="./tensorboard/temp",
                        help="Output directory for tensorboard")
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    with tf.Session() as sess:
        model_filename = args.modelPath
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)

    tensorboardDir = args.tensorboardOutputDir
    train_writer = tf.summary.FileWriter(tensorboardDir)
    train_writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
