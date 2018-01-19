import _init_paths
import argparse
import os
from tensorflow.python.tools.freeze_graph import freeze_graph

from config.ConfigParams import ConfigParams
from data.DatasetMetadata import DatasetMetadata
from data.DatasetTFReader import DatasetTFReader
from model.TensorflowModel import TensorflowModel
from model.ModelFactory import ModelFactory
from trainProcess.trainDevice import selectTrainDevice
from trainProcess.TrainProcess import TrainProcess


def doParsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Tensorflow image classification fine tuning')
    parser.add_argument('--datasetDir', required=True, default=None, help='Dataset directory')
    parser.add_argument('--baseModelDir', type=str, required=True, help='Base model folder')
    parser.add_argument('--checkpointOutputDir', required=False, default="./export",
                        help='Output folder that will contains checkpoints')
    parser.add_argument('--modelOutputDir', required=False, default="./export",
                        help='Output folder that will contains final trained model graph.pb')
    parser.add_argument('--configFile', required=True, help='Config File for training')
    parser.add_argument('--tensorboardDir', required=False, default=None, help="TensorBoard directory")
    parser.add_argument('--useGpu', type=str, required=False, default=None,
                        help='GPU ID to use for the training (default CPU training)')
    return parser.parse_args()


def main():

    args = doParsing()
    print(args)

    # Read training configuration (config file is in common for dataset creation and training hyperparameters)
    configParams = ConfigParams(args.configFile)

    # Select train device
    trainDevice = selectTrainDevice(args.useGpu)

    # Load DataProvider
    dataProvider = DatasetTFReader(
        datasetDir=args.datasetDir,
        datasetMetadata=DatasetMetadata().initFromJson(os.path.join(args.datasetDir, "metadata.json")),
        configParams=configParams)

    # Load base model graph
    baseTFModel = TensorflowModel(os.path.join(args.baseModelDir, "graph.pb"))

    # Append classifier for fine tuning training
    trainingModel = ModelFactory.create(config=configParams, tfmodel=baseTFModel,
                                        dataProvider=dataProvider, trainDevice=trainDevice)

    # Run training
    trainProcess = TrainProcess(config=configParams, trainingModel=trainingModel,
                                dataProvider=dataProvider, outputDir=args.checkpointOutputDir,
                                tensorboardDir=args.tensorboardDir)
    trainProcess.runTrain()

    # Freeze graph (graphdef plus parameters),
    # this includes in the graph only the layers needed to provide the output_node_names
    freeze_graph(input_graph=args.checkpointOutputDir + "/model_graph.pb", input_saver="", input_binary=True,
                 input_checkpoint=args.checkpointOutputDir + "/model", output_node_names=configParams.outputName,
                 restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                 output_graph=args.modelOutputDir + "/graph.pb", clear_devices=True, initializer_nodes="")


if __name__ == '__main__':

    main()
