import _init_paths
import argparse

from config.ConfigParams import ConfigParams
from tfutils.export import exportModelToTF
from model.ModelsFactory import ModelsFactory
from preprocess.preprocess import getPreprocessFunction
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


def doParsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Keras training script")
    parser.add_argument("--datasetTrainDir", required=True, type=str, help="Dataset train directory")
    parser.add_argument("--datasetValDir", required=True, type=str, help="Dataset validation directory")
    parser.add_argument("--configFile", required=True, type=str, help="Config file path")
    parser.add_argument("--modelOutputPath", required=False, type=str, default="./export/mobilenet_fn.h5",
                        help="Filepath where to save the keras model")
    parser.add_argument("--tfModelOutputDir", required=False, type=str, default=None,
                        help="Optional directory where to export model in TF format (checkpoint + graph)")
    args = parser.parse_args()
    return args


def main():

    args = doParsing()
    print(args)

    config = ConfigParams(args.configFile)

    preprocessFunction = getPreprocessFunction(config.preprocessType)

    # Image Generator using preprocess function, e.g. MobileNet needs [-1.0, 1.0] range (Inception like preprocessing)
    trainImageGenerator = ImageDataGenerator(preprocessing_function=preprocessFunction,
                                             rotation_range=10,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             zoom_range=.1,
                                             horizontal_flip=True)
    valImageGenerator = ImageDataGenerator(preprocessing_function=preprocessFunction)

    trainGenerator = trainImageGenerator.flow_from_directory(
        args.datasetTrainDir,
        # height, width
        target_size=(config.inputSize, config.inputSize),
        batch_size=config.batchSize,
        class_mode='categorical',
        shuffle=True)

    valGenerator = valImageGenerator.flow_from_directory(
        args.datasetValDir,
        # height, width
        target_size=(config.inputSize, config.inputSize),
        batch_size=config.batchSize,
        class_mode='categorical',
        shuffle=False)

    # Load model using config file
    model = ModelsFactory.create(config, trainGenerator.num_class)

    print(model.summary())

    # Train as categorical crossentropy (works also with numclasses > 2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=config.optimizer,
                  metrics=['categorical_accuracy'])

    # Callbacks for early stopping and best model save
    earlyStoppingCB = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=config.patience, verbose=1, mode='auto')
    modelChkptCB = ModelCheckpoint(args.modelOutputPath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                   save_weights_only=False, mode='auto', period=1)

    # fine-tune the model
    model.fit_generator(
        trainGenerator,
        steps_per_epoch=trainGenerator.samples//trainGenerator.batch_size,
        epochs=config.epochs,
        validation_data=valGenerator,
        validation_steps=valGenerator.samples//valGenerator.batch_size,
        callbacks=[earlyStoppingCB, modelChkptCB])

    print("Training finished")

    print("Model saved to " + args.modelOutputPath)

    # Export model to TF format
    if args.tfModelOutputDir is not None:
        exportModelToTF(args.tfModelOutputDir)


if __name__ == '__main__':
    main()
