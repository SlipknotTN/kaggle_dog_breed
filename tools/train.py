import _init_paths
import argparse

from config.ConfigParams import ConfigParams
from tfutils.export import exportModelToTF
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Activation
from keras.layers.pooling import AveragePooling2D
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras import optimizers
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

    # Image Generator, MobileNet needs [-1.0, 1.0] range (Inception like preprocessing)
    trainImageGenerator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
    valImageGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)

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

    # TODO: Create a model factory to load different architectures

    # Load MobileNet Full, with output shape of (None, 7, 7, 1024), final classifier is excluded
    baseModel = MobileNet(input_shape=(config.inputSize, config.inputSize, 3), alpha=config.mobilenetAlpha,
                          depth_multiplier=1, dropout=1e-3, include_top=False,
                          weights='imagenet', input_tensor=None, pooling=None)

    # Final classifier definition
    fineTunedModel = Sequential()

    fineTunedModel.add(baseModel)

    # Global average output shape (None, 1, 1, 1024).
    # Global pooling with AveragePooling2D to have a 4D Tensor and apply Conv2D.
    fineTunedModel.add(AveragePooling2D(pool_size=(baseModel.output_shape[1], baseModel.output_shape[2]),
                                        strides=(1, 1), padding='valid', name="global_pooling"))

    fineTunedModel.add(Dropout(rate=0.5))

    # Convolution layer that acts like fully connected, with 2 classes, output shape (None, 1, 1, 2)
    fineTunedModel.add(Conv2D(filters=trainGenerator.num_class, kernel_size=(1, 1), name="fc_conv"))

    # Reshape to (None, 2) to match the one hot encoding target and final softmax
    fineTunedModel.add(Flatten())

    # Final sofmax for deploy stage
    fineTunedModel.add(Activation('softmax', name="softmax"))

    # Freeze the base model layers, train only the last convolution
    for layer in fineTunedModel.layers[0].layers:
        layer.trainable = False

    # Train as categorical crossentropy (works also with numclasses > 2)
    fineTunedModel.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.SGD(lr=config.learningRate, momentum=config.momentum),
                           metrics=['categorical_accuracy'])

    # Callbacks for early stopping and best model save
    earlyStoppingCB = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=config.patience, verbose=1, mode='auto')
    modelChkptCB = ModelCheckpoint(args.modelOutputPath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                   save_weights_only=False, mode='auto', period=1)

    # fine-tune the model
    fineTunedModel.fit_generator(
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
