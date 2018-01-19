import tensorflow as tf
import os

from .DatasetReader import DatasetReader
from constants.Constants import Constants as constants
from image.ImageUtilsTF import ImageUtilsTF


class DatasetTFReader(DatasetReader):

    def __init__(self, datasetDir, datasetMetadata, configParams):

        DatasetReader.__init__(self, datasetDir=datasetDir, datasetMetadata=datasetMetadata, configParams=configParams)

    # TFRec files have standard names
    def readTFExamplesTraining(self):
        return self.readTFExamples(tfrecordsFile=os.path.join(self.datasetDir, "data_train.tfrecords"), shuffle=True)

    def readTFExamplesValidation(self):
        return self.readTFExamples(tfrecordsFile=os.path.join(self.datasetDir, "data_val.tfrecords"), shuffle=False)

    def readTFExamples(self, tfrecordsFile, shuffle=True):
        """
        Read TFRecords (only images and labels, not proposals).
        Use this for classification and detection RCNN.
        :param tfrecordsFile:
        :param shuffle:
        :return:
        """

        feature = {constants.DatasetFeatures.image: tf.FixedLenFeature([], tf.string),
                   constants.DatasetFeatures.label: tf.FixedLenFeature([], tf.int64)}

        # Create a list of filenames and pass it to a queue
        filenameQueue = tf.train.string_input_producer([tfrecordsFile],
                                                       num_epochs=self.configParams.epochs,
                                                       shuffle=shuffle)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filenameQueue)

        # Parsing and preprocessing
        image = None
        label = None
        with tf.device('/cpu:0'):

            # Decode the record read by the reader
            features = tf.parse_single_example(serialized_example, features=feature)

            if self.configParams.imageEncoding is None:
                # Convert the image data from string back to the numbers RGB
                image = tf.decode_raw(features[constants.DatasetFeatures.image], tf.uint8)
            elif str.lower(str(self.configParams.imageEncoding)) == "jpeg":
                # Image stored as uint8 RGB
                image = tf.image.decode_jpeg(features[constants.DatasetFeatures.image], channels=3)
            else:
                raise Exception("Image encoding " + self.configParams.imageEncoding + " not supported")

            # Cast label data into int64
            label = tf.cast(features[constants.DatasetFeatures.label], tf.int64)

            # Reshape image data into the original shape of the JPEG image, HWC
            image = tf.reshape(image, [self.configParams.inputSize, self.configParams.inputSize, 3])

            # Optional grayscale conversion if the model has BW input
            if self.configParams.inputChannels == 1:
                image = tf.image.rgb_to_grayscale(image)

            # Convert image format to float32 to match model input type
            image = tf.cast(image, tf.float32)

            # Label One Hot
            label = tf.one_hot(label, depth=self.datasetMetadata.numClasses)

            # Preprocess image (resize, mean subtraction, ...)
            # For simplicity we create the dataset with images ready for the model, no need to further resize
            # So width = height = inputSize
            modelInputHeight = self.configParams.inputSize
            modelInputWidth = self.configParams.inputSize
            image = ImageUtilsTF.preprocessing(image, width=modelInputWidth, height=modelInputHeight,
                                               preprocessingType=self.configParams.preprocessType,
                                               meanRGB=self.configParams.meanRGB)

            # Optional BGR conversion (depending on the model)
            if self.configParams.inputFormat == "BGR":
                image = tf.reverse(image, axis=[-1])

            # Random mirroring
            image = tf.image.random_flip_left_right(image, seed=0)

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch([image, label], batch_size=self.configParams.batchSize,
                                                capacity=self.configParams.batchSize * 4, num_threads=8,
                                                min_after_dequeue=self.configParams.batchSize)

        return images, labels