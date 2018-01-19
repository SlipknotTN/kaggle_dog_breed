import os
import tensorflow as tf
from tqdm import tqdm


class TrainProcess(object):

    def __init__(self, config, trainingModel, dataProvider, outputDir, tensorboardDir=None):

        self.config = config
        self.trainingModel = trainingModel
        self.dataProvider = dataProvider
        self.outputDir = outputDir
        self.summaryWriter = self.tensorboard(tensorboardDir)

    def tensorboard(self, tensorboardDir):
        # Enable the tensorboard
        if tensorboardDir is not None:
            print("Writing on tensorboard")
            return tf.summary.FileWriter(tensorboardDir, graph=self.trainingModel.getGraph())
        return None

    def runTrain(self):

        numEpochs = self.config.epochs
        with self.trainingModel.getSession() as sess:

            trainImagesProvider, trainLabelsProvider = self.dataProvider.readTFExamplesTraining()
            valImagesProvider, valLabelsProvider = self.dataProvider.readTFExamplesValidation()

            # We must avoid initialization for layers not trained from scratch (freezed or fine tuned)
            init_op = self.getInitializers()
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:

                # Save the whole graph definition
                saver = tf.train.Saver()
                if self.outputDir is not None:
                    if os.path.exists(self.outputDir) is False:
                        os.makedirs(self.outputDir)

                tf.train.write_graph(sess.graph.as_graph_def(), "", self.outputDir + "/model_graph.pb", False)

                trainingBatchesNum = self.dataProvider.getTrainBatchesNumber()
                validationBatchesNum = self.dataProvider.getValidationBatchesNumber()

                maxValidationAccuracy = 0.0
                for epoch in tqdm(range(numEpochs), desc="Training epochs progress"):

                    self.trainEpoch(session=sess, epochIndex=epoch, trainingBatchesNum=trainingBatchesNum,
                                    trainImagesProvider=trainImagesProvider, trainLabelsProvider=trainLabelsProvider)

                    currentValidationAccuracy = self.valEpoch(session=sess, epochIndex=epoch,
                                                              validationBatchesNum=validationBatchesNum,
                                                              valImagesProvider=valImagesProvider,
                                                              valLabelsProvider=valLabelsProvider)

                    if self.config.saveBestEpoch is True:
                        if currentValidationAccuracy > maxValidationAccuracy:
                            maxValidationAccuracy = currentValidationAccuracy

                            print("Epoch {:d}: saving new checkpoint file".format(epoch))
                            save_path = saver.save(sess, self.outputDir + "/model")
                    else:
                        print("Epoch {:d}: saving new checkpoint file".format(epoch))
                        save_path = saver.save(sess, self.outputDir + "/model")
            finally:

                # Stop threads also when the training process is stopped before the end

                # Stop the threads
                coord.request_stop()

                # Wait for threads to stop
                coord.join(threads)

    def getInitializers(self):

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        return init_op

    def trainEpoch(self, session, epochIndex, trainingBatchesNum, trainImagesProvider, trainLabelsProvider):

        # Training phase for each epoch
        trainingBatchesAverageLoss = 0.0
        trainingBatchesCount = 0
        for batchIndex in tqdm(range(trainingBatchesNum), desc="Training batches progress"):

            # Extract batch
            imagesBatch, gtBatch = session.run([trainImagesProvider, trainLabelsProvider])

            _, trainLoss, lrateSummary = session.run([self.trainingModel.optimizer,
                                                      self.trainingModel.cost,
                                                      self.trainingModel.lrateSummary],
                                                     feed_dict={self.trainingModel.x: imagesBatch,
                                                                self.trainingModel.y: gtBatch}
                                                     )

            # Update average training loss
            trainingBatchesAverageLoss += trainLoss
            #print("intermediate batch loss:" + str(trainingBatchesAverageLoss))
            trainingBatchesCount += 1

            # Display learning rate of the last batch on the tensorboard
            if batchIndex == trainingBatchesNum - 1:
                if self.summaryWriter is not None:
                    self.summaryWriter.add_summary(lrateSummary, epochIndex)


        # Get final average training loss for each epoch
        trainingBatchesAverageLoss /= trainingBatchesCount

        # Display validation accuracy on the tensorboard
        costSummary = session.run(self.trainingModel.costSummary,
                                  feed_dict={self.trainingModel.scalarInputForSummary: trainingBatchesAverageLoss})

        if self.summaryWriter is not None:
            self.summaryWriter.add_summary(costSummary, epochIndex)

        print("Training Loss: {:.4f}...".format(trainingBatchesAverageLoss))

    def valEpoch(self, session, epochIndex, validationBatchesNum, valImagesProvider, valLabelsProvider):

        # Validation phase for current epoch
        validationBatchesAccuracyTotal = 0.0
        validationBatchesCount = 0

        for _ in tqdm(range(validationBatchesNum), desc="Validation batches progress"):
            imagesBatchValidation, gtBatchValidation = session.run([valImagesProvider, valLabelsProvider])

            validationBatchesAccuracyTotal += session.run(self.trainingModel.accuracy,
                                                          feed_dict={self.trainingModel.x: imagesBatchValidation,
                                                                     self.trainingModel.y: gtBatchValidation})
            validationBatchesCount += 1

        validationBatchesAccuracyTotal /= validationBatchesCount
        # Display validation accuracy on the tensorboard
        summary = session.run(self.trainingModel.accuracySummary,
                              feed_dict={self.trainingModel.scalarInputForSummary: validationBatchesAccuracyTotal})

        if self.summaryWriter is not None:
            self.summaryWriter.add_summary(summary, epochIndex)

        print('Validation accuracy: {}'.format(validationBatchesAccuracyTotal))

        return validationBatchesAccuracyTotal
