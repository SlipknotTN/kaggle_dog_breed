import tensorflow as tf
from tensorflow.python.platform import gfile


class TensorflowModel(object):

    def __init__(self, modelPath):
        self.sess = None
        self.loadModel(modelPath)

    def loadModel(self, modelPath):
        # Load the saved graph
        with gfile.GFile(modelPath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess = tf.Session()
            tf.import_graph_def(graph_def, name="")

            return

    def getGraph(self):
        return self.sess.graph

    def getSession(self):
        return self.sess
