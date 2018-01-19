import os

from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph


def exportModelToTF(tfModelOutputDir):

    if not os.path.exists(tfModelOutputDir):
        os.makedirs(tfModelOutputDir)

    # Save checkpoint
    saver = tf.train.Saver()
    save_path = saver.save(K.get_session(), tfModelOutputDir + "/model")

    # Save metagraph
    tf.train.write_graph(K.get_session().graph.as_graph_def(), "", tfModelOutputDir + "/metagraph.pb", False)

    # Freeze graph
    freeze_graph(input_graph=tfModelOutputDir + "/metagraph.pb", input_saver="", input_binary=True,
                 input_checkpoint=tfModelOutputDir + "/model", output_node_names='softmax/Softmax',
                 restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                 output_graph=tfModelOutputDir + "/graph.pb", clear_devices=True, initializer_nodes="")
