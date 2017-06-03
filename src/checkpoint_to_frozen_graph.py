import os, argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(input_checkpoint, output_graph):
    output_node_names = "DecodedSequence"
    clear_devices = True
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as session:
        saver.restore(session, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(
            session,
            input_graph_def,
            output_node_names.split(",")
        ) 

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a checkpoint to a frozen graph.")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint", default="models/saved_model_epoch-50")
    parser.add_argument('--output', type=str, help="Output frozen graph path", default="models/frozen_model.pb")
    args = parser.parse_args()
    freeze_graph(args.checkpoint, args.output)