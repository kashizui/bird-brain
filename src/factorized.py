import tensorflow as tf
import numpy as np


# load instance of basic_model and extract weights
# run SVD to factorize RNN weight matrices


def factorize(config):
    # model = config.get_model()
    # init = tf.global_variables_initializer()

    with tf.Session() as session:
        if config.load_from_file is None:
            raise Exception('specify model with --load-from-file')
        saver = tf.train.import_meta_graph('%s.meta' % config.load_from_file, clear_devices=True)
        saver.restore(session, config.load_from_file)
        session.run(tf.global_variables_initializer())

        fw_cell_weights = session.run("bidirectional_rnn/fw/lstm_cell/weights:0")
        bw_cell_weights = session.run("bidirectional_rnn/bw/lstm_cell/weights:0")
        print(bw_cell_weights)
        print(bw_cell_weights.shape)
