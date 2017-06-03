import tensorflow as tf
import numpy as np


# load instance of basic_model and extract weights
# run SVD to factorize RNN weight matrices
# instantiate a model that


def factorize(config):
    with tf.Session() as session:
        if config.load_from_file is None:
            raise Exception('specify model with --load-from-file')
        saver = tf.train.import_meta_graph('%s.meta' % config.load_from_file, clear_devices=True)
        saver.restore(session, config.load_from_file)
        session.run(tf.global_variables_initializer())

        for v in tf.trainable_variables():
            print(v.name)

        # TODO: measure model size (num parameters, bytes)

        w_fw_cell = session.run("bidirectional_rnn/fw/lstm_cell/weights:0")
        w_bw_cell = session.run("bidirectional_rnn/bw/lstm_cell/weights:0")
        w_project = session.run("final/W:0")

        # hidden_size = num_units = 512
        # the recurrent matrix should be 512 x 512??

        num_units = config.hidden_size

        w_fw_recurrent = w_fw_cell[num_units:2*num_units, :]
        w_bw_recurrent = w_bw_cell[num_units:2*num_units, :]

        w_fw_project = w_project[:num_units, :]




