import os
import re

import tensorflow as tf
import numpy as np

from basic_model import CTCModel
from layers import FactorizedLSTMCell


class FactorizedCTCModel(CTCModel):
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIMIT (e.g. z1039) for a given audio wav file.
    """

    def add_prediction_op(self):
        """Applies a GRU RNN over the input data, then an affine layer projection. Steps to complete
        in this function:

        - Roll over inputs_placeholder with GRUCell, producing a Tensor of shape [batch_s, max_timestep,
          hidden_size].
        - Apply a W * f + b transformation over the data, where f is each hidden layer feature. This
          should produce a Tensor of shape [batch_s, max_timesteps, num_classes]. Set this result to
          "logits".

        Remember:
            * Use the xavier initialization for matrices (W, but not b).
            * W should be shape [hidden_size, num_classes].
        """
        # Non-recurrent hidden layers
        inputs = self.inputs_placeholder
        for i in range(self.config.num_hidden_layers):
            with tf.variable_scope('hidden%d' % (i + 1)) as vs:
                inputs = self.apply_affine_over_sequence(
                    inputs=inputs,
                    output_size=self.config.hidden_size,
                    activation=tf.nn.relu)

        # Construct forward and backward cells of bidirectional RNN
        fwdcell = FactorizedLSTMCell(
            self.config.hidden_size,
            self.config.svd_rank,
            activation=self.config.activation_func,
        )
        bckcell = FactorizedLSTMCell(
            self.config.hidden_size,
            self.config.svd_rank,
            activation=self.config.activation_func,
        )
        # TODO: look into non-zero initial hidden states?
        rnn_outputs, rnn_last_states = tf.nn.bidirectional_dynamic_rnn(
            fwdcell, bckcell,
            inputs=inputs,
            dtype=tf.float32,
            sequence_length=self.seq_lens_placeholder)

        # Sum the forward and backward hidden states together for the scores
        # scores.shape = [batch_s, max_timestep, 2*num_hidden]
        scores = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2, name='scores')

        # Push the scores through an affine layer
        # logits.shape = [batch_s, max_timestep, num_classes]
        with tf.variable_scope('final') as vs:
            self.logits = self.apply_affine_over_sequence(
                inputs=scores,
                output_size=self.config.num_classes)

# load instance of basic_model and extract weights
# run SVD to factorize RNN weight matrices
# instantiate a model that


def svd_truncate(X, r):
    """

    Args:
        X: matrix to truncate with shape (N, M)
        r: number of singular values to include

    Returns:
        (Z, P) where np.dot(Z, P) approximates X
            Z has shape (N, r) and P has shape (r, M)

    """
    U, s, W = np.linalg.svd(X, full_matrices=False)
    U_trunc = U[:, :r]
    S_trunc = np.diag(s[:r])
    W_trunc = W[:r, :]
    Z = np.dot(U_trunc, S_trunc)
    P = W_trunc

    # diagnostics
    # X_trunc = np.dot(Z, P)
    # diff = np.abs(X - X_trunc)
    # print(np.std(diff), np.max(diff))

    return Z, P


def pick_variable(name):
    try:
        return next(v for v in tf.global_variables() if v.name == name)
    except StopIteration:
        raise KeyError('variable %r not found' % name)


def factorize(config):
    """

    Args:
        config (config.Config):

    Returns:

    """

    # Load the original model to extract the weights and compute the factorized weights
    with tf.Graph().as_default() as g_original:
        with tf.Session(graph=g_original) as session:
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

            num_units = w_project.shape[0] // 2

            # Transforms:
            # hidden state -> input gate, the output gate, the forget gate and the cell state
            w_fw_recurrent = w_fw_cell[num_units:2*num_units, :].T
            w_bw_recurrent = w_bw_cell[num_units:2*num_units, :].T
            # hidden state -> input to next layer
            w_fw_project = w_project[:num_units, :].T
            w_bw_project = w_project[num_units:2*num_units, :].T

            # recurrent projection matrix for forward RNN
            Z_fw_recurrent, P_fw = svd_truncate(w_fw_recurrent, r=config.svd_rank)
            Z_bw_recurrent, P_bw = svd_truncate(w_bw_recurrent, r=config.svd_rank)
            print('Z_fw_recurrent.shape =', Z_fw_recurrent.shape)
            print('Z_bw_recurrent.shape =', Z_bw_recurrent.shape)

            # TODO: Compute least-squares


    # Instantiate a factorized version of the model
    with tf.Graph().as_default() as g_factorized:
        model = FactorizedCTCModel(config)
        variables_to_restore = [v for v in tf.global_variables()
                                if not re.match(r"bidirectional_rnn/[fb]w/lstm_cell/((anti_)?proj_)?weights(.*)", v.name)]
        restorer = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(tf.global_variables())

        # Load the weights from the original model into the new factorized graph
        # TODO: and manually load in the factorized weights computed above
        with tf.Session(graph=g_factorized) as session:
            session.run(tf.global_variables_initializer())
            restorer.restore(session, config.load_from_file)
            for v in tf.global_variables():
                print(v.name)

            pick_variable("bidirectional_rnn/fw/lstm_cell/weights:0").load(w_fw_cell[:num_units, :])
            pick_variable("bidirectional_rnn/fw/lstm_cell/proj_weights:0").load(P_fw.T)
            pick_variable("bidirectional_rnn/fw/lstm_cell/anti_proj_weights:0").load(Z_fw_recurrent.T)

            pick_variable("bidirectional_rnn/bw/lstm_cell/weights:0").load(w_bw_cell[:num_units, :])
            pick_variable("bidirectional_rnn/bw/lstm_cell/proj_weights:0").load(P_bw.T)
            pick_variable("bidirectional_rnn/bw/lstm_cell/anti_proj_weights:0").load(Z_bw_recurrent.T)

            os.makedirs(os.path.dirname(config.save_to_file), exist_ok=True)
            saver.save(session, config.save_to_file)

