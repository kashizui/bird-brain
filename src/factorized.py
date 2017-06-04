import tensorflow as tf
import numpy as np


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


def factorize(config):
    """

    Args:
        config (config.Config):

    Returns:

    """
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

        num_units = w_project.shape[0] // 2

        # Transforms:
        # hidden state -> input gate, the output gate, the forget gate and the cell state
        w_fw_recurrent = w_fw_cell[num_units:2*num_units, :].T
        w_bw_recurrent = w_bw_cell[num_units:2*num_units, :].T
        # hidden state -> input to next layer
        w_fw_project = w_project[:num_units, :].T
        w_bw_project = w_project[num_units:2*num_units, :].T

        print(w_fw_recurrent.shape)
        print(w_bw_recurrent.shape)
        print(w_fw_project.shape)
        print(w_bw_project.shape)

        # recurrent projection matrix for forward RNN
        w_fw_recurrent_trunc = svd_truncate(w_fw_recurrent, r=512)
        w_bw_recurrent_trunc = svd_truncate(w_bw_recurrent, r=512)

        # print(v)
        # u, v, w = np.linalg.svd(w_bw_recurrent)
        # print(v)
        # u, v, w = np.linalg.svd(w_fw_project)
        # print(v)
        # u, v, w = np.linalg.svd(w_bw_project)
        # print(v)





