#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import gmtime, strftime
import argparse
import json
import math
import pdb
import random
import time
# noinspection PyUnresolvedReferences,PyShadowingBuiltins
from six.moves import xrange as range

import numpy as np
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from utils import *


class Config(argparse.Namespace):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters.

    Values defined here are the default values which can be overridden by the
    command-line arguments.
    """
    #########################
    # BEGIN PARAM DEFS
    #########################
    # To define a help string associated with a parameter just make it a tuple
    # with the second value as the help string.
    train_path = './data/train/train.dat', "Give path to training data - this should not need to be changed if you are running from the assignment directory"
    val_path = './data/test/test.dat', "Give path to val data - this should not need to be changed if you are running from the assignment directory"
    save_every = 50, "Save model every x epochs. 0 means not saving at all."
    print_every = 10, "Print some training and val examples (true and predicted sequences) every x epochs. 0 means not printing at all."
    save_to_file = 'models/saved_model_epoch', "Provide filename prefix for saving intermediate models"
    load_from_file = None, "Provide filename to load saved model"

    context_size = 0
    num_mfcc_features = 24

    batch_size = 16
    num_classes = 28  # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12
    hidden_size = 128
    num_hidden_layers = 1

    activation = 'relu', "Activation type, either relu or tanh"
    cell_type = 'GRUCell', "RNN cell type, can be any member of tf.contrib.rnn, such as GRUCell, LSTMCell, or BasicRNNCell"

    num_epochs = 50
    l2_lambda = 0.0000001
    learning_rate = 1e-3

    # Define derived parameters as properties
    @property
    def num_final_features(self):
        return self.num_mfcc_features * (2 * self.context_size + 1)

    #########################
    # END PARAM DEFS
    #########################

    @classmethod
    def _build_parser(cls):
        """Build an ArgumentParser based on the params defined above."""
        parser = argparse.ArgumentParser()

        import inspect
        for key, value in inspect.getmembers(cls):
            if inspect.isroutine(value):  # skip methods
                continue
            if key.startswith('_'):  # skip magics
                continue
            if isinstance(value, property):  # skip properties
                continue

            doc = ''
            if isinstance(value, tuple):
                value, doc = value
            doc += ' (default: {})'.format(value)

            parser.add_argument(
                '--' + key.replace('_', '-'),
                nargs='?',
                default=NotImplemented,
                type=(str if value is None else type(value)),
                dest=key,
                help=doc,
            )
        return parser

    def __init__(self):
        """Load in params from system command line."""
        super().__init__()
        parser = Config._build_parser()
        args = parser.parse_args()
        for key, value in vars(args).items():
            if value is NotImplemented:
                default_value = getattr(Config, key)
                if isinstance(default_value, tuple):
                    default_value, _ = default_value
                setattr(self, key, default_value)
            else:
                setattr(self, key, value)

    def save(self, path):
        """Save config to JSON file."""
        with open(path, 'w') as fp:
            json.dump(vars(self), fp)


class CTCModel(object):
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIDIGITS (e.g. z1039) for a given audio wav file.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph:

        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32

        HINTS:
            - Use tf.sparse_placeholder(tf.int32) for targets_placeholder. This is required by TF's ctc_loss op.
            - Inputs is of shape [batch_size, max_timesteps, num_final_features], but we allow flexible sizes for
              batch_size and max_timesteps (hence the shape definition as [None, None, num_final_features].
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=[None, None, self.config.num_final_features])
        self.targets_placeholder = tf.sparse_placeholder(tf.int32)
        self.seq_lens_placeholder = tf.placeholder(tf.int32, shape=[None])

    def create_feed_dict(self, inputs_batch, targets_batch, seq_lens_batch):
        """Creates the feed_dict for the digit recognizer.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch:  A batch of input data.
            targets_batch: A batch of targets data.
            seq_lens_batch: A batch of seq_lens data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        return {
            self.inputs_placeholder: inputs_batch,
            self.targets_placeholder: targets_batch,
            self.seq_lens_placeholder: seq_lens_batch,
        }

    def apply_affine_over_sequence(self, name, inputs, output_size, activation=None):
        # inputs.shape = [batch_s, max_timestep, input_size]
        input_size = inputs.shape.as_list()[2]
        inputs_shape = tf.shape(inputs)  # get shape at runtime as well for batch_s and max_timestep

        self.W[name] = tf.get_variable('W' + name, shape=[input_size, output_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b[name] = tf.get_variable('b' + name, shape=[output_size],
                                       initializer=tf.contrib.layers.xavier_initializer())

        # Flatten the sequence into a long matrix and apply affine transform
        inputs_flat = tf.reshape(inputs, [-1, input_size])                  # shape = [batch_s * max_timestep, input_size]
        outputs_flat = tf.matmul(inputs_flat, self.W[name]) + self.b[name]  # shape = [batch_s * max_timestep, output_size]
        outputs = tf.reshape(outputs_flat, [inputs_shape[0], inputs_shape[1], output_size])  # shape = [batch_s, max_timestep, output_size]

        if activation is not None:
            outputs = activation(outputs)

        return outputs

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
            * W should be shape [hidden_size, num_classes]. num_classes for our dataset is 12
            * tf.contrib.rnn.GRUCell, tf.contrib.rnn.MultiRNNCell and tf.nn.dynamic_rnn are of interest
        """
        # Non-recurrent hidden layers
        inputs = self.inputs_placeholder
        for i in range(self.config.num_hidden_layers):
            inputs = self.apply_affine_over_sequence(
                name=str(i+1),
                inputs=inputs,
                output_size=self.config.hidden_size,
                activation=tf.nn.relu)

        # Construct forward and backward cells of bidirectional RNN
        construct_cell = getattr(tf.contrib.rnn, self.config.cell_type)
        fwdcell = construct_cell(
            self.config.hidden_size,
            activation=tf.nn.relu if self.config.activation == 'relu' else tf.tanh
        )
        bckcell = construct_cell(
            self.config.hidden_size,
            activation=tf.nn.relu if self.config.activation == 'relu' else tf.tanh
        )
        # TODO: look into non-zero initial hidden states?
        rnn_outputs, rnn_last_states = tf.nn.bidirectional_dynamic_rnn(
            fwdcell, bckcell,
            inputs=inputs,
            dtype=tf.float32,
            sequence_length=self.seq_lens_placeholder)

        # Sum the forward and backward hidden states together for the scores
        # scores.shape = [batch_s, max_timestep, num_hidden]
        scores = tf.add(rnn_outputs[0], rnn_outputs[1], name='scores')

        # Push the scores through an affine layer
        # logits.shape = [batch_s, max_timestep, num_classes]
        self.logits = self.apply_affine_over_sequence(
            name='final',
            inputs=scores,
            output_size=self.config.num_classes)

    def add_loss_op(self):
        """Adds Ops for the loss function to the computational graph.

        - Use tf.nn.ctc_loss to calculate the CTC loss for each example in the batch. You'll need self.logits,
          self.targets_placeholder, self.seq_lens_placeholder for this. Set variable ctc_loss to
          the output of tf.nn.ctc_loss
        - You will need to first tf.transpose the data so that self.logits is shaped [max_timesteps, batch_s,
          num_classes].
        - self.configure tf.nn.ctc_loss so that identical consecutive labels are allowed
        - Compute L2 regularization cost for all trainable variables. Use tf.nn.l2_loss(var).

        """
        # logitsT.shape = [max_timesteps, batch_s, num_classes]
        self.logitsT = tf.transpose(self.logits, perm=[1, 0, 2])

        ctc_loss = tf.nn.ctc_loss(
            labels=self.targets_placeholder,
            inputs=self.logitsT,
            sequence_length=self.seq_lens_placeholder,
            preprocess_collapse_repeated=False,  # FIXME?
            ctc_merge_repeated=True,
        )

        # Accumulate l2 cost over the weight matrices
        for _, W in self.W.items():
            l2_cost = tf.nn.l2_loss(W)

        # Remove inf cost training examples (no path found, yet)
        loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
        self.num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
        cost = tf.reduce_mean(loss_without_invalid_paths)

        self.loss = self.config.l2_lambda * l2_cost + cost

    def add_training_op(self):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables. The Op returned by this
        function is what must be passed to the `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model. Call optimizer.minimize() on self.loss.

        """
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate
        ).minimize(self.loss)

    def add_decoder_and_wer_op(self):
        """Setup the decoder and add the word error rate calculations here.

        Tip: You will find tf.nn.ctc_beam_search_decoder and tf.edit_distance methods useful here.
        Also, report the mean WER over the batch in variable wer

        """
        result = tf.nn.ctc_beam_search_decoder(
            self.logitsT,
            self.seq_lens_placeholder,
            top_paths=1,
        )
        decoded_sequence = tf.to_int32(result[0][0])

        # FIXME: Calculate actual WER?
        # edit_distance is no longer a proxy for WER, this is now character error rate
        wer = tf.reduce_mean(tf.edit_distance(
            hypothesis=decoded_sequence,
            truth=self.targets_placeholder,
            normalize=True,
        ))

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("wer", wer)

        self.decoded_sequence = decoded_sequence
        self.wer = wer

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()

    # This actually builds the computational graph
    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()
        self.add_decoder_and_wer_op()
        self.add_summary_op()

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        batch_cost, wer, batch_num_valid_ex, summary = session.run([self.loss, self.wer, self.num_valid_examples, self.merged_summary_op], feed)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped
            return 0
        if train:
            _ = session.run([self.optimizer], feed)

        return batch_cost, wer, summary

    def print_results(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)

    def __init__(self, config):
        self.W = {}
        self.b = {}
        self.config = config
        self.build()


def main():
    config = Config()
    config.save('config.json')

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    train_dataset = load_dataset(config.train_path)
    val_dataset = load_dataset(config.val_path)

    train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = make_batches(train_dataset, batch_size=Config.batch_size)
    val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(train_dataset, batch_size=len(val_dataset[0]))

    def pad_all_batches(batch_feature_array):
        for batch_num in range(len(batch_feature_array)):
            batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
        return batch_feature_array

    train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))

    with tf.Graph().as_default():
        model = CTCModel(config)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            # Initializate the weights and biases
            session.run(init)
            if config.load_from_file is not None:
                new_saver = tf.train.import_meta_graph('%s.meta'%config.load_from_file, clear_devices=True)
                new_saver.restore(session, config.load_from_file)

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

            global_start = time.time()

            step_ii = 0

            for curr_epoch in range(config.num_epochs):
                total_train_cost = total_train_wer = 0
                start = time.time()

                for batch in random.sample(range(num_batches_per_epoch),num_batches_per_epoch):
                    cur_batch_size = len(train_seqlens_minibatches[batch])

                    batch_cost, batch_ler, summary = model.train_on_batch(session, train_feature_minibatches[batch], train_labels_minibatches[batch], train_seqlens_minibatches[batch], train=True)
                    total_train_cost += batch_cost * cur_batch_size
                    total_train_wer += batch_ler * cur_batch_size

                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1

                train_cost = total_train_cost / num_examples
                train_wer = total_train_wer / num_examples

                val_batch_cost, val_batch_ler, _ = model.train_on_batch(session, val_feature_minibatches[0], val_labels_minibatches[0], val_seqlens_minibatches[0], train=False)

                log = "Epoch {}/{}, train_cost = {:.3f}, train_ed = {:.3f}, val_cost = {:.3f}, val_ed = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch+1, config.num_epochs, train_cost, train_wer, val_batch_cost, val_batch_ler, time.time() - start))

                # Write out status to JSON for CodaLab table display
                with open('status.json', 'w') as fp:
                    json.dump({
                        'epoch': curr_epoch + 1,
                        'train_cost': float(train_cost),
                        'train_wer': float(train_wer),
                        'val_batch_cost': float(val_batch_cost),
                        'val_batch_ler': float(val_batch_ler),
                    }, fp)

                if config.print_every > 0 and (curr_epoch + 1) % config.print_every == 0:
                    batch_ii = 0
                    model.print_results(session, train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii], train_seqlens_minibatches[batch_ii])

                if config.save_every > 0 and config.save_to_file and (curr_epoch + 1) % config.save_every == 0:
                    os.makedirs(os.path.dirname(config.save_to_file), exist_ok=True)
                    saver.save(session, config.save_to_file, global_step=curr_epoch + 1)


if __name__ == "__main__":
    main()
