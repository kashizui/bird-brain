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
from models import *

from six.moves import xrange as range

import numpy as np
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from utils import *


class CTCModel(Model):
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIMIT (e.g. z1039) for a given audio wav file.
    """

    def apply_affine_over_sequence(self, inputs, output_size, activation=None):
        # inputs.shape = [batch_s, max_timestep, input_size]
        input_size = inputs.shape.as_list()[2]
        inputs_shape = tf.shape(inputs)  # get shape at runtime as well for batch_s and max_timestep

        W = tf.get_variable('W', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[output_size])

        # Flatten the sequence into a long matrix and apply affine transform
        inputs_flat = tf.reshape(inputs, [-1, input_size])      # shape = [batch_s * max_timestep, input_size]
        outputs_flat = tf.matmul(inputs_flat, W) + b            # shape = [batch_s * max_timestep, output_size]
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
            * W should be shape [hidden_size, num_classes].
        """
        # Non-recurrent hidden layers
        inputs = self.inputs_placeholder
        for i in range(self.config.num_hidden_layers):
            with tf.variable_scope('hidden%d' % (i+1)) as vs:
                inputs = self.apply_affine_over_sequence(
                    inputs=inputs,
                    output_size=self.config.hidden_size,
                    activation=tf.nn.relu)

        # Construct forward and backward cells of bidirectional RNN
        construct_cell = getattr(tf.contrib.rnn, self.config.cell_type)
        fwdcell = construct_cell(
            self.config.hidden_size,
            activation=self.config.activation_func,
        )
        bckcell = construct_cell(
            self.config.hidden_size,
            activation=self.config.activation_func,
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
        with tf.variable_scope('final') as vs:
            self.logits = self.apply_affine_over_sequence(
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
        l2_cost = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

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

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        batch_cost, wer, batch_num_valid_ex, summary = session.run([self.loss, self.wer, self.num_valid_examples, self.merged_summary_op], feed)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped
            raise BatchSkipped
        if train:
            _ = session.run([self.optimizer], feed)

        return batch_cost, wer, summary

    def print_results(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)

    def __init__(self, config):
        self.config = config
        self.build()
