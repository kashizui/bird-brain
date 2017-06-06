import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import (
    _checked_scope,
    _linear,
    LSTMStateTuple,
)

from tensorflow.python.platform import tf_logging as logging


def dense(inputs, output_size, activation=None, bias=True):
    # inputs.shape = [batch_s, max_timestep, input_size]
    input_size = inputs.shape.as_list()[2]
    inputs_shape = tf.shape(inputs)  # get shape at runtime as well for batch_s and max_timestep

    W = tf.get_variable('W', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
    if bias:
        b = tf.get_variable('b', shape=[output_size])

    # Flatten the sequence into a long matrix and apply affine transform
    inputs_flat = tf.reshape(inputs, [-1, input_size])      # shape = [batch_s * max_timestep, input_size]
    if bias:
        outputs_flat = tf.matmul(inputs_flat, W) + b            # shape = [batch_s * max_timestep, output_size]
    else:
        outputs_flat = tf.matmul(inputs_flat, W)                # shape = [batch_s * max_timestep, output_size]
    outputs = tf.reshape(outputs_flat, [inputs_shape[0], inputs_shape[1], output_size])  # shape = [batch_s, max_timestep, output_size]

    if activation is not None:
        outputs = activation(outputs)

    return outputs


def leaky_relu(alpha):
    # From https://groups.google.com/a/tensorflow.org/forum/#!msg/discuss/V6aeBw4nlaE/VUAgE-nXEwAJ
    def apply(inputs):
        return tf.maximum(alpha*inputs, inputs)
    return apply


def clipped_relu(mu):
    def apply(inputs):
        return tf.minimum(tf.maximum(inputs, 0), mu)
    return apply


class FactorizedLSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  The default non-peephole implementation is based on:
    http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
  The peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  """

  def __init__(self, num_units, input_size=None,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=tanh, reuse=None):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: Deprecated and unused.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._reuse = reuse

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
      scope: VariableScope for the created subgraph; defaults to "lstm_cell".
    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, self._num_units])



    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    with _checked_scope(self, scope or "lstm_cell",
                        initializer=self._initializer,
                        reuse=self._reuse) as unit_scope:
      if self._num_unit_shards is not None:
        unit_scope.set_partitioner(
            partitioned_variables.fixed_size_partitioner(
                self._num_unit_shards))
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      input_contributions = _linear([inputs], 4 * self._num_units, bias=True)
      with tf.variable_scope('projection'):
          mprev_projected = _linear([m_prev], self._num_proj, bias=False)
      with tf.variable_scope('antiprojection'):
          mprev_contributions = _linear([mprev_projected], 4 * self._num_units, bias=False)
      lstm_matrix = input_contributions + mprev_contributions
      i, j, f, o = array_ops.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)
      # Diagonal connections
      if self._use_peepholes:
        with vs.variable_scope(unit_scope) as projection_scope:
          if self._num_unit_shards is not None:
            projection_scope.set_partitioner(None)
          w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
             sigmoid(i + w_i_diag * c_prev) * self._activation(j))
      else:
        c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
             self._activation(j))

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with vs.variable_scope("projection", reuse=True) as proj_scope:
          if self._num_proj_shards is not None:
            proj_scope.set_partitioner(
                partitioned_variables.fixed_size_partitioner(
                    self._num_proj_shards))
          out = _linear(m, self._num_proj, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          out = clip_ops.clip_by_value(out, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type
      else:
        out = m

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))

    return out, new_state
