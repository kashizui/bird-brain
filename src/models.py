import tensorflow as tf
import argparse
import json

from utils import compare_predicted_to_true, compute_wer

class BatchSkipped(Exception): pass

def leaky_relu(alpha):
    # From https://groups.google.com/a/tensorflow.org/forum/#!msg/discuss/V6aeBw4nlaE/VUAgE-nXEwAJ
    def apply(inputs):
        return tf.maximum(alpha*inputs, inputs)
    return apply


def clipped_relu(mu):
    def apply(inputs):
        return tf.minimum(tf.maximum(inputs, 0), mu)
    return apply


ACTIVATION_ALIAS = {
    'relu': tf.nn.relu,
    'tanh': tf.tanh,
    'leaky': leaky_relu,
    'clipped': clipped_relu,
}


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
    test_path = './data/test/test.dat', "Give path to val data - this should not need to be changed if you are running from the assignment directory"
    phase = 'train'
    save_every = 10, "Save model every x epochs. 0 means not saving at all."
    print_every = 10, "Print some training and val examples (true and predicted sequences) every x epochs. 0 means not printing at all."
    save_to_file = 'models/saved_model_epoch', "Provide filename prefix for saving intermediate models"
    load_from_file = None, "Provide filename to load saved model"

    context_size = 0
    num_mfcc_features = 26

    batch_size = 16
    num_classes = 28  # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12
    hidden_size = 128
    num_hidden_layers = 1

    activation = 'tanh', "Activation type for the recurrent layers [options: " + ', '.join(ACTIVATION_ALIAS.keys()) + "]"
    leaky_alpha = 0.01, "alpha value for leaky ReLU activation"
    clipped_mu = 1., "max value for clipped ReLU activation"
    cell_type = 'LSTMCell', "RNN cell type, can be any member of tf.contrib.rnn, such as GRUCell, LSTMCell, or BasicRNNCell"

    num_epochs = 50
    l2_lambda = 0.0000001
    learning_rate = 1e-3
    
    model = "basic", "Can be basic or quantized"

    # Define derived parameters as properties
    @property
    def num_final_features(self):
        return self.num_mfcc_features * (2 * self.context_size + 1)

    @property
    def activation_func(self):
        func = ACTIVATION_ALIAS[self.activation]
        if self.activation == 'leaky':
            return func(self.leaky_alpha)
        if self.activation == 'clipped':
            return func(self.clipped_mu)
        return func

    #########################
    # END PARAM DEFS
    #########################

    @classmethod
    def _build_parser(cls):
        """Build an ArgumentParser based on the params defined above."""
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--config', nargs='?', default=None, type=str, dest='config',
            help='Load config from this file')

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
        old_config = {}

        # Load in any existing config file specified
        if args.config is not None:
            with open(args.config, 'r') as fp:
                delattr(args, 'config')
                old_config = json.load(fp)

        for key, value in vars(args).items():
            if value is NotImplemented:
                # First try old config, then try global defaults
                default_value = old_config.get(key, getattr(Config, key))
                if isinstance(default_value, tuple):
                    default_value, _ = default_value
                setattr(self, key, default_value)
            else:
                setattr(self, key, value)

    def save(self, path):
        """Save config to JSON file."""
        with open(path, 'w') as fp:
            json.dump(vars(self), fp)
            
class Model(object):

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).
        Adds following nodes to the computational graph:
        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32
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
        
        
    def add_prediction_op(self):
        pass
    
    def add_loss_op(self):
        pass
     
    def add_training_op(self):
        pass
     
    def add_decoder_and_wer_op(self):
        """Setup the decoder and add the word error rate calculations here.

        Tip: You will find tf.nn.ctc_beam_search_decoder and tf.edit_distance methods useful here.
        Also, report the mean WER over the batch in variable wer

        """
        decoded, log_probability = tf.nn.ctc_beam_search_decoder(
            self.logitsT,
            self.seq_lens_placeholder,
            top_paths=1,
        )
        decoded_sequence = tf.to_int32(decoded[0])

        # FIXME: Calculate actual WER?
        # edit_distance is no longer a proxy for WER, this is now character error rate
        ler = tf.reduce_mean(tf.edit_distance(
            hypothesis=decoded_sequence,
            truth=self.targets_placeholder,
            normalize=True,
        ))
        
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("ler", ler)

        self.decoded_sequence = decoded_sequence
        self.wer = ler
     
    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()
     
    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()
        self.add_decoder_and_wer_op()
        self.add_summary_op()
        
    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch, train=True):
        pass
    
    def test_on_batch(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        self.train_on_batch(session, train_inputs_batch, train_targets_batch, train_seq_len_batch, train=False)
    
    def print_results(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)
        result = compute_wer(train_first_batch_preds, train_targets_batch)
        print("WER {:.3f}".format(result))
    
    def __init__(self, config):
        pass