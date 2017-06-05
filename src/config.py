import tensorflow as tf
import argparse
import json
import os
import warnings

from basic_model import CTCModel, CTCModelNoSum
from factorized import FactorizedCTCModel
from quantized_model import QuantizedCTCModel
from layers import leaky_relu, clipped_relu


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
    num_classes = 28 # 62  # 11 (TIDIGITS - 0-9 + oh) + 1 (blank) = 12
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

    svd_rank = 28

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

    def get_model(self):
        if self.model == 'basic':
            return CTCModel(self)
        if self.model == 'nosum':
            return CTCModelNoSum(self)
        if self.model == 'quantized':
            return QuantizedCTCModel(self)
        if self.model == 'factorized':
            return FactorizedCTCModel(self)
        raise Exception('unknown model type %r' % (self.model,))

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
        if os.path.exists(path):
            warnings.warn('Overwriting config at %s' % path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(vars(self), fp)
