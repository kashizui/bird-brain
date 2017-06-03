import tensorflow as tf

from utils import compare_predicted_to_true, compute_wer

class BatchSkipped(Exception): pass


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