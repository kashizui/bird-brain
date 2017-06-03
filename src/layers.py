import tensorflow as tf


def leaky_relu(alpha):
    # From https://groups.google.com/a/tensorflow.org/forum/#!msg/discuss/V6aeBw4nlaE/VUAgE-nXEwAJ
    def apply(inputs):
        return tf.maximum(alpha*inputs, inputs)
    return apply


def clipped_relu(mu):
    def apply(inputs):
        return tf.minimum(tf.maximum(inputs, 0), mu)
    return apply
