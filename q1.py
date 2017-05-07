import numpy as np
import argparse
import math
import random
import sys
import time
import os


DEBUG = int(os.environ.get('DEBUG', 0))


dot = ['.', 'X']
def display(arr):
    if DEBUG >= 2:
        print '============'
        for row in xrange(arr.shape[0]):
            for col in xrange(arr.shape[1]):
                sys.stdout.write(dot[int(arr[row,col] > 0)])
            sys.stdout.write('\n')
        sys.stdout.flush()
        time.sleep(1)


def softmax(x, axis):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class CTC(object):
    """
    Contains functionality for calculating CTC
    """

    def compute_ctc_loss(self, logits, target):
        """
        Given the RNN outputs (pre-softmax) at each time step and a target labeling,
        compute the negative of the CTC loss for one example, where the loss itself can be defined
        as the negative of the probability p(target | logits) as defined in equations 3 & 8
        in the paper ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf, using the forward-backward algorithm

        Inputs:

        - logits is a numpy float matrix of size (num_time_steps, num_labels + 1) representing
          the UNNORMALIZED outputs of the underlying RNN. Note that num_labels as used here doesn't include the blank label

        - target is a numpy integer array of size (seq_len, ), where seq_len <= num_time_steps,
          with each element of target represents a label and is an integer between 0 (inclusive) and num_labels (exclusive)

        Outputs:

        - a single float representing the negative log of p(target | logits) as defined in the paper

        Notes:

        - you should use compute_forward_variables or compute_backward_variables to compute the loss here

        - logits is unnormalized and may need to be converted to a probability distribution with a softmax operation
          to obtain y as defined in the paper

        """

        num_time_steps = logits.shape[0]
        num_labels = logits.shape[1] - 1
        num_labels_with_blank = num_labels + 1

        # sanity check to ensure targets are all right
        assert (target < num_labels).all()

        ######################
        ### YOUR CODE HERE ###
        ######################
        # Normalize by softmax
        normalized_logits = softmax(logits, axis=1)
        alpha = self.compute_forward_variables(normalized_logits, target)

        # Compute forward or backward pass and cost
        cost_forward = -np.log(alpha[-1, -1] + alpha[-2, -1])
        if DEBUG:
            beta = self.compute_backward_variables(normalized_logits, target)
            cost_backward = -np.log(beta[0, 0] + beta[1, 0])
            assert abs(cost_forward - cost_backward) < 1e-5
        return cost_forward

    def compute_forward_variables(self, normalized_logits, target):
        """
        Given the normalized RNN outputs (post-softmax) at each time step and a target labeling,
        compute the forward variables alpha_t(s) as defined in equation 5 in the paper
        ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf

        Inputs:

        - normalized_logits is a numpy float matrix of size (num_time_steps, num_labels + 1) representing
          the normalized/softmaxed activations of the underlying RNN, where num_labels does not include the blank label (hence the +1)

        - target is a numpy integer array of size (seq_len, ), where seq_len <= num_time_steps,
          with each element of target represents a label and is an integer between 0 (inclusive) and num_labels (exclusive)

        Outputs:

        - a numpy float matrix of size (2 * target_length + 1, max_time_steps) where the (s,t)th entry of the matrix
          equals alpha_t(s) as defined in the paper

        """

        target_length = target.shape[0]
        num_time_steps = normalized_logits.shape[0]
        num_labels = normalized_logits.shape[1] - 1

        ######################
        ### YOUR CODE HERE ###
        ######################
        # For clarification (ignoring 0- and 1-indexing differences):
        #   \alpha_t(s) = alpha[s, t]
        #   y_c^t       = normalized_logits[t, c]
        #   \pi_t       = target[t]
        # Other notes:
        # blank label is the last label:
        #   y_b^t       = normalized_logits[num_labels, BLANK]
        BLANK = num_labels
        S_MAX = 2 * target_length + 1

        # initialize alpha
        alpha = np.zeros((S_MAX + 1, num_time_steps))  # +1 for zero padding since alpha = 0 for s < 0 and s >= S_MAX
        touched = np.zeros((S_MAX, num_time_steps))

        alpha[0, 0] = normalized_logits[0, BLANK]
        alpha[1, 0] = normalized_logits[0, target[0]]

        # iterate in an order that allowed recurrence to be used
        for t in xrange(1, num_time_steps):
            s_beg = max(0, S_MAX - 2*(num_time_steps-t))
            s_end = min(S_MAX, 2*t + 2)
            for s in xrange(s_beg, s_end):
                alpha_bar = alpha[s, t-1] + alpha[s-1, t-1]
                target_idx = s / 2
                if s % 2 == 0:  # target is BLANK
                    alpha[s, t] = alpha_bar * normalized_logits[t, BLANK]
                elif target_idx > 0 and target[target_idx] == target[target_idx - 1]:  # target is same as prev
                    alpha[s, t] = alpha_bar * normalized_logits[t, target[target_idx]]
                else:
                    alpha[s, t] = (alpha_bar + alpha[s-2, t-1]) * normalized_logits[t, target[target_idx]]
                touched[s, t] = 1

            display(touched)

        return alpha[:S_MAX, :]


    def compute_backward_variables(self, normalized_logits, target):
        """
        Given the normalized RNN outputs (post-softmax) at each time step and a target labeling,
        compute the backward variables beta_t(s) as defined in equation 9 in the paper
        ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf

        Inputs:

        - normalized_logits is a numpy float matrix of size (num_time_steps, num_labels + 1) representing
          the normalized/softmaxed activations of the underlying RNN (y as defined in the paper), where
          num_labels does not include the blank label!

        - target is a numpy integer array of size (seq_len, ), where seq_len <= num_time_steps,
          with each element of target represents a label and is an integer between 0 (inclusive) and num_labels (exclusive)

        Outputs:

        - a numpy float matrix of size (2 * target_length + 1, max_time_steps) where the (s,t)th entry of the matrix
          equals beta_t(s) as defined in the paper

        """

        target_length = target.shape[0]
        num_time_steps = normalized_logits.shape[0]
        num_labels = normalized_logits.shape[1] - 1

        ######################
        ### YOUR CODE HERE ###
        ######################
        BLANK = num_labels
        S_MAX = 2 * target_length + 1

        # initialize beta
        beta = np.zeros((S_MAX + 1, num_time_steps))  # +1 for zero padding since alpha = 0 for s < 0 and s >= S_MAX
        touched = np.zeros((S_MAX, num_time_steps))
        beta[-1-1, -1] = normalized_logits[-1, BLANK]  # extra -1 to account for zero padding above
        beta[-2-1, -1] = normalized_logits[-1, target[-1]]

        # iterate in an order that allowed recurrence to be used
        for t in reversed(xrange(0, num_time_steps-1)):
            s_beg = max(0, S_MAX - 2*(num_time_steps-t))
            s_end = min(S_MAX, 2*t + 2)
            for s in reversed(xrange(s_beg, s_end)):
                beta_bar = beta[s, t+1] + beta[s+1, t+1]
                target_idx = s / 2
                if s % 2 == 0:  # target is blank
                    beta[s, t] = beta_bar * normalized_logits[t, BLANK]
                elif target_idx < target_length-1 and \
                        target[target_idx] == target[target_idx+1]:  # target is same as next
                    beta[s, t] = beta_bar * normalized_logits[t, target[target_idx]]
                else:
                    beta[s, t] = (beta_bar + beta[s+2, t+1]) * normalized_logits[t, target[target_idx]]
                touched[s, t] = 1
            display(touched)

        return beta[:S_MAX, :]

    def compute_gradients(self, logits, target):
        """
        Given the RNN outputs (pre-softmax) at each time step and a target labeling,
        compute the gradients of the CTC loss w.r.t. the unnormalized logits

        Inputs:

        - logits is a numpy float matrix of size (num_time_steps, num_labels + 1) representing
          the outputs y (as in the paper) of the underlying RNN, and num_labels does not include the blank label!

        - target is a numpy integer array of size (seq_len, ), where seq_len <= num_time_steps,
          with each element of target represents a label and is an integer between 0 (inclusive) and num_labels (exclusive)

        Outputs:

        - a numpy float matrix of size (num_time_steps, num_labels + 1) where the (s,t)th entry of the matrix
          equals the gradient of the CTC loss w.r.t. the UNNORMALIZED activation of label t at timestep s

        Notes:

        - you would probably call both compute_forward_variables and compute_backward_variables here

        - logits is unnormalized and may need to be converted to a probability distribution with a softmax operation
          to obtain y as in the paper

        """

        target_length = target.shape[0]
        num_time_steps = logits.shape[0]
        num_labels = logits.shape[1] - 1

        ######################
        ### YOUR CODE HERE ###
        ######################
        BLANK = num_labels
        normalized_logits = softmax(logits, axis=1)

        # Compute forward and backward passes
        alpha = self.compute_forward_variables(normalized_logits, target)
        beta = self.compute_backward_variables(normalized_logits, target)

        # Compute rescaled versions of forward and backward variables
        # alpha_hat.shape = beta_hat.shape = (2 * target_length + 1, num_time_steps)
        alpha_hat = alpha / np.sum(alpha, axis=0, keepdims=True)
        beta_hat = beta / np.sum(beta, axis=0, keepdims=True)
        gamma_hat = alpha_hat * beta_hat

        # Generate the expanded target ("l-prime")
        expanded_target = np.full((2*target_length+1,), BLANK, dtype=np.int32)
        expanded_target[1::2] = target

        # Extract the probabilities for each character in the expanded target,
        # for each time step
        # target_probs.shape = (2 * target_length + 1, num_time_steps)
        target_probs = normalized_logits[:, expanded_target].T

        # Compute Z  thingy
        # Z.shape = (num_time_steps,)
        Z = np.sum(gamma_hat / target_probs, axis=0)

        dlogit = np.empty_like(logits)

        for label in xrange(num_labels+1):
            s, = np.where(expanded_target == label)
            term = np.sum(gamma_hat[s, :], axis=0)

            for t in xrange(num_time_steps):
                 dlogit[t, label] = normalized_logits[t, label] - term[t] / (normalized_logits[t, label] * Z[t])

        return dlogit


if __name__ == "__main__":

    ctc_instance = CTC()

    t_sample = np.array([0, 3, 1, 2, 0], dtype=np.int32)
    inp_sample = np.array(
        [[0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
        [0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
        [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
        [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
        [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]])
    print("----------------------\nCTC loss on sample input:\nCalculated = %f\nExpected = %f" % (ctc_instance.compute_ctc_loss(inp_sample, t_sample), 8.4266024))

    expected_gradient_sample = np.array([[-0.84611112,  0.24801813,  0.18197855,  0.13847439,  0.13849108,  0.13914907],
        [ 0.25880966,  0.17131636,  0.15051439, -0.86088479,  0.13929759,  0.14094676],
        [ 0.14179565, -0.7421388,   0.18867508,  0.13715352,  0.13718593,  0.13732868],
        [ 0.14627296,  0.26060188, -0.81886262,  0.13727479,  0.13737293,  0.13734008],
        [-0.78102362,  0.20589428,  0.15666439,  0.13938184,  0.13973717,  0.13934597]])
    print("\nCTC loss gradient w.r.t. unnormalized sample input:\nCalculated =")
    print(ctc_instance.compute_gradients(inp_sample, t_sample))
    print("\nExpected = ")
    print(expected_gradient_sample)

    print('\n----------------------\n')

    t1 = np.array([0, 1, 1, 0], dtype=np.int32)
    inp1 = np.asarray(
        [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
         [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
         [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
         [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
         [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
        dtype=np.float32)

    print("CTC loss on input 1: %f" % ctc_instance.compute_ctc_loss(inp1, t1))
    print("CTC loss gradient w.r.t. unnormalized input 1:")
    print(ctc_instance.compute_gradients(inp1, t1))

    t2 = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    inp2 = np.array([[ -8.,  23.,  -5.,   7.,  17.,  -9.],
        [ 12.,  -7.,   9.,  24.,  24.,  -8.],
        [ 23.,  28.,  -93.,  27.,  18.,  14.],
        [ 12.,  23.,  -5.,  28.,   2.,  27.],
        [  3.,  27., -10.,  14.,   7.,  15.]], dtype=np.float32)

    print("\nCTC loss on input 2: %f\n----------------------" % ctc_instance.compute_ctc_loss(inp2, t2))



