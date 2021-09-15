import tensorflow as tf
from tensorflow.python.keras.backend import epsilon
from tensorflow.python.keras.losses import Loss

EPSILON = tf.constant(1e-12, dtype=tf.float32)


class AveragedMMILoss(Loss):
    """
    The Maximum Mutual Information loss for left-to-right stationary Hidden Markov Model
    This function takes a windowed predictions and returns the overlapping average
    as the prediction

    Based on:
    Fritz, L. and Burshtein, D., 2017.
    Simplified end-to-end MMI training and voting for ASR. arXiv preprint arXiv:1703.10356.

    Attributes
    ----------
    trans_mat : tf.Variable
        a tensor containing the transition matrix for the markov chain
    p_states : tf.Variable
        a tensor containing the state probabilities
    alpha : tf.Variable
        a tensor containing the forward probabilities, i.e., likelihood until time t

    Methods
    -------
    call(y_true, y_pred)
        The call build for Keras, returning the value of the MMI given a markov, conditional state probabilities
        from a DNN and an observation

    _forward_backward(y_pred)
        The forward algorithm that returns P(O|Model)

    _get_averaged_prediction(target, y_pred, patch_size=64, stride=8, num_classes=4)
        computes the overlapping window average and returns predictions


    """

    def __init__(self, trans_mat, p_states):
        super().__init__()
        self.trans_mat = trans_mat
        self.p_states = p_states
        self.alpha = tf.Variable(tf.zeros(tf.shape(p_states), dtype=tf.float32))

    def call(self, y_true, y_pred):
        """
        Computes -MMI(P(O, S) - P(O)) for minimization under gradient descent

        Parameters
        ----------
        y_true : tf.Tensor
            The tensor of one-hot-encoded ground-truth
        y_pred : tf.Tensor
            The tensor of the output layer of a DNN

        Returns
        -------
        tf.Variable
            The negative MMI value for optimization
        """
        y_pred_ = self._get_averaged_prediction(y_true, y_pred)
        classes = tf.argmax(y_true, axis=1)
        outputs = tf.reduce_max(y_true[1:] * y_pred_[1:], axis=1)
        states = tf.reduce_max(self.p_states * y_true[1:], axis=1)
        transitions_ = tf.tensordot(self.trans_mat + EPSILON, tf.transpose(y_true[1:]), axes=1)
        tf_time = tf.range(0, tf.shape(classes)[0], 1, dtype=tf.int64)
        full_indices = tf.stack([classes[:-1], tf_time[:-1]], axis=1)
        transitions = tf.gather_nd(transitions_, full_indices)

        log_p_os = tf.reduce_sum(tf.math.log(transitions + EPSILON)) + \
                   tf.reduce_sum(tf.math.log(outputs + EPSILON)) - tf.reduce_sum(tf.math.log(states + EPSILON))

        log_p_o = -tf.math.reduce_sum(self._forward_backward(y_pred_) +
                                      epsilon())  # log p(o|HMM) = - sum_t log(c_t | HMM)
        loss = -(log_p_os - log_p_o)

        return loss

    @staticmethod
    def _get_averaged_prediction(target, y_pred, patch_size=64, stride=8, num_classes=4):
        """
        Computes the average predictions per overlapping windows

        Parameters
        ----------
        y_true : tf.Tensor
            The tensor of one-hot-encoded ground-truth
        y_pred : tf.Tensor
            The tensor of the output layer of a DNN

        Returns
        -------
        tf.Variable
            The negative MMI value for optimization
        """
        length = tf.shape(target)[0]

        # Compute the indices were the windows start and end
        start = tf.range(start=0, limit=length - patch_size, delta=stride)
        end = tf.range(start=patch_size, limit=length, delta=stride)
        cond = (length - patch_size) % stride > 0  # condition for squeezing the last window
        if cond:
            last_window_start, last_window_end = length - patch_size, length
            start = tf.pad(start, [[0, 1]], constant_values=last_window_start)
            end = tf.pad(end, [[0, 1]], constant_values=last_window_end)

        cum_sum = tf.zeros(shape=[length, num_classes], dtype=tf.float32)
        overlapping_windows = tf.zeros(shape=[length], dtype=tf.float32)
        for i in range(tf.shape(y_pred)[0]):
            # pad the indices that are not in the window and add the window value to the sum
            cum_sum = cum_sum + tf.cast(tf.pad(y_pred[i], paddings=[[start[i], length - end[i]], [0, 0]]),
                                        dtype=tf.float32)
            # add a mask of LENGTH zeros with PATCH_SIZE ones where the window is defined
            # thus memorizing the number of windows that contribute to a prediction
            overlapping_windows = overlapping_windows + tf.pad(
                tf.ones(shape=[patch_size]), paddings=[[start[i], length - end[i]]]
            )

        # Compute the final average. Transpose operations used to make it broadcastable
        logits = tf.transpose(tf.transpose(cum_sum) / overlapping_windows)
        return logits

    def _forward_backward(self, y_pred):
        """
        Calculates the likelihood P(O) using a scaled forward algorithm.
        Bayes theorem is used given DNN output = P(S_t|O_t) to model the emissions.
        That is: p(O_t|S_t) = [P(S_t|O_t)P(O_t)]/P(S)

        Scaled implementation based on section 5.A of:
        Rabiner, L.R., 1989. A tutorial on hidden Markov models and selected applications in speech recognition.
        Proceedings of the IEEE, 77(2), pp.257-286.

        Parameters
        ----------
        y_pred : tf.Tensor
            The tensor of the output layer of a DNN

        Returns
        -------
        tf.Variable
            The likelihood P(O) given the markov chain and neural network output
        """

        T = tf.shape(y_pred)[0]
        y_bar = tf.math.divide(y_pred[0, :] + EPSILON, self.p_states + EPSILON)
        p_bar = self.p_states
        self.alpha.assign(tf.cast(tf.math.multiply_no_nan(self.p_states, y_bar), dtype=tf.float32))
        one = tf.constant(1.0, dtype=tf.float32)
        scaling_factors = 0
        scale = tf.math.divide_no_nan(one, tf.reduce_sum(self.alpha))
        scaling_factors += tf.math.log(scale)
        self.alpha.assign(tf.math.multiply(self.alpha, scale))
        for t in range(1, T):
            # we assume that the chain can only transition or to the same state or to the s+1 state
            alpha_lhs_ = tf.math.multiply(
                tf.roll(self.alpha, shift=1, axis=0),
                tf.linalg.tensor_diag_part(tf.roll(self.trans_mat, shift=1, axis=0))
            )
            alpha_rhs_ = tf.math.multiply_no_nan(self.alpha, tf.linalg.tensor_diag_part(self.trans_mat))

            alpha_ = alpha_lhs_ + alpha_rhs_
            # we assume that the markov process is stationary.
            # i.e. that the state probabilities are the principal eigenvector of the transition matrix
            p_bar = tf.tensordot(p_bar, self.trans_mat, axes=1)
            y_bar = tf.math.divide_no_nan(y_pred[t, :] + 1e-12, p_bar + 1e-12)
            self.alpha.assign(tf.cast(tf.math.multiply_no_nan(y_bar, alpha_), dtype=tf.float32))
            scale = tf.math.divide_no_nan(one, tf.reduce_sum(self.alpha))
            self.alpha.assign(tf.math.multiply_no_nan(self.alpha, scale))
            scaling_factors += tf.math.log(scale)

        return scaling_factors


class MMILoss(Loss):
    """
    The Maximum Mutual Information loss for left-to-right stationary Hidden Markov Model

    Based on:
    Fritz, L. and Burshtein, D., 2017.
    Simplified end-to-end MMI training and voting for ASR. arXiv preprint arXiv:1703.10356.

    Attributes
    ----------
    trans_mat : tf.Variable
        a tensor containing the transition matrix for the markov chain
    p_states : tf.Variable
        a tensor containing the state probabilities
    alpha : tf.Variable
        a tensor containing the forward probabilities, i.e., likelihood until time t

    Methods
    -------
    call(y_true, y_pred)
        The call build for Keras, returning the value of the MMI given a markov, conditional state probabilities
        from a DNN and an observation

    _forward_backward(y_pred)
        The forward algorithm that returns P(O|Model)
    """

    def __init__(self, trans_mat, p_states):
        super().__init__()
        self.trans_mat = trans_mat
        self.p_states = p_states
        self.alpha = tf.Variable(tf.zeros(tf.shape(p_states), dtype=tf.float32))

    def call(self, y_true, y_pred):
        """
        Computes -MMI(P(O, S) - P(O)) for minimization under gradient descent

        Parameters
        ----------
        y_true : tf.Tensor
            The tensor of one-hot-encoded ground-truth
        y_pred : tf.Tensor
            The tensor of the output layer of a DNN

        Returns
        -------
        tf.Variable
            The negative MMI value for optimization
        """
        classes = tf.argmax(y_true, axis=1)
        outputs = tf.reduce_max(y_true[1:] * y_pred[1:], axis=1)
        states = tf.reduce_max(self.p_states * y_true[1:], axis=1)
        tf.print("classes", tf.shape(classes), "outputs", tf.shape(outputs), "states", tf.shape(states))
        transitions_ = tf.tensordot(self.trans_mat + EPSILON, tf.transpose(y_true[1:]), axes=1)
        tf_time = tf.range(0, tf.shape(classes)[0], 1, dtype=tf.int64)
        full_indices = tf.stack([classes[:-1], tf_time[:-1]], axis=1)
        transitions = tf.gather_nd(transitions_, full_indices)

        log_p_os = tf.reduce_sum(tf.math.log(transitions + EPSILON)) + \
                   tf.reduce_sum(tf.math.log(outputs + EPSILON)) - tf.reduce_sum(tf.math.log(states + EPSILON))

        log_p_o = -tf.math.reduce_sum(self._forward_backward(y_pred) +
                                      epsilon())  # log p(o|HMM) = - sum_t log(c_t | HMM)
        loss = -(log_p_os - log_p_o)

        return loss

    def _forward_backward(self, y_pred):
        """
        Calculates the likelihood P(O) using a scaled forward algorithm.
        Bayes theorem is used given DNN output = P(S_t|O_t) to model the emissions.
        That is: p(O_t|S_t) = [P(S_t|O_t)P(O_t)]/P(S)

        Scaled implementation based on section 5.A of:
        Rabiner, L.R., 1989. A tutorial on hidden Markov models and selected applications in speech recognition.
        Proceedings of the IEEE, 77(2), pp.257-286.

        Parameters
        ----------
        y_pred : tf.Tensor
            The tensor of the output layer of a DNN

        Returns
        -------
        tf.Variable
            The likelihood P(O) given the markov chain and neural network output
        """

        T = tf.shape(y_pred)[0]
        y_bar = tf.math.divide(y_pred[0, :] + EPSILON, self.p_states + EPSILON)
        p_bar = self.p_states
        self.alpha.assign(tf.cast(tf.math.multiply_no_nan(self.p_states, y_bar), dtype=tf.float32))
        one = tf.constant(1.0, dtype=tf.float32)
        scaling_factors = 0
        scale = tf.math.divide_no_nan(one, tf.reduce_sum(self.alpha))
        scaling_factors += tf.math.log(scale)
        self.alpha.assign(tf.math.multiply(self.alpha, scale))
        for t in range(1, T):
            # we assume that the chain can only transition or to the same state or to the s+1 state
            alpha_lhs_ = tf.math.multiply(
                tf.roll(self.alpha, shift=1, axis=0),
                tf.linalg.tensor_diag_part(tf.roll(self.trans_mat, shift=1, axis=0))
            )
            alpha_rhs_ = tf.math.multiply_no_nan(self.alpha, tf.linalg.tensor_diag_part(self.trans_mat))

            alpha_ = alpha_lhs_ + alpha_rhs_
            # we assume that the markov process is stationary.
            # i.e. that the state probabilities are the principal eigenvector of the transition matrix
            p_bar = tf.tensordot(p_bar, self.trans_mat, axes=1)
            y_bar = tf.math.divide_no_nan(y_pred[t, :] + 1e-12, p_bar + 1e-12)
            self.alpha.assign(tf.cast(tf.math.multiply_no_nan(y_bar, alpha_), dtype=tf.float32))
            scale = tf.math.divide_no_nan(one, tf.reduce_sum(self.alpha))
            self.alpha.assign(tf.math.multiply_no_nan(self.alpha, scale))
            scaling_factors += tf.math.log(scale)

        return scaling_factors


class MMILossUnet(Loss):
    """
    The Maximum Mutual Information loss for left-to-right stationary Hidden Markov Model
    given an U-Net segmenting a window of o_t = [o_{t-F}, ..., o_t, ..., o_{t+F}]

    Based on:
    Fritz, L. and Burshtein, D., 2017.
    Simplified end-to-end MMI training and voting for ASR. arXiv preprint arXiv:1703.10356.

    Attributes
    ----------
    trans_mat : tf.Variable
        a tensor containing the transition matrix for the markov chain
    p_states : tf.Variable
        a tensor containing the state probabilities
    alpha : tf.Variable
        a tensor containing the forward probabilities, i.e., likelihood until time t

    Methods
    -------
    call(y_true, y_pred)
        The call build for Keras, returning the value of the MMI given a markov, conditional state probabilities
        from a DNN and an observation

    _forward_backward(y_pred)
        The forward algorithm that returns P(O|Model)
    """
    def __init__(self, trans_mat, p_states):
        super().__init__()
        self.trans_mat = trans_mat
        self.p_states = p_states
        self.alpha = tf.Variable(tf.zeros(tf.shape(p_states), dtype=tf.float32))

    def call(self, y_true, y_pred):
        """
        Computes -MMI(P(O, S) - P(O)) for minimization under gradient descent

        Parameters
        ----------
        y_true : tf.Tensor
            The tensor of one-hot-encoded ground-truth
        y_pred : tf.Tensor
            The tensor of the output layer of a DNN

        Returns
        -------
        tf.Variable
            The negative MMI value for optimization
        """
        y_pred, y_true = tf.squeeze(y_pred), tf.squeeze(y_true)  # [1, patch_size, L] -> [patch_size, L]
        classes = tf.argmax(y_true, axis=1)
        outputs = tf.reduce_max(y_true[1:] * y_pred[1:], axis=1)
        states = tf.reduce_max(self.p_states * y_true[1:], axis=1)
        transitions_ = tf.tensordot(self.trans_mat + EPSILON, tf.transpose(y_true[1:]), axes=1)

        tf_time = tf.range(0, tf.shape(classes)[0], 1, dtype=tf.int64)
        full_indices = tf.stack([classes[:-1], tf_time[:-1]], axis=1)
        transitions = tf.gather_nd(transitions_, full_indices)

        log_p_os = tf.reduce_sum(tf.math.log(transitions + EPSILON)) + \
                   tf.reduce_sum(tf.math.log(outputs + EPSILON)) - tf.reduce_sum(tf.math.log(states + EPSILON))

        log_p_o = -tf.math.reduce_sum(self._forward_backward(y_pred) +
                                      epsilon())  # log p(o|HMM) = - sum_t log(c_t | HMM)
        loss = -(log_p_os - log_p_o)

        return loss

    def _forward_backward(self, y_pred):
        """
        Calculates the likelihood P(O) using a scaled forward algorithm.
        Bayes theorem is used given DNN output = P(S_t|O_t) to model the emissions.
        That is: p(O_t|S_t) = [P(S_t|O_t)P(O_t)]/P(S)

        Scaled implementation based on section 5.A of:
        Rabiner, L.R., 1989. A tutorial on hidden Markov models and selected applications in speech recognition.
        Proceedings of the IEEE, 77(2), pp.257-286.

        Parameters
        ----------
        y_pred : tf.Tensor
            The tensor of the output layer of a DNN

        Returns
        -------
        tf.Variable
            The likelihood P(O) given the markov chain and neural network output
        """

        T = tf.shape(y_pred)[0]
        y_bar = tf.math.divide(y_pred[0, :] + EPSILON, self.p_states + EPSILON)
        p_bar = self.p_states
        self.alpha.assign(tf.cast(tf.math.multiply_no_nan(self.p_states, y_bar), dtype=tf.float32))
        one = tf.constant(1.0, dtype=tf.float32)
        scaling_factors = 0
        scale = tf.math.divide_no_nan(one, tf.reduce_sum(self.alpha))
        scaling_factors += tf.math.log(scale)
        self.alpha.assign(tf.math.multiply(self.alpha, scale))
        for t in range(1, T):
            # we assume that the chain can only transition or to the same state or to the s+1 state
            alpha_lhs_ = tf.math.multiply(
                tf.roll(self.alpha, shift=1, axis=0),
                tf.linalg.tensor_diag_part(tf.roll(self.trans_mat, shift=1, axis=0))
            )
            alpha_rhs_ = tf.math.multiply_no_nan(self.alpha, tf.linalg.tensor_diag_part(self.trans_mat))

            alpha_ = alpha_lhs_ + alpha_rhs_
            # we assume that the markov process is stationary.
            # i.e. that the state probabilities are the principal eigenvector of the transition matrix
            p_bar = tf.tensordot(p_bar, self.trans_mat, axes=1)
            y_bar = tf.math.divide_no_nan(y_pred[t, :] + 1e-12, p_bar + 1e-12)
            self.alpha.assign(tf.cast(tf.math.multiply_no_nan(y_bar, alpha_), dtype=tf.float32))
            scale = tf.math.divide_no_nan(one, tf.reduce_sum(self.alpha))
            self.alpha.assign(tf.math.multiply_no_nan(self.alpha, scale))
            scaling_factors += tf.math.log(scale)

        return scaling_factors
