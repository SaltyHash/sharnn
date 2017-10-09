import numpy as np

from .activation import Activation


class Layer:
    """Represents a single layer in an ANN."""

    def __init__(self, size, activation, dropout_prob=0.0):
        """
        Args:
        - size        : The number of nodes in the layer.
        - activation  : An Activation instance.
        - dropout_prob: The probability of each node being dropped during training.
        """
        if size < 1:
            raise ValueError('"size" must be >= 1')
        self.size = size

        if not isinstance(activation, Activation):
            raise ValueError('"activation" must be an Activation instance')
        self.activation = activation

        if (dropout_prob < 0) or (dropout_prob > 1):
            raise ValueError('"dropout_prob" must be in range [0, 1]')
        self.dropout_prob = dropout_prob

        self.prev_weights = self.weights = None
        self.prev_biases = self.biases = None

    def backward(self, learning_rate, prev_activation, linear, d_activation):
        # Determine d_W, d_b, and prev_d_activation
        m = prev_activation.shape[1]
        d_linear = d_activation * self.activation.prime(linear)
        d_weights = (1 / m) * d_linear.dot(prev_activation.T)
        assert d_weights.shape == self.weights.shape, \
            'weights.shape = {} / d_weights.shape = {}'.format(self.weights.shape, d_weights.shape)
        d_b = (1 / m) * np.sum(d_linear, axis=1, keepdims=True)
        assert d_b.shape == self.biases.shape
        prev_d_activation = self.weights.T.dot(d_linear)

        # Update parameters and return prev_d_activation
        self.prev_weights = np.copy(self.weights)
        self.prev_biases = np.copy(self.biases)
        self.weights = self.weights - learning_rate * d_weights
        self.biases = self.biases - learning_rate * d_b
        return prev_d_activation

    def forward(self, x):
        """Run the input "x" through the layer, where "x" must be of shape
        (prev_layer_size, examples).  Returns tuple (linear, activation).
        """
        linear = self.weights.dot(x) + self.biases
        return linear, self.activation(linear)

    def init_params(self, prev_layer_size):
        # He initialization scalar
        he_scalar = np.sqrt(2 / prev_layer_size)
        self.weights = np.random.normal(
            size=(self.size, prev_layer_size),
            scale=he_scalar)
        self.biases = np.zeros((self.size, 1))

    def rollback_parameters(self):
        """Returns the parameters to their previous state.  Useful in the
        event that the cost is higher after a training iteration.  Can only
        be called once after a backwards pass.
        """
        self.weights = self.prev_weights
        self.biases = self.prev_biases
        self.prev_weights = None
        self.prev_biases = None
