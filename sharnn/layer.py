import numpy as np

from .activation import Activation

class Layer:
    '''Represents a single layer in an ANN.'''
    
    def __init__(self, size, activation, dropout_prob=0.0):
        '''
        Args:
        - size        : The number of nodes in the layer.
        - activation  : An Activation instance.
        - dropout_prob: The probability of each node being dropped during training.
        '''
        if (size < 1):
            raise ValueError('"size" must be >= 1')
        self.size = size
        
        if not isinstance(activation, Activation):
            raise ValueError('"activation" must be an Activation instance')
        self.activation = activation
        
        if (dropout_prob < 0) or (dropout_prob > 1):
            raise ValueError('"dropout_prob" must be in range [0, 1]')
        self.dropout_prob = dropout_prob
    
    def backward(self, learning_rate, prev_activation, linear, d_activation):
        # Determine d_W, d_b, and prev_d_activation
        m = prev_activation.shape[1]
        d_linear = d_activation * self.activation.prime(linear)
        d_W      = (1/m)*d_linear.dot(prev_activation.T)
        assert d_W.shape == self.W.shape,\
            'W.shape = {} / d_W.shape = {}'.format(self.W.shape, d_W.shape)
        d_b      = (1/m)*np.sum(d_linear, axis=1, keepdims=True)
        assert d_b.shape == self.b.shape
        prev_d_activation = self.W.T.dot(d_linear)
        
        # Update parameters and return prev_d_activation
        self.W = self.W - learning_rate*d_W
        self.b = self.b - learning_rate*d_b
        return prev_d_activation
    
    def forward(self, x):
        '''Run the input "x" through the layer, where "x" must be of shape
        (prev_layer_size, examples).  Returns tuple (linear, activation).
        '''
        linear = self.W.dot(x) + self.b
        return (linear, self.activation(linear))
    
    def init_params(self, prev_layer_size, scale=0.01):
        self.W = np.random.normal(
            size=(self.size, prev_layer_size),
            scale=scale)
        self.b = np.zeros((self.size, 1))
