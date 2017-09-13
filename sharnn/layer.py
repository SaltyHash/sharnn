import numpy as np

from .activation import Activation

class Layer:
    '''Represents a single layer in an ANN.'''
    
    class Cache:
        def __init__(self):
            self.clear()
        
        def clear(self):
            self.linear_forward     = None
            self.activation_forward = None
    
    class Parameters:
        W = None
        b = None
    
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
        
        self.cache = Layer.Cache()
    
    def forward(self, x, cache=True):
        '''Run the input "x" through the layer and return the output.
        "x" must be of shape (prev_layer_size, examples).
        '''
        lf = self.params.W.dot(x) + self.params.b
        af = self.activation(lf)
        self.cache.linear_forward     = lf if cache else None
        self.cache.activation_forward = af if cache else None
        return af
    
    def init_params(self, prev_layer_size, scale=0.01):
        self.params = Layer.Parameters()
        self.params.W = np.random.normal(
            size=(self.size, prev_layer_size),
            scale=scale)
        self.params.b = np.zeros((self.size, 1))
        return self.params
