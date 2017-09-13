import numpy as np

from .layer import Layer

class ANN:
    '''A multi-layer neural network.'''
    
    def __init__(self, input_size, layers):
        '''Initialize a new ANN.
        
        Args:
        - input_size: The number of input features of the network.
        - layers    : Layer instances comprising the network.
        '''
        if (input_size <= 0):
            raise ValueError('"input_size" must be > 0.')
        self.input_size = input_size
        
        if not len(layers):
            raise ValueError('"layers" must have at least one Layer.')
        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError('"layers" contains a non-Layer object.')
        self.layers = layers
        
        # Initialize layers
        prev_layer_size = input_size
        for layer in layers:
            layer.init_params(prev_layer_size)
            prev_layer_size = layer.size
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer.forward(x, cache=False)
        return x
