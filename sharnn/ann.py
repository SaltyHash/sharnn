import numpy as np

class ANN:
    '''Represents an L-layer neural network.'''
    
    def __init__(self, layer_dims):
        '''Initialize a new ANN.
        
        Args:
        - layer_dims: A list of layer dimensions of the form
                      [<inputs>, ..., <outputs>].
        '''
        if (len(layer_dims) < 2):
            raise ValueError('layer_dims must have at least input and output layers.')
        self.layer_dims = layer_dims
        
        # Build layers
        self.layers = []
