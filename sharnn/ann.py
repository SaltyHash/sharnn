import numpy as np

from .cost  import Cost
from .layer import Layer

class ANN:
    '''A multi-layer neural network.'''
    
    def __init__(self, input_size, layers, cost_func):
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
        
        if not isinstance(cost_func, Cost):
            raise ValueError('"cost_func" must be a Cost instance')
        self.cost_func = cost_func
        
        # Initialize layers
        prev_layer_size = input_size
        for layer in layers:
            layer.init_params(prev_layer_size)
            prev_layer_size = layer.size
    
    def __call__(self, x):
        return self._forward(x, cache_outputs=False)
    
    def _forward(self, x, cache_outputs):
        '''If "cache_outputs" is True, a tuple of (y_predict, caches) is
        returned; otherwise, only y_predict is returned.
        '''
        caches = []
        for layer in self.layers:
            output = layer.forward(x)
            if cache_outputs: caches.append(output)
            x = output[1]
        return (x, caches) if cache_outputs else x
    
    def train(self, x, y, iters, learning_rate):
        '''Train the network.
        
        Args:
        - x    : Input array of shape (<input features>, <# examples>).
        - y    : Output array of shape (<outputs>, <# examples>).
        - iters: Number of training iterations to run.
        - learning_rate: Rate at which the gradient is descended.
        '''
        assert x.shape[0] == self.input_size, \
            'Number of inputs in "x" does not match number of network inputs'
        assert y.shape[0] == self.layers[-1].size, \
            'Number of outputs in "y" does not match number of network outputs'
        assert x.shape[1] == y.shape[1], \
            'Number of examples in "x" and "y" do not match'
        assert learning_rate > 0, '"learning_rate" must be > 0'
        
        y_predict, caches = self._forward(x, cache_outputs=True)
        pre_cost = self.cost_func(y, y_predict)
        
        # Run backprop
        for i in range(iters):
            d_activation = self.cost_func.prime(y, y_predict)
            for l in reversed(range(len(self.layers))):
                layer = self.layers[l]
                prev_activation = caches[l-1][1] if l else x
                linear          = caches[l][0]
                d_activation    = layer.backward(
                    learning_rate, prev_activation, linear, d_activation)
            y_predict, caches = self._forward(x, cache_outputs=True)
        
        # Return pre- and post-costs
        post_cost = self.cost_func(y, y_predict)
        return (pre_cost, post_cost)
