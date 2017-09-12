import numpy as np

from .activation import Activation

class ReLU(Activation):
    '''Rectified Linear Unit'''
    
    def function(self, x):
        return np.maximum(0, x)
    
    def prime(self, x):
        # Assume x is an instance of np.ndarray; otherwise, treat as single
        try:
            return (x > 0).astype('float')
        except AttributeError:
            return float(x > 0)
