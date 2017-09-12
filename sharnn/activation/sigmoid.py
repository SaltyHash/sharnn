import numpy as np

from .activation import Activation

class Sigmoid(Activation):
    '''Sigmoid (aka softmax)'''
    
    def function(self, x):
        return 1/(1+np.exp(-x))
    
    def prime(self, x):
        x = self(x)
        return x*(1-x)
