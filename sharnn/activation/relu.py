import numpy as np

from .activation import Activation

class ReLU(Activation):
    def __init__(self):
        Activation.__init__(self)
    
    def call(self, arg):
        return np.max((0.0, arg))
    
    def call_prime(self, arg):
        return (arg > 0)
