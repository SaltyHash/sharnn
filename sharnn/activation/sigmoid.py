import numpy as np

from .activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        Activation.__init__(self)
    
    def call(self, arg):
        return 1/(1+np.exp(-arg))
    
    def call_prime(self, arg):
        arg = self.call(arg)
        return arg*(1-arg)
