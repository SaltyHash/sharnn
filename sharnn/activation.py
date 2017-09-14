'''Contains several common activation functions.'''

import numpy as np

class Activation:
    '''Base class for representing an activation and its first derivative.'''
    
    def __call__(self, x):
        return self.function(x)
    
    def function(self, x):
        '''The activation function.'''
        raise NotImplementedError('"function" method needs to be overridden.')
    
    def prime(self, x):
        '''The first derivative of function() w.r.t. "x".'''
        raise NotImplementedError('"prime" method needs to be overridden.')
    
    def check_prime(self,
            checks=1000, epsilon=1e-7, min_value=-100, max_value=100):
        '''Checks if prime() appears to be the derivative of function().
        
        Derivative estimates are made "checks" times using this formula,
        with x randomly generated in the range of [min_value, max_value):
            y_prime_est = (self(x+epsilon) - self(x-epsilon)) / (2*epsilon)
        
        The maximum difference between y_prime_est and self.prime(x) is returned.
        If the value is "too high", you may have a problem with your prime().
        '''
        # Choose x in range [min_value, max_value)
        x = np.random.random((1, checks))*(max_value-min_value)+min_value
        y_prime_est = (self(x+epsilon) - self(x-epsilon)) / (2*epsilon)
        return np.max(np.abs(self.prime(x)-y_prime_est))

class Identity(Activation):
    '''Identity'''
    
    def function(self, x):
        return x
    
    def prime(self, x):
        return 1.0
identity = Identity()

class LeakyReLU(Activation):
    '''Leaky Rectified Linear Unit'''
    
    def __init__(self, slope):
        '''slope must be in range [0, 1).'''
        Activation.__init__(self)
        if (slope < 0) or (slope >= 1):
            raise ValueError('slope must be in range [0, 1).')
        self.slope = slope
    
    def function(self, x):
        return np.maximum(self.slope*x, x)
    
    def prime(self, x):
        # Assume x is an instance of np.ndarray
        try:
            prime = (x > 0).astype('float')
            if self.slope: prime[prime == 0.0] = self.slope
        # Otherwise, treat as single
        except AttributeError:
            prime = 1.0 if (x > 0) else self.slope
        return prime

class ReLU(LeakyReLU):
    '''Rectified Linear Unit'''
    
    def __init__(self):
        ''''''
        LeakyReLU.__init__(self, 0.0)
relu = ReLU()

class Sigmoid(Activation):
    '''Sigmoid'''
    
    def function(self, x):
        return 1/(1+np.exp(-x))
    
    def prime(self, x):
        x = self(x)
        return x*(1-x)
sigmoid = Sigmoid()

class Tanh(Activation):
    '''Hyperbolic Tangent'''
    
    def function(self, x):
        return np.tanh(x)
    
    def prime(self, x):
        return 1 - np.power(np.tanh(x), 2)
tanh = Tanh()
