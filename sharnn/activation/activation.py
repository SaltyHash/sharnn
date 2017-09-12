import numpy as np

class Activation:
    '''Base class for representing an activation function and its derivative.
    
    Example:
    >>> a = sharnn.activation.<Activation Class>()
    >>> assert a.check_prime() < 1e-5
    >>> y = a(x)
    >>> y_prime = a.prime(x)
    '''
    
    def __call__(self, x):
        return self.function(x)
    
    def function(self, x):
        '''[Override] The actual activation function.'''
        raise NotImplementedError('"function" method needs to be overridden.')
    
    def prime(self, x):
        '''[Override] The first derivative of the activation function.'''
        raise NotImplementedError('"prime" method needs to be overridden.')
    
    def check_prime(self,
            checks=1000, epsilon=1e-7, min_value=-100, max_value=100):
        '''Checks if prime() appears to be the derivative of function().
        
        Derivative estimates are made 'checks' times using this formula,
        with x randomly generated in the range of [min_value, max_value):
            y_prime_est = (self(x+epsilon) - self(x-epsilon)) / (2*epsilon)
        
        The maximum difference between y_prime_est and self.prime(x) is returned.
        If the value is "too high", you may have a problem with your prime().
        '''
        # Choose x in range [min_value, max_value)
        x = np.random.random((1, checks))*(max_value-min_value)+min_value
        y_prime_est = (self(x+epsilon) - self(x-epsilon)) / (2*epsilon)
        return np.max(np.abs(self.prime(x)-y_prime_est))
