class Activation:
    '''Represents an activation function and its derivative.'''
    
    def __init__(self, call=None, call_prime=None):
        if callable(call)      : self.call       = call
        if callable(call_prime): self.call_prime = call_prime
    
    def call(self, arg):
        raise NotImplementedError()
    
    def call_prime(self, arg):
        raise NotImplementedError()
    
    def check(self):
        '''Checks if call_prime appears to be the derivative of call.'''
        raise NotImplementedError()
