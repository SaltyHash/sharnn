import numpy as np
import sharnn

# Activation Test
if 0:
    class TestActivation(sharnn.activation.Activation):
        def function(self, x):
            return 1/(1+np.exp(-x))
        
        def prime(self, x):
            y = self(x)
            return y*(1-y)

    a = TestActivation()
    x = np.array((1, -2, 4, -5))
    print('x          = {}'.format(x))
    print('a(x)       = {}'.format(a(x)))
    print('a.prime(x) = {}'.format(a.prime(x)))
    print()
    diff = a.check_prime(checks=3)
    print('a.check_prime() = {}'.format(diff))

# ReLU Test
if 0:
    x_vals = [3, 0, -2, np.array((1, 0, -2))]
    
    relu = sharnn.activation.ReLU()
    for x in x_vals:
        print('relu({}) = {}'.format(x, relu(x)))
        print('relu.prime({}) = {}'.format(x, relu.prime(x)))
        print()
    
    print('relu.check_prime() = {}'.format(relu.check_prime()))

# Sigmoid Test
if 0:
    x_vals = [0, 0.5, 1, np.array((10, -10, 3.14, -42))]
    
    sigmoid = sharnn.activation.Sigmoid()
    for x in x_vals:
        print('sigmoid({}) = {}'.format(x, sigmoid(x)))
        print('sigmoid.prime({}) = {}'.format(x, sigmoid.prime(x)))
        print()
    
    print('sigmoid.check_prime() = {}'.format(sigmoid.check_prime()))


