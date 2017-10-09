"""sharnn - Salty Hash Artificial Neural Network"""

import numpy as np

from . import activation
from . import cost
from .ann import ANN
from .layer import Layer

__author__  = 'Austin Bowen <austin.bowen.314@gmail.com>'
__version__ = '0.1'
__all__     = ['activation', 'cost', 'ANN', 'Layer']

np.random.seed(0)
