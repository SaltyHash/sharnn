'''sharnn - Salty Hash Artificial Neural Network'''

__author__  = 'Austin Bowen <austin.bowen.314@gmail.com>'
__version__ = '0.1'
__all__     = ['activation', 'cost', 'ANN', 'Layer']

import numpy as np
np.random.seed(0)

from .      import activation
from .      import cost
from .ann   import ANN
from .layer import Layer
