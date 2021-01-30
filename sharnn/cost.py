"""Contains common cost functions."""

import numpy as np


class Cost:
    """Base class representing a cost function."""

    def __call__(self, y, y_predict):
        return self.function(y, y_predict)

    def function(self, y, y_predict):
        """The cost function."""
        raise NotImplementedError()

    def prime(self, y, y_predict):
        """The derivative of function() w.r.t. "y_predict"."""
        raise NotImplementedError()


class CrossEntropy(Cost):
    def function(self, y, y_predict):
        assert np.all(y_predict > 0), np.min(y_predict)
        assert np.all(y_predict < 1), np.max(y_predict)
        try:
            m = y.shape[1]
        except AttributeError:
            m = 1
        cost = (-1 / m) * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
        return np.squeeze(cost)

    def prime(self, y, y_predict):
        return (1 - y) / (1 - y_predict) - y / y_predict

cross_entropy = CrossEntropy()


class MeanSquareError(Cost):
    def function(self, y, y_predict):
        return 0.5 * np.sum(np.power(y - y_predict, 2))

    def prime(self, y, y_predict):
        return (y - y_predict) * -y_predict

mse = MeanSquareError()

