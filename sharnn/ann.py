import numpy as np
import os

from datetime import datetime
from time import process_time, sleep

from .cost import Cost
from .layer import Layer


class ANN:
    """A multi-layer neural network."""

    def __init__(self, input_size, layers, cost_func):
        """Initialize a new ANN.

        Args:
        - input_size: The number of input features of the network.
        - layers    : Layer instances comprising the network.
        """
        if input_size <= 0:
            raise ValueError('"input_size" must be > 0.')
        self.input_size = input_size

        if not len(layers):
            raise ValueError('"layers" must have at least one Layer.')
        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError('"layers" contains a non-Layer object.')
        self.layers = layers

        if not isinstance(cost_func, Cost):
            raise ValueError('"cost_func" must be a Cost instance')
        self.cost_func = cost_func

        # Initialize layers
        prev_layer_size = input_size
        for layer in layers:
            layer.init_params(prev_layer_size)
            prev_layer_size = layer.size

    def __call__(self, x):
        return self._forward(x, cache_outputs=False)

    def _forward(self, x, cache_outputs):
        """If "cache_outputs" is True, a tuple of (y_predict, caches) is
        returned; otherwise, only y_predict is returned.
        """
        caches = []
        for layer in self.layers:
            output = layer.forward(x)
            if cache_outputs:
                caches.append(output)
            x = output[1]
        return (x, caches) if cache_outputs else x

    def get_accuracy(self, x, y):
        """Determines classification accuracy of the model in percent."""
        assert x.shape[1] == y.shape[1]
        examples = x.shape[1]
        y_predict = self(x) >= 0.5
        # TODO: Vectorize this
        correct = 0
        for i in range(examples):
            if np.all(y_predict[:, i] == y[:, i]):
                correct += 1
        return 100.0 * correct / examples

    def train(self, x, y,
              iters=None, stop_date=None,
              learning_rate=0.01,
              learning_rate_gain=1.01,
              learning_rate_decay=0.5,
              callback=None,
              max_cpu_usage=100,
              print_every=1):
        """Train the network.

        Args:
        - x    : Input array of shape (<input features>, <# examples>).
        - y    : Output array of shape (<outputs>, <# examples>).
        - iters: Number of training iterations to run.
        - learning_rate: Rate at which the gradient is descended.
        """
        assert iters or stop_date, \
            'Either "iters" or "stop_date" must be given'
        assert x.shape[0] == self.input_size, \
            'Number of inputs in "x" does not match number of network inputs'
        assert y.shape[0] == self.layers[-1].size, \
            'Number of outputs in "y" does not match number of network outputs'
        assert x.shape[1] == y.shape[1], \
            'Number of examples in "x" and "y" do not match'
        assert learning_rate > 0, '"learning_rate" must be > 0'
        assert max_cpu_usage > 0, '"max_cpu_usage" must be > 0'
        assert print_every > 0, '"print_every" must be > 0'

        # Used for printing
        iters_len = len(str(iters - 1))
        print_prefix = ' ' * (iters_len + 2)

        # Get initial predictions, caches, and cost
        y_predict, caches = self._forward(x, cache_outputs=True)
        old_y_predict, old_caches = y_predict, caches
        pre_train_cost = self.cost_func(y, y_predict)

        cost_history = [pre_train_cost] * 4
        increased_learning_rate = False
        successes = 0
        hit_max = False

        # Run backprop until iters or stop_date is reached
        start_date = datetime.now()
        i = 0
        while (not iters or i < iters) and (not stop_date or datetime.now() < stop_date):
            # Perform single parameter update across all layers
            t0 = process_time()
            for _ in range(1):
                i += 1
                d_activation = self.cost_func.prime(y, y_predict)
                for l in reversed(range(len(self.layers))):
                    layer = self.layers[l]
                    prev_activation = caches[l - 1][1] if l else x
                    linear = caches[l][0]
                    d_activation = layer.backward(
                        learning_rate, prev_activation, linear, d_activation)
                y_predict, caches = self._forward(x, cache_outputs=True)
                cost = self.cost_func(y, y_predict)
            i -= 1

            # Determine new learning rate
            # - Cost went down?
            if cost < cost_history[0]:
                successes += 1

                # Save these in case of rollback at next iteration
                old_y_predict = y_predict
                old_caches = caches

                #new_learning_rate = learning_rate * (1 + 0.02 * (3 + successes ** 2 - 1) / successes ** 2)
                if hit_max:
                    new_learning_rate = learning_rate_gain * learning_rate
                else:
                    new_learning_rate = 2 * learning_rate

            # - Cost went UP?
            else:
                successes = 0
                hit_max = True

                # Pretend this NEVER happened...
                for layer in self.layers:
                    layer.rollback_parameters()
                y_predict, caches = old_y_predict, old_caches

                new_learning_rate = learning_rate_decay * learning_rate

            # Update cost history and derivatives
            cost_history = [cost] + cost_history[:3]
            d_cost = cost_history[0]  - cost_history[1]
            d2_cost = cost_history[0] - 2 * cost_history[1] + cost_history[2]
            d3_cost = cost_history[0] - 3 * cost_history[1] + 3 * cost_history[2] - cost_history[3]

            # Limit CPU usage
            if max_cpu_usage < 100:
                t = process_time() - t0
                sleep_time = (t * 100 / max_cpu_usage - t) / os.cpu_count()
                sleep(sleep_time)

            # Print results
            if i % print_every == 0:
                now = datetime.now()
                est_stop_date = start_date + (now - start_date) * iters / (i + 1)

                #print()
                #for i in range(1, len(self.layers)):
                #    print(f'Layer {i}:')
                #    self.layers[i].print_info()
                for layer in self.layers:
                    print(layer)

                print('\n{:0{}}: Cost={:.3e} ({:+.3e})'.format(
                    i, iters_len, cost, d_cost))
                print(print_prefix + 'learning_rate={:.3e}'.format(
                    learning_rate))
                print(print_prefix + 'new_learning_rate={:.3e} ({:+.3e})'.format(
                    new_learning_rate, new_learning_rate - learning_rate))
                print('Est. stop date: {} ({} remaining)'.format(
                    str(est_stop_date)[:-7], str(est_stop_date - now)[:-7]))

                # Execute the callback
                if callable(callback):
                    callback({
                        'cost': cost,
                        'd_cost': d_cost,
                        'd2_cost': d2_cost,
                        'iter': i,
                        'learning_rate': learning_rate,
                    })

            # Update variables
            i += 1
            learning_rate = new_learning_rate

        print('Total training time: {}'.format(datetime.now() - start_date))

        # Return pre- and post-training costs
        post_train_cost = self.cost_func(y, y_predict)
        return pre_train_cost, post_train_cost

