import numpy as np
import sharnn

if 1:
    iters         = 1000
    learning_rate = 1
    x = np.array((
        (0, 1, 0, 1, 0.1),
        (0, 0, 1, 1, 0.7),
    ))
    y = np.array((
        (0, 1, 0, 1, 0),
    ))
    print('x =\n{}\ny =\n{}'.format(x, y))
    
    ann = sharnn.ANN(
        input_size=x.shape[0],
        layers=(
            sharnn.layer.Layer(2, sharnn.activation.relu),
            sharnn.layer.Layer(1, sharnn.activation.sigmoid),
        ),
        cost_func=sharnn.cost.cross_entropy
    )
    
    print()
    print('ann(x) = {}'.format(ann(x)))
    print()
    print('ann.train(...) -> {}'.format(ann.train(x, y, iters, learning_rate)))
    print()
    y_predict = ann(x)
    print('y_predict =\n{}'.format(y_predict))
    print('(y_predict >= 0.5) =\n{}'.format(y_predict >= 0.5))
