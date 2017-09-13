import numpy as np
import sharnn

if 1:
    ann = sharnn.ANN(
        input_size=3,
        layers=[
            sharnn.layer.Layer(3, sharnn.activation.relu),
            sharnn.layer.Layer(2, sharnn.activation.relu),
            sharnn.layer.Layer(1, sharnn.activation.tanh),
        ]
    )
    
    x = np.array((
        (0, 1),
        (0, 2),
        (0, 3)
    ))
    print('x =')
    print(x)
    print('\nann(x) =')
    print(ann(x))
