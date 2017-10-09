import numpy as np
import os
import pickle
import sharnn

from time import sleep

if 0:
    iters         = 1000
    learning_rate = 1
    x = np.array((
        (0, 1, 0, 1, 0, 1, 0, 1),
        (0, 0, 1, 1, 0, 0, 1, 1),
        (0, 0, 0, 0, 1, 1, 1, 1)
    ))
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    y = np.array((
        (0, 1, 1, 0, 1, 0, 0, 1),
    ))
    print('x =\n{}\ny =\n{}'.format(x, y))
    
    ann = sharnn.ANN(
        input_size=x.shape[0],
        layers=(
            sharnn.layer.Layer(3, sharnn.activation.tanh),
            sharnn.layer.Layer(y.shape[0], sharnn.activation.sigmoid),
        ),
        cost_func=sharnn.cost.cross_entropy
    )
    
    print()
    print('ann(x) = {}'.format(ann(x)))
    print()
    output = ann.train(x, y,
        iters=iters,
        learning_rate=learning_rate,
        learning_rate_gain=1.5,
        learning_rate_decay=0.5
    )
    print('ann.train(...) -> {}'.format(output))
    print()
    y_predict = ann(x)
    print('y_predict =\n{}'.format(y_predict))
    y_predict = (y_predict >= 0.5)
    print('(y_predict >= 0.5) =\n{}'.format(y_predict))
    print()
    print('Accuracy: {}%'.format(100.0*np.sum(y_predict == y)/y.shape[1]))
    print()

# Handwritten digit classifier using MNIST dataset
if 1:
    from datetime import datetime, timedelta
    import idx2numpy
    import matplotlib.pyplot as plt
    
    
    ## Get MNIST data
    
    # Get MNIST training and test sets as numpy arrays
    base_dir = '/home/austin/Downloads/mnist'
    mnist_train_data   = idx2numpy.convert_from_file(
            base_dir+'/train-images-idx3-ubyte/data')
    mnist_train_labels = idx2numpy.convert_from_file(
            base_dir+'/train-labels-idx1-ubyte/data')
    mnist_test_data    = idx2numpy.convert_from_file(
            base_dir+'/t10k-images-idx3-ubyte/data')
    mnist_test_labels  = idx2numpy.convert_from_file(
            base_dir+'/t10k-labels-idx1-ubyte/data')
    
    # Reshape the data from (m, x, y) to (x*y, m) with values in range [0, 1]
    d = mnist_train_data
    _, x, y = d.shape
    d = np.reshape(d, (-1, x*y)).T / 255
    mnist_train_data = d
    
    d = mnist_test_data
    _, x, y = d.shape
    d = np.reshape(d, (-1, x*y)).T / 255
    mnist_test_data = d
    
    # Reshape the labels from (m,) to one-hot (10, m)
    mnist_train_labels = np.eye(10)[mnist_train_labels].T
    mnist_test_labels  = np.eye(10)[mnist_test_labels].T
    
    # Faux mini batching
    #minibatch_size = 2**14
    minibatch_size = mnist_train_data.shape[1]
    print('Minibatch size : {}'.format(minibatch_size))
    mnist_train_data   = mnist_train_data[:, :minibatch_size]
    mnist_train_labels = mnist_train_labels[:, :minibatch_size]
    
    input_features,  train_size = mnist_train_data.shape
    output_features, test_size  = mnist_test_labels.shape
    print('input_features : {}'.format(input_features))
    print('output_features: {}'.format(output_features))
    print('train_size     : {}'.format(train_size))
    print('test_size      : {}'.format(test_size))
    print()
    
    
    ## Setup plotting stuff
    
    plt.ion()
    figure = plt.figure()
    
    # Cost plot
    cost_history = []
    cost_subplot = figure.add_subplot(411)
    cost_subplot.grid()
    cost_subplot.set_ylabel('Cost')
    cost_line,   = cost_subplot.plot([], [], 'b-')
    
    # Plot 1st derivative of cost
    d_cost_history = []
    d_cost_subplot = figure.add_subplot(412)
    d_cost_subplot.grid()
    d_cost_subplot.set_ylabel("Cost'")
    d_cost_line,   = d_cost_subplot.plot([], [], 'g-')
    
    # Plot 2nd derivative of cost
    d2_cost_history = []
    d2_cost_subplot = figure.add_subplot(413)
    d2_cost_subplot.grid()
    d2_cost_subplot.set_ylabel("Cost''")
    d2_cost_line,   = d2_cost_subplot.plot([], [], 'g-')
    
    # Learning rate plot
    learning_rate_history = []
    learning_rate_subplot = figure.add_subplot(414)
    learning_rate_subplot.grid()
    learning_rate_subplot.set_xlabel('Training Iterations')
    learning_rate_subplot.set_ylabel('Learning Rate')
    learning_rate_line,   = learning_rate_subplot.plot([], [], 'k-')
    
    def training_callback(args):
        cost          = args['cost']
        d_cost        = args['d_cost']
        d2_cost       = args['d2_cost']
        i             = args['iter']
        learning_rate = args['learning_rate']
        
        x = list(range(i+1))
        
        # Update cost plot
        cost_history.append(cost)
        min_height = min(cost_history); min_height -= 0.1*abs(min_height)
        max_height = max(cost_history); max_height += 0.1*abs(max_height)
        cost_subplot.axis([0, max(i, 1), min_height, max_height])
        cost_line.set_data(x, cost_history)
        
        # Update d_cost plot
        d_cost_history.append(d_cost)
        min_height = min(d_cost_history); min_height -= 0.1*abs(min_height)
        max_height = max(d_cost_history); max_height += 0.1*abs(max_height)
        d_cost_subplot.axis([0, max(i, 1), min_height, max_height])
        d_cost_line.set_data(x, d_cost_history)
        
        # Update d2_cost plot
        d2_cost_history.append(d2_cost)
        min_height = min(d2_cost_history); min_height -= 0.1*abs(min_height)
        max_height = max(d2_cost_history); max_height += 0.1*abs(max_height)
        d2_cost_subplot.axis([0, max(i, 1), min_height, max_height])
        d2_cost_line.set_data(x, d2_cost_history)
        
        # Update learning_rate plot
        learning_rate_history.append(learning_rate)
        min_height  = min(learning_rate_history)
        min_height -= 0.1*abs(min_height)
        max_height  = max(learning_rate_history)
        max_height += 0.1*abs(max_height)
        learning_rate_subplot.axis([0, max(i, 1), min_height, max_height])
        learning_rate_line.set_data(x, learning_rate_history)
        
        figure.canvas.draw()
    
    
    ## Build model
    
    digit_classifier_file = './models/digit_classifier.pickle'
    
    # Load classifier?
    if input('Load classifier from file (Y/n)? ').lower() in {'', 'y'}:
        with open(digit_classifier_file, 'rb') as f:
            digit_classifier = pickle.load(f)
    
    # Create fresh classifier?
    else:
        digit_classifier = sharnn.ANN(
            input_size=input_features,
            layers=(
                sharnn.layer.Layer(100, sharnn.activation.relu),
                sharnn.layer.Layer(output_features, sharnn.activation.sigmoid),
            ),
            cost_func=sharnn.cost.cross_entropy
        )
    
    iters = int(input('\nEnter training iterations: '))
    
    
    ## Train model
    
    output = digit_classifier.train(
        mnist_train_data,
        mnist_train_labels,
        iters=iters,
        #stop_date=datetime.now() + timedelta(0, 15, 0),
        learning_rate=0.4,#1.2,
        learning_rate_gain=1.02,
        learning_rate_decay=0.5,
        callback=training_callback,
        max_cpu_usage=100
    )
    print('digit_classifier.train(...) -> {}'.format(output))
    
    
    ## Print the results
    
    # Get training and testing accuracy
    train_accuracy = digit_classifier.get_accuracy(
            mnist_train_data, mnist_train_labels)
    test_accuracy  = digit_classifier.get_accuracy(
            mnist_test_data, mnist_test_labels)
    print()
    print('Training accuracy: {}%'.format(train_accuracy))
    print('Testing  accuracy: {}%'.format(test_accuracy))
    
    # Offer to save the model
    if input('Save classifier (Y/n)? ').lower() in {'', 'y'}:
        print('Saving classifier to "{}"...'.format(digit_classifier_file))
        with open(digit_classifier_file, 'wb') as f:
            pickle.dump(digit_classifier, f)

