# Soni, Parth
# 1002_053_647
# 2023_02_26
# Assignment_01_01

import numpy as np


def mse(y_hat, y):
    return np.mean(np.square(y_hat - y))

def activationLayer(weight, input):
    net = np.dot(weight, input)
    transferValue = sigmoid(net)
    return transferValue

def feedForward(weights, X, layers):
    dt = np.dtype('float64') 
    l = X
    for j in range(0, len(layers)-1):
        ones = np.ones([1,l.shape[1]])
        l = np.insert(l,0,ones, axis = 0)
        l = activationLayer(weights[j], l)
    return l

def predict(weights, X, Y, layers):
    Y_hat = feedForward(weights, X, layers)
    # error = (Y - Y_hat)**2
    cost = mse(Y_hat, Y)
    return cost, Y_hat

def sigmoid(net):
    return (1./(1.+np.exp(-net)))

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    # This function creates and trains a multi-layer neural Network
    # X_train: Array of input for training [input_dimensions,nof_train_samples]
    # Y_train: Array of desired outputs for training samples [output_dimensions,nof_train_samples]
    # X_test: Array of input for testing [input_dimensions,nof_test_samples]
    # Y_test: Array of desired outputs for test samples [output_dimensions,nof_test_samples]
    # layers: array of integers representing number of nodes in each layer
    # alpha: learning rate  
    # epochs: number of epochs for training.
    # h: step size
    # seed: random number generator seed for initializing the weights.
    # return: This function should return a list containing 3 elements:
        # The first element of the return list should be a list of weight matrices.
        # Each element of the list corresponds to the weight matrix of the corresponding layer.

        # The second element should be a one dimensional array of numbers
        # representing the average mse error after each epoch. Each error should
        # be calculated by using the X_test array while the network is frozen.
        # This means that the weights should not be adjusted while calculating the error.

        # The third element should be a two-dimensional array [output_dimensions,nof_test_samples]
        # representing the actual output of network when X_test is used as input.

    # Notes:
    # DO NOT use any other package other than numpy
    # Bias should be included in the weight matrix in the first column.
    # Assume that the activation functions for all the layers are sigmoid.
    # Use MSE to calculate error.
    # Use gradient descent for adjusting the weights.
    # use centered difference approximation to calculate partial derivatives.
    # (f(x + h)-f(x - h))/2*h
    # Reseed the random number generator when initializing weights for each layer.
    # i.e., Initialize the weights for each layer by:

    dt = np.dtype('float64')     
    epochsCost = []
    Y_pred = []
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test= np.array(X_test)
    Y_test = np.array(Y_test)
    weights = []
    layers.insert(0, X_train.shape[0])
    print(layers)
    for i in range(1,len(layers)):
        np.random.seed(seed)
        weights.append(np.random.randn(layers[i],layers[i-1]+1))
    print(weights)
    e = 0
    if epochs > 0:
        for e in range(1,epochs+1):
            for k in range(X_train.shape[1]):
                deltaw = []
                x= X_train[:,k].reshape(-1,1)
                y = Y_train[:,k]
                for weight in weights:
                    dw = np.zeros((weight.shape[0], weight.shape[1]))
                    for i in range(weight.shape[0]):
                        for j in range(weight.shape[1]):
                            weight[i,j] = weight[i,j] + h
                            y_hat = feedForward(weights, x, layers)
                            cost1 = mse(y_hat, y)
                            weight[i,j] = weight[i,j] - 2*h
                            y_hat = feedForward(weights, x, layers)
                            cost2 = mse(y_hat, y)
                            dw[i,j] = alpha*(cost1-cost2)/(2*h)
                            weight[i,j] = weight[i,j] + h
                    deltaw.append(dw)
                for weight, dw in zip(weights, deltaw):
                    for i in range(weight.shape[0]):
                        for j in range(weight.shape[1]):
                            weight[i,j] = weight[i,j] - dw[i,j]
            c, Y_pred = predict(weights, X_test, Y_test, layers)
            print(Y_pred.shape)
            epochsCost.append(c)
            print(f"Cost after {e} epoch ", predict(weights, X_test, Y_test, layers)[0])
    else:
        c, Y_pred = predict(weights, X_test, Y_test, layers)
        print(Y_pred.shape)
        epochsCost.append(c)

    return [weights, epochsCost, Y_pred]

def create_toy_data_nonlinear_2d(n_samples=1000):
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.linspace(-1, 1, n_samples)
    X[:, 1] = np.linspace(-1, 1, n_samples)
    X[:, 2] = np.linspace(-1, 1, n_samples)
    X[:, 3] = np.linspace(-1, 1, n_samples)

    y = np.zeros((n_samples, 2))
    y[:, 0] = 0.5*X[:, 0] -0.2 * X[:, 1]**2 - 0.2*X[:, 2] + X[:, 3]*X[:,1] - 0.1
    y[:, 1] = 1.5 * X[:, 0] + 1.25 * X[:, 1]*X[:, 0] + 0.4 * X[:, 2] * X[:, 0]

    # shuffle X and y
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X.T, y.T



def test_can_fit_data_test_2d():
    np.random.seed(1234)
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.35,epochs=10,h=1e-8,seed=1234)
    print(err)
    assert err[1] < err[0]
    assert err[2] < err[1]
    assert err[3] < err[2]
    assert err[9] < 0.005
    assert abs(err[9] - 0.0022172173471326816) < 1e-5

# test_can_fit_data_test_2d()




                

                    

 

        
    


                 


