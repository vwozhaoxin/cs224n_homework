#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    #print(Dx,H,Dy)
    #print(params.shape)
    #print(type(params))
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    #print(labels.shape)
    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    Z1 = X.dot(W1)+b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2)+b2
    proba = softmax(Z2)
    y = np.argmax(labels, axis=1)
    N=X.shape[0]
    correct_log_loss = np.log(proba[np.arange(N), y]) 
    cost = -np.sum(correct_log_loss)/N
    #predict = np.zeros(labels.shape)
    #predict[range(predict.shape[0]),np.argmax(proba,axis=1)]=1
    #cost = 0.5*np.sum(np.sum(np.square(predict-labels)))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    gradZ2 = proba.copy()
    gradZ2[np.arange(N),y]-=1
    gradZ2 /=N
    gradW2 = A1.T.dot(gradZ2)
    gradb2 = np.sum(gradZ2,axis=0)
    gradA1 = gradZ2.dot(W2.T)
    gradZ1 = gradA1*sigmoid_grad(A1)
    gradW1 = X.T.dot(gradZ1)
    gradb1 = np.sum(gradZ1,axis=0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1
    print('ok')
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2])

    #print(params.shape)
    #print(type(params))
    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    pass
    #print "Running your sanity checks..."
    ### YOUR CODE HERE

    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
