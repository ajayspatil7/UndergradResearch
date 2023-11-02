import numpy as np

"""
This module provides functions for linear regression.
"""


class SupportVectorMachine:

    """
    A class for performing linear regression.
    """

    # Create a docstring
    """
    Support Vector Machine classifier.

    Parameters:
        -----------
        learningRate: float
            The step length that will be taken when following the negative gradient during training.
        lambd: float
            Regularization parameter for the l2 penalty.
        numIterations: int
            The number of iterations that the classifier will train over the dataset.
    """

    def __init__(self, learningRate=0.001, lambd=0.01, numIterations=1000):
        self.bias = None
        self.weights = None
        self.learningRate = learningRate
        self.lambd = lambd
        self.numIterations = numIterations


    def fit(self, x, y):
        num_samples, num_features = x.shape
        y = np.where(y <= 0, -1, 1)

        # Initialise weights
        self.weights = np.zeros(num_features)
        self.bias = 0

        for x in range(self.numIterations):
            for i, xi in enumerate(x):
                cond = y[i] * (np.dot(xi, self.weights) - self.bias) >= 1
                if cond:
                    self.weights -= (2 * self.learningRate * self.weights)
                else:
                    self.weights -= self.learningRate * (2 * self.lambd * self.weights - np.dot(xi, y[i]))
                    self.bias -= self.learningRate * y[i]


    def predict(self, X):
        pred = np.dot(X, self.weights) - self.bias
        return np.sign(pred)
