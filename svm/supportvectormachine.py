import numpy as np

class SupportVectorMachine:

    def __init__(self, learningRate=0.001, lambd=0.01, numIterations=1000):
        self.bias = None
        self.weights = None
        self.learningRate = learningRate
        self.lambd = lambd
        self.numIterations = numIterations


    def fit(self, X, y):
        num_samples, num_features = X.shape
        y = np.where(y <= 0, -1, 1)

        # Initialise weights
        self.weights = np.zeros(num_features)
        self.bias = 0

        for x in range(self.numIterations):
            for i, xi in enumerate(X):
                cond = y[i] * (np.dot(xi, self.weights) - self.bias) >= 1
                if cond:
                    self.weights -= (2 * self.learningRate * self.weights)
                else:
                    self.weights -= self.learningRate * (2 * self.lambd * self.weights - np.dot(xi, y[i]))
                    self.bias -= self.learningRate * y[i]


    def predict(self, X):
        pred = np.dot(X, self.weights) - self.bias
        return np.sign(pred)
