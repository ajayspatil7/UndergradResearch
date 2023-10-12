import numpy as np

class Logistic():


    def __init__(self, learningRate=0.001, itterations=1000):
        self.learningRate = learningRate
        self.itterations = itterations
        self.bias = None
        self.weights = None


    def fit(self, X, y):
        # Initialize weights and bias as Zero
        numSamples, numFeatures = X.shape
        self.weights = np.zeros(numFeatures)
        self.bias = 0

        for _ in range(self.itterations):
            lin_pred = np.dot(X, self.weights) + self.bias
            log_pred = self.sigmoid(lin_pred)

            # Calculating the gradients
            # 1/N SUM(2xi * (y_pred - yi))
            # Taking the transpose of the matrix to match the dimensions
            dw = (1 / numSamples) * np.dot(X.T, (log_pred - y))

            # 1/N SUM(2 * (y_pred - yi))
            db = (1 / numSamples) * np.sum(log_pred - y)

            self.weights = self.weights - self.learningRate * dw
            self.bias = self.bias - self.learningRate * db


    def predict(self, X):
        lin_pred = np.dot(X, self.weights) + self.bias
        log_pred = self.sigmoid(lin_pred)
        prediction = [1 if i > 0.5 else 0 for i in log_pred]
        return prediction



    def sigmoid(self, x):
        #  s(x) = 1 / 1 + e^(-x)
        return 1 / (1 + np.exp(-x))

