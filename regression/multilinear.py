import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6]])
y = np.array([[1], [2], [3], [4]])

class Multilinear:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)
        self.n = X.shape[1]
        self.theta = np.zeros((self.n, 1))
        self.alpha = 0.01
        self.iterations = 1500
        self.cost_history = np.zeros((self.iterations, 1))

    def compute_cost(self):
        h = self.X.dot(self.theta)
        return (1 / (2 * self.m)) * np.sum(np.square(h - self.y))

    def gradient_descent(self):
        for i in range(self.iterations):
            h = self.X.dot(self.theta)
            self.theta = self.theta - (self.alpha / self.m) * (self.X.T.dot(h - self.y))
            self.cost_history[i] = self.compute_cost()
        return self.theta

    def predict(self, X):
        return X.dot(self.theta)

    def plot_cost(self):
        plt.plot(range(len(self.cost_history)), self.cost_history, 'r')
        plt.title('Cost History')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

model = Multilinear(X, y)
print(f"Gradient descent : {model.gradient_descent()}")
model.plot_cost()