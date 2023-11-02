import numpy as np
from collections import Counter


class KNearestNeighbour:

    def __init__(self, k=3):
        self.k = k
        self.Y_train = None
        self.X_train = None


    def calculateDistance(self, x, y):
        dist = np.sqrt(np.sum((x-y)**2))
        return dist


    def fit(self, x, y):
        self.X_train = x
        self.Y_train = y


    def predict(self, X):
        # Predict for all data points
        pred = [self.computeForEachDataPoint(x) for x in X]
        return pred


    def computeForEachDataPoint(self, dataPoints):
        # Calculate Distance
        distance = [self.calculateDistance(dataPoints, x_train) for x_train in self.X_train]
        # Get Closest K
        k_index = np.argsort(distance)[:self.k]
        k_label = [self.Y_train[i] for i in k_index]
        counter = Counter(k_label).most_common()
        return counter[0][0]


    @staticmethod
    def accuracyScore(preds, testData):
        return np.sum(preds == testData) / len(testData)
