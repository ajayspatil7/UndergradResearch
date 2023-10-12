import statistics


class LinearRegression:

    def __init__(self):
        self.m = None
        self.b = None
        self.rmse = None
        self.r_squared = None
        self.mse = None
        self.mae = None

    def fit(self, X, Y):
        """
        Fit linear regression model to data
        """
        if len(X) != len(Y):
            raise ValueError("Input arrays must have the same length")

        x_mean = statistics.mean(X)
        y_mean = statistics.mean(Y)
        x_squared_mean = statistics.mean([x ** 2 for x in X])
        xy_mean = statistics.mean([x * y for x, y in zip(X, Y)])

        self.m = (x_mean * y_mean - xy_mean) / (x_mean ** 2 - x_squared_mean)
        self.b = y_mean - self.m * x_mean

        # Calculate R-squared
        y_predicted = [self.m * x + self.b for x in X]
        ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(Y, y_predicted))
        ss_tot = sum((y - y_mean) ** 2 for y in Y)
        self.r_squared = 1 - (ss_res / ss_tot)

        # Calculate RMSE
        self.rmse = sum((y - y_pred) ** 2 for y, y_pred in zip(Y, y_predicted)) / len(Y)

        # Calculate MSE
        self.mse = self.rmse**2

        # Calculate MAE
        self.mae = sum(abs(y - y_pred) for y, y_pred in zip(Y, y_predicted)) / len(Y)

        return self

    def predict(self, X):
        """
        Make predictions on data using fitted model
        """
        y_predicted = [self.m * x + self.b for x in X]
        return y_predicted

    def R_squared(self):
        """
        Return R-squared score of model fit
        """
        return self.r_squared

    def RMSE(self):
        """
        Return root mean squared error of model fit
        """
        return self.rmse

    def MSE(self):
        """
        Return mean squared error of model fit
        """
        self.mse = self.rmse**2
        return self.rmse**2

    def MAE(self):
        """
        Return mean absolute error of model fit
        """
        return self.mae

if __name__ == "__main__":
    # Sample data
    X = [x for x in range(0, 10)]
    Y = [2 * x + 1 for x in range(0, 10)]

    # Fit model
    model = LinearRegression().fit(X, Y)

    # Make predictions
    predictions = model.predict([10, 11, 12, 13])

    # Score model
    r_squared = model.R_squared()
    rmse = model.RMSE()
    mse = model.MSE()
    mae = model.mae

    print('-' * 40)
    print("X:", X)
    print("Y:", Y)
    print("Predictions:", predictions)
    print('-'*40)
    print("R-squared:", r_squared)
    print("RMSE:", rmse)
    print("MSE:", mse)
    print("MAE:", mae)
