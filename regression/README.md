# LinearRegression

This class implements a simple linear regression model, which fits a linear relationship between a dependent variable (Y) and an independent variable (X).

## Attributes

- `m`: Slope of the linear regression line.
- `b`: Intercept of the linear regression line.
- `rmse`: Root Mean Squared Error, a measure of the model's predictive accuracy.
- `r_squared`: Coefficient of determination (R-squared), indicating the proportion of the variance in the dependent variable explained by the independent variable.
- `mse`: Mean Squared Error, an alternate measure of the model's predictive accuracy.
- `mae`: Mean Absolute Error, a measure of the average magnitude of errors between actual and predicted values.

### Implementation
```python
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
```


## Methods

### `fit(X, Y)`

Fits the linear regression model to the provided data.

- **Parameters:**
  - `X`: List or array of independent variable values.
  - `Y`: List or array of dependent variable values.
- **Returns:**
  - Returns the instance of the LinearRegression class with updated attributes.

### `predict(X)`

Makes predictions using the fitted linear regression model.

- **Parameters:**
  - `X`: List or array of independent variable values for which predictions are to be made.
- **Returns:**
  - List of predicted dependent variable values corresponding to the provided independent variable values.

### `R_squared()`

Calculates and returns the R-squared score of the model fit.

- **Returns:**
  - R-squared score, a value between 0 and 1 indicating the goodness of fit of the model.

### `RMSE()`

Calculates and returns the Root Mean Squared Error of the model fit.

- **Returns:**
  - RMSE, a measure of the average magnitude of errors between actual and predicted values.

### `MSE()`

Calculates and returns the Mean Squared Error of the model fit.

- **Returns:**
  - MSE, a measure of the average squared errors between actual and predicted values.

### `MAE()`

Calculates and returns the Mean Absolute Error of the model fit.

- **Returns:**
  - MAE, a measure of the average magnitude of errors between actual and predicted values.

## Example Usage

```python
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
mae = model.MAE()

# Print results
print("Predictions:", predictions)
print("R-squared:", r_squared)
print("RMSE:", rmse)
print("MSE:", mse)
print("MAE:", mae)
```

# Multilinear regression

This class implements a multilinear regression model, which fits a linear relationship between a dependent variable (Y) and multiple independent variables (X).
## Attributes

- `theta`: Vector of coefficients for the linear regression model.
- `alpha`: Learning rate for gradient descent.
- `iterations`: Number of iterations for gradient descent.
- `cost_history`: Vector of cost values for each iteration of gradient descent.
- `m`: Number of training examples.
- `n`: Number of features.
- `X`: Matrix of training examples.
- `y`: Vector of target values.
- `y_pred`: Vector of predicted values.
### Implementation and use case
```python
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
```

## Functions

- `compute_cost()`: Computes the cost of the current model.
- `gradient_descent()`: Performs gradient descent to learn the optimal values for theta.
- `predict(X)`: Makes predictions using the fitted linear regression model.
- `plot_cost()`: Plots the cost history of the model.