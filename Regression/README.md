# LinearRegression Class

This class implements a simple linear regression model, which fits a linear relationship between a dependent variable (Y) and an independent variable (X).

## Attributes

- `m`: Slope of the linear regression line.
- `b`: Intercept of the linear regression line.
- `rmse`: Root Mean Squared Error, a measure of the model's predictive accuracy.
- `r_squared`: Coefficient of determination (R-squared), indicating the proportion of the variance in the dependent variable explained by the independent variable.
- `mse`: Mean Squared Error, an alternate measure of the model's predictive accuracy.
- `mae`: Mean Absolute Error, a measure of the average magnitude of errors between actual and predicted values.

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
