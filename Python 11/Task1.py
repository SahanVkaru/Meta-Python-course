import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Size': [1, 2, 3],
    'Price': [150, 200, 250]
}

df = pd.DataFrame(data) # create a dataframe from the data set

# Prepare the data
x = df[['Size']] # select the 'Size' column as the feature
y = df['Price'] # select the 'Price' column as the target variable

# Train the model
model = LinearRegression() # create an instance of the LinearRegression class
model.fit(x, y) # fit the model to the data

# View the learned coefficients
print("Intercept (β0):", model.intercept_) # This is the point where the
print("Coefficient (β1):", model.coef_[0]) # This is the slope of the line (change in y per unit change in x

# Predict on training data
y_pred = model.predict(x)

# Predict on new data
size = [[2.5]]
predicted_price = model.predict(size)
print("Predicted price of a 2.5 (1000 sqft) house: ", predicted_price[0]) # Give me the first value inside the array

X = np.array([1, 2, 3, 4]) # feature (input)
y = np.array([40, 45, 50, 55]) # target variable (output)

def compute_cost(X, y, theta0, theta1):
    m = len(y)
    predictions = theta0 + theta1 * X
    cost = (1/m) * np.sum((predictions - y) ** 2)
    return cost

# Implement Gradient Descent to find the best fit line
def gradient_descent(X, y, alpha = 0.01, iterations = 1000):
    m = len(y) # number of training points
    theta0 = 0 # initial intercept
    theta1 = 0 # initial slope
    cost_history = [] # to store cost at each iteration

    for i in range(iterations):
        predictions = theta0 + theta1 * X
        error = predictions - y

        # compute gradients
        grad0 = (1/m) * np.sum(error)
        grad1 = (1/m) * np.sum(error * X)

        # update parameters
        theta0 -= alpha * grad0
        theta1 -= alpha * grad1

        # Track cost
        cost = compute_cost(X, y, theta0, theta1)
        cost_history.append(cost)

    return theta0, theta1, cost_history

theta0, theta1, cost_history = gradient_descent(X, y, alpha=0.01, iterations=1000)

print("Optimal Intercept (theta0): ", theta0)
print("Optimal Slope (theta1): ", theta1)

# Plot the cost history to see how the cost decreased over iterations
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()
