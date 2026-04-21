import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])  # Replace with your actual x data
y = np.array([2, 4, 5, 4, 6])  # Replace with your actual y data

slope = 0
intercept = 0 

def predict(x_val):
    return slope * x_val + intercept #(y = mx + c)  

model = list(map(predict, x))
plt.scatter(x, y)
plt.plot(x, model)
plt.show()