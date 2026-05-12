import numpy as np

# Inputs (XOR truth table)
X = np.array([[0, 0],
[0, 1],
[1, 0],
[1, 1]])

# Output labels
Y = np.array([[0],
[1],
[1],
[0]])

input_neurons = 2 # 2 input features
hidden_neurons = 4 # 4 hidden neurons
output_neurons = 1 # 1 output neuron
learning_rate = 0.1 # learning rate
epochs = 10000 # number of training epochs

# Network architecture: 1 -> 8 -> 1
n_in, n_hidden, n_out = 1, 8, 1

# Xavier-like init for stability
W1 = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_hidden))
b1 = np.zeros((1, n_hidden))
W2 = rng.normal(0.0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, n_out))
b2 = np.zeros((1, n_out))

# Activation and its derivative

def tanh(x):
    return np.tanh(x)


def dtanh(x):
    t = np.tanh(x)
    return 1.0 - t * t

# Training loop (batch gradient descent)
lr = 0.05
epochs = 2000

for epoch in range(epochs):
    # Forward pass
    z1 = xn @ W1 + b1
    a1 = tanh(z1)
    y_pred = a1 @ W2 + b2

    # Mean squared error
    loss = np.mean((y_pred - y) ** 2)

    # Backprop
    d_y_pred = (2.0 / len(xn)) * (y_pred - y)
    dW2 = a1.T @ d_y_pred
    db2 = np.sum(d_y_pred, axis=0, keepdims=True)

    da1 = d_y_pred @ W2.T
    dz1 = da1 * dtanh(z1)
    dW1 = xn.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Gradient step
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch + 1:4d} | loss {loss:.6f}")