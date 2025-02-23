"""
PERCEPTRON ALGORITHM

1. Start with random weights: W1, ..., Wn, b
2. For every misclassified point (X1, ..., Xn):

   2.1 If prediction = 0:

       - For i = 1 ...n

           - Change Wi + alpha * Xi

       - Change b to b + alpha

   2.2 If prediction = 1:

       - For i = 1 ...n

           - Change Wi - alpha * Xi

       - Change b to b - alpha

"""
import numpy as np
import pandas as pd
import os


def step_function(value): # Suitable for discrete/binary classification but for continuous, we need a Sigmoid activation function.
    """Step activation function."""
    return 1 if value > 0 else -1


def perceptron(X, y, learning_rate=0.01, epochs=100):
    """
    Perceptron algorithm to classify binary data:

    Parameters:
    X (Numpy array): Feature matrix of shape (n_samples, n_features)
    y (Numpy array): Target labels of shape (n_sample), with values -1 or 1
    Learning_rate (float): Step size for weight updates
    Epochs (int): Number of times to iterate over the dataset

    Returns:
    weights (Numpy array): Final weights vector
    bias (float): Final bias term
    """
    # Initialize weighs and bias to random values
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(epochs):
        for i in range(len(X)):  # Sequential update (after each sample)
            linear_output = np.dot(X[i], weights) + bias
            prediction = step_function(linear_output)

            if prediction != y[i]:
                weights += learning_rate * y[i] * X[i]
                bias += learning_rate * y[i]

    return weights, bias


# Usage
if __name__ == "__main__":
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", "diabetes_1.csv")

    data = pd.read_csv(file_path)

    # Using last column as label
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Labels

    # Convert labels to -1 and 1 if needed
    y = np.where(y == 0, -1, 1)

    # Train perceptron with sequential updates
    weights, bias = perceptron(X, y, learning_rate=0.1, epochs=10)

    print("Final weights:", weights)
    print("Final bias:", bias)

    # Test predictions
    for x, target in zip(X[:10], y[:10]):  # Displaying first 10 for verification
        prediction = step_function(np.dot(x, weights) + bias)
        print(f"Input: {x}, Target: {target}, Prediction: {prediction}")
