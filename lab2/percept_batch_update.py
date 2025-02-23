import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def step_function(value):
    """Step activation function."""
    return 1 if value > 0 else -1


def perceptron_batch(X, y, learning_rate=0.03, epochs=100):
    """
    Perceptron algorithm with batch weight updates.

    Parameters:
    X (Numpy array): Feature matrix of shape (n_samples, n_features)
    y (Numpy array): Target labels of shape (n_samples), with values -1 or 1
    learning_rate (float): Step size for weight updates
    epochs (int): Number of times to iterate over the dataset

    Returns:
    weights (Numpy array): Final weights vector
    bias (float): Final bias term
    """
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(epochs):
        total_weight_update = np.zeros(n_features)
        total_bias_update = 0

        for i in range(len(X)):
            linear_output = np.dot(X[i], weights) + bias
            prediction = step_function(linear_output)

            if prediction != y[i]:
                total_weight_update += learning_rate * y[i] * X[i]
                total_bias_update += learning_rate * y[i]

        # Apply batch updates after iterating through all samples
        weights += total_weight_update
        bias += total_bias_update

    return weights, bias


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

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Normalize feature columns excluding the target column
    X = scaler.fit_transform(X)

    # Train perceptron with batch updates
    weights, bias = perceptron_batch(X, y, learning_rate=0.03, epochs=50)

    print("Final weights:", weights)
    print("Final bias:", bias)

    # Test predictions and evaluate model
    y_pred = [step_function(np.dot(x, weights) + bias) for x in X]

    # Print first 10 predictions for verification
    for x, target, pred in zip(X[:10], y[:10], y_pred[:10]):
        print(f"Input: {x}, Target: {target}, Prediction: {pred}")

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y, y_pred))
