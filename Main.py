import argparse
import numpy as np


def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y


def split_data(X, y, train_ratio=0.8):
    idx = int(len(X) * train_ratio)
    return X[:idx], y[:idx], X[idx:], y[idx:]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.activation = activation

    def _activation(self, x):
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)

    def _activation_derivative(self, x):
        if self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation == 'relu':
            return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]
        error = output - y
        dW2 = self.a1.T @ error / m
        db2 = np.mean(error, axis=0, keepdims=True)

        dz1 = (error @ self.W2.T) * self._activation_derivative(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train_batch(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def train_online(self, X, y, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                xi = X[i:i+1]
                yi = y[i:i+1]
                output = self.forward(xi)
                self.backward(xi, yi, output, learning_rate)

    def evaluate(self, X, y, name=""):
        predictions = self.forward(X)
        mse = np.mean((predictions - y) ** 2)
        print(f"{name} MSE: {mse:.4f}")
        return mse

    def assess_fit(self, mse_train, mse_test):
        print("Ocena dopasowania:")
        ratio = mse_test / (mse_train + 1e-8)
        if mse_test < 0.01 and mse_train < 0.01:
            print(" - Dopasowanie: dobre")
        elif ratio < 1.5:
            print(" - Dopasowanie: optymalne")
        elif ratio < 3.0:
            print(" - Dopasowanie: lekkie przeuczenie")
        else:
            print(" - Dopasowanie: zbyt duże przeuczenie")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="ścieżka do pliku")
    args = parser.parse_args()

    X, y = load_data(args.filename)
    X_train, y_train, X_test, y_test = split_data(X, y)

    print("--- Sieć z tanh ---")
    nn_tanh = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1, activation='tanh')
    nn_tanh.train_batch(X_train, y_train, epochs=1000, learning_rate=0.01)
    mse_train = nn_tanh.evaluate(X_train, y_train, name="Train")
    mse_test = nn_tanh.evaluate(X_test, y_test, name="Test")
    nn_tanh.assess_fit(mse_train, mse_test)

    print("--- Sieć z sigmoid ---")
    nn_sigmoid = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1, activation='sigmoid')
    nn_sigmoid.train_batch(X_train, y_train, epochs=1000, learning_rate=0.01)
    mse_train = nn_sigmoid.evaluate(X_train, y_train, name="Train")
    mse_test = nn_sigmoid.evaluate(X_test, y_test, name="Test")
    nn_sigmoid.assess_fit(mse_train, mse_test)

    print("--- Sieć z ReLU ---")
    nn_relu = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1, activation='relu')
    nn_relu.train_batch(X_train, y_train, epochs=1000, learning_rate=0.01)
    mse_train = nn_relu.evaluate(X_train, y_train, name="Train")
    mse_test = nn_relu.evaluate(X_test, y_test, name="Test")
    nn_relu.assess_fit(mse_train, mse_test)

    print("--- Sieć online z tanh ---")
    nn_online_tanh = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1, activation='tanh')
    nn_online_tanh.train_online(X_train, y_train, epochs=10, learning_rate=0.01)
    mse_train = nn_online_tanh.evaluate(X_train, y_train, name="Train")
    mse_test = nn_online_tanh.evaluate(X_test, y_test, name="Test")
    nn_online_tanh.assess_fit(mse_train, mse_test)

    print("--- Sieć online z ReLU ---")
    nn_online_relu = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1, activation='relu')
    nn_online_relu.train_online(X_train, y_train, epochs=10, learning_rate=0.01)
    mse_train = nn_online_relu.evaluate(X_train, y_train, name="Train")
    mse_test = nn_online_relu.evaluate(X_test, y_test, name="Test")
    nn_online_relu.assess_fit(mse_train, mse_test)
