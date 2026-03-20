import numpy as np
from activation import relu, relu_derivative, softmax
from losses import cross_entropy_loss
from utils import create_mini_batches

class DeepNeuralNetwork:
    def __init__(self, layer_sizes, seed=42):
        """
        layer_sizes example:
        [4, 16, 8, 3]
        input layer = 4 features
        hidden1 = 16 neurons
        hidden2 = 8 neurons
        output = 3 classes
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        self.weights = {}
        self.biases = {}

        self.train_loss_history = []
        self.train_acc_history = []

        self._initialize_parameters()

    def _initialize_parameters(self):
        for l in range(1, len(self.layer_sizes)):
            # He initialization for ReLU layers
            self.weights[l] = np.random.randn(self.layer_sizes[l - 1], self.layer_sizes[l]) * np.sqrt(2.0 / self.layer_sizes[l - 1])
            self.biases[l] = np.zeros((1, self.layer_sizes[l]))

    def forward(self, X):
        self.activations = {}
        self.z_values = {}

        self.activations[0] = X

        # Hidden layers with ReLU
        for l in range(1, self.num_layers):
            self.z_values[l] = np.dot(self.activations[l - 1], self.weights[l]) + self.biases[l]
            self.activations[l] = relu(self.z_values[l])

        # Output layer with Softmax
        self.z_values[self.num_layers] = np.dot(self.activations[self.num_layers - 1], self.weights[self.num_layers]) + self.biases[self.num_layers]
        self.activations[self.num_layers] = softmax(self.z_values[self.num_layers])

        return self.activations[self.num_layers]

    def backward(self, y_true):
        m = y_true.shape[0]
        self.dW = {}
        self.db = {}

        # Output layer gradient (Softmax + Cross Entropy simplifies)
        dZ = self.activations[self.num_layers] - y_true

        self.dW[self.num_layers] = np.dot(self.activations[self.num_layers - 1].T, dZ) / m
        self.db[self.num_layers] = np.sum(dZ, axis=0, keepdims=True) / m

        # Hidden layers backprop
        for l in range(self.num_layers - 1, 0, -1):
            dA = np.dot(dZ, self.weights[l + 1].T)
            dZ = dA * relu_derivative(self.z_values[l])

            self.dW[l] = np.dot(self.activations[l - 1].T, dZ) / m
            self.db[l] = np.sum(dZ, axis=0, keepdims=True) / m

    def update_parameters(self, learning_rate):
        for l in range(1, self.num_layers + 1):
            self.weights[l] -= learning_rate * self.dW[l]
            self.biases[l] -= learning_rate * self.db[l]

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def train(self, X_train, y_train, y_train_labels, epochs=500, learning_rate=0.01, batch_size=16, print_every=50):
        for epoch in range(1, epochs + 1):
            mini_batches = create_mini_batches(X_train, y_train, batch_size)

            for X_batch, y_batch in mini_batches:
                y_pred = self.forward(X_batch)
                self.backward(y_batch)
                self.update_parameters(learning_rate)

            # Track full-train metrics
            full_pred = self.forward(X_train)
            loss = cross_entropy_loss(y_train, full_pred)
            preds = np.argmax(full_pred, axis=1)
            acc = np.mean(preds == y_train_labels) * 100

            self.train_loss_history.append(loss)
            self.train_acc_history.append(acc)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Train Accuracy: {acc:.2f}%")

    def save_model(self, filepath="model_weights.npz"):
        save_dict = {}
        for l in range(1, self.num_layers + 1):
            save_dict[f"W{l}"] = self.weights[l]
            save_dict[f"b{l}"] = self.biases[l]
        save_dict["layer_sizes"] = np.array(self.layer_sizes)
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="model_weights.npz"):
        data = np.load(filepath, allow_pickle=True)
        self.layer_sizes = data["layer_sizes"].tolist()
        self.num_layers = len(self.layer_sizes) - 1

        self.weights = {}
        self.biases = {}

        for l in range(1, self.num_layers + 1):
            self.weights[l] = data[f"W{l}"]
            self.biases[l] = data[f"b{l}"]

        print(f"Model loaded from {filepath}")