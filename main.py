import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import DeepNeuralNetwork
from utils import one_hot_encode, accuracy_score, confusion_matrix

# -----------------------------
# 1. Load Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

print("Dataset shape:", X.shape)
print("Classes:", class_names)

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train_labels, y_test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 4. One-Hot Encode Labels
# -----------------------------
num_classes = len(np.unique(y))
y_train = one_hot_encode(y_train_labels, num_classes)
y_test = one_hot_encode(y_test_labels, num_classes)

# -----------------------------
# 5. Create Model
# Architecture: 4 -> 16 -> 8 -> 3
# -----------------------------
model = DeepNeuralNetwork(layer_sizes=[4, 16, 8, 3], seed=42)

# -----------------------------
# 6. Train Model
# -----------------------------
model.train(
    X_train,
    y_train,
    y_train_labels,
    epochs=500,
    learning_rate=0.01,
    batch_size=16,
    print_every=50
)

# -----------------------------
# 7. Evaluate on Test Set
# -----------------------------
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test_labels, test_predictions)

print("\nTest Accuracy: {:.2f}%".format(test_accuracy))

# -----------------------------
# 8. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test_labels, test_predictions, num_classes)

print("\nConfusion Matrix:")
print(cm)

print("\nClass-wise Prediction Results:")
for i, class_name in enumerate(class_names):
    print(f"{i} -> {class_name}")

# -----------------------------
# 9. Save Model
# -----------------------------
model.save_model("iris_dnn_model.npz")

# -----------------------------
# 10. Test Load Model
# -----------------------------
loaded_model = DeepNeuralNetwork(layer_sizes=[4, 16, 8, 3])
loaded_model.load_model("iris_dnn_model.npz")

loaded_predictions = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test_labels, loaded_predictions)

print("\nLoaded Model Test Accuracy: {:.2f}%".format(loaded_accuracy))

# -----------------------------
# 11. Plot Training Loss
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(model.train_loss_history)
plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.show()

# -----------------------------
# 12. Plot Training Accuracy
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(model.train_acc_history)
plt.title("Training Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()