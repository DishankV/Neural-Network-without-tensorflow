# Deep Neural Network From Scratch using NumPy

A **placement-level deep learning project** where a **multi-layer neural network** is implemented completely **from scratch using only NumPy**, without using high-level deep learning frameworks such as **TensorFlow**, **PyTorch**, or **Keras**.

This project demonstrates the **core internal working of neural networks**, including:

- Forward Propagation
- Backpropagation
- ReLU Activation
- Softmax Output Layer
- Cross-Entropy Loss
- He Weight Initialization
- Mini-Batch Gradient Descent
- Multi-Class Classification
- Model Save/Load
- Training Metrics Visualization

The model is trained on the **Iris dataset** for **multi-class classification**.

---

## 🚀 Features

- Built using **NumPy only**
- No TensorFlow / No PyTorch / No Keras
- Supports **multiple hidden layers**
- Uses **ReLU** for hidden layers
- Uses **Softmax** for output layer
- Implements **Cross-Entropy Loss**
- Uses **He Initialization** for better convergence
- Trains using **Mini-Batch Gradient Descent**
- Performs **Train/Test Split**
- Includes **Accuracy Evaluation**
- Includes **Confusion Matrix**
- Supports **Model Persistence (.npz save/load)**
- Plots **Training Loss** and **Training Accuracy**

---

## 📌 Why This Project?

Most neural network projects rely on frameworks like TensorFlow or PyTorch, which abstract away the internal mechanics of deep learning.

This project was built to deeply understand and demonstrate:

- How neurons compute outputs
- How gradients flow backward
- How weights are updated
- How multi-class classification works
- How modern neural networks are trained internally

This makes it a strong **interview-ready** and **resume-worthy** project for **AI / ML / Data Science roles**.

---

## 🧠 Concepts Implemented

### 1. Forward Propagation
Each layer computes:

- Weighted sum of inputs
- Bias addition
- Activation transformation

### 2. ReLU Activation
Used in hidden layers:

\[
f(x) = \max(0, x)
\]

### 3. Softmax Activation
Used in output layer for multi-class classification:

\[
P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

### 4. Cross-Entropy Loss
Used for multi-class classification:

\[
L = - \frac{1}{m} \sum \sum y \log(\hat{y})
\]

### 5. Backpropagation
Gradients are computed manually for:

- Output layer
- Hidden layers
- Weights
- Biases

### 6. He Initialization
Weights are initialized using:

\[
W \sim \mathcal{N}(0, \sqrt{2/n})
\]

This improves stability when using ReLU activations.

### 7. Mini-Batch Gradient Descent
Training is performed in mini-batches instead of:

- Full batch (slow)
- Pure SGD (too noisy)

---

## 📂 Project Structure

```bash
deep_nn_from_scratch/
│
├── main.py              # Main training and evaluation script
├── model.py             # Deep neural network implementation
├── activations.py       # ReLU, ReLU derivative, Softmax
├── losses.py            # Cross-Entropy loss
├── utils.py             # One-hot encoding, accuracy, confusion matrix, mini-batches
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

Input Layer (4)
    ↓
Hidden Layer 1 (16) + ReLU
    ↓
Hidden Layer 2 (8) + ReLU
    ↓
Output Layer (3) + Softmax

📚 Learning Outcomes

By building this project, the following concepts were implemented and understood in depth:
Neural network architecture design
Hidden vs output layer behavior
Activation functions and their derivatives
Softmax probability interpretation
Cross-entropy loss computation
Manual backpropagation
Gradient-based optimization
Weight initialization strategies
Batch training techniques
Model evaluation and persistence