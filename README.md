# Micrograd Implementation

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A lightweight, scalar-valued Autograd engine and Neural Network library implemented from scratch in Python. 

This project implements **Reverse-Mode Automatic Differentiation** (Backpropagation) over a dynamically built Directed Acyclic Graph (DAG). It is designed to demystify the "black box" of modern deep learning frameworks like PyTorch.

**Heavily inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).**

## 🧠 The Core Logic

Unlike high-level frameworks that operate on tensors, this engine operates on **scalars** (single numbers). It builds a computational graph where:
1.  **Nodes** are values (data).
2.  **Edges** represent mathematical operations (+, -, *, tanh, etc.).
3.  **Gradients** flow backward from the loss function to the input weights using the **Chain Rule**.

### The Math
For any node `y = f(x)`, we compute the gradient `∂L/∂x` using the Chain Rule:

**∂L/∂x = (∂L/∂y) * (∂y/∂x)**

Where:
* **∂L/∂y**: The gradient flowing from upstream (the parent node).
* **∂y/∂x**: The local derivative of the operation at the current node.

## 🚀 Features

* **Autograd Engine:** Tracks operations (add, mul, pow, tanh) and recursively computes gradients.
* **Neural Network API:** Includes `Neuron`, `Layer`, and `MLP` classes mimicking PyTorch's API.
* **Zero Dependencies:** The core engine (`engine.py` and `nn.py`) requires only standard Python libraries (`math`, `random`).
* **Topological Sort:** Implements graph traversal to ensure correct backpropagation order.

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/tuhinangshugoswami/micrograd-implementation.git](https://github.com/tuhinangshugoswami/micrograd-implementation.git)
cd micrograd-implementation

pip install -r requirements.txt
from engine import Value

# Build a simple graph
a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).tanh()

# Backward pass
d.backward()

print(f"Value of d: {d.data}")
print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")

from nn import MLP

# Define a simple dataset
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0] 

# Initialize the Multi-Layer Perceptron
# 3 input neurons, two hidden layers of 4 neurons, 1 output neuron
model = MLP(3, [4, 4, 1]) 

# Training Loop
for k in range(20):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Zero grad
    for p in model.parameters():
        p.grad = 0.0
        
    # Backward pass
    loss.backward()
    
    # Update (SGD)
    for p in model.parameters():
        p.data += -0.05 * p.grad
        
    print(f"Step {k} | Loss: {loss.data}")
