# 🧠 NumPy Neural Network from Scratch

A modular Deep Learning implementation built entirely in **Python** and **NumPy**. This project demonstrates the core mathematics of neural networks, including forward propagation, backpropagation, and manual gradient calculation without high-level frameworks.

### 🚀 Key Features
- **Dense Layers**: Fully connected layers with random weight initialization.
- **Activations**: ReLU for hidden layers; Softmax for probability outputs.
- **Loss**: Categorical Cross-Entropy with numerical stability (log-clipping).
- **Backpropagation**: Manual implementation of the Chain Rule.
- **Vectorized**: Efficient batch processing using NumPy matrix operations.

### 🛠 Architecture
The network uses a modular design where each layer handles its own forward and backward pass.



### 💻 Quick Start
```python
import numpy as np

# Initialize network components
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3) # 3 Output classes
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

# Backward pass (Gradients)
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
