import numpy as np

class activation_softmax:
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.output = probabilities

class Loss_categoricalCrossEntropy:
  def forward(self, y_pred, y_true):
    samples = len(y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    if len(y_true.shape) == 1:
      correct_confidences = y_pred_clipped[range(samples), y_true]
      negative_log_likelihoods = -np.log(correct_confidences)
      return negative_log_likelihoods
    elif len(y_true.shape) == 2:
      correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
      negative_log_likelihoods = -np.log(correct_confidences)
      return negative_log_likelihoods

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases

  def backward(self, dvalues):
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

class Activation_ReLU:
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)

  def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax_Loss_CategoricalCrossentropy:
  def __init__(self):
    self.activation = activation_softmax()
    self.loss = Loss_categoricalCrossEntropy()
    self.inputs = []

  def forward(self, inputs, y_true):
    self.inputs = inputs
    self.activation.forward(inputs)
    return self.loss.forward(self.activation.output, y_true)

  def backward(self, dvalues, y_true):
    samples = len(self.inputs)
    if len(y_true.shape) == 2:
      y_true = np.argmax(y_true, axis=1)
    self.dinputs = self.activation.output.copy()
    self.dinputs[range(samples), y_true] -= 1
    self.dinputs /= samples

# Test
X = np.array([[1, 2], [5, 1], [2, 3]], dtype=np.float32)
y = np.array([0, 1, 1])
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)
print('Loss:', loss)
print('Loss mean:', np.mean(loss))  # ~0.70
