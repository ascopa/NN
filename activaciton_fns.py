import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def glu(x):
    return x * sigmoid(x)

# Generate input values
x = np.linspace(-4, 4, 100)

# Calculate outputs for each activation function
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_glu = glu(x)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label='Sigmoide', linewidth=3)
plt.plot(x, y_tanh, label='Tanh', linewidth=3)
plt.plot(x, y_relu, label='ReLU', linewidth=3)
plt.plot(x, y_glu, label='GLU', linewidth=3)

plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.legend()
#plt.ylim(-1, 3)  # Limit y-axis to 0 to 3
plt.grid(True)
plt.savefig('activation_functions_comparison.png', dpi=400)