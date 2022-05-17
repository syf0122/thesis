# Import matplotlib, numpy and math
import matplotlib.pyplot as plt
import numpy as np
import math

# x = np.linspace(-10, 10, 1000)
# y = 1 / (1 + np.exp(-x) )

# plt.figure(figsize=(10, 5))
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#
x = np.linspace(-10, 10, 1000)
y = np.maximum(0, x)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='ReLu')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

def leaky_ReLU(x):
  data = [max(0.2*value,value) for value in x]
  return np.array(data, dtype=float)
y2 = leaky_ReLU(x)

# plt.figure(figsize=(10, 5))
plt.plot(x, y2, label = 'Leaky ReLu')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
