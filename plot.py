import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import MultipleLocator

result = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

result['train_acc'] = [77.5, 88.1, 88.5, 88.9, 90.4, 91.7]
result['test_acc'] = [77.7, 88.7, 89.1, 89.4, 91.1, 92.6]
fig, ax = plt.subplots(figsize=(4, 3.5))
plt.subplot(111)
#plt.title("Accuarcy / bits")
plt.xlabel("Bits")
plt.ylabel("Accuarcy(%)")
x = [1, 2, 3, 5, 9, 17]
line1, = plt.plot(x, result['test_acc'], marker='o', label='Test')
line2, = plt.plot(x, result['train_acc'], marker='o', label='Train')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=32)}, prop={'size': 13})
plt.show()
fig.savefig("Accuarcy.png")
plt.close(fig)

result['train_loss'] = [0.606, 0.560, 0.561, 0.563, 0.548, 0.545]
result['test_loss'] = [0.602, 0.555, 0.556, 0.559, 0.543, 0.539]
fig, ax = plt.subplots(figsize=(4, 3.5))
plt.subplot(111)
plt.xlabel("Bits")
plt.ylabel("Loss")
x = [1, 2, 3, 5, 9, 17]
line1, = plt.plot(x, result['test_loss'],  marker='o', label='Test')
line2, = plt.plot(x, result['train_loss'],  marker='o', label='Train')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=32)}, prop={'size': 13})
plt.show()
fig.savefig("Loss.png")
plt.close(fig)
