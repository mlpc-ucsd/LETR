#!/usr/bin/env python3
import sys
import json
from matplotlib import pyplot as plt

with open(sys.argv[1], 'r') as file:
    data = json.load(file)

title = ''
if len(sys.argv) == 3:
    title = sys.argv[2]

# Print the data
#print(data)

train = data['train']
epochs = list(range(1, len(train)+1))

test = data['test']
epochs = list(range(1, len(test)+1))

print()

fig, ax = plt.subplots()
ax.plot(epochs, train, label='train loss')
ax.plot(epochs, test, label='test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title(title)

plt.show()
