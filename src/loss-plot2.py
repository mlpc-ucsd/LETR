#!/usr/bin/env python3
import sys
import json
from matplotlib import pyplot as plt

with open(sys.argv[1], 'r') as file:
    data = json.load(file)

# Print the data
print(data)


def plot_loss(loss_dict, note, ax):
    keys = loss_dict[0].keys()

    losses = {}
    for i in range(len(loss_dict)):
        for key in keys:
            if key in losses:
                losses[key].append(loss_dict[i][key])
            else:
                losses[key] = [loss_dict[i][key]]
    epochs = list(range(1, len(loss_dict)+1))
    for key in keys:
        ax.plot(epochs, losses[key], label=f'{note} {key}')



print()

fig, ax = plt.subplots()
#ax.plot(epochs, train, label='train loss')
#ax.plot(epochs, test, label='test loss')
plot_loss(data['train'], 'Train' , ax)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
