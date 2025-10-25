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




weight_dict = {
    'loss_ce': 1,
    'loss_line': 5,
    'loss_ce_0': 1,
    'loss_line_0': 5,
    'loss_ce_1': 1,
    'loss_line_1': 5,
    'loss_ce_2': 1,
    'loss_line_2': 5,
    'loss_ce_3': 1,
    'loss_line_3': 5,
    'loss_ce_4': 1,
    'loss_line_4': 5
}


#exit()

def compute_loss(loss_dict_arr):
    l = []
    for i in range(len(loss_dict_arr)):
        loss_dict = loss_dict_arr[i]
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        l.append(losses)
    return l

#print(len(data['train']))
#exit()

train_loss = compute_loss(data['train'])
test_loss = compute_loss(data['test'])
epochs = list(range(1, len(train_loss)+1))

fig, ax = plt.subplots()
ax.plot(epochs, train_loss, label='train loss')
ax.plot(epochs, test_loss, label='test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(title)
plt.legend()

plt.show()
