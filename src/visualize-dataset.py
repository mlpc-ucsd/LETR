from datasets.bin_dataset import Dataset
from matplotlib import pyplot as plt
import torch
import numpy as np


def visualize_xyz_with_lines(xyz, lines, subsample_factor=0.2):
    img = np.moveaxis(xyz, 0, -1)

    non_zeros = np.prod(img, axis=2) != 0
    points = img[non_zeros]

    print(points.shape)

    subsampled = points[np.random.rand(points.shape[0]) < subsample_factor]

    print(subsampled.shape)

    #exit()

    zeros = np.prod(img, axis=2) == 0
    img -= np.min(img)
    img /= np.max(img)
    img[zeros] = np.array([0. , 0. , 0.])

    plt.imshow(img)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(subsampled[:,0], subsampled[:,1], subsampled[:,2], marker='o')

    for line in lines:
        #line = np.array([
        #    np.concatenate((line[:3], np.array([1.0]))),
        #    np.concatenate((line[3:], np.array([1.0]))),
        #])
        print(line)
        x1, y1, z1, x2, y2, z2 = line
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='r', marker='o')
    plt.show()

dataset = Dataset('data/Gajdosech_etal_2021_dataset/VISIGRAPP_TRAIN/dataset.json',
                  'train',
                  256, 191,
                  preload=False)


xyz, target = dataset[np.random.randint(len(dataset))]
#xyz, target = dataset[10]

visualize_xyz_with_lines(xyz.numpy(), target['lines'].numpy())
