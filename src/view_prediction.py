#!/usr/bin/env python3
import sys
import json
import numpy as np
from matplotlib import pyplot as plt

def plot_lines_3D(lines, ax, color, label):
    for i, line in enumerate(lines):
        x1, y1, z1, x2, y2, z2 = line
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, marker='o', label=label if i == 0 else '')

def view_structured(xyz):
    plt.imshow(np.moveaxis(xyz, 0, -1))


def view3D(points, pred_lines, target_lines):
    #img = np.moveaxis(xyz, 0, -1)
    #non_zeros = np.prod(img, axis=2) != 0
    #points = img[non_zeros]

    points = points[np.random.rand(points.shape[0]) < 0.05]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], marker='o')

    plot_lines_3D(target_lines, ax, 'g', 'annotation')
    plot_lines_3D(pred_lines, ax, 'r', 'prediction')

    plt.legend()
    return plt, ax

def view2D(img, pred_lines, target_lines):
    from PIL import Image, ImageDraw

    mean = np.array([0.538, 0.494, 0.453])
    std = np.array([0.257, 0.263, 0.273])

    img = np.moveaxis(img, 0, -1)

    img = std * img + mean
    img = np.clip(img, 0, 1)


    pil_img = Image.fromarray(np.uint8(img*255))
    draw = ImageDraw.Draw(pil_img)

    width = img.shape[1]
    height = img.shape[0]

    for line in target_lines:
        x1, y1, x2, y2 = line
        draw.line((x1*width, y1*height, x2*width, y2*height), fill=(0, 255, 0), width=2)

    for line in pred_lines:
        x1, y1, x2, y2 = line
        draw.line((x1*width, y1*height, x2*width, y2*height), fill=(255, 0, 0), width=2)

    plt.imshow(np.asarray(pil_img))
    return plt

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('No input file')
        exit()
    file = sys.argv[1]

    with open(file, 'r') as file:
        data = json.load(file)

    normalized = data['entry']['normalized']
    means = np.array(normalized['means'])
    stds = np.array(normalized['stds'])

    xyz = np.array(data['xyz'])

    img = np.moveaxis(xyz, 0, -1)
    non_zeros = np.prod(img, axis=2) != 0
    points = img[non_zeros]

    #points = points[np.random.rand(points.shape[0]) < 0.05]

    pred_scores = np.array(data['prediction']['scores'])
    pred_lines = np.array(data['prediction']['lines'])
    target_lines = np.array(data['targets']['lines'])
    keep = np.array(np.argsort(pred_scores)[::-1][:4])
    print(f'displaying {len(keep)} lines with scores: ', pred_scores[keep])
    pred_lines = pred_lines[keep]


    pred_lines = ((pred_lines.reshape(-1,3) * stds) + means).reshape(-1, 6)
    target_lines = ((target_lines.reshape(-1,3) * stds) + means).reshape(-1, 6)
    points = (points * stds) + means


    print(pred_lines.shape)
    if pred_lines.shape[1] == 4:
        view2D(xyz, pred_lines, target_lines).show()
    else:
        view_structured(xyz)
        view3D(points, pred_lines, target_lines)
        plt.show()
