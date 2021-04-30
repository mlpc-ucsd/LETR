#!/usr/bin/env python3
"""Process YorkUrban dataset for L-CNN network
Usage:
    york.py <src> <dst>
    york.py (-h | --help )
Examples:
    python dataset/york.py /datadir/york data/york
Arguments:
    <src>                Original data directory of YorkUrban
    <dst>                Directory of the output
Options:
   -h --help             Show this screen.
"""

import os
import sys
import glob
import json
import os.path as osp
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.io import loadmat
from scipy.ndimage import zoom

def main():
    args = docopt(__doc__)
    src_dir = args["<src>"]
    tar_dir = args["<dst>"]

    os.makedirs(tar_dir, exist_ok=True)
    dataset = sorted(glob.glob(osp.join(src_dir, "*/*.jpg")))
    image_id = 0
    anno_id = 0
    for mode in ["train", "val"]:
        batch = f"{mode}2017"
        os.makedirs(os.path.join(tar_dir, batch), exist_ok=True)

        anno = {}
        anno['images'] = []
        anno['annotations'] = []
        anno['categories'] = [{'supercategory': "line", "id": "0", "name": "line"}]

        def handle(iname, image_id, anno_id, batch):

            im = cv2.imread(iname)
            filename = iname.split("/")[-1]

            anno['images'].append({'file_name': filename,
                                'height': im.shape[0], 'width': im.shape[1], 'id': image_id})
            mat = loadmat(iname.replace(".jpg", "LinesAndVP.mat"))
            lines = np.array(mat["lines"]).reshape(-1, 2, 2)
            lines = lines.astype('float')
            os.makedirs(os.path.join(tar_dir, batch), exist_ok=True)

            image_path = os.path.join(tar_dir, batch, filename)
            line_set = save_and_process(f"{image_path}", filename, im[::, ::], lines)
            for line in line_set:
                info = {}
                info['id'] = anno_id
                anno_id += 1
                info['image_id'] = image_id
                info['category_id'] = 0
                info['line'] = line
                info['area'] = 1
                anno['annotations'].append(info)

            image_id += 1
            print(f"Finishing {image_path}")
            return anno_id

        if mode == "val":
            for img in dataset:
                anno_id = handle(img, image_id, anno_id, batch)
                image_id += 1

        os.makedirs(os.path.join(tar_dir, "annotations"), exist_ok=True)
        anno_path = os.path.join(tar_dir, "annotations", f"lines_{batch}.json")
        with open(anno_path, 'w') as outfile:
            json.dump(anno, outfile)


def save_and_process(image_path, image_name, image, lines):
    # change the format from x,y,x,y to x,y,dx, dy
    # order: top point > bottom point
    #        if same y coordinate, right point > left point

    new_lines_pairs = []
    for line in lines:  # [ #lines, 2, 2 ]
        p1 = line[0]    # xy
        p2 = line[1]    # xy
        if p1[0] < p2[0]:
            new_lines_pairs.append([p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]])
        elif p1[0] > p2[0]:
            new_lines_pairs.append([p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]])
        else:
            if p1[1] < p2[1]:
                new_lines_pairs.append(
                    [p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]])
            else:
                new_lines_pairs.append(
                    [p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]])

    cv2.imwrite(f"{image_path}", image)
    return new_lines_pairs

if __name__ == "__main__":
    main()