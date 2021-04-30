#!/usr/bin/env python3
"""Process data for LETR
Usage:
    wireframe.py <src> <dst>
    wireframe.py (-h | --help )

Examples:
    python wireframe.py wireframe_raw wireframe_processed

Arguments:
    <src>                Source directory that stores preprocessed wireframe data
    <dst>                Temporary output directory

Options:
   -h --help             Show this screen.
"""

import json
import cv2
import os
import numpy as np
import math
from docopt import docopt

def main():
    args = docopt(__doc__)
    src_dir = args["<src>"]
    tar_dir = args["<dst>"]

    image_id = 0
    anno_id = 0
    for batch in ["train2017", "val2017"]:
        if batch == "train2017":
            anno_file = os.path.join(src_dir, "train.json")
        else:
            anno_file = os.path.join(src_dir, "valid.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        def handle(data, image_id, anno_id):
            im = cv2.imread(os.path.join(src_dir, "images", data["filename"]))
            anno['images'].append({'file_name': data['filename'], 'height': im.shape[0], 'width': im.shape[1], 'id': image_id})
            lines = np.array(data["lines"]).reshape(-1, 2, 2)
            os.makedirs(os.path.join(tar_dir, batch), exist_ok=True)

            image_path = os.path.join(tar_dir, batch, data['filename'])
            line_set = save_and_process(f"{image_path}", data['filename'], im[::, ::], lines)
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
            print("Finishing", image_path)
            return anno_id

        anno = {}
        anno['images'] = []
        anno['annotations'] = []
        anno['categories'] = [{'supercategory':"line", "id": "0", "name": "line"}]
        for img_dict in dataset:
            anno_id = handle(img_dict, image_id, anno_id)
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
    for line in lines: # [ #lines, 2, 2 ]
        p1 = line[0]    # xy
        p2 = line[1]    # xy
        if p1[0] < p2[0]:
            new_lines_pairs.append( [p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]] ) 
        elif  p1[0] > p2[0]:
            new_lines_pairs.append( [p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]] )
        else:
            if p1[1] < p2[1]:
                new_lines_pairs.append( [p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]] )
            else:
                new_lines_pairs.append( [p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]] )

    cv2.imwrite(f"{image_path}", image)
    return new_lines_pairs


if __name__ == "__main__":
    main()