"""
Modified based on Detr: https://github.com/facebookresearch/detr/blob/master/datasets/coco.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T
import math
import numpy as np

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, args):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask() 
        self.args = args

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target, self.args)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class ConvertCocoPolysToMask(object):

    def __call__(self, image, target, args):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno]
 
        lines = [obj["line"] for obj in anno]
        lines = torch.as_tensor(lines, dtype=torch.float32).reshape(-1, 4)

        lines[:, 2:] += lines[:, :2] #xyxy

        lines[:, 0::2].clamp_(min=0, max=w)
        lines[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["lines"] = lines
        

        target["labels"] = classes
        
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 680, 690, 704, 736, 768, 788, 800]
    test_size = 1100
    max = 1333 

    if args.eval:
        return T.Compose([
            T.RandomResize([test_size], max_size=max),
            normalize,
        ])
    else:
        if image_set == 'train':
            return T.Compose([
                T.RandomSelect(
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                ),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=max),
                    ])
                ),
                T.ColorJitter(),
                normalize,
            ])

        if image_set == 'val':
            return T.Compose([
                T.RandomResize([test_size], max_size=max),
                normalize,
            ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'lines'

    if args.eval:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        }    
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, args), args=args)
    return dataset
