# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .bin_dataset import build_bins
from .mtevents_rgb import build_mtevents_rgb
from .test2d import build_test2d



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_name, image_set, args):
    if dataset_name == 'coco':
        return build_coco(image_set, args)
    elif dataset_name == 'bins':
        return build_bins(image_set, args)
    elif dataset_name == 'mtevents_rgb':
        return build_mtevents_rgb(image_set, args)
    elif dataset_name == 'test2d':
        return build_test2d(image_set, args)
