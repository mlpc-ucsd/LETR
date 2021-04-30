"""
Transforms and data augmentation for both image + line.
modfied based on https://github.com/facebookresearch/detr/blob/master/datasets/transforms.py
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numbers
import warnings
from typing import Tuple, List, Optional
from PIL import Image
from torch import Tensor
import math

from util.misc import interpolate
import numpy as np

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "lines" in target:
        lines = target["lines"]
        cropped_lines = lines - torch.as_tensor([j, i, j, i])
        
        eps = 1e-12

        # In dataset, we assume the left point has smaller x coord
        remove_x_min = cropped_lines[:, 2] < 0
        remove_x_max = cropped_lines[:, 0] > w
        remove_x = torch.logical_or(remove_x_min, remove_x_max)
        keep_x = ~remove_x

        # there is no assumption on y, so remove lines that have both y coord out of bound
        remove_y_min = torch.logical_and(cropped_lines[:, 1] < 0, cropped_lines[:, 3] < 0)
        remove_y_max = torch.logical_and(cropped_lines[:, 1] > h, cropped_lines[:, 3] > h)
        remove_y = torch.logical_or(remove_y_min, remove_y_max)
        keep_y = ~remove_y

        keep = torch.logical_and(keep_x, keep_y)
        cropped_lines = cropped_lines[keep]
        clamped_lines = torch.zeros_like(cropped_lines)

        for i,line in enumerate(cropped_lines):
            x1, y1, x2, y2 = line
            slope = (y2 - y1) / (x2 - x1 + eps)
            if x1 < 0:
                x1 = 0
                y1 = y2 + (x1 - x2) * slope
            if y1 < 0:
                y1 = 0
                x1 = x2 - (y2 - y1) / slope
            if x2 > w:
                x2 = w
                y2 = y1 + (x2 - x1) * slope
            if y2 > h:
                y2 = h
                x2 = x1 + (y2 - y1) / slope

            clamped_lines[i, :] = torch.tensor([x1, y1, x2, y2])

        target["lines"] = clamped_lines
        
    for field in fields:
        target[field] = target[field][keep]
    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    
    if "lines" in target:
        lines = target["lines"]   
        lines = lines[:, [2, 3, 0, 1]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["lines"] = lines


    return flipped_image, target


def vflip(image, target):
    flipped_image = F.vflip(image)

    w, h = image.size

    target = target.copy()

    if "lines" in target:
        lines = target["lines"]

        # in dataset, we assume if two points with same x coord, we assume first point is the upper point
        lines = lines * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        vertical_line_idx = (lines[:, 0] == lines[:, 2])
        lines[vertical_line_idx] = torch.index_select(lines[vertical_line_idx], 1, torch.tensor([2,3,0,1]))
        target["lines"] = lines

    return flipped_image, target


def ccw_rotation(image, target):
    rotateded_image = F.rotate(image, 90, expand=True)
    w, h = rotateded_image.size

    target = target.copy()

    target["size"] = torch.tensor([h, w])

    if "lines" in target:
        lines = target["lines"]
        lines = lines[:, [1, 0, 3, 2]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        # in dataset, we assume the first point is the left point
        x_switch_idx = lines[:, 0] > lines[:, 2]
        lines[x_switch_idx] = torch.index_select(lines[x_switch_idx], 1, torch.tensor([2,3,0,1]))

        # in dataset, if two points have same x coord, we assume the first point is the upper point
        y_switch_idx = torch.logical_and(lines[:, 0] == lines[:, 2], lines[:, 1] > lines[:, 3])
        lines[y_switch_idx] = torch.index_select(lines[y_switch_idx], 1, torch.tensor([2,3,0,1]))

        target["lines"] = lines

    return rotateded_image, target


def cw_rotation(image, target):
    rotateded_image = F.rotate(image, -90, expand=True)
    w, h = rotateded_image.size

    target = target.copy()

    target["size"] = torch.tensor([h, w])

    if "lines" in target:
        lines = target["lines"]
        lines = lines[:, [1, 0, 3, 2]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        
        # in dataset, we assume the first point is the left point
        x_switch_idx = lines[:, 0] > lines[:, 2]
        lines[x_switch_idx] = torch.index_select(
            lines[x_switch_idx], 1, torch.tensor([2, 3, 0, 1]))

        # in dataset, if two points have same x coord, we assume the first point is the upper point
        y_switch_idx = torch.logical_and(
            lines[:, 0] == lines[:, 2], lines[:, 1] > lines[:, 3])
        lines[y_switch_idx] = torch.index_select(
            lines[y_switch_idx], 1, torch.tensor([2, 3, 0, 1]))

        target["lines"] = lines

    return rotateded_image, target

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    
    if "lines" in target:
        lines = target["lines"]
        scaled_lines = lines * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["lines"] = scaled_lines
    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


def pad(image, target, padding):
    assert False
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
        return img, target

class RandomCounterClockwiseRotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return ccw_rotation(img, target)
        return img, target

class RandomClockwiseRotation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return cw_rotation(img, target)
        return img, target
        
class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))

class RandomErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, fill=False):


        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    @staticmethod
    def get_params(img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
        ) -> Tuple[int, int, int, int, Tensor]:

        if isinstance(img, Tensor):
            img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
            area = img_h * img_w
        elif isinstance(img, Image.Image):
            img_c = 3
            img_w, img_h = img.size
            area = img_h * img_w
        else:
            raise TypeError("img is not type Tensor or Image")

        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(ratio[0], ratio[1]).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1, )).item()
            j = torch.randint(0, img_w - w + 1, size=(1, )).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img, target):
        i, j, h, w, v = RandomErasing.get_params(img, self.scale, self.ratio)
        img_tensor = torch.tensor(np.transpose(np.asarray(img), (2, 0, 1)))
        new_img = F.erase(img_tensor, i, j, h, w, v)
        new_img = new_img.numpy()
        new_img = Image.fromarray(new_img)
        return new_img, target

class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img, target):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

        return img, target

class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]

        if "lines" in target:
            lines = target["lines"]
            lines = lines / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["lines"] = lines

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
