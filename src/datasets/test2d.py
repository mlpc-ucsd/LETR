import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

class Test2D(Dataset):
    def __init__(self, args):
        self.entries = []
        w, h = 640, 480
        im = Image.new(mode="RGB", size=(w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.line((170, 390, 470, 390), fill=(0, 0, 0), width=8)
        draw.line((170, 390, 170, 90), fill=(0, 0, 0), width=8)
        #im.show()

        t = torch.Tensor(np.asarray(im, dtype=float)) / 255
        t = torch.moveaxis(t, 2, 0)
        e = (t, {
            'image_id': torch.tensor(1),
            'labels': torch.tensor([0, 0]),
            'area': torch.tensor([1, 1]),
            'iscrowd': torch.tensor([0, 0]),
            'lines': torch.Tensor([
                [170/w, 390/h, 470/w, 390/h],
                [170/w, 390/h, 170/w, 90/h]
            ])
        })
        self.entries.append(e)

        im = Image.new(mode="RGB", size=(w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.line((170, 90, 470, 90), fill=(0, 0, 0), width=8)
        draw.line((470, 390, 470, 90), fill=(0, 0, 0), width=8)
        #im.show()

        t = torch.Tensor(np.asarray(im, dtype=float)) / 255
        t = torch.moveaxis(t, 2, 0)
        e = (t, {
            'image_id': torch.tensor(2),
            'labels': torch.tensor([0, 0]),
            'area': torch.tensor([1, 1]),
            'iscrowd': torch.tensor([0, 0]),
            'lines': torch.Tensor([
                [170/w, 90/h, 470/w, 90/h],
                [470/w, 390/h, 470/w, 90/h]
            ])
        })
        self.entries.append(e)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


def build_test2d(image_set, args):
    return Test2D(args)


if __name__ == '__main__':
    dataset = Test2D({})
    print(dataset[0])
