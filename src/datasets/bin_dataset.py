import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np

def visualize_xyz(xyz):
    img = np.moveaxis(xyz, 0, -1)

    non_zeros = np.prod(img, axis=2) != 0
    points = img[non_zeros]

    zeros = np.prod(img, axis=2) == 0
    img -= np.min(img)
    img /= np.max(img)
    img[zeros] = np.array([0. , 0. , 0.])

    plt.imshow(img)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], marker='o')

    plt.show()

def visualize_xyz_with_lines(xyz, lines):
    img = np.moveaxis(xyz, 0, -1)

    non_zeros = np.prod(img, axis=2) != 0
    points = img[non_zeros]

    zeros = np.prod(img, axis=2) == 0
    img -= np.min(img)
    img /= np.max(img)
    img[zeros] = np.array([0. , 0. , 0.])

    plt.imshow(img)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], marker='o')

    for line in lines:
        #line = np.array([
        #    np.concatenate((line[:3], np.array([1.0]))),
        #    np.concatenate((line[3:], np.array([1.0]))),
        #])
        print(line)
        x1, y1, z1, x2, y2, z2 = line
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='b', marker='o')
    plt.show()

def get_canonical_transform(transform):
    """
    Unused - Takes rotation matrix and finds canonical representation w.r.t. symmetries as per:
    https://arxiv.org/pdf/1908.07640.pdf check eq (22) for this case specifically
    """

    rot = transform[:3, :3]

    # we need to consider only one symmetry e.g. 180 deg around z axis
    sym_rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    if np.linalg.norm(sym_rot @ rot - np.eye(3), ord='fro') < np.linalg.norm(rot - np.eye(3), ord='fro'):
        sym_rot_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        regressor = 1
        if np.linalg.norm(sym_rot @ rot - sym_rot_90, ord='fro') < np.linalg.norm(rot - sym_rot_90, ord='fro'):
            rot = sym_rot @ rot
    else:
        regressor = 0

    transform[:3, :3] = rot
    return transform, np.array([regressor], dtype=np.float32)


class Dataset(Dataset):
    def __init__(self, args, path, split,
                 width, height, keep_dim_aspect_ratio=True, preload=True, use_resize_cache=True,
                 cutout_prob=0.0, cutout_inside=True,
                 max_cutout_size=0.8, min_cutout_size=0.2,
                 noise_sigma=None, t_sigma=0.0, random_rot=False):
        self.dataset_dir = os.path.dirname(path)
        self.split = split
        self.width = width
        self.height = height
        self.keep_dim_aspect_ratio = keep_dim_aspect_ratio
        self.use_resize_cache = use_resize_cache
        self.preload = preload
        self.noise_sigma = noise_sigma
        self.t_sigma = t_sigma
        self.random_rot = random_rot

        self.cutout_prob = cutout_prob
        self.use_cutout = cutout_prob > 0.0
        self.cutout_inside = cutout_inside
        self.max_cutout_size = max_cutout_size
        self.min_cutout_size = min_cutout_size

        self.used_size = None

        if self.split != 'train' and self.cutout_prob > 0.0:
            print("***** Split is not train, but cutout is enabled! *****")

        print("Loading dataset from path: ", path)
        with open(path, 'r') as f:
            self.entries = json.load(f)

        # convert paths to host format
        for i in range(len(self.entries)):
            for p in {'exr_normals_path', 'exr_positions_path', 'txt_path'}:
                self.entries[i][p] = os.path.join(*self.entries[i][p].split('\\'))

        for i in range(len(self.entries)):
            # load lines
            self.entries[i]['lines'] = self.load_lines(self.entries[i])
            # add synthetic flag
            self.entries[i]['synthetic'] = self.is_synthetic(self.entries[i])
            # add ids
            self.entries[i]['sample_id'] = i

        if 'train' not in path and 'val' not in path:
            if self.split == 'train':
                self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 != 0]
            elif self.split == 'val':
                self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 == 0]

        if args.bins_pick_samples != 'all':
            picked = []
            print('picking', args.bins_pick_samples, 'samples')
            for i in range(len(self.entries)):
                if self.entries[i]['synthetic'] is True and args.bins_pick_samples == 'synthetic':
                    picked.append(self.entries[i])
                elif self.entries[i]['synthetic'] is False and args.bins_pick_samples == 'real':
                    picked.append(self.entries[i])
            self.entries = picked
            print('Picked', len(self.entries), 'samples')

        if args.bins_subsample_batch < 1.0:
            reduced = []
            for i in range(len(self.entries)):
                if np.random.rand() < args.bins_subsample_batch:
                    reduced.append(self.entries[i])
            #reduced = [self.entries[0], self.entries[1]]
            #reduced = [self.entries[0]]
            self.entries = reduced

        print("Split: ", self.split)
        print("Size: ", len(self))
        if self.preload:
            print("Preloading exrs to memory")
            for entry in self.entries:
                print(entry)
                entry['xyz'] = self.load_xyz(entry)


        self.means = [-7.317206859588623, -7.509462833404541, 621.6871337890625]
        self.stds = [222.503662109375, 165.90419006347656, 681.2403564453125]
        #self.compute_normalization_constants()

    def get_nomalization_constants(self):
        return self.means, self.stds

    def compute_normalization_constants(self):
        xyzs = []
        for i in range(len(self.entries)):
            xyz, _ = self.__getitem__(i)
            xyzs.append(xyz.tolist())
        xyzs = torch.tensor(xyzs)
        means = [xyzs[:, 0, :, :].mean().item(), xyzs[:, 1, :, :].mean().item(), xyzs[:, 2, :, :].mean().item()]
        stds = [xyzs[:, 0, :, :].std().item(), xyzs[:, 1, :, :].std().item(), xyzs[:, 2, :, :].std().item()]

        self.means = means
        self.stds = stds

        print('Per channel means:', means)
        print('Per channel stds:', stds)

    def cutout(self, xyz):
        mask_width = np.random.randint(int(self.min_cutout_size * self.width), int(self.max_cutout_size * self.width))
        mask_height = np.random.randint(int(self.min_cutout_size * self.height), int(self.max_cutout_size * self.height))

        mask_width_half = mask_width // 2
        offset_width = 1 if mask_width % 2 == 0 else 0

        mask_height_half = mask_height // 2
        offset_height = 1 if mask_height % 2 == 0 else 0

        xyz = xyz.copy()

        h, w = self.height, self.width

        if self.cutout_inside:
            cxmin, cxmax = mask_width_half, w + offset_width - mask_width_half
            cymin, cymax = mask_height_half, h + offset_height - mask_height_half
        else:
            cxmin, cxmax = 0, w + offset_width
            cymin, cymax = 0, h + offset_height

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_width_half
        ymin = cy - mask_height_half
        xmax = xmin + mask_width
        ymax = ymin + mask_height
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        xyz[:, ymin:ymax, xmin:xmax] = 0.0

        return xyz

    def __len__(self):
        """
        Length of dataset
        :return: number of elements in dataset
        """
        return len(self.entries)

    def is_synthetic(self, entry):
        sample_dir = os.path.dirname(entry['exr_positions_path'])
        flag_file = os.path.join(self.dataset_dir, sample_dir, 'synthetic')
        return os.path.exists(flag_file)

    def load_lines(self, entry):
        sample_dir = os.path.dirname(entry['exr_positions_path'])
        lines_file = os.path.join(self.dataset_dir, sample_dir, 'bin_lines.txt')
        lines = np.loadtxt(lines_file)
        return lines
        
    def get_transformed_lines(self, lines, transform):
        #lines = entry['lines']
        
        starts = np.c_[(lines[:, :3], np.ones(lines.shape[0]).T)]
        ends = np.c_[(lines[:, 3:], np.ones(lines.shape[0]).T)]
        
        t_starts = transform @ starts.T
        t_ends = transform @ ends.T
        
        return np.hstack((t_starts.T[:, :3], t_ends.T[:, :3]))

    def get_resized_size(self, orig_width, orig_height):
        width = self.width
        height = self.height
        if self.keep_dim_aspect_ratio:
            if orig_width > orig_height:
                width = self.width
                height = int(orig_height * (self.width / orig_width))
            else:
                width = int(orig_width * (self.height / orig_height))
                height = self.height
        return width, height



    def load_xyz(self, entry):
        """
        Loads pointcloud for a given entry
        :param entry: entry from self.entries
        :return: pointcloud wit shape (3, height, width)
        """
        exr_path = None
        xyz = None
        cached = False
        if self.use_resize_cache:
            exr_path = os.path.join(self.dataset_dir, f'_cache_{self.width}x{self.height}', entry['exr_positions_path'])
            if os.path.exists(exr_path):
                xyz = cv2.imread(exr_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if xyz is None:
                    print(exr_path)
                    raise ValueError("Image at path ", exr_path)
                #print('Loaded resized sample from cache!')
                cached = True

        if xyz is None:
            #print('reading original')
            exr_path = os.path.join(self.dataset_dir, entry['exr_positions_path'])
            xyz = cv2.imread(exr_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if xyz is None:
                print(exr_path)
                raise ValueError("Image at path ", exr_path)

        width, height = self.get_resized_size(xyz.shape[1], xyz.shape[0])
        if self.used_size is not None:
            if width != self.used_size[0] or height != self.used_size[1]:
                raise ValueError("Image at path ", exr_path, "has different aspect ratio")
        else:
            self.used_size = (width, height)
            #print(f'Corrected width and height aspect ratio. {xyz.shape[1]} x {xyz.shape[0]} -> {width} x {height}')

        if xyz.shape[1] != width or xyz.shape[0] != height:
            xyz = cv2.resize(xyz, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)

        if self.use_resize_cache and cached is False:
            exr_path = os.path.join(self.dataset_dir, f'_cache_{width}x{height}', entry['exr_positions_path'])
            exr_dir = os.path.dirname(exr_path)
            Path(exr_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(exr_path, xyz)
            #print('Saved resized sample to cache')

        xyz = np.transpose(xyz, [2, 0, 1])
        return xyz

    def get_aug_transform(self):
        """
        Generates random transformation using. R is from SO(3) thanks to QR decomposition.
        :return: random transformation matrix
        """
        if self.random_rot:
            R, _ = np.linalg.qr(np.random.randn(3, 3))
        else:
            R = np.eye(3)

        t = self.t_sigma * np.random.randn(3)

        out = np.zeros([4, 4])
        out[:3, :3] = R
        out[:3, 3] = t
        out[3, 3] = 1

        #if np.random.rand() < 0.5:
        #    out[0, 0] = out[0, 0] * -1

        return out

    def aug(self, xyz_gt, transform):
        """
        Applies transformation matrix to pointcloud
        :param xyz_gt: original pointcloud with shape (3, height, width)
        :param transform: (4, 4) transformation matrix
        :return: Transformed pointcloud with shape (3, height, width)
        """
        orig_shape = xyz_gt.shape
        xyz = np.reshape(xyz_gt, [-1, 3])
        xyz = np.concatenate([xyz, np.ones([xyz.shape[0], 1])], axis=-1)

        xyz_t = (transform @ xyz.T).T

        xyz_t = xyz_t[:, :3] / xyz_t[:, 3, np.newaxis]
        xyz_t = np.reshape(xyz_t, orig_shape)
        return xyz_t

    def normalize_lines(self, lines):
        div = torch.tensor(self.stds + self.stds)
        sub = torch.tensor(self.means + self.means) / div
        #print('sub:', sub)
        #print('div:', div)
        return (lines / div) - sub


    def normalize_xyz(self, xyz):
        x_mean, y_mean, z_mean = self.means
        x_std, y_std, z_std = self.stds

        sub = torch.tensor([x_mean/x_std, y_mean/y_std, z_mean/z_std], dtype=torch.float32).view(3, 1, 1)
        div = torch.tensor([x_std, y_std, z_std], dtype=torch.float32).view(3, 1, 1)
        return (xyz / div) - sub

    def __getitem__(self, index):
        """
        Returns one sample for training
        :param index: index of entry
        :return: dict containing sample data
        """
        entry = self.entries[index]

        gt_transform = np.array(entry['proper_transform'])
        orig_transform = np.array(entry['orig_transform'])

        if gt_transform[0, 1] < 0.0:
            gt_transform[:, :2] *= -1

        if self.split == 'train':
            aug_transform = self.get_aug_transform()
            transform = aug_transform @ gt_transform
        else:
            transform = gt_transform

        transform = transform.astype(np.float32)

        rot = Rotation.from_matrix(transform[:3, :3])
        rotvec = torch.from_numpy(rot.as_rotvec())
        t = torch.from_numpy(transform[:3, 3])

        if self.preload:
            xyz = entry['xyz']
        else:
            xyz = self.load_xyz(entry)

        if self.split == 'train':
            xyz = self.aug(xyz, aug_transform)

        xyz = xyz.astype(np.float32)

        if self.noise_sigma is not None:
            xyz += self.noise_sigma * np.random.randn(*xyz.shape)

        if self.use_cutout:
            if np.random.rand() < self.cutout_prob:
                xyz = self.cutout(xyz)

        #visualize_xyz(xyz)

        #return {'xyz': xyz, 'bin_rotvec': rotvec, 'bin_translation': t, 'bin_transform': torch.from_numpy(transform),
        #        'orig_transform': torch.from_numpy(orig_transform), 'txt_path': entry['txt_path']}

        target = {}
        lines = entry['lines']#[[0, 2]]
        target['image_id'] = torch.tensor(entry['sample_id'])
        target['labels'] = torch.tensor([0 for _ in lines], dtype=torch.int64)
        target['area'] = torch.tensor([1 for _ in lines])
        target['iscrowd'] = torch.tensor([0 for _ in lines])
        target['lines'] = torch.tensor(self.get_transformed_lines(lines, transform), dtype=torch.float32)
        #target['exr_file'] = entry['exr_positions_path']

        #print('Lines are', target['lines'])
        #self.normalize_lines(target['lines'])
        #exit()
        #return self.normalize(torch.tensor(xyz)), target

        #xyz = torch.tensor(xyz)

        if True:
            xyz = self.normalize_xyz(torch.tensor(xyz))
            target['lines'] = self.normalize_lines(target['lines'])
            entry['normalized'] = {'means': self.means, 'stds': self.stds}
        else:
            xyz = torch.tensor(xyz)
            entry['normalized'] = {'means': [0.0, 0.0, 0.0], 'stds': [1.0, 1.0, 1.0]}

        #print('target[\'lines\']', target['lines'][:2].view(2, 6))
        #print(target['lines'][:1].shape)
        #exit()
        #target['lines'] = target['lines'][:2].view(2, 6)

        #print('xyz shape', xyz.shape)

        #visualize_xyz_with_lines(xyz.numpy(), target['lines'].numpy())

        #exit()

        #return torch.tensor(self.load_xyz(entry)), target
        return xyz, target, entry

def build_bins(image_set, args):
    if image_set == 'train':
        return Dataset(args, args.bins_path, 'train', args.bins_input_width, args.bins_input_height,
                            cutout_prob=args.bins_cutout_prob, cutout_inside=args.bins_cutout_inside,
                            max_cutout_size=args.bins_cutout_max_size, min_cutout_size=args.bins_cutout_min_size,
                            noise_sigma=args.bins_noise_sigma, t_sigma=args.bins_t_sigma, random_rot=args.bins_random_rot,
                            preload=not args.bins_no_preload)
    elif image_set == 'val':
        return Dataset(args, args.bins_path, 'val', args.bins_input_width, args.bins_input_height, preload=not args.bins_no_preload)
    elif image_set == 'test':
        return Dataset(args, args.bins_path, 'test', args.bins_input_width, args.bins_input_height, preload=not args.bins_no_preload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='Path to dataset json file.')
    args = parser.parse_args()
    json_path = args.json

    dataset = Dataset(json_path, 'train', 258, 193, preload=False, noise_sigma=0.0, random_rot=True)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for item in data_loader:
        print(item['xyz'].size())
        xyz = item['xyz'][0].cpu().detach().numpy()

        print(np.mean(xyz))

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel(), marker='o')

        #plt.show()
