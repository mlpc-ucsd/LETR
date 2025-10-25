import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d


class Annotation():
    def __init__(self, dataset_root, matching_name, rgb_file_name):
        sample_name = os.path.basename(rgb_file_name).split('.')[0]
        annotation_path = os.path.join(dataset_root, 'matched_poses', matching_name, sample_name + '.json')
        with open(annotation_path) as f:
            self.annotation = json.load(f)

    def tracked_objects_ids(self):
        return [obj['id'] for obj in self.annotation['tracked_objects']]

    def _compute_rotation_mat_from_pose(self, pose):
        return R.from_euler('xyz', pose['euler_angles_deg'], degrees=True).as_matrix()

    def get_transform(self, object_id):
        if str(object_id) not in self.annotation['matched_objects']:
            raise(Exception('This image do not have match for object with id' + int(object_id)))

        poses = self.annotation['matched_objects'][str(object_id)]
        poses = sorted(poses, key=lambda p: p['pose_ts'])

        if len(poses) > 1:
            if self.annotation['rgb_ts'] >= poses[0]['pose_ts'] and self.annotation['rgb_ts'] <= poses[-1]['pose_ts']:
                key_times = [p['pose_ts'] for p in poses]
                key_rots = R.from_matrix([ self._compute_rotation_mat_from_pose(p) for p in poses ])
                slerp = Slerp(key_times, key_rots)
                r = slerp(self.annotation['rgb_ts']).as_matrix()

                key_positions = np.array([p['center_3d'] for p in poses])
                t = interp1d(key_times, key_positions, axis=0, kind='linear')(self.annotation['rgb_ts'])
            else:
                raise(Exception('Now what?'))
        elif len(poses) == 1:
            r = self._compute_rotation_mat_from_pose(poses[0])
            t = np.array(poses[0]['center_3d'])
        else:
            raise(Exception('Missing poses for object ' + object_id))

        transform = np.eye(4)
        transform[:3, :3] = r
        transform[:3, 3] = t
        return transform

class Dataset(Dataset):
    def __init__(self, args, path, split,
                 width, height, matching_name = 'best_of_two_nearest_and_avg', lines_name = 'outside', keep_dim_aspect_ratio=True, preload=True, use_resize_cache=True):
        self.dataset_dir = os.path.dirname(path)
        self.split = split
        self.width = width
        self.height = height
        self.keep_dim_aspect_ratio = keep_dim_aspect_ratio
        self.use_resize_cache = use_resize_cache

        self.dataset_root = os.path.dirname(path)
        self.rgbs_dir = 'extracted_rgb'
        self.poses_dir = 'matched_poses'
        self.lines_dir = 'obj_lines'
        self.camera_params_dir = 'camera_params'
        self.matching_name = matching_name
        self.lines_name = lines_name
        self.preload = preload
        self.used_size = None

        self.normalize_output_3d_space = False

        print("Loading dataset from path: ", path)
        with open(path, 'r') as f:
            self.entries = json.load(f)[split]

        if args.mtevents_rgb_subsample_batch < 1.0:
            reduced = []
            for i in range(len(self.entries)):
                if np.random.rand() < args.mtevents_rgb_subsample_batch:
                    reduced.append(self.entries[i])
            #reduced = [self.entries[0], self.entries[1]]
            #reduced = [self.entries[0]]
            self.entries = reduced
            print('Reduced sample count to', len(self.entries))

        print("Loading annotations")
        for i in range(len(self.entries)):
            # load lines
            self.entries[i]['lines'] = self.load_lines(self.entries[i])
            self.entries[i]['transform'] = self.load_transform(self.entries[i])
            self.entries[i]['sample_id'] = i
            # not used for training but for prediction visualization
            self.entries[i]['camera_params'] = self.load_camera_params(self.entries[i]['img_path'])

        print("Split: ", self.split)
        print("Size: ", len(self))
        if self.preload:
            print("Preloading exrs to memory")
            i, total = 0, len(self.entries)
            p, pc = int(total * 0.1), 0
            for entry in self.entries:
                #print(entry)
                entry['rgb'] = self.load_rgb(entry)
                if i == p:
                    pct = ((i+1)/total)*100
                    print(f'{pct:.2f} %')
                    pc += 1
                    p = int(total * 0.1 * (pc+1))-1
                i += 1


        # for RGB normalization
        self.means = [125.834, 134.014, 134.206]
        self.stds = [67.9709, 66.5856, 62.999]
        # for prediction 3D space normalization
        self.means3d = [-0.04032357, -0.14958457, 5.70839085]
        self.stds3d = [0.74611402, 0.3836728,  1.21448497]

    def get_nomalization_constants(self):
        return self.means, self.stds, self.means3d, self.stds3d

    def __len__(self):
        """
        Length of dataset
        :return: number of elements in dataset
        """
        return len(self.entries)

    def load_camera_params(self, img_path):
        scene_name = img_path.split('/')[-1].split('_')[0]
        camera_params_json_path = os.path.join(self.dataset_root, self.camera_params_dir, scene_name + '.json')
        with open(camera_params_json_path) as f:
            return json.load(f)

    def load_entry_annotation(self, entry):
        sample_name = entry['img_path'].split('.')[0]
        annotation_path = os.path.join(self.dataset_root, self.poses_dir, self.matching_name, sample_name + '.json')
        with open(annotation_path) as f:
            return json.load(f)

    def load_transform(self, entry):
        annotation = Annotation(self.dataset_root, self.matching_name, entry['img_path'])
        return annotation.get_transform(entry['object_id'])

    def load_lines(self, entry):
        lines_file = os.path.join(self.dataset_root, self.lines_dir, self.lines_name,  f'obj_{entry['object_id']:06}.txt')
        lines = np.loadtxt(lines_file)
        return lines / 1000

    def get_transformed_lines(self, lines, transform):
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



    def load_rgb(self, entry):
        """
        Loads rgb matrix for a given entry
        :param entry: entry from self.entries
        :return: rgb matrix with shape (3, height, width)
        """
        img_path = None
        rgb = None
        cached = False
        if self.use_resize_cache:
            img_path = os.path.join(self.dataset_root, f'_cache_{self.width}x{self.height}', entry['img_path'])
            if os.path.exists(img_path):
                rgb = cv2.imread(img_path)
                if rgb is None:
                    print(img_path)
                    raise ValueError("Image at path ", img_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                #print('Loaded resized sample from cache!')
                cached = True

        if rgb is None:
            #print('reading original')
            img_path = os.path.join(self.dataset_root, self.rgbs_dir, entry['img_path'])
            rgb = cv2.imread(img_path)
            if rgb is None:
                print(img_path)
                raise ValueError("Image at path ", img_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        width, height = self.get_resized_size(rgb.shape[1], rgb.shape[0])
        if self.used_size is not None:
            if width != self.used_size[0] or height != self.used_size[1]:
                raise ValueError("Image at path ", img_path, "has different aspect ratio")
        else:
            self.used_size = (width, height)
            #print(f'Corrected width and height aspect ratio. {rgb.shape[1]} x {rgb.shape[0]} -> {width} x {height}')

        img_resized = False
        if rgb.shape[1] != width or rgb.shape[0] != height:
            rgb = cv2.resize(rgb, (width, height))#, interpolation=cv2.INTER_NEAREST_EXACT)
            img_resized = True

        if self.use_resize_cache and cached is False and img_resized is True:
            img_path = os.path.join(self.dataset_dir, f'_cache_{width}x{height}', entry['img_path'])
            img_dir = os.path.dirname(img_path)
            Path(img_dir).mkdir(parents=True, exist_ok=True)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, rgb)
            #print('Saved resized sample to cache')

        rgb = np.transpose(rgb, [2, 0, 1])
        return rgb

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

    def normalize_lines(self, lines):
        div = torch.tensor(self.stds3d + self.stds3d)
        sub = torch.tensor(self.means3d + self.means3d) / div
        #print('sub:', sub)
        #print('div:', div)
        return (lines / div) - sub


    def normalize_rgb(self, rgb):
        r_mean, g_mean, b_mean = self.means
        r_std, g_std, b_std = self.stds

        sub = torch.tensor([r_mean/r_std, g_mean/g_std, b_mean/b_std], dtype=torch.float32).view(3, 1, 1)
        div = torch.tensor([r_std, g_std, b_std], dtype=torch.float32).view(3, 1, 1)
        return (rgb / div) - sub

    def save_prediction_data(model_output):
        pass

    def save_prediction_visualization(model_output):
        pass

    def __getitem__(self, index):
        """
        Returns one sample for training
        :param index: index of entry
        :return: dict containing sample data
        """
        entry = self.entries[index]

        transform = np.array(entry['transform'])
        transform = transform.astype(np.float32)

        if self.preload:
            rgb = entry['rgb']
        else:
            rgb = self.load_rgb(entry)

        rgb = rgb.astype(np.float32)

        target = {}
        lines = entry['lines']
        target['image_id'] = torch.tensor(entry['sample_id'])
        target['labels'] = torch.tensor([0 for _ in lines], dtype=torch.int64)
        target['area'] = torch.tensor([1 for _ in lines])
        target['iscrowd'] = torch.tensor([0 for _ in lines])
        target['lines'] = torch.tensor(self.get_transformed_lines(lines, transform), dtype=torch.float32)

        rgb = self.normalize_rgb(torch.tensor(rgb))
        if self.normalize_output_3d_space:
            target['lines'] = self.normalize_lines(target['lines'])

        entry['normalized'] = {
            'means': self.means, 'stds': self.stds,
            'means3d': self.means3d, 'stds3d': self.stds3d
        }

        return rgb, target, entry

def build_mtevents_rgb(image_set, args):
    if image_set == 'train':
        return Dataset(args, args.mtevents_rgb_dataset_json_path, 'train', args.mtevents_rgb_input_width, args.mtevents_rgb_input_height,
                            preload=not args.mtevents_rgb_no_preload)
    elif image_set == 'val':
        return Dataset(args, args.mtevents_rgb_dataset_json_path, 'val', args.mtevents_rgb_input_width, args.mtevents_rgb_input_height, preload=not args.mtevents_rgb_no_preload)
    elif image_set == 'test':
        return Dataset(args, args.mtevents_rgb_dataset_json_path, 'test', args.mtevents_rgb_input_width, args.mtevents_rgb_input_height, preload=not args.mtevents_rgb_no_preload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='Path to dataset json file.')
    args = parser.parse_args()
    json_path = args.json

    args.mtevents_rgb_subsample_batch = 1.0

    dataset = Dataset(args, json_path, 'val', 800, 600, preload=False)

    #print(dataset[0])

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for item in data_loader:
        #print(item)

        rgbs, target, entries = item

        rgb = rgbs[0]
        rgb_orig = dataset.load_rgb({'img_path': entries['img_path'][0]})
        camera_params = dataset.load_camera_params(entries['img_path'][0])
        lines = target['lines'][0]
        print('orig lines', entries['lines'][0])
        print('transformed lines', lines)
        lines = lines.reshape((8, 3))

        orig_width, orig_height = entries['original_size']['width'][0], entries['original_size']['height'][0] #entries['original_size'].values()

        camera_matrix = np.array(camera_params['camera_mtx_cam1'])
        distortion_coefficients = np.array(camera_params['distortion_coeffs_cam1'])

        lines_2d, _ = cv2.projectPoints(lines.numpy(), np.eye(3), np.zeros(3), camera_matrix,
                                     distortion_coefficients)

        img = np.moveaxis(rgb_orig, 0, -1)
        img_h, img_w = img.shape[:2]
        sx = img_w / orig_width
        sy = img_h / orig_height
        lines_2d = lines_2d * np.array([sx, sy])
        for i in range(4):
            p1 = tuple(np.round(lines_2d[2*i][0]).astype(int))
            p2 = tuple(np.round(lines_2d[2*i+1][0]).astype(int))

            print('Line from', p1, 'to', p2)

            if 0 <= p1[0] < img_w and 0 <= p1[1] < img_h and 0 <= p2[0] < img_w and 0 <= p2[1] < img_h:
                cv2.line(img, p1, p2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                print('drawing')


        print(entries['img_path'][0])
        print('object id', entries['object_id'][0])

        plt.imshow(rgb.moveaxis(0, -1))
        plt.imshow(np.moveaxis(rgb_orig, 0, -1))
        plt.show()



        #print(item['xyz'].size())
        #xyz = item['xyz'][0].cpu().detach().numpy()

        #print(np.mean(xyz))

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel(), marker='o')

        #plt.show()
        break
