import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.letr import build
import argparse
from args import get_args_parser
from datasets import build_dataset
import util.misc as utils
from util.save_prediction import save_prediction_data, save_prediction_visualization
from util.calculate_pose import calculate_pose
import json

def save_json(obj, file):
    json_object = json.dumps(obj, indent=4)
    with open(file, "w") as outfile:
        outfile.write(json_object)

def read_transform_file(file):
    #print(file)
    with open(file, 'r') as tfile:
        P = tfile.readline().strip().split(' ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                      [float(P[1]), float(P[5]), float(P[9])],
                      [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
        return R, t


def calculate_eTE(gt_t, pr_t):
    return np.linalg.norm((pr_t-gt_t), ord=2)/10


def calculate_eRE(gt_R, pr_R):
    numerator = np.trace(np.matmul(gt_R, np.linalg.inv(pr_R))) - 1
    numerator = np.clip(numerator, -2, 2)
    return np.arccos(numerator/2)


def get_bin_z_offset(bin_dir):
    dir2type = {
        'TestBin': 3,
        'TestCarton': 7,
        'TestGold': 3,
        'TestGray': 0,
        'TestSynth': 5,
        'dataset0': 0,
        'dataset1': 1,
        'dataset2': 0,
        'dataset3': 2,
        'dataset4': 2,
        'ElavatedGrayBox': 0,
        'ElevatedGreyBox': 0,
        'ElevatedGreyFullBeer': 0,
        'FirstRealSet': 3,
        'GoldBinAdditional': 3,
        'GrayBoxPad': 0,
        'LargeWoodenBoxDynamic': 2,
        'LargeWoodenBoxStatic': 2,
        'ShallowGreyBox': 0,
        'SmalGreyBasket': 4,
        'SmallGoldenBox': 3,
        'SmallWhiteBasket': 1,
        'synth_dataset5_random_origin': 5,
        'synth_dataset6_random_origin': 6,
        'synth_dataset7_random_origin': 5,
        'synth_dataset8_random_origin': 5,
    }
    type2offset = {
        0: 67.8125,
        1: 95.115,
        2: 160.398,
        3: 75.6255,
        4: 119.425,
        5: 108.75,
        6: 56.25,
        7: 48.0,
    }
    return type2offset[dir2type[bin_dir]]


def create_letr(args2):
    print(f'create_letr({args2.model})')
    # obtain checkpoints
    checkpoint = torch.load(args2.model, weights_only=False, map_location='cpu')

    # load model
    args = checkpoint['args']

    print ('creating letr with args', args)

    if args2.set_cuda is False:
        args.device = 'cpu'
    else:
        args.device = 'cuda'
    model, _, _ = build(args)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def main(args):
    model = create_letr(args)

    args.bins_no_preload = True
    dataset = build_dataset('bins', args.split, args)
    dataset_dir = os.path.dirname(args.bins_path)

    means, stds = dataset.get_nomalization_constants()

    i = 0

    eTEs = []
    eREs = []

    data_loader = DataLoader(dataset, args.batch_size, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.set_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    for xyz, targets, entries in data_loader:

        xyz = xyz.to(device)
        targets = [{k: (v.to(device) if k != 'exr_file' else v) for k, v in t.items()} for t in targets]

        outputs = model(xyz)
        for index in range(len(entries)):
            entry = entries[index]

            pred_lines = outputs[0]['pred_lines'][index].detach().cpu().numpy()
            pred_logits = outputs[0]['pred_logits'][index].detach().cpu()
            pred_scores, _ = F.softmax(pred_logits, -1)[..., :-1].max(-1)
            pred_scores = pred_scores.numpy()

            # undo normalization
            pred_lines = ((pred_lines.reshape(-1,3) * stds) + means).reshape(-1, 6)

            R, T = calculate_pose(pred_lines, pred_scores, get_bin_z_offset(entry['dir'])*2)

            gt_R1, gt_T = read_transform_file(os.path.join(dataset_dir, entry['txt_path']))
            gt_R2 = np.matrix.copy(gt_R1)
            gt_R2[:, :2] *= -1


            eRE = min(calculate_eRE(gt_R1, R), calculate_eRE(gt_R2, R))
            eTE = calculate_eTE(gt_T, T)

            print(f"eval_vis/eval_{i:03}.json [{entry['txt_path']}], synthetic ",  entry['synthetic'],': eTE =', eTE, ', eRE =', eRE)

            eTEs.append(eTE)
            eREs.append(eRE)

            sample, _, _ = dataset[i]
            sample = utils.nested_tensor_from_tensor_list([sample])

            if args.save_prediction_data:
                save_prediction_data(xyz, outputs[0], targets, entries, args.bins_path, f'eval_vis/eval_{i:03}.json', index=index)
            if args.save_prediction_visualization:
                save_prediction_visualization(xyz, outputs[0], f'eval_vis/eval_{i:03}.png', entries, index=index)

            i+=1
    print('mean eTE:', np.array(eTEs).mean(), 'mean eRE:', np.array(eREs).mean())
    save_json({'eTE': eTEs, 'eRE': eREs}, 'errs.json')

if __name__ == '__main__':

    parser = argparse.ArgumentParser('LETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--split', type=str, choices=('test', 'val'), default='test')
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_png_visualization', action='store_true', default=False)
    parser.add_argument('--set_cuda', action='store_true', default=False)
    #parser.add_argument('--save_prediction_data', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
