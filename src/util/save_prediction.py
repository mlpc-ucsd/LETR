import numpy as np
import os
import json
import torch.nn.functional as F
from view_prediction import view2D, view3D

def read_transform_file(file):
    with open(file, 'r') as tfile:
        P = tfile.readline().strip().split(' ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                      [float(P[1]), float(P[5]), float(P[9])],
                      [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
        return R, t

def jsonify_value(v):
    tolist = getattr(v, 'tolist', None)
    if callable(tolist):
        return tolist()
    else:
        return v

def save_prediction_data(samples, outputs, targets,
                         entry, out_filename,
                         criterion=None, index=0):
    out_logits, out_line = outputs['pred_logits'][index].detach().cpu(), outputs['pred_lines'][index].detach().cpu()

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    lines = out_line
    xyz, _ = samples.decompose()

    #ddir = os.path.dirname(bins_path)
    #gt_R1, gt_T = read_transform_file(os.path.join(ddir, entry[index]['txt_path']))
    #gt_R2 = np.matrix.copy(gt_R1)
    #gt_R2[:, :2] *= -1

    entry[index]['lines'] = entry[index]['lines'].tolist()
    data = {
        'entry': {k:jsonify_value(v) for (k, v) in entry[index].items() if k != 'xyz'},
        #'gt_transform': {
        #    'R1': gt_R1.tolist(),
        #    'R2': gt_R2.tolist(),
        #    'T': gt_T.tolist(),
        #},
        'targets': {key: value.tolist() for (key,value) in targets[index].items()},
        'prediction': {
            'scores': scores.tolist(),
            'labels': labels.tolist(),
            'lines': lines.tolist(),
        },
        'xyz': xyz[index].tolist(),
    }

    with open(out_filename, 'w') as f:
        json.dump(data, f)


def save_prediction_visualization(samples, outputs, filename, entry, index=0):
    # find lines
    out_logits, out_line = outputs['pred_logits'][index], outputs['pred_lines'][index]
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    lines = out_line.detach().cpu()
    scores = scores.detach().cpu().numpy()
    keep = np.array(np.argsort(scores)[::-1][:12])
    lines = lines[keep]

    #target_lines = targets[index]['lines']

    xyz, _ = samples.decompose()
    xyz = xyz[index].detach().cpu().tolist()
    #print(xyz.shape)
    #print('xyz is ', xyz)
    #exit()
    if lines.shape[1] == 4:
        plt = view2D(xyz, lines, [])
    else:
        plt, _ = view3D(np.array(xyz), lines, [], entry[index])
    plt.savefig(filename)
    plt.close()
