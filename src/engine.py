"""
Train and eval functions used in main.py

modified based on https://github.com/facebookresearch/detr/blob/master/engine.py
"""
import math
import os
import sys
from typing import Iterable
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import datetime
import util.misc as utils

def train_one_epoch(model, criterion, postprocessors, data_loader, optimizer, device, epoch, max_norm, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10


    counter = 0
    torch.cuda.empty_cache()
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        try:
            if args.LETRpost:
                outputs, origin_indices = model(samples, postprocessors, targets, criterion)
                loss_dict = criterion(outputs, targets, origin_indices)
            else:
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
        except RuntimeError as e:
            if "out of memory" in str(e):
                sys.exit('Out Of Memory')   
            else:
                raise e
            
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v   for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]  for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    id_to_img = {}
    f = open(os.path.join(args.coco_path, "annotations", "lines_{}2017.json".format(args.dataset)))
    data = json.load(f)
    for d in data['images']:
        id_to_img[d['id']] = d['file_name'].split('.')[0]
    counter = 0
    num_images = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.LETRpost:
            outputs, origin_indices = model(samples, postprocessors, targets, criterion)
            loss_dict = criterion(outputs, targets, origin_indices)
        else:
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        if args.benchmark:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
            results = postprocessors['line'](outputs, orig_target_sizes, "prediction")

            pred_logits = outputs['pred_logits']
            bz = pred_logits.shape[0]
            assert bz ==1 
            query = pred_logits.shape[1]

            rst = results[0]['lines']
            pred_lines = rst.view(query, 2, 2)

            pred_lines = pred_lines.flip([-1]) # this is yxyx format

            h, w = targets[0]['orig_size'].tolist()
            pred_lines[:,:,0] = pred_lines[:,:,0]*(128)   
            pred_lines[:,:,0] = pred_lines[:,:,0]/h
            pred_lines[:,:,1] = pred_lines[:,:,1]*(128)
            pred_lines[:,:,1] = pred_lines[:,:,1]/w

            
            
            score = results[0]['scores'].cpu().numpy()
            line = pred_lines.cpu().numpy()

            score_idx = np.argsort(-score)
            line = line[score_idx]
            score = score[score_idx]

            os.makedirs(args.output_dir+'/benchmark' , exist_ok=True)
            if 'data/york_processed' in args.coco_path:
                append_path = '/benchmark/benchmark_york_'+args.append_word
                os.makedirs(args.output_dir+append_path , exist_ok=True)
                checkpoint_path = args.output_dir+append_path+'/{}.npz'
                curr_img_id = targets[0]['image_id'].tolist()[0]
                np.savez(checkpoint_path.format(id_to_img[curr_img_id]),**{'lines': line, 'score':score})
            elif 'data/wireframe_processed' in args.coco_path:
                append_path = '/benchmark/benchmark_val_'+args.append_word
                os.makedirs(args.output_dir+append_path , exist_ok=True)
                checkpoint_path = args.output_dir+append_path+'/{:08d}.npz'
                curr_img_id = targets[0]['image_id'].tolist()[0]
                np.savez(checkpoint_path.format(int(id_to_img[curr_img_id])),**{'lines': line, 'score':score})
            else:
                assert False
        num_images +=1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats