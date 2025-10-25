# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from args import get_args_parser

import json


def save_json(file, data):
    f = open(file, "w")
    json.dump(data, f, indent = 6)
    f.close()


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    output_dir = Path(args.output_dir)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.sgd_weight_decay)
    else:
        assert False

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.eval:
        dataset_val = build_dataset(dataset_name=args.dataset_name, image_set=args.dataset, args=args)

        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    else:
        dataset_train = build_dataset(dataset_name=args.dataset_name, image_set='train', args=args)
        dataset_val = build_dataset(dataset_name=args.dataset_name, image_set='val', args=args)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.SequentialSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)



    if args.resume and args.frozen_weights:
        assert False
    elif args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
            new_state_dict = {}
            for k in checkpoint['model']:
                if ("class_embed" in k) or ("bbox_embed" in k) or ("query_embed" in k):
                    continue
                if  ("input_proj" in k) and args.layer1_num != 3:
                    continue
                new_state_dict[k] = checkpoint['model'][k]

            # Compare load model and current model
            current_param = [n for n,p in model_without_ddp.named_parameters()]
            current_buffer = [n for n,p in model_without_ddp.named_buffers()]
            load_param = new_state_dict.keys()
            for p in load_param:
                if p not in current_param and p not in current_buffer:
                    print(p, 'NOT appear in current model.  ')
            for p in current_param:
                if p not in load_param:
                    print(p, 'NEW parameter.  ')
            model_without_ddp.load_state_dict(new_state_dict, strict=False)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

            # this is to compromise old implementation
            new_state_dict = {}
            for k in checkpoint['model']:
                if "bbox_embed" in k:
                    print("bbox_embed from OLD implementation has been replaced with lines_embed")
                    new_state_dict["lines_embed."+'.'.join(k.split('.')[1:])] = checkpoint['model'][k]
                else:
                    new_state_dict[k] = checkpoint['model'][k]

            # compare resume model and current model
            current_param = [n for n,p in model_without_ddp.named_parameters()]
            current_buffer = [n for n,p in model_without_ddp.named_buffers()]
            load_param = new_state_dict.keys()
            #for p in load_param:
                #if p not in current_param and p not in current_buffer:
                    #print(p, 'not been loaded to current model. Strict == False?')
            for p in current_param:
                if p not in load_param:
                    print(p, 'is a new parameter. Not found from load dict.')

            # load model
            model_without_ddp.load_state_dict(new_state_dict)

            # load optimizer
            if not args.no_opt and not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                checkpoint['lr_scheduler']['step_size'] = args.lr_drop  # change the lr_drop epoch
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
    elif args.frozen_weights:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu', weights_only=False)
        new_state_dict = {}
        for k in checkpoint['model']:
            if "bbox_embed" in k:
                new_state_dict["lines_embed."+'.'.join(k.split('.')[1:])] = checkpoint['model'][k]
            else:
                new_state_dict[k] = checkpoint['model'][k]

        model_without_ddp.letr.load_state_dict(new_state_dict)

        # params
        encoder = {k:v for k,v in new_state_dict.items() if "encoder" in k}
        decoder = {k:v for k,v in new_state_dict.items() if "decoder" in k}
        class_embed = {k:v for k,v in new_state_dict.items() if "class_embed" in k}
        line_embed = {k:v for k,v in new_state_dict.items() if "lines_embed" in k}

        model_without_ddp.load_state_dict(encoder, strict=False)
        model_without_ddp.load_state_dict(decoder, strict=False)
        model_without_ddp.load_state_dict(class_embed, strict=False)
        model_without_ddp.load_state_dict(line_embed, strict=False)
        print('Finish load frozen_weights')
    else:
        print("NO RESUME. TRAIN FROM SCRATCH")

    if args.eval:
        test_stats = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args, epoch)
        #print('checkpoint'+ str(checkpoint['epoch']))
        return

    print("Start training")
    start_time = time.time()

    train_losses = []
    test_losses = []
    losses_all = {
        'train': [],
        'test': []
    }

    save_json(os.path.join(output_dir, 'info.json'), {'weight_dict': criterion.weight_dict, 'args': vars(args)})

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, postprocessors, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoints/checkpoint.pth']
            # extra checkpoint before LR drop and every several epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_freq == 0:
                checkpoint_paths.append(output_dir / f'checkpoints/checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args, epoch)

        log_stats = {**{f'train_{k}': format(v, ".6f") for k, v in train_stats.items()},
                     **{f'test_{k}': format(v, ".6f") for k, v in test_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        print('******')
        print('******')
        print('******')
        print('******')

        print('Train loss:', train_stats['loss'])
        print('Test loss:', test_stats['loss'])

        train_losses.append(train_stats['loss'])
        test_losses.append(test_stats['loss'])
        losses_all['train'].append(train_stats)
        losses_all['test'].append(test_stats)

        save_json(os.path.join(output_dir, 'train.json'), {'train': train_losses})
        save_json(os.path.join(output_dir, 'test.json'), {'test': test_losses})
        save_json(os.path.join(output_dir, 'losses_all.json'), losses_all)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir == '<auto>':
        args.output_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args.output_dir = os.path.join(args.output_dir_prefix, args.output_dir)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.output_dir+'/checkpoints').mkdir(parents=True, exist_ok=True)

    main(args)
