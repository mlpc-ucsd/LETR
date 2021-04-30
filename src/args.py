import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--save_freq', default=5, type=int)
    
    parser.add_argument('--benchmark', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--append_word', default=None, type=str, help="Name of the convolutional backbone to use")
    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Load
    parser.add_argument('--layer1_frozen', action='store_true')
    parser.add_argument('--layer2_frozen', action='store_true')

    parser.add_argument('--frozen_weights', default='', help='resume from checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--no_opt', action='store_true')

    # Transformer
    parser.add_argument('--LETRpost', action='store_true')
    parser.add_argument('--layer1_num', default=3, type=int)
    parser.add_argument('--layer2_num', default=2, type=int)

    # First Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1000, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')


    # Second Transformer
    parser.add_argument('--second_enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--second_dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--second_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--second_hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--second_dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--second_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--second_pre_norm', action='store_true')
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_line', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=5, type=float,
                        help="L1 box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--point_loss_coef', default=5, type=float)
    parser.add_argument('--line_loss_coef', default=5, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--label_loss_func', default='cross_entropy', type=str)
    parser.add_argument('--label_loss_params', default='{}', type=str)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
   

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dataset', default='train', type=str, choices=('train', 'val'))

    return parser