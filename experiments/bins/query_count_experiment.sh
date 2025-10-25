#!/bin/sh

DEBUG_ARGS=(
    --save_prediction_probability 0.01
    --save_prediction_data
    --save_prediction_visualization
    --save_test_prediction_probability 0.005
    --save_test_prediction_data
    --save_test_prediction_visualization
)

ADD_ARGS=()
if [ -n "$DEBUG" -a "$DEBUG" == "1" ] ; then
    ADD_ARGS+=${DEBUG_ARGS[@]}
fi

set -x

for q in {4,16,32,64} ; do
    OPENCV_IO_ENABLE_OPENEXR=1 python src/main.py \
        --dataset_name bins \
        --bins_input_width 512 \
        --bins_input_height 383 \
        --bins_path ./data/bin_dataset/data/VISIGRAPP_TRAIN/dataset.json \
        --backbone resnet50 \
        --resume  https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
        --layer1_num 3 \
        --output_dir bins_output/queries/$q/stage1 \
        --num_queries $q \
        --lr_drop 200 \
        --epochs 449 \
        ${ADD_ARGS[@]}

    OPENCV_IO_ENABLE_OPENEXR=1 python src/main.py \
        --dataset_name bins \
        --bins_input_width 512 \
        --bins_input_height 383 \
        --bins_path ./data/bin_dataset/data/VISIGRAPP_TRAIN/dataset.json \
        --backbone resnet50 \
        --layer1_num 3 \
        --output_dir bins_output/queries/$q/stage2 \
        --num_queries $q \
        --LETRpost \
        --no_opt \
        --layer1_frozen \
        --frozen_weights bins_output/queries/$q/stage1/checkpoints/checkpoint0444.pth \
        --lr_drop 30 \
        --epochs 99 \
        ${ADD_ARGS[@]}

    OPENCV_IO_ENABLE_OPENEXR=1 python src/main.py \
        --dataset_name bins \
        --bins_input_width 512 \
        --bins_input_height 383 \
        --bins_path ./data/bin_dataset/data/VISIGRAPP_TRAIN/dataset.json \
        --backbone resnet50 \
        --layer1_num 3 \
        --output_dir bins_output/queries/$q/stage3 \
        --num_queries $q \
        --LETRpost \
        --no_opt \
        --layer1_frozen \
        --resume bins_output/queries/$q/stage2/checkpoints/checkpoint.pth \
        --label_loss_func focal_loss \
        --label_loss_params '{"gamma":2.0}' \
        --save_freq 1 \
        --lr 1e-5 \
        --epochs 20 \
        ${ADD_ARGS[@]}

done
