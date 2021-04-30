# Fail the script if there is any failure
set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi

# The name of this experiment.
name=$1

# Save logs and models under snap/gqa; make backup.
output=exp/$name
if [ ! -d "$output"  ]; then
    echo "folder not exist"
    mkdir -p $output/src
    cp -r src/* $output/src/
    cp $0 $output/run.bash

    PYTHONPATH=$PYTHONPATH:./src python -m torch.distributed.launch \
    --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=8 --use_env  src/main.py --coco_path data/wireframe_processed \
    --output_dir $output --LETRpost --backbone resnet50 --layer1_frozen --frozen_weights exp/res50_stage1/checkpoints/checkpoint.pth --no_opt \
    --batch_size 1 ${@:2} --epochs 300 --lr_drop 120 --num_queries 1000 --num_gpus 8 | tee -a $output/history.txt  

else
    echo "folder already exist"
fi



