# Fail the script if there is any failure
set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name and Epoch'
    exit 1
fi

name=$1
epoch=$2

output=exp/$name

cd evaluation

echo "Post-processing"
python eval-aph-post-wireframe.py --plot --thresholds="0.010,0.015" ../$output/benchmark/benchmark_val_$epoch ../$output/post/$epoch

echo "Evaluation AP-H"
mkdir -p ../$output/score/eval-APH-0_010
python eval-aph-score-wireframe.py ../$output/post/$epoch/0_010 ../$output/post/$epoch/0_010-APH | tee ../$output/score/eval-APH-0_010/$epoch.txt