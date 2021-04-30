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
python eval-aph-post-york.py --plot --thresholds="0.010,0.015" ../$output/benchmark/benchmark_york_$epoch ../$output/post_york/$epoch

echo "Evaluation AP-H"
mkdir -p ../$output/score/eval-APH-0_010-york
python eval-aph-score-york.py ../$output/post_york/$epoch/0_010 ../$output/post_york/$epoch/0_010-APH | tee ../$output/score/eval-APH-0_010-york/$epoch.txt