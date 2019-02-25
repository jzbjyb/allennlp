#!/bin/bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --time=0

config=$1
out_dir=$2
eval_dir=$3
eval_args="${@:4}"

./run.sh ${config} ${out_dir}
./scripts/openie_extract.sh ${out_dir} ${eval_dir} ${eval_args}
