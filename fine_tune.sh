#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=0

config=$1
model_dir=$2
out_dir=$3
args="${@:4}"

allennlp fine-tune -c ${config} -m ${model_dir} -s ${out_dir} --include-package multitask ${args}
