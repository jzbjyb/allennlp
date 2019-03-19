#!/bin/bash
#SBATCH --mem=15000
#SBATCH --gres=gpu:1
#SBATCH --time=0

config=$1
out_dir=$2

allennlp train ${config} --serialization-dir ${out_dir} --include-package multitask
