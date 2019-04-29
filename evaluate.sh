#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=0

model_dir=$1
data=$2

allennlp evaluate ${model_dir} ${data} --include-package=multitask --cuda-device=0
