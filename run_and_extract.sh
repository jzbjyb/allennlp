#!/bin/bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --time=0

config=$1
out_dir=$2
eval_dir=$3

allennlp train ${config} --serialization-dir ${out_dir} --include-package multitask &&
./scripts/openie_extract.sh ${out_dir}/model.tar.gz ${eval_dir}
