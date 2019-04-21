#!/usr/bin/env bash
#SBATCH --mem=15000
#SBATCH --gres=gpu:1
#SBATCH --time=0
set -e

config=$1
out_dir=$2
srl_model=$3
eval_dir=$4

./run.sh ${config} ${out_dir}
./scripts/openie_extract.sh ${srl_model}:${out_dir}/model.tar.gz ${eval_dir} --method=xsrl