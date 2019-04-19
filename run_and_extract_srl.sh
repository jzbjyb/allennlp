#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=0

config=$1
out_dir=$2
srl_model=$3
eval_dir=$4
eval_args="${@:5}"

./run.sh ${config} ${out_dir}
./scripts/openie_extract.sh ${srl_model}:${out_dir}/model.tar.gz ${eval_dir} --method=xsrl ${eval_args}
