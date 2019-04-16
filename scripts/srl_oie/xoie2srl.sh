#!/usr/bin/env bash
#SBATCH --mem=15000
#SBATCH --gres=gpu:1
#SBATCH --time=0

model="pretrain/srl-model-2018.05.25"
train_dir="data/openie/conll_for_allennlp/train_split_rm_coor"
dev_dir="data/openie/conll_for_allennlp/dev_split_rm_coor"

mkdir -p ${dev_dir}/parallel
mkdir -p ${train_dir}/parallel

python scripts/srl_oie/oie_srl_parallel.py --direction oie2srl --model ${model} \
    --inp ${dev_dir}/oie2016.dev.gold_conll \
    --out ${dev_dir}/parallel/oie2016.dev.all_tagging

python scripts/srl_oie/oie_srl_parallel.py --direction oie2srl --model ${model} \
    --inp ${train_dir}/oie2016.train.gold_conll \
    --out ${train_dir}/parallel/oie2016.train.all_tagging
