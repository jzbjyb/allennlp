#!/usr/bin/env bash
#SBATCH --mem=15000
#SBATCH --gres=gpu:1
#SBATCH --time=0

model="output/openie/multitask_vocab_shuffle/small_srl_oie_mt_large_task_encoder_sr3/model.tar.gz"
train_dir="data/openie/conll_for_allennlp/train_parallel_on_srl_nw_wsj"
dev_dir="data/openie/conll_for_allennlp/dev_parallel_on_srl_nw_wsj"

mkdir -p ${dev_dir}
mkdir -p ${train_dir}

python scripts/srl_oie/oie_srl_parallel.py --direction srl2oie --model ${model} \
    --inp data/srl/conll-formatted-ontonotes-5.0/data/development/data/english/annotations/nw/wsj/ \
    --out ${dev_dir}/oie2016.dev.all_tagging \
    --is_dir

python scripts/srl_oie/oie_srl_parallel.py --direction srl2oie --model ${model} \
    --inp data/srl/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/nw/wsj/ \
    --out ${train_dir}/oie2016.train.all_tagging \
    --is_dir
