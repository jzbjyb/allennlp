#!/usr/bin/env bash
#SBATCH --mem=15000
#SBATCH --gres=gpu:1
#SBATCH --time=0

model=$1

mkdir -p data/openie/conll_for_allennlp/train_parallel_shuffle_on_srl

python scripts/oie_to_srl.py --direction srl2oie --model ${model} \
    --inp data/openie/conll_for_allennlp/train_srl_oie_mt/srl/ontonotes.shuffle.gold_conll \
    --out data/openie/conll_for_allennlp/train_parallel_shuffle_on_srl/oie.train.all_tagging