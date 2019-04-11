#!/usr/bin/env bash
#SBATCH --mem=15000
#SBATCH --gres=gpu:1
#SBATCH --time=0

model="output/openie/multitask_vocab/small_srl_oie_mt_large_task_encoder/model.tar.gz"

mkdir -p data/openie/conll_for_allennlp/train_parallel_on_srl2

mkdir -p data/openie/conll_for_allennlp/dev_parallel_on_srl2

python scripts/srl_oie/oie_srl_parallel.py --direction srl2oie --model ${model} \
    --inp data/srl/conll-formatted-ontonotes-5.0/data/development/ \
    --out data/openie/conll_for_allennlp/dev_parallel_on_srl2/oie2016.dev.all_tagging \
    --is_dir

python scripts/srl_oie/oie_srl_parallel.py --direction srl2oie --model ${model} \
    --inp data/srl/conll-formatted-ontonotes-5.0/data/train/ \
    --out data/openie/conll_for_allennlp/train_parallel_on_srl2/oie2016.train.all_tagging \
    --is_dir
