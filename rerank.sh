#!/usr/bin/env bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --time=0

openie_model_dir=$1
rerank_data_dir=$2
rerank_config=$3
rerank_model_dir=$4
eval_from=$5
eval_to=$6

cur_dir=$(pwd)

openie_model_dir=${cur_dir}/${openie_model_dir}
rerank_data_dir=${cur_dir}/${rerank_data_dir}
rerank_config=${cur_dir}/${rerank_config}
rerank_model_dir=${cur_dir}/${rerank_model_dir}
eval_from=${cur_dir}/${eval_from}
eval_to=${cur_dir}/${eval_to}

conda_act=/home/zhengbaj/anaconda3/bin/activate
conda_dea=/home/zhengbaj/anaconda3/bin/deactivate

# beam search to generate extractions
mkdir -p ${rerank_data_dir}
for split in train dev
do
    # TODO: add beam search
    python scripts/openie_extract.py --model ${openie_model_dir}/model.tar.gz \
        --inp data/openie/raw_sent/oie2016_${split}.txt \
        --out ${rerank_data_dir}/oie2016.${split}.beam --keep_one
done &&

# converting extraction files to pseudo conll files
pushd ~/exp/supervised-oie/supervised-oie-benchmark
source $conda_dea
source $conda_act sup_oie
for split in train dev
do
    python benchmark.py --gold=./oie_corpus/${split}.oie.orig.correct.head --out=/dev/null \
        --tabbed=${rerank_data_dir}/oie2016.${split}.beam \
        --label=${rerank_data_dir}/oie2016.${split}.beam.conll \
        --predArgHeadMatch
done &&
source $conda_dea
source $conda_act allennlp
popd

# train reranking model
# TODO: automatically generate config file?
./run.sh ${rerank_config} ${rerank_model_dir} &&

# generate pseudo conll file for prediction
# all the extractions in eval_from should have contiguous arguments and predicates
mkdir -p ${eval_to}
pushd ~/exp/supervised-oie/src
source $conda_dea
source $conda_act sup_oie
./extraction_to_conll.sh ${eval_from} ${eval_to} &&
source $conda_dea
source $conda_act allennlp
popd

# predict (re-compute confidence score)
for ds in oie2016 web nyt penn
do
    python scripts/rerank.py --model ${rerank_model_dir}/model.tar.gz \
        --inp ${eval_to}/${ds}.txt.conll:${eval_from}/${ds}.txt \
        --out ${eval_to}/${ds}.txt
done