#!/usr/bin/env bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --time=0
set -e

niter=$1
data_dir=$2 # reranking data root dir. Initial conll file (if have)
            # should be ${data_dir}/iter0/oie2016.[train|dev].iter.conll
model_dir=$3 # reranking model root dir. Initial model should be put in
             # ${model_dir}/iter0/tag_model/model.tar.gz
             # (both compressed and uncompressed)
eval_dir=$4 # evaluation root dir. Initial extractions should be put in
            # ${eval_dir}/iter0/tag_model
rerank_conf=$5 # rerank json config file
beam=$6 # size of beam search

# make path absolute
pwd_dir=$(pwd)
data_dir=${pwd_dir}/${data_dir}
model_dir=${pwd_dir}/${model_dir}
eval_dir=${pwd_dir}/${eval_dir}

# set up conda
conda_act=/home/zhengbaj/anaconda3/bin/activate
conda_dea=/home/zhengbaj/anaconda3/bin/deactivate

for (( e=1; e<=$niter; e++ ))
do
    echo "=== Iter $e ==="

    cur_dir=iter$((e-1))
    next_dir=iter${e}

    echo "beam search to generate extractions"

    mkdir -p ${data_dir}/${next_dir}
    for split in train dev
    do
        python scripts/openie_extract.py --model ${model_dir}/${cur_dir}/tag_model \
            --inp data/openie/raw_sent/oie2016_${split}.txt \
            --out ${data_dir}/${next_dir}/oie2016.${split}.beam \
            --keep_one --beam_search ${beam}
    done

    echo "converting extraction files to pseudo conll files and combine with previous epoch"

    pushd ~/exp/supervised-oie/supervised-oie-benchmark
    source $conda_dea
    source $conda_act sup_oie
    for split in train dev
    do
        last_conll=${data_dir}/${cur_dir}/oie2016.${split}.iter.conll
        this_beam=${data_dir}/${next_dir}/oie2016.${split}.beam
        this_beam_conll=${data_dir}/${next_dir}/oie2016.${split}.beam.conll
        this_conll=${data_dir}/${next_dir}/oie2016.${split}.iter.conll

        python benchmark.py --gold=./oie_corpus/${split}.oie.orig.correct.head --out=/dev/null \
            --tabbed=${this_beam} --label=${this_beam_conll} --predArgHeadMatch

        if [ -f "$last_conll" ]
        then
            python combine_conll.py -inp=${last_conll}:${this_beam_conll} -out=${this_conll}
        else
            cp ${this_beam_conll} ${this_conll}
        fi
    done
    source $conda_dea
    source $conda_act allennlp
    popd

    echo "train reranking model"

    # TODO: automatically handle config with external variables?
    conf_data_dir=${data_dir}/${next_dir} # use new data
    cond_model_dir=${model_dir}/${cur_dir}/tag_model # initialize from old tagging model
    conf_conf=${conf_data_dir}/iter_conf.jsonnet # place generated conf in data dir
    sed "s|ITER_DATA_ROOT|${conf_data_dir}|g" ${rerank_conf} > ${conf_conf}
    sed -i "s|ITER_MODEL_ROOT|${cond_model_dir}|g" ${conf_conf}
    mkdir -p ${model_dir}/${next_dir}
    ./run.sh ${conf_conf} ${model_dir}/${next_dir}

    echo "rerank extractions"

    eval_from=${eval_dir}/${cur_dir}/tag_model
    eval_to=${eval_dir}/${next_dir}
    mkdir -p ${eval_to}
    pushd ~/exp/supervised-oie/src
    source $conda_dea
    source $conda_act sup_oie
    ./extraction_to_conll.sh ${eval_from} ${eval_to}
    source $conda_dea
    source $conda_act allennlp
    popd

    for ds in oie2016 web nyt penn
    do
        python scripts/rerank.py --model ${model_dir}/${next_dir}/model.tar.gz \
            --inp ${eval_to}/${ds}.txt.conll:${eval_from}/${ds}.txt \
            --out ${eval_to}/${ds}.txt
    done

    echo "generate extractions using reranking model"

    ./scripts/rerank_to_tag.sh ${model_dir}/iter0/tag_model ${model_dir}/${next_dir}
    ./scripts/openie_extract.sh ${model_dir}/${next_dir}/tag_model ${eval_to}/tag_model
done
