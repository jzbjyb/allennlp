#!/usr/bin/env bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --time=0

model=$1
out_dir=$2

mkdir -p ${out_dir} &&
python scripts/openie_extract.py --model ${model} --inp data/openie/raw_sent/oie2016_test.txt \
    --out ${out_dir}/oie2016.txt --unmerge &&
python scripts/openie_extract.py --model ${model} --inp data/openie/raw_sent/web.txt \
    --out ${out_dir}/web.txt --unmerge &&
python scripts/openie_extract.py --model ${model} --inp data/openie/raw_sent/nyt.txt \
    --out ${out_dir}/nyt.txt --unmerge &&
python scripts/openie_extract.py --model ${model} --inp data/openie/raw_sent/penn.txt \
    --out ${out_dir}/penn.txt --unmerge