#!/usr/bin/env bash
model=$1
out_dir=$2

mkdir -p ${out_dir} &&
python scripts/openie_extract.py --model ${model} --inp data/openie/raw_sent/oie2016_test.txt \
    --out ${out_dir}/oie2016.txt &&
python scripts/openie_extract.py --model ${model} --inp data/openie/raw_sent/web.txt \
    --out ${out_dir}/web.txt &&
python scripts/openie_extract.py --model ${model} --inp data/openie/raw_sent/nyt.txt \
    --out ${out_dir}/nyt.txt &&
python scripts/openie_extract.py --model ${model} --inp data/openie/raw_sent/penn.txt \
    --out ${out_dir}/penn.txt