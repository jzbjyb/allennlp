#!/usr/bin/env bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --time=0

tag_model=$1
rerank_model=$2

# create dir to hold tagging model
mkdir -p ${rerank_model}/tag_model
# copy original tagging model
cp ${tag_model}/model.tar.gz ${rerank_model}/tag_model/.
# uncompress
pushd ${rerank_model}/tag_model/
tar xzf model.tar.gz
rm weights.th
popd
# replace the weight
python scripts/rerank_to_tag.py --inp ${rerank_model}/best.th --out ${rerank_model}/tag_model/weights.th
