#!/bin/bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1

out_dir=test
allennlp train training_config/srl_openie.jsonnet --serialization-dir ${test}
