#!/bin/bash
#SBATCH --job-name=gamma-train
#SBATCH -p qdagnormal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o gamma.log   # 标准输出日志

set HF_ENDPOINT=https://hf-mirror.com
python src/train.py