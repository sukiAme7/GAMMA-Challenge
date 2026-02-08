#!/bin/bash
#SBATCH --job-name=gamma-inf
#SBATCH -p qdagnormal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o gamma-inf.log   # 标准输出日志

source /work/home/shiyan_dong/miniconda3/etc/profile.d/conda.sh
conda activate gamma

export PYTHONUNBUFFERED=1
python src/inference.py --evaluate --checkpoint outputs/checkpoints/best_model.pth
# python src/train.py