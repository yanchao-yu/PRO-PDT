#!/bin/bash
#SBATCH --job-name=resnet_tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/resnet_tune%j.out
#SBATCH --error=logs/resnet_tune%j.err

source ~/.bashrc
conda activate dissertation

cd /users/40771994/projects/dissertation/codebase

python src/inference/parameter_tune/resnet_tune.py
