#!/bin/bash
#SBATCH --job-name=vit_tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/vit_tune%j.out
#SBATCH --error=logs/vit_tune%j.err

source ~/.bashrc
conda activate dissertation

cd /users/40771994/projects/dissertation/codebase || exit

python src/inference/parameter_tune/vit_tune.py
