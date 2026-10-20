#!/bin/bash
#SBATCH --job-name=vit_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/vit_eval_%j.out
#SBATCH --error=logs/vit_eval_%j.err

source ~/.bashrc
conda activate dissertation

cd /users/40771994/projects/dissertation/codebase

python  notebooks/vit_eval.py