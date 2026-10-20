#!/bin/bash
#SBATCH --job-name=vit_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/vit_%j.out
#SBATCH --error=logs/vit_%j.err

source ~/.bashrc
conda activate dissertation

cd /users/40771994/projects/dissertation/codebase

python src/models/vision_transformer.py
