#!/bin/bash

#SBATCH --job-name=resnet_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/resnet_%j.out
#SBATCH --error=logs/resnet_%j.err

source ~/.bashrc
conda activate dissertation

cd /users/40771994/projects/dissertation/codebase

python src/models/inception_resnet.py
