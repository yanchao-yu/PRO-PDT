#!/bin/bash
#SBATCH --job-name=resnet_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00-05:00:00
#SBATCH --output=logs/resnet_eval_vggface_%j.out
#SBATCH --error=logs/resnet_eval_vggface_%j.err

source ~/.bashrc
conda activate dissertation

cd /users/40771994/projects/dissertation/codebase

python  notebooks/eval.py