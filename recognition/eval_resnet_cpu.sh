#!/bin/bash
#SBATCH --job-name=resnet_eval_cpu
#SBATCH --partition=nodes
#SBATCH --time=00-05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192M
#SBATCH --output=logs/resnet_eval_vggface_%j.out
#SBATCH --error=logs/resnet_eval_vggface_%j.err

source ~/.bashrc
conda activate dissertation

cd /users/40771994/projects/dissertation/codebase

python  notebooks/eval.py