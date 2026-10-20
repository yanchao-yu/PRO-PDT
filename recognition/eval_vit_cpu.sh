#!/bin/bash
#SBATCH --job-name=vit_eval_cpu
#SBATCH --partition=nodes
#SBATCH --time=00-05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192M
#SBATCH --output=logs/vit_eval_vggface_%j.out
#SBATCH --error=logs/vit_eval_vggface_%j.err

source ~/.bashrc
conda activate dissertation

cd /users/40771994/projects/dissertation/codebase

python  notebooks/vit_eval.py