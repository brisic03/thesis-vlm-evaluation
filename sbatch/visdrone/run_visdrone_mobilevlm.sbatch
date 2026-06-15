#!/bin/bash
#SBATCH --job-name=vis_mobile
#SBATCH --output=/home/brisic03/visdrone_mobile_%j.out
#SBATCH --error=/home/brisic03/visdrone_mobile_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate mobilevlm

python /home/brisic03/visdrone_mobilevlm_eval.py

