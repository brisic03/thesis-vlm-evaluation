#!/bin/bash
#SBATCH --job-name=tinyllava_p2_jpeg20_run3
#SBATCH --output=/home/brisic03/Phase_2/tinyllava_robustness/logs/tinyllava_p2_jpeg20_run3_%j.out
#SBATCH --error=/home/brisic03/Phase_2/tinyllava_robustness/logs/tinyllava_p2_jpeg20_run3_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --qos=standard

mkdir -p /home/brisic03/Phase_2/tinyllava_robustness/logs

source ~/.bashrc

conda activate tinyllava

python /home/brisic03/Phase_2/tinyllava_robustness.py
