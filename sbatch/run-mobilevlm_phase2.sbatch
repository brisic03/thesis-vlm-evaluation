#!/bin/bash
#SBATCH --job-name=mobilevlm_p2_occ03_run3
#SBATCH --output=/home/brisic03/Phase_2/mobilevlm_robustness/logs/mobilevlm_p2_occ03_run3_%j.out
#SBATCH --error=/home/brisic03/Phase_2/mobilevlm_robustness/logs/mobilevlm_p2_occ03_run3_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=standard
#SBATCH --time=12:00:00

mkdir -p /home/brisic03/Phase_2/mobilevlm_robustness/logs

source ~/.bashrc

conda activate mobilevlm

python /home/brisic03/Phase_2/mobilevlm_robustness.py


