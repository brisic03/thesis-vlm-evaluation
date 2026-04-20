#!/bin/bash
#SBATCH --job-name=latency_mvlm
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/brisic03/latency_mobilevlm.log

source /home/brisic03/miniconda3/etc/profile.d/conda.sh
conda activate mobilevlm
cd /home/brisic03/MobileVLM
python latency_mobilevlm.py
