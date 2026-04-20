#!/bin/bash
#SBATCH --job-name=latency_tllava
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/brisic03/latency_tinyllava.log

source /home/brisic03/miniconda3/etc/profile.d/conda.sh
conda activate tinyllava
cd /home/brisic03/TinyLLaVA_Factory
python latency_tinyllava.py
