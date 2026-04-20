#!/bin/bash
#SBATCH --job-name=eval_nextqa
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=eval_%j.log

source /home/brisic03/miniconda3/etc/profile.d/conda.sh
conda activate tinyllava
cd /home/brisic03/TinyLLaVA_Factory
python eval_nextqa.py
