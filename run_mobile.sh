#!/bin/bash
#SBATCH --job-name=mobile_eval
#SBATCH --output=mobile_results.log
#SBATCH --error=mobile_error.log
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=18:00:00

source /home/brisic03/miniconda3/etc/profile.d/conda.sh
conda activate mobilevlm

export PYTHONPATH=$PYTHONPATH:/home/brisic03/MobileVLM

python /home/brisic03/MobileVLM/eval_mobilevlm.py
