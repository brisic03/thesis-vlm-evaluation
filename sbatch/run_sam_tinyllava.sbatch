#!/bin/bash
#SBATCH --job-name=sam_tinyllava
#SBATCH --output=/home/brisic03/sam_tinyllava_%j.out
#SBATCH --error=/home/brisic03/sam_tinyllava_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate tinyllava

python /home/brisic03/sam_counting_tinyllava.py

