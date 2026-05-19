#!/bin/bash
#SBATCH --job-name=sam_resnet_tiny
#SBATCH --output=/home/brisic03/sam_resnet_tiny_%j.out
#SBATCH --error=/home/brisic03/sam_resnet_tiny_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate tinyllava

python /home/brisic03/sam_resnet_tinyllava.py

