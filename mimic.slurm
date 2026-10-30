#!/bin/bash
#SBATCH --job-name=mimic
#SBATCH --output=mimic.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuscavenger

module load anaconda
source activate PhDResearch

python fit_nuisance_models.py