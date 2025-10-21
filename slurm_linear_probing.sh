#!/bin/bash
#SBATCH -p gpu24            # Partition name
#SBATCH -t 20:00:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/linear_probing-%j.out   # Standard output log
#SBATCH --job-name=linear_probing       # Job name (used in log filenames)
#SBATCH --gres=gpu:h100:1              # Request 1 GPU

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

export PYTHONPATH=$PYTHONPATH:$(pwd)

python /home/azhuravl/work/TrajectoryCrafter/notebooks/15_10_25_depth/collect_dataset.py
