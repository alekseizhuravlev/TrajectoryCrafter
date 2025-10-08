#!/bin/bash
#SBATCH -p gpu24            # Partition name
#SBATCH -t 12:00:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/jupyter-%j.out   # Standard output log
#SBATCH --job-name=jupyter       # Job name (used in log filenames)
#SBATCH --gres=gpu:h100:1              # Request 1 GPU

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

export PYTHONPATH=$PYTHONPATH:$(pwd)

jupyter notebook --no-browser --port 5998 --ip $(hostname -f)
