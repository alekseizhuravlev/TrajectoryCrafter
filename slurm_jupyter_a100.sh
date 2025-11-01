#!/bin/bash
#SBATCH -p gpu22            # Partition name
#SBATCH -t 14:00:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/jupyter-%j.out   # Standard output log
#SBATCH --job-name=jupyter       # Job name (used in log filenames)
#SBATCH --gres=gpu:a100:1              # Request 1 GPU

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter_ssd

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

export PYTHONPATH=$PYTHONPATH:$(pwd)

jupyter notebook --no-browser --port 5998 --ip $(hostname -f)
