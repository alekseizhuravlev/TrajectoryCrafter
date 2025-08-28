#!/bin/bash
#SBATCH -p gpu22                   # Partition name
#SBATCH -t 01:00:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/%x_%j.out   # Standard output log
#SBATCH -e /home/azhuravl/nobackup/slurm_err/%x_%j.err   # Standard error log
#SBATCH --job-name=trajcraft       # Job name (used in log filenames)
#SBATCH --gres=gpu:1                                 # Request 1 GPU


# Activate conda environment
source ~/.bashrc
conda activate trajcrafter

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

# Run your script
python gradio_app.py
