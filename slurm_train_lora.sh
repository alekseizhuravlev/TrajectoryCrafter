#!/bin/bash
#SBATCH -p gpu22            # Partition name
#SBATCH -t 12:00:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/train_lora-%j.out   # Standard output log
#SBATCH --job-name=train_lora       # Job name (used in log filenames)
#SBATCH --gres=gpu:a100:1              # Request 1 GPU

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter_ssd

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

export PYTHONPATH=$PYTHONPATH:$(pwd)



BOT_TOKEN="1908938238:AAFjWmxj6nzVM08P9MTMM0PFKsreVTgVYKI"
CHAT_ID="840237381"
MSG_START="🚀 Job $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) just started on $(hostname)"
MSG_END="✅ Job $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) finished on $(hostname)"

curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
     -d chat_id="${CHAT_ID}" -d text="${MSG_START}"



sh /home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/train_lora.sh



curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
     -d chat_id="${CHAT_ID}" -d text="${MSG_END}"