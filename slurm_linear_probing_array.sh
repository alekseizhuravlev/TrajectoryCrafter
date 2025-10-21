#!/bin/bash
#SBATCH -p gpu24,gpu22                        # Partition name
#SBATCH --array=0-9                     # Array job with 8 processes (0-7)
#SBATCH -t 8:00:00                     # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/linear_probing-%A_%a.out   # Standard output log (%A=job_id, %a=array_index)
#SBATCH --job-name=linear_probing       # Job name (used in log filenames)
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Dynamic calculation of samples per process
TOTAL_SAMPLES=250
SAMPLES_PER_PROCESS=$(( $TOTAL_SAMPLES / $SLURM_ARRAY_TASK_COUNT ))

echo "Process $SLURM_ARRAY_TASK_ID: Generating $SAMPLES_PER_PROCESS samples (Total: $TOTAL_SAMPLES, Processes: $SLURM_ARRAY_TASK_COUNT)"

# Run the script with array job parameters
python /home/azhuravl/work/TrajectoryCrafter/notebooks/15_10_25_depth/collect_dataset.py \
    --process_id $SLURM_ARRAY_TASK_ID \
    --n_processes $SLURM_ARRAY_TASK_COUNT \
    --num_samples $SAMPLES_PER_PROCESS

echo "Array job $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT completed"