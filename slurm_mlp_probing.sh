#!/bin/bash
#SBATCH -p gpu20                        # Partition name
#SBATCH --array=0-5                     # Array job with 6 processes (0-5) - one per timestep
#SBATCH -t 8:00:00                     # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/mlp_probing-%A_%a.out
#SBATCH --job-name=mlp_probing
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

export PYTHONPATH=$PYTHONPATH:$(pwd)

TIMESTEPS=(
    timestep_39
    timestep_199
    timestep_359
    timestep_519
    timestep_679
    timestep_839
)
FEATURES=(
    cross_attn_0
    cross_attn_1
    cross_attn_2
    final_norm
    pos_embed
    transformer_block_8
    transformer_block_16
    transformer_block_24
    transformer_block_32
    transformer_block_40
)

# Get the timestep for this array task
TIMESTEP=${TIMESTEPS[$SLURM_ARRAY_TASK_ID]}

echo "Array task $SLURM_ARRAY_TASK_ID: Processing timestep $TIMESTEP with all features"

# Loop through all features for this timestep
for FEATURE in "${FEATURES[@]}"; do
    echo "Running with timestep=$TIMESTEP, feature=$FEATURE"
    
    python /home/azhuravl/work/TrajectoryCrafter/notebooks/15_10_25_depth/mlp_probing.py \
        --timestep $TIMESTEP \
        --feature_name $FEATURE \
        --num_epochs 10 \
        --exp_name mlp_probes_fixed_8_lr_10 \
        --data_dir linear_probing_fixed_8
    
    echo "Completed $TIMESTEP with $FEATURE"
done

echo "Completed all features for array task $SLURM_ARRAY_TASK_ID (timestep: $TIMESTEP)"