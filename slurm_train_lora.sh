#!/bin/bash
#SBATCH -p gpu24            # Partition name
#SBATCH -t 12:00:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/train_lora-%j.out   # Standard output log
#SBATCH --job-name=train_lora       # Job name (used in log filenames)
#SBATCH --gres=gpu:h100:1              # Request 1 GPU

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter_ssd

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

export PYTHONPATH=$PYTHONPATH:$(pwd)


##############################
# Telegram notification setup
###############################

BOT_TOKEN="1908938238:AAFjWmxj6nzVM08P9MTMM0PFKsreVTgVYKI"
CHAT_ID="840237381"
MSG_START="🚀 Job $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) just started on $(hostname)"
MSG_END="✅ Job $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) finished on $(hostname)"

curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
     -d chat_id="${CHAT_ID}" -d text="${MSG_START}"

###############################


# sh /home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/train_lora.sh



export MODEL_NAME="/home/azhuravl/scratch/checkpoints/CogVideoX-Fun-V1.1-5b-InP"
export DATASET_META_NAME="datasets/Minimalism/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# Create output directory with date and time
BASE_OUTPUT_DIR="/home/azhuravl/work/TrajectoryCrafter/experiments"
DATE_DIR=$(date +"%d-%m-%Y")
TIME_DIR=$(date +"%H-%M-%S")
OUTPUT_DIR="$BASE_OUTPUT_DIR/$DATE_DIR/$TIME_DIR"

# OUTPUT_DIR="/home/azhuravl/work/TrajectoryCrafter/experiments/07-11-2025/13-33-14_copy"

echo "Output directory: $OUTPUT_DIR"
# Create the directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"




export TRAIN_DATASET_NAME="/home/azhuravl/scratch/datasets_latents/monkaa_wrev_ref_latents"
export VAL_DATASET_NAME="/home/azhuravl/scratch/datasets_latents/driving_1000"
seed=42

accelerate launch --mixed_precision="bf16" notebooks/05_11_25_training/lora_utils_ours/main.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATASET_NAME \
  --val_data_dir=$VAL_DATASET_NAME \
  --max_val_samples=8 \
  --train_data_meta=$DATASET_META_NAME \
  --use_depth \
  --num_ref_frames 10 \
  --save_state \
  --rank 8 \
  --network_alpha 8 \
  --resume_from_checkpoint "latest" \
  --seed=$seed \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=50 \
  --checkpointing_steps=5000 \
  --validation_steps=5000 \
  --validation_epochs=50 \
  --learning_rate=1e-04 \
  --output_dir="$OUTPUT_DIR" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --train_mode="inpaint" \
  --use_deepspeed



########################################


curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
     -d chat_id="${CHAT_ID}" -d text="${MSG_END}"