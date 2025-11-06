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

echo "Output directory: $OUTPUT_DIR"
# Create the directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"


export TRAIN_DATASET_NAME="/home/azhuravl/scratch/datasets_latents/monkaa_1000"
export VAL_DATASET_NAME="/home/azhuravl/scratch/datasets_latents/driving_1000"

# --use_depth \
accelerate launch --mixed_precision="bf16" notebooks/05_11_25_training/lora_utils_ours/main.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATASET_NAME \
  --val_data_dir=$VAL_DATASET_NAME \
  --max_val_samples=8 \
  --train_data_meta=$DATASET_META_NAME \
  --use_depth \
  --rank 4 \
  --network_alpha 4 \
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
  --checkpointing_steps=10000 \
  --validation_steps=10000 \
  --validation_epochs=5 \
  --learning_rate=1e-04 \
  --seed=42 \
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
  --train_mode="inpaint" 
  # --low_vram \

  

# Training command for CogVideoX-Fun-V1.5
# export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"
# export DATASET_NAME="datasets/internal_datasets/"
# export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# NCCL_DEBUG=INFO

# accelerate launch --mixed_precision="bf16" scripts/cogvideox_fun/train_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATASET_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --image_sample_size=1024 \
#   --video_sample_size=256 \
#   --token_sample_size=512 \
#   --video_sample_stride=3 \
#   --video_sample_n_frames=85 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=8 \
#   --num_train_epochs=100 \
#   --checkpointing_steps=50 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --enable_bucket \
#   --low_vram \
#   --train_mode="inpaint" 