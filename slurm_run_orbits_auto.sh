#!/bin/bash

# First, calculate the number of active videos
videos=(
    # "0-NNvgaTcVzAG0-r.mp4"
    # "p7.mp4"
    # "part-2-3.mp4"
    # "tUfDESZsQFhdDW9S.mp4"
    # "UST-fn-RvhJwMR5S.mp4"
    # "bmx-bumps.mp4"
    # "india.mp4"
    "judo.mp4"
    # "koala.mp4"
    # "rhino.mp4"
)

# Calculate array size dynamically
num_videos=${#videos[@]}
max_index=$((num_videos - 1))

# If running this script directly (not via sbatch), submit the job with correct array size
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Detected ${num_videos} active video(s)"
    echo "Submitting SLURM job with array range 0-${max_index}"
    
    if [ $num_videos -eq 1 ]; then
        # Single video - no array needed
        sbatch --array=0 "$0"
    else
        # Multiple videos - use array
        sbatch --array=0-${max_index} "$0"
    fi
    exit 0
fi

# SLURM directives (will be used when script is submitted)
#SBATCH -p gpu22                   # Partition name
#SBATCH -t 06:00:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/%x_%A_%a.out   # Standard output log
#SBATCH -e /home/azhuravl/nobackup/slurm_err/%x_%A_%a.err   # Standard error log
#SBATCH --job-name=trajcraft       # Job name (used in log filenames)
#SBATCH --gres=gpu:1               # Request 1 GPU

# This part runs when the job is actually executed by SLURM
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Total videos: ${num_videos}"

# Validate array task ID
if [ "$SLURM_ARRAY_TASK_ID" -ge "$num_videos" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) >= number of videos ($num_videos)"
    exit 1
fi

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

# Get the video file for this array task
video_file=${videos[$SLURM_ARRAY_TASK_ID]}

video_path="/home/azhuravl/nobackup/DAVIS_testing/trainval/$video_file"

# Set radius
radius=0

echo "Processing video: $video_file (Task ID: $SLURM_ARRAY_TASK_ID)"
echo "Video path: $video_path"

# Check if video file exists
if [ ! -f "$video_path" ]; then
    echo "Error: Video file $video_path not found!"
    exit 1
fi

# Run the inference
python inference_orbits.py --video_path "$video_path" --radius $radius --test_run

echo "Completed processing $video_file"