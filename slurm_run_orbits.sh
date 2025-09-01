#!/bin/bash
#SBATCH -p gpu22             # Partition name
#SBATCH -t 06:00:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/%x_%A_%a.out   # Standard output log
#SBATCH -e /home/azhuravl/nobackup/slurm_err/%x_%A_%a.err   # Standard error log
#SBATCH --job-name=trajcraft       # Job name (used in log filenames)
#SBATCH --gres=gpu:a100:1              # Request 1 GPU
#SBATCH --array=0-0                # Array job indices (0 to 4 for 5 videos)

# Activate conda environment
source ~/.bashrc
conda activate trajcrafter

# Navigate to working directory
cd /home/azhuravl/work/TrajectoryCrafter

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define video files array
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
# Define video paths array
video_paths=(
    # "./test/videos/"
    # "./test/videos/"
    # "./test/videos/"
    # "./test/videos/"
    # "./test/videos/"
    # "/home/azhuravl/nobackup/DAVIS_testing/trainval/"
    # "/home/azhuravl/nobackup/DAVIS_testing/trainval/"
    "/home/azhuravl/nobackup/DAVIS_testing/trainval/"
    # "/home/azhuravl/nobackup/DAVIS_testing/trainval/"
    # "/home/azhuravl/nobackup/DAVIS_testing/trainval/"
)

# Get the video file and path for this array task
video_file=${videos[$SLURM_ARRAY_TASK_ID]}
video_path="${video_paths[$SLURM_ARRAY_TASK_ID]}$video_file"

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
python inference_orbits.py \
    --video_path "$video_path" \
    --radius $radius \
    --mode direct \
    # --test_run \


# python inference_autoregressive.py \
#     --video_path "$video_path" \
#     --n_splits 20 \
#     --overlap_frames 0 \
#     --test_run \
#     --radius $radius 

# python notebooks/28_08_25_trajectories/test_autoregressive.py

echo "Completed processing $video_file"