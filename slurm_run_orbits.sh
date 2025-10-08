#!/bin/bash
#SBATCH -p gpu24             # Partition name
#SBATCH -t 07:55:00                # Time limit (HH:MM:SS)
#SBATCH -o /home/azhuravl/nobackup/slurm_out/%x_%A_%a.out   # Standard output log
#SBATCH -e /home/azhuravl/nobackup/slurm_err/%x_%A_%a.err   # Standard error log
#SBATCH --job-name=trajcraft       # Job name (used in log filenames)
#SBATCH --gres=gpu:h100:1              # Request 1 GPU
#SBATCH --array=0-2                # Array job indices (0 to 4 for 5 videos)

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

    # trainval

    # "judo.mp4"
    # "koala.mp4"
    # "rhino.mp4"

    # "crossing.mp4"
    # "dog-agility.mp4"
    # "dogs-jump.mp4"

    # "drift-chicane.mp4"
    # "longboard.mp4"
    # "miami-surf.mp4"

    # "pigs.mp4"
    # "schoolgirls.mp4"
    # "sheep.mp4"

    # "soccerball.mp4"

    # test-dev
    # "carousel.mp4"
    # "orchid.mp4"
    # "slackline.mp4"

    # test-challenge
    # "boxing.mp4"
    # "demolition.mp4"
    "dog-control.mp4"

    # "lions.mp4"
    # "monkeys.mp4"
    # "ocean-birds.mp4"

    "skydive.mp4"

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
    # "/home/azhuravl/nobackup/DAVIS_testing/trainval/"
    # "/home/azhuravl/nobackup/DAVIS_testing/trainval/"

    # "/home/azhuravl/nobackup/DAVIS_testing/test-dev/"
    # "/home/azhuravl/nobackup/DAVIS_testing/test-dev/"
    # "/home/azhuravl/nobackup/DAVIS_testing/test-dev/"
    # "/home/azhuravl/nobackup/DAVIS_testing/test-dev/"

    "/home/azhuravl/nobackup/DAVIS_testing/test-challenge/"
    "/home/azhuravl/nobackup/DAVIS_testing/test-challenge/"
    "/home/azhuravl/nobackup/DAVIS_testing/test-challenge/"
    # "/home/azhuravl/nobackup/DAVIS_testing/test-challenge"
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
    --mode gradual
    # --test_run

python inference_orbits.py \
    --video_path "$video_path" \
    --radius $radius \
    --mode direct

# python inference_autoregressive.py \
#     --video_path "$video_path" \
#     --n_splits 4 \
#     --overlap_frames 0 \
#     --radius $radius  \
#     --mode gradual 
    # --test_run \

# run autoregressive for 2, 4, 10 splits
split_counts=(2 4)
for n_splits in "${split_counts[@]}"; do
    echo "Running autoregressive inference with $n_splits splits..."
    python inference_autoregressive.py \
        --video_path "$video_path" \
        --n_splits $n_splits \
        --overlap_frames 0 \
        --radius $radius  \
        --mode direct 
        # --test_run 
done

# python notebooks/28_08_25_trajectories/test_autoregressive.py

echo "Completed processing $video_file"