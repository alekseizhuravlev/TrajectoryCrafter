"""
Autoregressive trajectory generation for large camera movements
"""

# from demo import TrajCrafter
import os
from datetime import datetime
import argparse
import torch
import copy
import time
import sys
import tempfile
from pathlib import Path


# Add core.py to path if needed
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/28_08_25_trajectories')
from core_autoregressive import TrajCrafterAutoregressive


def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--video_path', type=str, help='Input path')
    parser.add_argument(
        '--out_dir', type=str, default='./experiments/', help='Output dir'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='The device to use'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name, use video file name by default',
    )
    parser.add_argument(
        '--seed', type=int, default=43, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--video_length', type=int, default=49, help='Length of the video frames'
    )
    parser.add_argument('--fps', type=int, default=10, help='Fps for saved video')
    parser.add_argument(
        '--stride', type=int, default=1, help='Sampling stride for input video'
    )
    parser.add_argument('--server_name', type=str, help='Server IP address')

    ## autoregressive specific
    parser.add_argument(
        '--n_splits', 
        type=int, 
        default=4, 
        help='Number of parts to split trajectory into'
    )
    parser.add_argument(
        '--overlap_frames', 
        type=int, 
        default=8, 
        help='Number of frames to overlap between segments'
    )
    parser.add_argument(
        '--test_run',
        action='store_true',
        default=False,
        help='Run only the right_180 trajectory for testing'
    )

    ## render
    parser.add_argument(
        '--radius_scale',
        type=float,
        default=1.0,
        help='Scale factor for the spherical radius',
    )
    parser.add_argument('--camera', type=str, default='target', help='traj or target')
    parser.add_argument(
        '--mode', type=str, default='gradual', help='gradual, bullet or direct'
    )
    parser.add_argument(
        '--mask', action='store_true', default=True, help='Clean the pcd if true'
    )
    parser.add_argument(
        '--traj_txt',
        type=str,
        help="Required for 'traj' camera, a txt file that specify camera trajectory",
    )
    parser.add_argument(
        '--target_pose',
        nargs=5,
        type=float,
        help="Required for 'target' mode, specify target camera pose, <theta phi r x y>",
    )
    parser.add_argument(
        '--near', type=float, default=0.0001, help='Near clipping plane distance'
    )
    parser.add_argument(
        '--far', type=float, default=10000.0, help='Far clipping plane distance'
    )
    parser.add_argument('--anchor_idx', type=int, default=0, help='One GT frame')

    ## diffusion
    parser.add_argument(
        '--low_gpu_memory_mode',
        type=bool,
        default=False,
        help='Enable low GPU memory mode',
    )
    parser.add_argument('--model_name', type=str, default='checkpoints/CogVideoX-Fun-V1.1-5b-InP', help='Path to the model')
    parser.add_argument(
        '--sampler_name',
        type=str,
        choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
        default='DDIM_Origin',
        help='Choose the sampler',
    )
    parser.add_argument(
        '--transformer_path',
        type=str,
        default="checkpoints/TrajectoryCrafter",
        help='Path to the pretrained transformer model',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[384, 672],
        help='Sample size as [height, width]',
    )
    parser.add_argument(
        '--diffusion_guidance_scale',
        type=float,
        default=6.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--diffusion_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps',
    )
    parser.add_argument(
        '--prompt', type=str, default=None, help='Prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
        help='Negative prompt for video generation',
    )
    parser.add_argument(
        '--refine_prompt',
        type=str,
        default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        help='Prompt for video generation',
    )
    parser.add_argument('--blip_path', type=str, default="checkpoints/blip2-opt-2.7b")

    ## depth
    parser.add_argument(
        '--unet_path',
        type=str,
        default="checkpoints/DepthCrafter",
        help='Path to the UNet model',
    )
    parser.add_argument(
        '--pre_train_path',
        type=str,
        default="checkpoints/stable-video-diffusion-img2vid",
        help='Path to the pre-trained model',
    )
    parser.add_argument(
        '--cpu_offload', type=str, 
        default='model', 
        help='CPU offload strategy'
    )
    parser.add_argument(
        '--depth_inference_steps', type=int, default=5, help='Number of inference steps'
    )
    parser.add_argument(
        '--depth_guidance_scale',
        type=float,
        default=1.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--window_size', type=int, default=110, help='Window size for processing'
    )
    parser.add_argument(
        '--overlap', type=int, default=25, help='Overlap size for processing'
    )
    parser.add_argument(
        '--max_res', type=int, default=1024, help='Maximum resolution for processing'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=1.0,
        help='Radius for camera orbit motions',
    )

    return parser


if __name__ == "__main__":
    import torch, os
    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("GPU:", torch.cuda.get_device_name(0))
    print("SLURM node:", os.environ.get("SLURMD_NODENAME"))

    parser = get_parser()
    opts_base = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    video_basename = os.path.splitext(os.path.basename(opts_base.video_path))[0]

    # Setup
    opts_base.weight_dtype = torch.bfloat16
    opts_base.exp_name = f"{video_basename}_{timestamp}_autoregressive"
    opts_base.save_dir = os.path.join(opts_base.out_dir, opts_base.exp_name)

    # Create TrajCrafterVisualization instance for autoregressive generation
    vis_crafter = TrajCrafterAutoregressive(opts_base)

    radius = opts_base.radius


    if opts_base.test_run:
        print("🧪 TEST RUN MODE: Running only one trajectory")
        variants = [
            ("right_90", [0, 90, radius, 0, 0]),
        ]
    else:
        print("🎬 FULL RUN MODE: Running all trajectories")
        variants = [
            ("left_-90",     [0, -90, radius, 0, 0]),
            ("right_90",     [0, 90, radius, 0, 0]),
            ("top_90",       [90, 0, radius, 0, 0]),
            ("right_180",    [0, 180, radius, 0, 0]),
            ("left_-180",    [0, -180, radius, 0, 0]),
            ("top_180",      [180, 0, radius, 0, 0]),
            ("left_-360",    [0, -360, radius, 0, 0]),
            ("right_360",    [0, 360, radius, 0, 0]),
        ]

    print(f"Will run {len(variants)} large trajectory variant(s)")
    print(f"Autoregressive settings: {opts_base.n_splits} splits, {opts_base.overlap_frames} overlap frames")

    for name, pose in variants:
        print(f"\n=== Running Autoregressive {name} ===")
        opts = copy.deepcopy(opts_base)
        opts.exp_name = f"{video_basename}_{timestamp}_{name}_auto_s{opts_base.n_splits}"
        opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
        opts.camera = "target"
        opts.mode = "gradual"
        opts.mask = True
        opts.target_pose = pose
        opts.traj_txt = 'test/trajs/loop2.txt'

        # Make directories
        os.makedirs(opts.save_dir, exist_ok=True)

        print(f"Target trajectory: θ={pose[0]}°, φ={pose[1]}°, r={pose[2]}, dx={pose[3]}, dy={pose[4]}")
        print(f"Output dir: {opts.save_dir}")

        start_time = time.time()

        try:
            # Use autoregressive generation for large trajectories
            final_video = vis_crafter.infer_autoregressive(
                opts, 
                n_splits=opts_base.n_splits,
                overlap_frames=opts_base.overlap_frames
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Finished {name} in {elapsed:.2f} seconds")
            print(f"Final video: {final_video}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error in {name} after {elapsed:.2f} seconds: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎉 Autoregressive generation complete for {len(variants)} variant(s)!")