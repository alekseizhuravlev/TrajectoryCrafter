import torch
import numpy as np
import copy
import os
from pathlib import Path
import cv2
import argparse

from demo import TrajCrafter
from models.utils import Warper, read_video_frames, sphere2pose, save_video
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoProcessor, Blip2ForConditionalGeneration


from models.infer import DepthCrafterDemo

import sys 

sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/28_08_25_trajectories')
from core_autoregressive import TrajCrafterAutoregressive

class TrajectoryTester(TrajCrafterAutoregressive):
    """Test trajectory generation without video generation"""
    
    def __init__(self, opts, gradio=False):
        self.funwarp = Warper(device=opts.device)
        # self.depth_estimater = VDADemo(pre_train_path=opts.pre_train_path_vda,device=opts.device)
        self.depth_estimater = DepthCrafterDemo(
            unet_path=opts.unet_path,
            pre_train_path=opts.pre_train_path,
            cpu_offload=opts.cpu_offload,
            device=opts.device,
        )
        # self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        # self.captioner = Blip2ForConditionalGeneration.from_pretrained(
        #     opts.blip_path, torch_dtype=torch.float16
        # ).to(opts.device)
        # self.setup_diffusion(opts)
        if gradio:
            self.opts = opts
            
        self.cumulative_pose = [0.0, 0.0, 0.0, 0.0, 0.0]  # Track cumulative state

    
    def test_trajectory_only(self, opts, n_splits=3):
        """Test only trajectory generation and create render videos"""
        
        # Setup directories
        test_dir = Path(opts.save_dir) / "trajectory_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        original_target_pose = opts.target_pose.copy()
        dtheta, dphi, dr, dx, dy = original_target_pose
        
        # Calculate trajectory deltas (same as original)
        theta_deltas = [dtheta * (i + 1) / n_splits for i in range(n_splits)]
        phi_deltas = [dphi * (i + 1) / n_splits for i in range(n_splits)]
        dr_deltas = [dr * (i + 1) / n_splits for i in range(n_splits)]
        dx_deltas = [dx * (i + 1) / n_splits for i in range(n_splits)]
        dy_deltas = [dy * (i + 1) / n_splits for i in range(n_splits)]
        
        theta_deltas = [theta_deltas[0]] + [theta_deltas[i] - theta_deltas[i-1] for i in range(1, n_splits)]
        phi_deltas = [phi_deltas[0]] + [phi_deltas[i] - phi_deltas[i-1] for i in range(1, n_splits)]
        dr_deltas = [dr_deltas[0]] + [dr_deltas[i] - dr_deltas[i-1] for i in range(1, n_splits)]
        dx_deltas = [dx_deltas[0]] + [dx_deltas[i] - dx_deltas[i-1] for i in range(1, n_splits)]
        dy_deltas = [dy_deltas[0]] + [dy_deltas[i] - dy_deltas[i-1] for i in range(1, n_splits)]
        
        # Reset cumulative pose for testing
        self.cumulative_pose = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Initialize
        current_poses = None
        all_render_videos = []
        
        for segment_idx in range(n_splits):
            print(f"\n=== Testing Segment {segment_idx + 1}/{n_splits} ===")
            
            # Set segment directory
            segment_dir = test_dir / f"segment_{segment_idx:02d}"
            segment_dir.mkdir(exist_ok=True)
            opts.save_dir = str(segment_dir)
            
            # Set trajectory for this segment
            opts.target_pose = [theta_deltas[segment_idx], phi_deltas[segment_idx], 
                               dr_deltas[segment_idx], dx_deltas[segment_idx], dy_deltas[segment_idx]]
            
            # Test segment (only rendering, no video generation)
            current_poses = self._test_segment_render_only(opts, current_poses, segment_idx)
            
            # Track render video
            render_path = segment_dir / "render.mp4"
            if render_path.exists():
                all_render_videos.append(str(render_path))
                opts.video_path = str(render_path)  # Use for next segment input
        
        # Create trajectory summary
        self._create_trajectory_summary(test_dir, all_render_videos, n_splits)
        
        print(f"\n✅ Trajectory test completed! Check: {test_dir}")
        return str(test_dir)
    
    def _test_segment_render_only(self, opts, previous_poses, segment_idx):
        """Test single segment - only create render video"""
        
        # Read and prepare frames
        frames = read_video_frames(opts.video_path, opts.video_length, opts.stride, opts.max_res)
        if frames.shape[0] < opts.video_length:
            last_frame = frames[-1:]
            num_pad = opts.video_length - frames.shape[0]
            pad_frames = np.repeat(last_frame, num_pad, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
        
        # Get depth and poses
        depths = self.depth_estimater.infer(frames, opts.near, opts.far, 
                                           opts.depth_inference_steps, opts.depth_guidance_scale,
                                           window_size=opts.window_size, overlap=opts.overlap).to(opts.device)
        
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        
        # Get poses with continuation
        pose_s, pose_t = self._get_poses_with_continuation(opts, depths, previous_poses, segment_idx)
        
        # Get intrinsics and warp frames
        K = self._get_intrinsics(opts, depths)
        warped_images, masks = self._warp_frames(frames, depths, pose_s, pose_t, K, opts)
        
        # Save ONLY render videos (no diffusion generation)
        self._save_render_only(frames, warped_images, masks, opts)
        
        return pose_t  # Return poses for next segment
    
    def _save_render_only(self, frames, warped_images, masks, opts):
        """Save only input, render and mask videos - no diffusion generation"""
        
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        
        # Resize for consistent output
        frames = F.interpolate(frames, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_video = F.interpolate(cond_video, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        
        # Save videos
        save_video((frames.permute(0, 2, 3, 1) + 1.0) / 2.0, 
                   os.path.join(opts.save_dir, 'input.mp4'), fps=opts.fps)
        save_video(cond_video.permute(0, 2, 3, 1), 
                   os.path.join(opts.save_dir, 'render.mp4'), fps=opts.fps)
        save_video(cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1), 
                   os.path.join(opts.save_dir, 'mask.mp4'), fps=opts.fps)
        
        print(f"✅ Render videos saved to: {opts.save_dir}")
    
    def _create_trajectory_summary(self, test_dir, render_videos, n_splits):
        """Create summary of trajectory test"""
        
        summary_path = test_dir / "trajectory_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("TRAJECTORY TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of segments: {n_splits}\n")
            f.write(f"Cumulative poses tracked:\n")
            for i, pose in enumerate(self.cumulative_pose):
                labels = ['theta', 'phi', 'dr', 'dx', 'dy']
                f.write(f"  {labels[i]}: {pose:.3f}\n")
            f.write(f"\nRender videos created:\n")
            for i, video in enumerate(render_videos):
                f.write(f"  Segment {i+1}: {video}\n")
        
        print(f"📄 Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='/home/azhuravl/nobackup/DAVIS_testing/trainval/judo.mp4', help='Input video path')
    parser.add_argument('--save_dir', type=str, default='./experiments/trajectory_test_output', help='Output directory')
    parser.add_argument('--target_pose', type=float, nargs=5, default=[0.0, 90.0, 0.0, 0.0, 0.0], 
                       help='Target pose [theta, phi, dr, dx, dy]')
    parser.add_argument('--n_splits', type=int, default=6, help='Number of trajectory segments')
    parser.add_argument('--video_length', type=int, default=49, help='Video length')
    parser.add_argument('--stride', type=int, default=3, help='Video stride')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--max_res', type=int, default=1024, help='Max resolution')
    parser.add_argument('--sample_size', type=int, nargs=2, default=[576, 1024], help='Sample size [H, W]')
    parser.add_argument('--fps', type=int, default=8, help='Output FPS')
    
    # General parameters from get_parser()
    parser.add_argument('--out_dir', type=str, default='./experiments/', help='Output dir')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name, use video file name by default')
    parser.add_argument('--seed', type=int, default=43, help='Random seed for reproducibility')
    parser.add_argument('--server_name', type=str, help='Server IP address')
    
    # Autoregressive specific
    parser.add_argument('--overlap_frames', type=int, default=8, help='Number of frames to overlap between segments')
    parser.add_argument('--test_run', action='store_true', default=False, help='Run only the right_180 trajectory for testing')
    
    # Render parameters
    parser.add_argument('--radius_scale', type=float, default=1.0, help='Scale factor for the spherical radius')
    parser.add_argument('--camera', type=str, default='target', help='traj or target')
    parser.add_argument('--mode', type=str, default='gradual', help='gradual, bullet or direct')
    parser.add_argument('--mask', action='store_true', default=True, help='Clean the pcd if true')
    parser.add_argument('--traj_txt', type=str, help="Required for 'traj' camera, a txt file that specify camera trajectory")
    parser.add_argument('--near', type=float, default=0.0001, help='Near clipping plane distance')
    parser.add_argument('--far', type=float, default=10000.0, help='Far clipping plane distance')
    parser.add_argument('--anchor_idx', type=int, default=0, help='One GT frame')
    parser.add_argument('--radius', type=float, default=1.0, help='Radius for camera orbit motions')
    
    # Diffusion parameters
    parser.add_argument('--low_gpu_memory_mode', type=bool, default=False, help='Enable low GPU memory mode')
    parser.add_argument('--model_name', type=str, default='checkpoints/CogVideoX-Fun-V1.1-5b-InP', help='Path to the model')
    parser.add_argument('--sampler_name', type=str, choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
                       default='DDIM_Origin', help='Choose the sampler')
    parser.add_argument('--transformer_path', type=str, default="checkpoints/TrajectoryCrafter", help='Path to the pretrained transformer model')
    parser.add_argument('--diffusion_guidance_scale', type=float, default=6.0, help='Guidance scale for inference')
    parser.add_argument('--diffusion_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt for video generation')
    parser.add_argument('--negative_prompt', type=str, 
                       default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                       help='Negative prompt for video generation')
    parser.add_argument('--refine_prompt', type=str,
                       default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
                       help='Prompt for video generation')
    parser.add_argument('--blip_path', type=str, default="checkpoints/blip2-opt-2.7b")
    
    # Depth parameters (updated from get_parser)
    parser.add_argument('--unet_path', type=str, default="checkpoints/DepthCrafter", help='Path to the UNet model')
    parser.add_argument('--pre_train_path', type=str, default="checkpoints/stable-video-diffusion-img2vid", help='Path to the pre-trained model')
    parser.add_argument('--cpu_offload', type=str, default='model', help='CPU offload strategy')
    parser.add_argument('--depth_inference_steps', type=int, default=5, help='Number of inference steps')  # Updated default
    parser.add_argument('--depth_guidance_scale', type=float, default=1.0, help='Guidance scale for inference')  # Updated default
    parser.add_argument('--window_size', type=int, default=110, help='Window size for processing')
    parser.add_argument('--overlap', type=int, default=25, help='Overlap size for processing')
    
    # Additional parameters that might be needed
    parser.add_argument('--weight_dtype', type=str, default='bfloat16', help='Weight dtype')
    
    
    
    
    
    
    
    
    
    
    
    
    
    args = parser.parse_args()
    
    # SET MISSING ATTRIBUTES THAT CORE_AUTOREGRESSIVE EXPECTS:
    args.input_pose = [0.0, 0.0, 0.0, 0.0, 0.0]  # Starting pose
    args.source_poses = None  # Will be set in _get_poses_with_continuation
    
    # Convert weight_dtype string to torch dtype
    if hasattr(args, 'weight_dtype'):
        if args.weight_dtype == 'bfloat16':
            args.weight_dtype = torch.bfloat16
        elif args.weight_dtype == 'float16':
            args.weight_dtype = torch.float16
        else:
            args.weight_dtype = torch.float32
    
    # Initialize trajectory tester
    print("🚀 Initializing trajectory tester...")
    tester = TrajectoryTester(args)  # Pass args to constructor
    
    # Set up depth estimator
    # tester.depth_estimater = DepthCrafterDemo()
    # tester.funwarp = Warper()
    
    # Set input pose for autoregressive
    args.input_pose = [0.0, 0.0, 0.0, 0.0, 0.0]  # Starting pose
    args.camera = 'target'
    
    # Run trajectory test
    print(f"🎯 Testing trajectory: {args.target_pose} in {args.n_splits} segments")
    result_dir = tester.test_trajectory_only(args, args.n_splits)
    
    print(f"\n✅ Test completed! Results in: {result_dir}")
    print("📹 Check the render.mp4 files in each segment folder to see trajectory progression")

if __name__ == "__main__":
    main()