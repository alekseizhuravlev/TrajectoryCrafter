import sys
sys.path.append('/home/azhuravl/work')

import stereoanyvideo.datasets.video_datasets as video_datasets

from pytorch3d.renderer.camera_utils import join_cameras_as_batch

import torch

sys.path.append('/home/azhuravl/work/TrajectoryCrafter')

import models.utils as utils

sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/06_10_25_vggt')
from parsing import get_parser
import utils_autoregressive as utils_ar
from datetime import datetime
import os
import copy
import logging
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection
from tqdm import tqdm
from models.utils import save_video
import numpy as np
import argparse


def extract_video_data(data, baseline=1, image_size=(540, 960)):
    """
    Extract frames, depths, poses, and camera intrinsics from data object.
    
    Args:
        data: Data object containing 'img', 'disp', and 'viewpoint'
        baseline: Baseline for depth calculation (default: 1)
    
    Returns:
        frames_tensor: [T, 3, H, W] in [-1, 1] range
        depths: [T, 1, H, W] depth maps
        poses_tensor: [T, 4, 4] camera poses
        K_tensor: [T, 3, 3] camera intrinsics
    """
    # Convert to [-1, 1] range
    frames_tensor = data['img'][:,0] / 127.5 - 1.0  # [T, 3, H, W]
    disparity_tensor = data['disp'][:,0]  # [T, 1, H, W]
    
    image_size = torch.tensor(image_size)
    
    viewpoints_left = [data["viewpoint"][i][0] for i in range(frames_tensor.shape[0])]
    viewpoints_batch = join_cameras_as_batch(viewpoints_left)
    
    R, t, K = opencv_from_cameras_projection(
        viewpoints_batch,
        image_size=image_size.repeat(frames_tensor.shape[0], 1)
    )
    poses_tensor = torch.eye(4).unsqueeze(0).repeat(frames_tensor.shape[0], 1, 1)  # [T, 4, 4]
    poses_tensor[:, :3, :3] = R
    poses_tensor[:, :3, 3] = t.squeeze(1)
    
    transform_mat = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)
    poses_tensor = poses_tensor @ transform_mat
    
    # invert poses to get camera-to-world
    poses_tensor = torch.inverse(poses_tensor)
    
    K_tensor = K  # [T, 3, 3]
    
    # Calculate focal length from K tensor
    focal_length = K_tensor[0, 0, 0]
    
    depths = -(focal_length * baseline) / disparity_tensor  # [T, 1, H, W]
    
    return frames_tensor, depths, poses_tensor, K_tensor


class CameraMotionFilter:
    def __init__(self, device='cuda'):
        """
        Args:
            device: torch device to run computations on
        """
        self.min_total_translation = 10
        self.max_total_translation = 100
        
        self.min_total_rotation = 0.1
        self.max_total_rotation = 0.55  # ~30 degrees
        
        self.device = device
    
    def compute_motion_metrics(self, poses):
        """Compute motion metrics for a single video using PyTorch"""
        # Ensure tensor is on correct device
        if isinstance(poses, torch.Tensor):
            poses = poses.to(self.device)
        else:
            poses = torch.tensor(poses, dtype=torch.float32, device=self.device)
        
        if poses.shape[1:] == (4, 4):
            # Extract translations and rotations
            translations = poses[:, :3, 3]  # [n_frames, 3]
            rotations = poses[:, :3, :3]    # [n_frames, 3, 3]
        else:
            raise ValueError("Expected poses shape: (n_frames, 4, 4)")
        
        # Frame-to-frame translation distances
        trans_diffs = torch.diff(translations, dim=0)  # [n_frames-1, 3]
        trans_distances = torch.linalg.norm(trans_diffs, dim=1)  # [n_frames-1]
        
        # Frame-to-frame rotation angles (vectorized)
        rotations_curr = rotations[1:]  # [n_frames-1, 3, 3]
        rotations_prev = rotations[:-1] # [n_frames-1, 3, 3]
        
        # Relative rotations: R_rel = R_curr @ R_prev^T
        R_rel = torch.bmm(rotations_curr, rotations_prev.transpose(-1, -2))  # [n_frames-1, 3, 3]
        
        # Rotation angles from trace: angle = arccos((trace(R) - 1) / 2)
        traces = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(dim=-1)  # [n_frames-1]
        angles_arg = (traces - 1) / 2
        angles_arg = torch.clamp(angles_arg, -1.0, 1.0)  # Clamp for numerical stability
        rotation_angles = torch.acos(angles_arg)  # [n_frames-1]
        
        metrics = {
            'max_frame_translation': torch.max(trans_distances).item(),
            'mean_frame_translation': torch.mean(trans_distances).item(),
            'total_translation': torch.sum(trans_distances).item(),
            'max_frame_rotation': torch.max(rotation_angles).item(),
            'mean_frame_rotation': torch.mean(rotation_angles).item(),
            'total_rotation': torch.sum(rotation_angles).item(),
            # 'translation_std': torch.std(trans_distances).item(),
            # 'rotation_std': torch.std(rotation_angles).item()
        }
        
        return metrics
    
    def is_low_motion(self, poses):
        """Check if video meets motion criteria"""
        metrics = self.compute_motion_metrics(poses)
        
        # Check all criteria
        translation_ok = (metrics['total_translation'] <= self.max_total_translation)
        rotation_ok = (metrics['total_rotation'] <= self.max_total_rotation)
        
        non_zero = (metrics['total_translation'] >= self.min_total_translation) or \
                     (metrics['total_rotation'] >= self.min_total_rotation)
        
        criteria = [translation_ok, rotation_ok, non_zero]
        
        return all(criteria), metrics
    
    
    
def setup_trajcrafter():

    sys.argv = [
        "",
        "--video_path", "monkaa.mp4",
        "--n_splits", "4",
        "--overlap_frames", "0",
        "--radius", "0",
        "--mode", "gradual",
    ]

    parser = get_parser()
    opts_base = parser.parse_args()

    opts_base.exp_name = "monkaa"
    # opts_base.out_dir = '/home/azhuravl/scratch/linear_probing_fixed_8'
    
    # opts_base.save_dir = os.path.join(
    #     opts_base.out_dir, 
    #     opts_base.exp_name
    #     )
    
    opts_base.weight_dtype = torch.bfloat16
    opts_base.camera = "target"
    opts_base.target_pose = [90, 0, 0, 0, 1]
    opts_base.traj_txt = 'test/trajs/loop2.txt'
    
    return opts_base


if __name__ == '__main__':
    
    # python /home/azhuravl/work/TrajectoryCrafter/notebooks/15_10_25_depth/collect_dataset.py --process_id 1 --n_processes 4 --num_samples 10
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments for array job
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_id', type=int, required=True, help='Array job process ID')
    parser.add_argument('--n_processes', type=int, required=True, help='Total number of processes')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per process')
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    
    opts = setup_trajcrafter()
    opts.process_id = args.process_id
    opts.n_processes = args.n_processes
    opts.num_samples = args.num_samples
    opts.out_dir = f'/home/azhuravl/scratch/{args.output_dir}'
    os.makedirs(opts.out_dir)


    dataset_monkaa = video_datasets.SequenceSceneFlowDatasetCamera(
        aug_params=None,
        root="/home/azhuravl/scratch/SceneFlow",
        dstype="frames_cleanpass",
        sample_len=59,
        things_test=False,
        add_things=False,
        add_monkaa=True,
        add_driving=False,
        split="test",
        stride = 8,
    ) # 994 samples
    
    # Calculate start index and end index for this process
    total_samples = len(dataset_monkaa)
    samples_per_process = total_samples // opts.n_processes
    idx_start = samples_per_process * opts.process_id

    logger.info(f"Process {opts.process_id}/{opts.n_processes}: processing samples {idx_start}")
    
    filter_motion = CameraMotionFilter()
    warper_old = utils.Warper(device='cuda')
    trajcrafter = utils_ar.TrajCrafterAutoregressive(opts)
    
    current_idx = idx_start
    samples_processed = 0
    
    
    while samples_processed < opts.num_samples and current_idx < total_samples:
        
        logger.info(f"Processing sample index: {current_idx}")
        logger.info(f"Samples processed: {samples_processed}/{opts.num_samples}")
        
        # Extract video data        
        data_0 = dataset_monkaa[current_idx]
        frames_tensor, depths, poses_tensor, K_tensor = extract_video_data(data_0)
        
        # Check motion criteria
        low_motion, metrics = filter_motion.is_low_motion(poses_tensor)
        logger.info(f"Low motion: {low_motion}, Metrics: {metrics}")
        
        if not low_motion:
            logger.info("Skipping due to high motion.")
            current_idx += 1
            continue
        
        # make output directory
        opts.exp_name = f"monkaa_{current_idx:04d}"
        opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)     
        os.makedirs(opts.save_dir, exist_ok=True)
        
        # warping
        warped_images = []
        masks = []
        warped_depths = []

        for i in tqdm(range(10, frames_tensor.shape[0]), desc="Warping frames"):
            warped_frame2, mask2, warped_depth2, flow12 = warper_old.forward_warp(
                frame1=frames_tensor[i:i+1],
                mask1=None,
                depth1=depths[i:i+1],
                transformation1=poses_tensor[i:i+1],
                transformation2=poses_tensor[10:11],
                intrinsic1=K_tensor[i:i+1],
                intrinsic2=K_tensor[i:i+1],
                mask=False,
                twice=True,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
            warped_depths.append(warped_depth2)
            
        # save depths, warped depths, camera poses to save_dir         
        # as .pt files    
        logger.info("Saving depths, warped depths, and poses...")
        
        warped_depths_tensor = torch.cat(warped_depths)   
        depths_dir = os.path.join(opts.save_dir, 'depths')
        os.makedirs(depths_dir, exist_ok=True)
        
        torch.save(depths[10:].cpu(), os.path.join(depths_dir, 'depths.pt'))
        torch.save(warped_depths_tensor.cpu(), os.path.join(depths_dir, 'warped_depths.pt'))
        torch.save(poses_tensor[10:].cpu(), os.path.join(depths_dir, 'poses.pt'))
        
        # as rendered videos
        save_video(
            (warped_depths_tensor.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)) / warped_depths_tensor.max() ,
            os.path.join(depths_dir, 'warped_depths.mp4'),
            fps=10,
        )
        save_video(
            (depths[10:].permute(0, 2, 3, 1).repeat(1, 1, 1, 3)) / depths.max() ,
            os.path.join(depths_dir, 'depths.mp4'),
            fps=10,
        )
        
        # prompt
        frames_np = ((frames_tensor.cpu().permute(0, 2, 3, 1).numpy() + 1.0) / 2.0).astype(np.float32)
        trajcrafter.prompt = trajcrafter.get_caption(opts, frames_np[opts.video_length // 2])
        logger.info(f"Prompt: {trajcrafter.prompt}")
        
        # sample diffusion
        logger.info("Sampling diffusion...")
        frames_inpainted, segment_dir = utils_ar.sample_diffusion(
            trajcrafter,
            frames_tensor[10:],
            warped_images,
            frames_tensor[:10],
            masks,
            opts,
        )
        
        # collect the features
        logger.info("Collecting and saving features...")
        collected_features = trajcrafter.pipeline.collected_features
        
        # save the features
        for timestep in collected_features.keys():
            for keys in collected_features[timestep].keys():
                feature_tensor = collected_features[timestep][keys]
                
                feature_dir = os.path.join(opts.save_dir, 'features', timestep)
                os.makedirs(feature_dir, exist_ok=True)
                
                feature_path = os.path.join(feature_dir, f"{keys}.pt")
                torch.save(feature_tensor.cpu(), feature_path)
        
        
        samples_processed += 1
        current_idx += 1
