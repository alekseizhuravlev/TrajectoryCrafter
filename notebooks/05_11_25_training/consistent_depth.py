import sys
import os
import cv2
import copy
import time
import warnings
import argparse
import numpy as np
import torch
import torch.optim 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
from importlib import import_module 
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose
import logging
from datetime import datetime

# Add paths
sys.path.append('/home/azhuravl/work/Video-Depth-Anything')
sys.path.append('/home/azhuravl/work/Video-Depth-Anything/video_depth_anything/util')
# sys.path.append('/home/azhuravl/work/Video-Depth-Anything/loss')

# Video Depth Anything imports
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
from transform import Resize, NormalizeImage, PrepareForNet

from loss.loss import compute_scale_and_shift, VideoDepthLoss
import json

# Add this import at the top
import sys
sys.path.append('/home/azhuravl/work/Video-Depth-Anything/benchmark/eval')
from eval_tae import tae_torch


# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Consistent Depth Training')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing input_video.mp4, input_depths.pt, and warped_depths.pt')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory where all results will be saved')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    return parser.parse_args()


def setup_logging(debug_dir, exp_name="consistent_depth"):
    """
    Setup logging to both console and file.
    """
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(debug_dir, f"{exp_name}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger


def prepare_frames(frames, input_size=518):
    """
    Prepare frames for inference by resizing and normalizing.
    
    Args:
        frames: numpy array of shape [T, H, W, C] containing video frames
        input_size: target input size for the model
    
    Returns:
        torch.Tensor: processed frames ready for model input [1, T, C, H, W]
        tuple: original frame dimensions (height, width)
    """
    if frames.shape[0] != INFER_LEN:
        raise ValueError(f"Expected {INFER_LEN} frames, but got {frames.shape[0]} frames")
    
    frame_height, frame_width = frames[0].shape[:2]
    ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    
    # Adjust input size for very wide/tall videos
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14

    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Process all frames
    processed_frames = []
    for i in range(INFER_LEN):
        frame_tensor = torch.from_numpy(
            transform({'image': frames[i].astype(np.float32) / 255.0})['image']
        ).unsqueeze(0).unsqueeze(0)
        processed_frames.append(frame_tensor)
    
    input_tensor = torch.cat(processed_frames, dim=1)
    
    return input_tensor, (frame_height, frame_width)


def denormalize_rgb(tensor):
    """
    Denormalize RGB tensor that was normalized with ImageNet stats.
    
    Args:
        tensor: normalized tensor [B, T, C, H, W] or [T, C, H, W]
        
    Returns:
        denormalized tensor in range [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    
    if len(tensor.shape) == 4:  # [T, C, H, W]
        mean = mean.squeeze(0)  # [1, 3, 1, 1]
        std = std.squeeze(0)
    
    mean = mean.to(tensor.device, tensor.dtype)
    std = std.to(tensor.device, tensor.dtype)
    
    return tensor * std + mean


def save_debug_videos(frames_resized, depths_gt_inv_resized, depths_warped_inv_resized, 
                     depth_gt_inv, depth_warped_inv, debug_dir, fps=25):
    """
    Save debug videos using Video-Depth-Anything's save_video function.
    """
    os.makedirs(debug_dir, exist_ok=True)
    # print("Saving debug videos...")
    
    # Denormalize and convert RGB frames
    rgb_denorm = denormalize_rgb(frames_resized.cpu())
    rgb_clamped = torch.clamp(rgb_denorm, 0, 1)
    rgb_frames = (rgb_clamped[0] * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)  # [T, H, W, C]
    
    # Convert depth tensors to numpy arrays
    gt_depth_resized = depths_gt_inv_resized[0, :, 0].cpu().numpy()  # [T, H, W]
    warped_depth_resized = depths_warped_inv_resized[0, :, 0].cpu().numpy()  # [T, H, W]
    gt_depth_original = depth_gt_inv.cpu().numpy()  # [T, H, W]
    warped_depth_original = depth_warped_inv.cpu().numpy()  # [T, H, W]
    
    # Save videos using save_video function from Video-Depth-Anything
    save_video(rgb_frames, f'{debug_dir}/resized_video.mp4', fps=fps)
    save_video(gt_depth_resized, f'{debug_dir}/gt_depth_inv_resized.mp4', fps=fps, is_depths=True, grayscale=False)
    save_video(warped_depth_resized, f'{debug_dir}/warped_depth_inv_resized.mp4', fps=fps, is_depths=True, grayscale=False)
    save_video(gt_depth_original, f'{debug_dir}/gt_depth_inv_original.mp4', fps=fps, is_depths=True, grayscale=False)
    save_video(warped_depth_original, f'{debug_dir}/warped_depth_inv_original.mp4', fps=fps, is_depths=True, grayscale=False)
    
    # Also save as tensors for programmatic analysis
    torch.save(rgb_clamped, f'{debug_dir}/resized_video.pt')
    torch.save(depths_gt_inv_resized, f'{debug_dir}/gt_depth_inv_resized.pt')
    torch.save(depths_warped_inv_resized, f'{debug_dir}/warped_depth_inv_resized.pt')
    
    # print(f"Debug videos saved to {debug_dir}")


class Arguments:
    def __init__(self):
        self.gpu = '0'
        self.random_seed = 2025
        self.epochs = 50
        self.exp_name = 'base'
        self.mode = 'VP'  # choices=['VP', 'FT']
        self.dataset = 'ibims'  # choices=['ibims', 'ddad']
        self.dataset_path = '/workspace/data_all'


class ArgsVDA:
    def __init__(self):
        self.input_video = '/home/azhuravl/scratch/datasets_latents/monkaa_1000/000/videos/input_video.mp4'
        self.output_dir = '/home/azhuravl/work/Video-Depth-Anything/outputs'
        self.input_size = 256
        self.max_res = 1280
        self.encoder = 'vitl'
        self.max_len = -1
        self.target_fps = -1
        self.metric = False
        self.fp32 = False
        self.grayscale = False
        self.save_npz = False
        self.save_exr = False
        self.focal_length_x = 470.4
        self.focal_length_y = 470.4


# def compute_scale_and_shift(predicted_depth, sparse_depth):
#     valid_mask = (sparse_depth > 0)
    
#     pred_valid = predicted_depth[valid_mask]   
#     sparse_valid = sparse_depth[valid_mask]    
    
#     if pred_valid.numel() == 0:
#         device = predicted_depth.device
#         dtype = predicted_depth.dtype
#         return torch.tensor(1.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype)
    
#     X = torch.stack([pred_valid, torch.ones_like(pred_valid)], dim=1)
    
#     a = torch.pinverse(X) @ sparse_valid 
#     scale = a[0]
#     shift = a[1]
    
#     return scale, shift


# def compute_scale_and_shift(prediction, target, mask):
#     """
#     Compute scale and shift parameters using least squares method.
    
#     Args:
#         prediction: predicted depth [B, T, H, W] or any shape
#         target: target depth, same shape as prediction
#         mask: valid mask, same shape as prediction
        
#     Returns:
#         scale: scale parameter (tensor)
#         shift: shift parameter (tensor)
#     """
#     # Convert to float32 for numerical stability
#     prediction = prediction.float()
#     target = target.float()
#     mask = mask.float()
    
#     # Flatten tensors for computation
#     pred_flat = prediction.view(-1)
#     target_flat = target.view(-1)
#     mask_flat = mask.view(-1)
    
#     # System matrix: A = [[a_00, a_01], [a_10, a_11]]
#     a_00 = torch.sum(mask_flat * pred_flat * pred_flat)
#     a_01 = torch.sum(mask_flat * pred_flat)
#     a_11 = torch.sum(mask_flat)
    
#     # Right hand side: b = [b_0, b_1]
#     b_0 = torch.sum(mask_flat * pred_flat * target_flat)
#     b_1 = torch.sum(mask_flat * target_flat)
    
#     # Default values
#     x_0 = torch.tensor(1.0, device=prediction.device, dtype=prediction.dtype)
#     x_1 = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    
#     # Solve the system: A * x = b
#     det = a_00 * a_11 - a_01 * a_01
    
#     # Check if determinant is not zero (avoid division by zero)
#     eps = 1e-8
#     det_nonzero = torch.abs(det) > eps
    
#     if det_nonzero:
#         x_0 = (a_11 * b_0 - a_01 * b_1) / det
#         x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    
#     return x_0, x_1


class SimpleDepthLoss(nn.Module):
    def __init__(self, l1_weight=1.0, rmse_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.rmse_weight = rmse_weight

    def forward(self, prediction, target, mask):
        '''
            prediction: predicted depth tensor
            target: target depth tensor  
            mask: valid pixel mask
        '''
        loss_dict = {}
        
        if mask.sum() == 0:
            # No valid pixels, return zero losses
            device = prediction.device
            zero_loss = torch.tensor(0.0, device=device, dtype=prediction.dtype)
            loss_dict['l1_loss'] = zero_loss
            loss_dict['rmse_loss'] = zero_loss
            loss_dict['total_loss'] = zero_loss
            return loss_dict
        
        # Extract valid pixels
        pred_valid = prediction[mask]
        target_valid = target[mask]
        
        # Compute individual losses
        loss_dict['l1_loss'] = F.l1_loss(pred_valid, target_valid) * self.l1_weight
        loss_dict['rmse_loss'] = torch.sqrt(F.mse_loss(pred_valid, target_valid)) * self.rmse_weight
        
        # Total loss
        loss_dict['total_loss'] = loss_dict['l1_loss'] + loss_dict['rmse_loss']
        
        return loss_dict


# class CombinedDepthLoss(nn.Module):
#     def __init__(self, vda_weight=1.0, simple_weight=1.0, l1_weight=1.0, rmse_weight=1.0):
#         super().__init__()
#         self.vda_loss = VideoDepthLoss()
#         self.simple_loss = SimpleDepthLoss(l1_weight=l1_weight, rmse_weight=rmse_weight)
#         self.vda_weight = vda_weight
#         self.simple_weight = simple_weight

#     def forward(self, prediction, target, mask):
#         '''
#             prediction: predicted depth tensor
#             target: target depth tensor  
#             mask: valid pixel mask
#         '''
#         # Get VDA loss
#         vda_loss_dict = self.vda_loss(prediction, target, mask)
        
#         # Get Simple loss
#         simple_loss_dict = self.simple_loss(prediction, target, mask)
        
#         # Combine losses
#         loss_dict = {}
        
#         # Add individual VDA losses with prefix
#         for key, value in vda_loss_dict.items():
#             if key != 'total_loss':
#                 loss_dict[f'vda_{key}'] = value
        
#         # Add individual Simple losses with prefix  
#         for key, value in simple_loss_dict.items():
#             if key != 'total_loss':
#                 loss_dict[f'simple_{key}'] = value
        
#         # Weighted combination
#         vda_total = vda_loss_dict['total_loss'] * self.vda_weight
#         simple_total = simple_loss_dict['total_loss'] * self.simple_weight
        
#         loss_dict['vda_total'] = vda_total
#         loss_dict['simple_total'] = simple_total
#         loss_dict['total_loss'] = vda_total + simple_total
        
#         return loss_dict



# Updated Combined Loss Class
class CombinedDepthLossWithTAE(nn.Module):
    def __init__(self, vda_weight=1.0, simple_weight=1.0, tae_weight=0.1, 
                 l1_weight=1.0, rmse_weight=1.0):
        super().__init__()
        self.vda_loss = VideoDepthLoss()
        self.simple_loss = SimpleDepthLoss(l1_weight=l1_weight, rmse_weight=rmse_weight)
        self.tae_loss = DifferentiableTAELoss(weight=tae_weight)
        self.vda_weight = vda_weight
        self.simple_weight = simple_weight
        self.tae_weight = tae_weight

    def forward(self, prediction, target, mask, intrinsics=None, extrinsics=None):
        '''
            prediction: predicted depth tensor [B, T, H, W]
            target: target depth tensor [B, T, H, W]
            mask: valid pixel mask [B, T, H, W]
            intrinsics: [3, 3] camera intrinsics (for TAE)
            extrinsics: [T, 4, 4] camera poses (for TAE)
        '''
        # Get VDA and Simple losses
        vda_loss_dict = self.vda_loss(prediction, target, mask)
        simple_loss_dict = self.simple_loss(prediction, target, mask)
        
        # Combine losses
        loss_dict = {}
        
        # Add individual VDA and Simple losses
        for key, value in vda_loss_dict.items():
            if key != 'total_loss':
                loss_dict[f'vda_{key}'] = value
        
        for key, value in simple_loss_dict.items():
            if key != 'total_loss':
                loss_dict[f'simple_{key}'] = value
        
        # Compute TAE loss if intrinsics and extrinsics are provided
        if intrinsics is not None and extrinsics is not None:
            tae_loss_value = self.tae_loss(prediction, intrinsics, extrinsics)
            loss_dict['tae_loss'] = tae_loss_value
        else:
            loss_dict['tae_loss'] = torch.tensor(0.0, device=prediction.device)
        
        # Weighted combination
        vda_total = vda_loss_dict['total_loss'] * self.vda_weight
        simple_total = simple_loss_dict['total_loss'] * self.simple_weight
        tae_total = loss_dict['tae_loss'] * self.tae_weight
        
        loss_dict['vda_total'] = vda_total
        loss_dict['simple_total'] = simple_total
        loss_dict['tae_total'] = tae_total
        loss_dict['total_loss'] = vda_total + simple_total + tae_total
        
        return loss_dict


class DifferentiableTAELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, depth_sequence, intrinsics, extrinsics, mask_threshold=1e-3, max_depth=100.0):
        """
        Differentiable Temporal Alignment Error loss.
        
        Args:
            depth_sequence: [B, T, H, W] or [T, H, W] depth tensor
            intrinsics: [3, 3] intrinsic matrix
            extrinsics: [T, 4, 4] camera poses tensor
            mask_threshold: minimum valid depth value
            max_depth: maximum valid depth value
        
        Returns:
            TAE loss (scalar tensor)
        """
        if len(depth_sequence.shape) == 3:
            depth_sequence = depth_sequence.unsqueeze(0)  # Add batch dim
        
        B, T, H, W = depth_sequence.shape
        device = depth_sequence.device
        
        # Ensure float32 for numerical stability
        depth_sequence = depth_sequence.float()
        intrinsics = intrinsics.float().to(device)
        extrinsics = extrinsics.float().to(device)
        
        # Extract intrinsic parameters
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        total_error = 0.0
        valid_pairs = 0
        
        for i in range(T - 1):
            depth1 = depth_sequence[:, i]  # [B, H, W]
            depth2 = depth_sequence[:, i + 1]  # [B, H, W]
            
            # Camera poses
            T_1 = extrinsics[i]      # [4, 4]
            T_2 = extrinsics[i + 1]  # [4, 4]
            
            # Relative transformation
            T_2_1 = torch.linalg.inv(T_2) @ T_1
            R = T_2_1[:3, :3]  # [3, 3]
            t = T_2_1[:3, 3]   # [3]
            
            # Valid depth masks
            mask1 = (depth1 > mask_threshold) & (depth1 < max_depth)  # [B, H, W]
            mask2 = (depth2 > mask_threshold) & (depth2 < max_depth)  # [B, H, W]
            
            # Convert pixels to 3D points in frame 1
            X1 = (x_coords - cx) * depth1 / fx  # [B, H, W]
            Y1 = (y_coords - cy) * depth1 / fy  # [B, H, W]
            Z1 = depth1  # [B, H, W]
            
            # Stack to get 3D points [B, 3, H, W]
            points_3d_1 = torch.stack([X1, Y1, Z1], dim=1)
            
            # Transform points to frame 2 coordinate system
            # Reshape for matrix multiplication: [B, 3, H*W]
            points_flat = points_3d_1.reshape(B, 3, -1)
            
            # Apply rotation and translation
            points_2_flat = R @ points_flat + t.unsqueeze(-1)  # [3, H*W]
            points_2 = points_2_flat.reshape(B, 3, H, W)  # [B, 3, H, W]
            
            # Project back to frame 2 image coordinates
            X2, Y2, Z2 = points_2[:, 0], points_2[:, 1], points_2[:, 2]
            
            # Avoid division by zero
            Z2_safe = torch.clamp(Z2, min=1e-6)
            u2 = fx * X2 / Z2_safe + cx
            v2 = fy * Y2 / Z2_safe + cy
            
            # Check if projected coordinates are within image bounds
            valid_proj = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H) & (Z2 > mask_threshold)
            
            # Combined mask
            combined_mask = mask1 & valid_proj
            
            if combined_mask.sum() > 0:
                # Sample depth2 at projected coordinates using bilinear interpolation
                # Normalize coordinates to [-1, 1] for grid_sample
                grid_x = (u2 / (W - 1)) * 2.0 - 1.0  # [B, H, W]
                grid_y = (v2 / (H - 1)) * 2.0 - 1.0  # [B, H, W]
                grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]
                
                # Sample depth2 using bilinear interpolation
                depth2_sampled = F.grid_sample(
                    depth2.unsqueeze(1),  # [B, 1, H, W]
                    grid,  # [B, H, W, 2]
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                ).squeeze(1)  # [B, H, W]
                
                # Compute depth error only where mask is valid
                projected_depth = Z2  # Depth of projected point
                depth_error = torch.abs(projected_depth - depth2_sampled)
                
                # Relative error (more robust)
                relative_error = depth_error / (depth2_sampled + 1e-6)
                
                # Average over valid pixels
                error = (relative_error * combined_mask.float()).sum() / (combined_mask.float().sum() + 1e-6)
                total_error += error
                valid_pairs += 1
        
        if valid_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return (total_error / valid_pairs) * self.weight


def evaluate_tae(depth_sequence, intrinsics, extrinsics):
    """
    Evaluate Temporal Alignment Error (TAE) for a depth sequence.
    
    Args:
        depth_sequence: [T, H, W] depth tensor
        intrinsics: [3, 3] intrinsic matrix (shared by all frames)
        extrinsics: [T, 4, 4] camera poses tensor
    
    Returns:
        float: TAE score (percentage)
    """
    device = depth_sequence.device
    T, H, W = depth_sequence.shape
    
    # Convert to torch tensors if needed
    if isinstance(intrinsics, np.ndarray):
        K = torch.tensor(intrinsics, dtype=depth_sequence.dtype, device=device)
    else:
        K = intrinsics.to(device=device, dtype=depth_sequence.dtype)
    
    if isinstance(extrinsics, np.ndarray):
        poses = torch.tensor(extrinsics, dtype=depth_sequence.dtype, device=device)
    elif isinstance(extrinsics, list):
        poses = torch.stack([torch.tensor(pose, dtype=depth_sequence.dtype, device=device) 
                           for pose in extrinsics], dim=0)
    else:
        poses = extrinsics.to(device=device, dtype=depth_sequence.dtype)
    
    error_sum = 0.0
    valid_pairs = 0
    
    for i in range(T - 1):
        depth1 = depth_sequence[i]
        depth2 = depth_sequence[i + 1]
        
        # Get camera poses
        T_1 = poses[i]      # [4, 4]
        T_2 = poses[i + 1]  # [4, 4]
        
        # Compute relative transformation
        T_2_1 = torch.linalg.inv(T_2) @ T_1
        R_2_1 = T_2_1[:3, :3]
        t_2_1 = T_2_1[:3, 3]
        
        # Create masks (valid depth regions)
        mask1 = (depth1 > 1e-3) & (depth1 < 100.0)
        mask2 = (depth2 > 1e-3) & (depth2 < 100.0)
        
        # Compute TAE in both directions
        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K.cpu().numpy(), mask2)
        
        # Reverse transformation
        T_1_2 = torch.linalg.inv(T_2_1)
        R_1_2 = T_1_2[:3, :3]
        t_1_2 = T_1_2[:3, 3]
        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K.cpu().numpy(), mask1)
        
        if isinstance(error1, torch.Tensor) and isinstance(error2, torch.Tensor):
            error_sum += error1 + error2
            valid_pairs += 2
    
    if valid_pairs == 0:
        return 0.0
    
    return (error_sum / valid_pairs).item() 




if __name__ == '__main__':
    
    args_script = parse_args()
    
    args_ttt = Arguments()
    args_vda = ArgsVDA()

    args_ttt.epochs = args_script.epochs


    # print(args_script.__dict__, args_ttt.__dict__, args_vda.__dict__)
    # print(vars(args_script), vars(args_ttt), vars(args_vda))
    # exit(0)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup logging
    # debug_dir = '/home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/debug_data'
    debug_dir = args_script.output_dir
    
    logger = setup_logging(debug_dir, args_ttt.exp_name)
    
    logger.info("Starting consistent depth training")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Arguments - Epochs: {args_ttt.epochs}, Mode: {args_ttt.mode}, Dataset: {args_ttt.dataset}")
    logger.info(f"VDA Arguments - Input size: {args_vda.input_size}, Encoder: {args_vda.encoder}")
    
    # save args to debug dir, as json    
    args_dict_all = {
        'script_args': vars(args_script),
        'ttt_args': vars(args_ttt),
        'vda_args': vars(args_vda),
    }
    with open(f'{debug_dir}/args.json', 'w') as f:
        json.dump(args_dict_all, f, indent=4)    
        
    ##########################################################


    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args_vda.encoder], metric=args_vda.metric)
    video_depth_anything.load_state_dict(torch.load(
        f'/home/azhuravl/work/Video-Depth-Anything/checkpoints/{checkpoint_name}_{args_vda.encoder}.pth', 
        map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # disable grad for video_depth_anything
    for param in video_depth_anything.parameters():
        param.requires_grad = False

    frames, target_fps = read_video_frames(
        # args_vda.input_video,
        f'{args_script.input_dir}/input_video.mp4',
        args_vda.max_len, args_vda.target_fps, args_vda.max_res)

    with torch.cuda.amp.autocast():
        depths, fps = video_depth_anything.infer_video_depth(
            frames,
            target_fps, input_size=args_vda.input_size, device=DEVICE, fp32=args_vda.fp32)

    save_video(depths, f'{debug_dir}/depths_pred_before_opt.mp4', fps=fps, is_depths=True, grayscale=False)


    # depth_gt = torch.load('/home/azhuravl/scratch/datasets_latents/monkaa_1000/000/videos/input_depths.pt', weights_only=True).squeeze(1) 
    depth_gt = torch.load(f'{args_script.input_dir}/input_depths.pt', weights_only=True).squeeze(1)
    depth_gt_inv = 1.0 / (depth_gt + 1e-8)

    # depth_warped = torch.load('/home/azhuravl/scratch/datasets_latents/monkaa_1000/000/videos/warped_depths.pt', weights_only=True).squeeze(1)
    depth_warped = torch.load(f'{args_script.input_dir}/warped_depths.pt', weights_only=True).squeeze(1)
    depth_warped_inv = torch.where(depth_warped > 0, 1.0 / depth_warped, torch.zeros_like(depth_warped))
    
    # read extrinsics.json and intrinsics.json from input_dir
    with open(f'{args_script.input_dir}/extrinsics.json', 'r') as f:
        extrinsics = json.load(f)
        
    with open(f'{args_script.input_dir}/intrinsics.json', 'r') as f:
        intrinsics = json.load(f)
    
    # Convert to torch tensors instead of numpy
    extrinsics_torch = torch.stack([torch.tensor(pose, dtype=torch.float32) for pose in extrinsics], dim=0)  # [T, 4, 4]
    intrinsics_torch = torch.tensor(intrinsics, dtype=torch.float32)  # [3, 3]
    
    

    # clip gt and warped depths max values
    max_depth_value = 100.0
    depth_gt_inv = torch.clamp(depth_gt_inv, max=max_depth_value)
    depth_warped_inv = torch.clamp(depth_warped_inv, max=max_depth_value)

    # idxs = torch.linspace(0, frames.shape[0]-1, INFER_LEN).long()
    idxs = torch.linspace(0, INFER_LEN-1, INFER_LEN).long()

    logger.info(f"Using indexes: {idxs.tolist()}")


    frames_resized, orig_dims = prepare_frames(frames[idxs], input_size=args_vda.input_size)

    depths_gt_inv_resized = F.interpolate(
        depth_gt_inv[idxs].unsqueeze(1),
        size=frames_resized.shape[3:],
        mode='bilinear',
    ).unsqueeze(0)

    depths_warped_inv_resized = F.interpolate(
        depth_warped_inv[idxs].unsqueeze(1),
        size=frames_resized.shape[3:],
        mode='nearest',
    ).unsqueeze(0)
    
    
    extrinsics_torch = extrinsics_torch[idxs]  # [INFER_LEN, 4, 4]
    
    logger.info(f"Tensor shapes - Frames: {frames_resized.shape}, GT depth: {depths_gt_inv_resized.shape}, "
               f"Warped depth: {depths_warped_inv_resized.shape}, Extrinsics: {extrinsics_torch.shape}, "
               f"Intrinsics: {intrinsics_torch.shape}")
    
    # ===== SAVE DEBUG VIDEOS USING VIDEO-DEPTH-ANYTHING FUNCTIONS =====
    # debug_dir = '/home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/debug_data'
    save_debug_videos(frames_resized, depths_gt_inv_resized, depths_warped_inv_resized, 
                     depth_gt_inv[idxs], depth_warped_inv[idxs], debug_dir, fps=target_fps)
    
    # ===== END DEBUG SAVE =====


    rmse_mean_ft, mae_mean_ft = 0.0, 0.0
    rmse_mean_vp, mae_mean_vp = 0.0, 0.0

    rgb = frames_resized.to(torch.bfloat16).cuda()
    depth = depths_gt_inv_resized.squeeze(2).to(torch.bfloat16).cuda()
    sparse_depth = depths_warped_inv_resized.squeeze(2).to(torch.bfloat16).cuda()

    # single_frame_prompt = torch.nn.Parameter(torch.zeros_like(rgb[:, :1], dtype=torch.bfloat16, device='cuda'))
    single_frame_prompt = torch.nn.Parameter(torch.zeros_like(rgb[:, :], dtype=torch.bfloat16, device='cuda'))
    
    
    
    # freeze 90% of the parameters in single_frame_prompt
    # Create smaller prompt - use lower resolution
    # prompt_scale = 1  # Make prompt 1/4 resolution
    # prompt_shape = (1, rgb.shape[2], 
    #                rgb.shape[3] // prompt_scale, rgb.shape[4] // prompt_scale)
    # single_frame_prompt_reduced = torch.nn.Parameter(torch.zeros(prompt_shape, dtype=torch.float16, device='cuda'))

    # print("Single frame prompt reduced shape:", single_frame_prompt_reduced.shape)
    # print(rgb.shape, sparse_depth.shape)

    # single_frame_prompt = F.interpolate(
    #     single_frame_prompt_reduced, 
    #     size=rgb.shape[3:], mode='bilinear', align_corners=False
    # ).unsqueeze(0)
    
    # print(single_frame_prompt_reduced.shape, single_frame_prompt.shape, rgb.shape, sparse_depth.shape)
    
    # print(single_frame_prompt.shape, rgb.shape, sparse_depth.shape)


    gt_mask = depth > 0
    sparse_mask = sparse_depth > 0

    optimizer = torch.optim.AdamW([{'params': single_frame_prompt, 'lr': 2e-3}])
    # scaler = torch.cuda.amp.GradScaler()

    # pbar = tqdm.tqdm(total=args_ttt.epochs)
    
    # loss_fn = VideoDepthLoss()
    # loss_fn = SimpleDepthLoss(l1_weight=1.0, rmse_weight=1.0)
    # loss_fn = CombinedDepthLoss(vda_weight=1.0, simple_weight=1.0, l1_weight=1.0, rmse_weight=1.0)
    loss_fn = CombinedDepthLossWithTAE(
        vda_weight=1.0, 
        simple_weight=1.0, 
        tae_weight=0.0,  # Start with small weight
        l1_weight=1.0, 
        rmse_weight=1.0
    )
    
    loss_tae = DifferentiableTAELoss(weight=1.0)

    for epoch in range(args_ttt.epochs):
        # visual_prompt = single_frame_prompt.repeat(1, rgb.shape[1], 1, 1, 1)
        # visual_prompt = single_frame_prompt + torch.zeros_like(rgb)

        # print(single_frame_prompt.shape, rgb.shape, sparse_depth.shape)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            new_rgb = rgb + single_frame_prompt
            pre_depth_ = video_depth_anything.forward(new_rgb)
            
            # scale, shift = compute_scale_and_shift(pre_depth_, sparse_depth)
            # pre_depth = pre_depth_ * scale + shift
            
            # print(pre_depth_.shape, sparse_depth.shape, sparse_mask.shape)
            # exit(0)
            # # Use the new compute_scale_and_shift function with mask
            # scale, shift = compute_scale_and_shift(pre_depth_, sparse_depth, sparse_mask)
            
            # print(scale.shape, shift.shape)
            # pre_depth = pre_depth_ * scale + shift
            
            scale, shift = compute_scale_and_shift(pre_depth_.flatten(1,2), sparse_depth.flatten(1,2), sparse_mask.flatten(1,2))
            pre_depth = scale.view(-1, 1, 1, 1) * pre_depth_ + shift.view(-1, 1, 1, 1)
            
            
            # loss_l1 = F.l1_loss(pre_depth[sparse_mask], sparse_depth[sparse_mask])
            # loss_rmse = torch.sqrt(((pre_depth[sparse_mask] - sparse_depth[sparse_mask]) ** 2).mean())
            # loss = loss_l1 + loss_rmse
            
            # loss_dict = loss_fn(pre_depth, sparse_depth, sparse_mask)
            loss_dict = loss_fn(
                pre_depth, 
                sparse_depth, 
                sparse_mask,
                intrinsics=intrinsics_torch,  # Pass intrinsics
                extrinsics=extrinsics_torch   # Pass extrinsics
            )
            loss = loss_dict['total_loss']

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(pre_depth.shape, single_frame_prompt.shape)
        
        # calculate relative depth error (RDE)
        with torch.no_grad():
            rde = (torch.abs(pre_depth[sparse_mask] - sparse_depth[sparse_mask]) / (sparse_depth[sparse_mask] + 1e-8)).mean()


        # print('pre_depth shape:', pre_depth.shape)
        with torch.no_grad():
            tae = evaluate_tae(
                pre_depth.squeeze(0),
                intrinsics_torch,
                extrinsics_torch
            )
            tae_diff = loss_tae(
                pre_depth.squeeze(0),
                intrinsics_torch,
                extrinsics_torch
            ).item()


        # logger.info(f'{epoch:3d}/{args_ttt.epochs} l1: {loss_l1.item():.4f}, rmse: {loss_rmse.item():.4f}, rde: {rde.item():.4f}, scale: {scale.item():.4f}, shift: {shift.item():.4f}')
        
        # Loss-agnostic logging
        loss_str_parts = []
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':  # Log individual losses
                loss_str_parts.append(f"{loss_name}: {loss_value.item():.4f}")
        
        # Add RDE and scale/shift info
        loss_str_parts.extend([
            "\n"
            f"rde: {rde.item():.4f}",
            f"tae: {tae:.4f}",
            f"tae_diff: {tae_diff:.4f}",
            f"scale: {scale.item():.4f}",
            f"shift: {shift.item():.4f}"
        ])
        
        loss_str = ", ".join(loss_str_parts)
        logger.info(f'{epoch:3d}/{args_ttt.epochs} {loss_str}')


    # ===== SAVE CORRECTED DEPTH AFTER TRAINING =====
    logger.info("Saving corrected depth after training...")
    
    with torch.no_grad():
        # Final inference with optimized prompt
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            new_rgb = rgb + single_frame_prompt
            corrected_depth_ = video_depth_anything.forward(new_rgb)
            
            # Apply final scale and shift
            # final_scale, final_shift = compute_scale_and_shift(corrected_depth_, sparse_depth, sparse_mask)
            # corrected_depth = corrected_depth_ * final_scale + final_shift
            
            final_scale, final_shift = compute_scale_and_shift(corrected_depth_.flatten(1,2), sparse_depth.flatten(1,2), sparse_mask.flatten(1,2))
            corrected_depth = final_scale.view(-1, 1, 1, 1) * corrected_depth_ + final_shift.view(-1, 1, 1, 1)
    
    # Save corrected depth as tensor
    torch.save(corrected_depth.cpu(), f'{debug_dir}/corrected_depth.pt')
    
    # Save corrected depth as video
    corrected_depth_np = corrected_depth[0].cpu().numpy()  # [T, H, W]
    save_video(corrected_depth_np, f'{debug_dir}/corrected_depth.mp4', fps=target_fps, is_depths=True, grayscale=False)
    
    # save visual single frame prompt as matplotlib image
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    max_abs = torch.max(torch.abs(single_frame_prompt)).item()
    single_frame_prompt_vis = single_frame_prompt[0, 0].detach().cpu().permute(1, 2, 0).float().numpy()

    for i in range(3):
        im = axes[i].imshow(single_frame_prompt_vis[:, :, i], cmap='bwr', vmin=-0.1, vmax=0.1)
        axes[i].set_title(f"Channel {i}")
        axes[i].axis('off')  # Remove individual subplot axes
        plt.colorbar(im, ax=axes[i], shrink=0.4)

    # plt.tight_layout()
    
    # normalize and show the RGB prompt addition
    prompt_rgb = (single_frame_prompt_vis - single_frame_prompt_vis.min()) / (single_frame_prompt_vis.max() - single_frame_prompt_vis.min() + 1e-8)
    axes[3].imshow(prompt_rgb)
    axes[3].axis('off')  # Remove individual subplot axes
    
    
    plt.savefig(f'{debug_dir}/single_frame_prompt.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    #############
    
    # Create side-by-side comparison video
    comparison_frames = []
    for i in range(corrected_depth_np.shape[0]):
        # Get frames at index i
        gt_frame = depths_gt_inv_resized[0, i, 0].cpu().numpy()
        warped_frame = depths_warped_inv_resized[0, i, 0].cpu().numpy()
        corrected_frame = corrected_depth_np[i]
        
        # Normalize each frame to 0-1 for comparison
        def normalize_depth(depth):
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                return (depth - d_min) / (d_max - d_min)
            return depth
        
        gt_norm = normalize_depth(gt_frame)
        warped_norm = normalize_depth(warped_frame)
        corrected_norm = normalize_depth(corrected_frame)
        
        # Create side-by-side frame [H, W*3]
        combined = np.concatenate([gt_norm, warped_norm, corrected_norm], axis=1)
        comparison_frames.append(combined)
    
    comparison_frames = np.array(comparison_frames)
    save_video(comparison_frames, f'{debug_dir}/depth_comparison.mp4', fps=target_fps, is_depths=True, grayscale=False)
    
    # Compute final metrics
    with torch.no_grad():
        
        # print shapes
        logger.info("Computing final metrics...")
        logger.info(f"Corrected depth shape: {corrected_depth.shape}")
        logger.info(f"Ground truth depth shape: {depth.shape}")
        logger.info(f"Sparse depth shape: {sparse_depth.shape}")
        
        
        # Metrics against ground truth
        gt_mask_cpu = gt_mask.cpu()
        final_rmse_gt = torch.sqrt(((corrected_depth[gt_mask_cpu] - depth[gt_mask_cpu]) ** 2).mean())
        final_mae_gt = torch.abs(corrected_depth[gt_mask_cpu] - depth[gt_mask_cpu]).mean()
        
        # calculate relative depth error (RDE)
        final_rde_gt = (torch.abs(corrected_depth[gt_mask_cpu] - depth[gt_mask_cpu]) / depth[gt_mask_cpu]).mean()
                       
        
        # Metrics against sparse depth
        sparse_mask_cpu = sparse_mask.cpu()
        final_rmse_sparse = torch.sqrt(((corrected_depth[sparse_mask_cpu] - sparse_depth[sparse_mask_cpu]) ** 2).mean())
        final_mae_sparse = torch.abs(corrected_depth[sparse_mask_cpu] - sparse_depth[sparse_mask_cpu]).mean()
        
        final_rde_sparse = (torch.abs(corrected_depth[sparse_mask_cpu] - sparse_depth[sparse_mask_cpu]) / sparse_depth[sparse_mask_cpu]).mean()
        
        # Temporal Alignment Error (TAE)
        
        # print(corrected_depth.dtype, depth.dtype, intrinsics_torch.dtype, extrinsics_torch.dtype)
        
        # with torch.cuda.amp.autocast():
            
            
        final_tae = evaluate_tae(
            corrected_depth.squeeze(0),
            intrinsics_torch,
            extrinsics_torch
        )
        gt_tae = evaluate_tae(
            depth.float().squeeze(0),
            intrinsics_torch,
            extrinsics_torch
        )
        
        
        
        
    logger.info("Final Metrics:")
    logger.info(f"Ground Truth  - RMSE: {final_rmse_gt.item():.4f}, MAE: {final_mae_gt.item():.4f}, RDE: {final_rde_gt.item():.4f}")
    logger.info(f"Sparse Depth  - RMSE: {final_rmse_sparse.item():.4f}, MAE: {final_mae_sparse.item():.4f}, RDE: {final_rde_sparse.item():.4f}")
    logger.info(f"Temporal Alignment Error (TAE): {final_tae:.4f} (GT TAE: {gt_tae:.4f})")
    logger.info(f"Final Scale: {final_scale.item():.4f}, Final Shift: {final_shift.item():.4f}")
    
    
    
    
    
    # ===== TEMPORAL ALIGNMENT ERROR EVALUATION =====
    logger.info("Computing Temporal Alignment Error (TAE)...")



    
    
    
    
    
    
    
    
    
    logger.info("Training completed successfully!")
    logger.info(f"All outputs saved to: {debug_dir}")
    

    