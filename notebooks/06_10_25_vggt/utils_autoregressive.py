import os
import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.append('/home/azhuravl/work/TrajectoryCrafter')

from demo import TrajCrafter
from models.utils import Warper, read_video_frames, sphere2pose, save_video


# ============================================================================
# UTILITY FUNCTIONS (moved outside class)
# ============================================================================

def pad_video(frames, target_length):
    if frames.shape[0] < target_length:
        last_frame = frames[-1:]
        num_pad = target_length - frames.shape[0]
        pad_frames = np.repeat(last_frame, num_pad, axis=0)
        frames = np.concatenate([frames, pad_frames], axis=0)
    return frames


def generate_traj_specified(c2ws_anchor, target_pose, n_frames, device):
    theta, phi, d_r, d_x, d_y = target_pose
    
    thetas = np.linspace(0, theta, n_frames)  
    phis = np.linspace(0, phi, n_frames)          
    rs = np.linspace(0, d_r, n_frames)            
    xs = np.linspace(0, d_x, n_frames)            
    ys = np.linspace(0, d_y, n_frames)            
    
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose(
            c2ws_anchor,
            np.float32(th),
            np.float32(ph),
            np.float32(r),
            device,
            np.float32(x),
            np.float32(y),
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws


def save_poses_torch(c2ws, filepath):
    """Save camera poses as PyTorch tensor (.pth file)"""
    torch.save(c2ws.cpu(), filepath)

def save_point_clouds_torch(pc_list, color_list, dirpath):
    """Save point clouds as PyTorch tensors (much faster than text files)"""
    os.makedirs(dirpath, exist_ok=True)
    
    # Save as individual tensor files
    for idx, (pc, color) in enumerate(zip(pc_list, color_list)):
        # Save points and colors as separate tensors
        torch.save(pc.cpu(), os.path.join(dirpath, f'points_{idx:03d}.pth'))
        torch.save(color.cpu(), os.path.join(dirpath, f'colors_{idx:03d}.pth'))
    

def save_segment_results(pc_input, color_input, pc_inpainted, color_inpainted, 
                        pc_merged, color_merged, traj_segment, opts, segment_idx):
    # Function to save results for each segment
    # Implementation needed
    stage_dir = Path(opts.save_dir) / f'stage_{segment_idx+1}'
    stage_dir.mkdir(parents=True, exist_ok=True)
    

    save_point_clouds_torch(pc_input, color_input, stage_dir / 'point_cloud_input')
    # save_point_clouds_torch(pc_inpainted, color_inpainted, stage_dir / 'point_cloud_inpainted')
    # save_point_clouds_torch(pc_merged, color_merged, stage_dir / 'point_cloud_merged')
    save_poses_torch(traj_segment, stage_dir / 'cameras_target.pth')


def clean_single_mask_simple(mask_tensor, kernel_size=3, n_erosion_steps=3, n_dilation_steps=2):
    """Simpler mask cleaning without size mismatch issues"""

    # Handle different input dimensions
    if mask_tensor.dim() == 4:  # (B, C, H, W)
        frame_mask = mask_tensor
    elif mask_tensor.dim() == 3:  # (C, H, W)
        frame_mask = mask_tensor.unsqueeze(0)  # Add batch dim -> (1, C, H, W)
    elif mask_tensor.dim() == 2:  # (H, W)
        frame_mask = mask_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
    else:
        raise ValueError(f"Unsupported mask dimensions: {mask_tensor.shape}")
    
    # Ensure we have single channel
    if frame_mask.shape[1] > 1:
        frame_mask = frame_mask[:, 0:1]  # Take first channel
    
    # Binarize the mask
    binary_mask = (frame_mask > 0.5).float()
    
    # Simple morphological opening (erosion followed by dilation)
    padding = kernel_size // 2
    
    # Erosion
    
    for _ in range(n_erosion_steps):
        binary_mask = -F.max_pool2d(-binary_mask, kernel_size, stride=1, padding=padding)
    
    # Dilation
    for _ in range(n_dilation_steps):
        binary_mask = F.max_pool2d(binary_mask, kernel_size, stride=1, padding=padding)
    
    # print(binary_mask.shape, frame_mask.shape)
    
    cleaned = binary_mask * (frame_mask > 0.5).float()
    
    # Return in same format as input
    if mask_tensor.dim() == 2:
        return cleaned.squeeze()  # (H, W)
    elif mask_tensor.dim() == 3:
        return cleaned.squeeze(0)  # (C, H, W)
    else:
        return cleaned  # (B, C, H, W)
    
    
def align_depth_maps(depth1, depth2):
    """
    Align two depth maps from the same viewpoint
    
    Args:
        depth1, depth2: Depth maps (H, W) or (1, H, W)
        mask1, mask2: Valid pixel masks
    """
    
    # masks = 1 if depth < 10, 0 otherwise
    mask1 = (depth1 < 10).float()
    mask2 = (depth2 < 10).float()
    
    
    # Flatten depth maps
    if depth1.dim() == 3:
        depth1 = depth1.squeeze(0)
        depth2 = depth2.squeeze(0)
    
    d1_flat = depth1.flatten()
    d2_flat = depth2.flatten()
    
    
    # Apply masks if provided
    if mask1 is not None and mask2 is not None:
        if mask1.dim() == 3:
            mask1 = mask1.squeeze(0)
            mask2 = mask2.squeeze(0)
        
        valid_mask = (mask1.flatten() > 0.5) & (mask2.flatten() > 0.5)
        d1_flat = d1_flat[valid_mask]
        d2_flat = d2_flat[valid_mask]
    
    # Remove invalid depths
    valid_depths = (d1_flat > 0) & (d2_flat > 0) & torch.isfinite(d1_flat) & torch.isfinite(d2_flat)
    d1_valid = d1_flat[valid_depths]
    d2_valid = d2_flat[valid_depths]
    
    if len(d1_valid) == 0:
        return depth1, depth2, 1.0
    
    # Compute scale factor using robust estimation
    ratios = d1_valid / (d2_valid + 1e-8)
    
    # Remove outliers
    q25, q75 = torch.quantile(ratios, 0.25), torch.quantile(ratios, 0.75)
    iqr = q75 - q25
    inlier_mask = (ratios >= q25 - 1.5*iqr) & (ratios <= q75 + 1.5*iqr)
    
    if inlier_mask.sum() > 0:
        scale_factor = torch.median(ratios[inlier_mask])
    else:
        scale_factor = torch.median(ratios)
    
    # Scale the second depth map
    # depth2_aligned = depth2 * scale_factor
    
    return scale_factor


def load_video_frames(video_path, video_length, stride, max_res, device, reverse=False):
    """
    Load video frames with optional reversal
    
    Args:
        video_path: Path to video file
        video_length: Number of frames to load
        stride: Frame sampling stride
        max_res: Maximum resolution
        device: Target device
        reverse: Whether to reverse the video frames
    
    Returns:
        frames_np: NumPy array of frames (T, H, W, 3) in [0,1]
        frames_tensor: PyTorch tensor (T, 3, H, W) in [-1,1]
    """
    # Load frames
    frames_np = utils.read_video_frames(
        video_path, video_length, stride, max_res,
        # height=opts.sample_size[0], width=opts.sample_size[1],
    )

    # Pad if too short
    frames_np = pad_video(frames_np, video_length)
    
    # Reverse the video if requested
    if reverse:
        frames_np = frames_np[::-1, ...].copy()

    # Convert to tensor version
    frames_tensor = (
        torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
    )  # T H W 3 -> T 3 H W, [-1,1]
    
    assert frames_tensor.shape[0] == video_length
    
    return frames_np, frames_tensor


def sample_diffusion(
    vis_crafter,
    frames_tensor,    # [T, 3, H, W], in [-1, 1]
    warped_images,    # list of warped images tensors
    frames_ref,
    masks,            # list of mask tensors
    opts,
    segment_dir=None,
):
    """
    Run diffusion sampling for a given stage.
    
    TODO: ALWAYS PASS ORIGINAL FRAMES AS REFERENCE FRAMES
    """

    # --- determine output directory ---
    if not segment_dir:
        n_subdirs = len([
            name for name in os.listdir(opts.save_dir)
            if os.path.isdir(os.path.join(opts.save_dir, name))
        ])
        segment_dir = os.path.join(opts.save_dir, f'stage_{n_subdirs + 1}')
        os.makedirs(segment_dir, exist_ok=True)
        print(f"Saving to: {segment_dir}")

    # --- build conditioning tensors ---
    cond_video = (torch.cat(warped_images) + 1.0) / 2.0  # [T, 3, H, W] in [0,1]
    cond_masks = torch.cat(masks)  # [T, 1, H, W]

    # --- resize inputs to diffusion sample size ---
    frames_interp = F.interpolate(
        frames_tensor, size=opts.sample_size, mode='bilinear', align_corners=False
    )
    cond_video = F.interpolate(
        cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
    )
    cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')

    # --- save inputs for visualization ---
    save_video(
        (frames_interp.permute(0, 2, 3, 1) + 1.0) / 2.0,
        os.path.join(segment_dir, 'input.mp4'),
        fps=opts.fps,
    )
    save_video(
        cond_video.permute(0, 2, 3, 1),
        os.path.join(segment_dir, 'render.mp4'),
        fps=opts.fps,
    )
    save_video(
        cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
        os.path.join(segment_dir, 'mask.mp4'),
        fps=opts.fps,
    )

    # --- prepare for diffusion pipeline ---
    frames_interp = (frames_interp.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
    
    # frames_ref = frames_interp[:, :, :10, :, :]  # first few frames as reference
    frames_ref_interp = F.interpolate(
        frames_ref, size=opts.sample_size, mode='bilinear', align_corners=False
    )
    
    # save frames_ref
    save_video(
        (frames_ref_interp.permute(0, 2, 3, 1) + 1.0) / 2.0,
        os.path.join(segment_dir, 'reference.mp4'),
        fps=opts.fps,
    )
    
    
    frames_ref = (frames_ref_interp.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
    
    
    cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
    cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0

    generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

    # --- run diffusion model ---
    with torch.no_grad():
        sample = vis_crafter.pipeline(
            vis_crafter.prompt,
            num_frames=opts.video_length,
            negative_prompt=opts.negative_prompt,
            height=opts.sample_size[0],
            width=opts.sample_size[1],
            generator=generator,
            guidance_scale=opts.diffusion_guidance_scale,
            num_inference_steps=opts.diffusion_inference_steps,
            video=cond_video.to(opts.device),
            mask_video=cond_masks.to(opts.device),
            reference=frames_ref.to(opts.device),
        ).videos

    # --- save result ---
    save_video(
        sample[0].permute(1, 2, 3, 0),
        os.path.join(segment_dir, 'gen.mp4'),
        fps=opts.fps,
    )

    return sample, segment_dir


class TrajCrafterAutoregressive(TrajCrafter):
    def __init__(self, opts):
        super().__init__(opts)

        # self.funwarp = VisualizationWarper(device=opts.device)
        self.prompt = None
        
        self.K = torch.tensor(
            [[500, 0.0, 512.], [0.0, 500, 288.], [0.0, 0.0, 1.0]]
            ).repeat(opts.video_length, 1, 1).to(opts.device)