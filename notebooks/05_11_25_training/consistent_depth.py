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

# Add paths
sys.path.append('/home/azhuravl/work/Video-Depth-Anything')
sys.path.append('/home/azhuravl/work/Video-Depth-Anything/video_depth_anything/util')

# Video Depth Anything imports
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
from transform import Resize, NormalizeImage, PrepareForNet

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8


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
    print("Saving debug videos...")
    
    # Denormalize and convert RGB frames
    rgb_denorm = denormalize_rgb(frames_resized.cpu())
    rgb_clamped = torch.clamp(rgb_denorm, 0, 1)
    rgb_frames = (rgb_clamped[0] * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)  # [T, H, W, C]
    
    # Convert depth tensors to numpy arrays
    gt_depth_resized = depths_gt_inv_resized[0, :, 0].cpu().numpy()  # [T, H, W]
    warped_depth_resized = depths_warped_inv_resized[0, :, 0].cpu().numpy()  # [T, H, W]
    gt_depth_original = depth_gt_inv[:32].cpu().numpy()  # [T, H, W]
    warped_depth_original = depth_warped_inv[:32].cpu().numpy()  # [T, H, W]
    
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
    
    print(f"Debug videos saved to {debug_dir}")


class Arguments:
    gpu = '0'
    random_seed = 2025
    epochs = 100
    exp_name = 'base'
    mode = 'VP'  # choices=['VP', 'FT']
    dataset = 'ibims'  # choices=['ibims', 'ddad']
    dataset_path = '/workspace/data_all'


class ArgsVDA:
    input_video = '/home/azhuravl/scratch/datasets_latents/monkaa_1000/000/videos/input_video.mp4'
    output_dir = '/home/azhuravl/work/Video-Depth-Anything/outputs'
    input_size = 256
    max_res = 1280
    encoder = 'vitl'
    max_len = -1
    target_fps = -1
    metric = False
    fp32 = False
    grayscale = False
    save_npz = False
    save_exr = False
    focal_length_x = 470.4
    focal_length_y = 470.4


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


def compute_scale_and_shift(prediction, target, mask):
    """
    Compute scale and shift parameters using least squares method.
    
    Args:
        prediction: predicted depth [B, T, H, W] or any shape
        target: target depth, same shape as prediction
        mask: valid mask, same shape as prediction
        
    Returns:
        scale: scale parameter (tensor)
        shift: shift parameter (tensor)
    """
    # Convert to float32 for numerical stability
    prediction = prediction.float()
    target = target.float()
    mask = mask.float()
    
    # Flatten tensors for computation
    pred_flat = prediction.view(-1)
    target_flat = target.view(-1)
    mask_flat = mask.view(-1)
    
    # System matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask_flat * pred_flat * pred_flat)
    a_01 = torch.sum(mask_flat * pred_flat)
    a_11 = torch.sum(mask_flat)
    
    # Right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask_flat * pred_flat * target_flat)
    b_1 = torch.sum(mask_flat * target_flat)
    
    # Default values
    x_0 = torch.tensor(1.0, device=prediction.device, dtype=prediction.dtype)
    x_1 = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    
    # Solve the system: A * x = b
    det = a_00 * a_11 - a_01 * a_01
    
    # Check if determinant is not zero (avoid division by zero)
    eps = 1e-8
    det_nonzero = torch.abs(det) > eps
    
    if det_nonzero:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    
    return x_0, x_1


if __name__ == '__main__':
    args_ttt = Arguments()
    args_vda = ArgsVDA()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    frames, target_fps = read_video_frames(args_vda.input_video, args_vda.max_len, args_vda.target_fps, args_vda.max_res)

    with torch.cuda.amp.autocast():
        depths, fps = video_depth_anything.infer_video_depth(
            frames,
            target_fps, input_size=args_vda.input_size, device=DEVICE, fp32=args_vda.fp32)

    depth_gt = torch.load('/home/azhuravl/scratch/datasets_latents/monkaa_1000/000/videos/input_depths.pt', weights_only=True).squeeze(1) 
    depth_gt_inv = 1.0 / (depth_gt + 1e-8)

    depth_warped = torch.load('/home/azhuravl/scratch/datasets_latents/monkaa_1000/000/videos/warped_depths.pt', weights_only=True).squeeze(1)
    depth_warped_inv = torch.where(depth_warped > 0, 1.0 / depth_warped, torch.zeros_like(depth_warped))

    frames_resized, orig_dims = prepare_frames(frames[:32], input_size=args_vda.input_size)

    depths_gt_inv_resized = F.interpolate(
        depth_gt_inv[:32].unsqueeze(1),
        size=frames_resized.shape[3:],
        mode='bilinear',
    ).unsqueeze(0)

    depths_warped_inv_resized = F.interpolate(
        depth_warped_inv[:32].unsqueeze(1),
        size=frames_resized.shape[3:],
        mode='nearest',
    ).unsqueeze(0)

    print(frames_resized.shape, depths_gt_inv_resized.shape, depths_warped_inv_resized.shape)

    # ===== SAVE DEBUG VIDEOS USING VIDEO-DEPTH-ANYTHING FUNCTIONS =====
    debug_dir = '/home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/debug_data'
    save_debug_videos(frames_resized, depths_gt_inv_resized, depths_warped_inv_resized, 
                     depth_gt_inv, depth_warped_inv, debug_dir, fps=target_fps)
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

    for epoch in range(args_ttt.epochs):
        # visual_prompt = single_frame_prompt.repeat(1, rgb.shape[1], 1, 1, 1)
        # visual_prompt = single_frame_prompt + torch.zeros_like(rgb)

        # print(single_frame_prompt.shape, rgb.shape, sparse_depth.shape)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            new_rgb = rgb + single_frame_prompt
            pre_depth_ = video_depth_anything.forward(new_rgb)
            
            # scale, shift = compute_scale_and_shift(pre_depth_, sparse_depth)
            # pre_depth = pre_depth_ * scale + shift
            
            # Use the new compute_scale_and_shift function with mask
            scale, shift = compute_scale_and_shift(pre_depth_, sparse_depth, sparse_mask)
            pre_depth = pre_depth_ * scale + shift
            
            
            loss_l1 = F.l1_loss(pre_depth[sparse_mask], sparse_depth[sparse_mask])
            loss_rmse = torch.sqrt(((pre_depth[sparse_mask] - sparse_depth[sparse_mask]) ** 2).mean())
            loss = loss_l1 + loss_rmse

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(pre_depth.shape, single_frame_prompt.shape)

        print(f'{epoch}) l1: {loss_l1.item():.4f}, rmse: {loss_rmse.item():.4f}, scale: {scale.item():.4f}, shift: {shift.item():.4f}')
    
    
    
    # ===== SAVE CORRECTED DEPTH AFTER TRAINING =====
    print("\nSaving corrected depth after training...")
    
    with torch.no_grad():
        # Final inference with optimized prompt
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            new_rgb = rgb + single_frame_prompt
            corrected_depth_ = video_depth_anything.forward(new_rgb)
            
            # Apply final scale and shift
            final_scale, final_shift = compute_scale_and_shift(corrected_depth_, sparse_depth, sparse_mask)
            corrected_depth = corrected_depth_ * final_scale + final_shift
    
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
        print("Corrected depth shape:", corrected_depth.shape)
        print("Ground truth depth shape:", depth.shape)
        print("Sparse depth shape:", sparse_depth.shape)
        
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
        
    
    print(f"\nFinal Metrics:")
    print(f"Ground Truth  - RMSE: {final_rmse_gt.item():.4f}, MAE: {final_mae_gt.item():.4f}, RDE: {final_rde_gt.item():.4f}")
    print(f"Sparse Depth  - RMSE: {final_rmse_sparse.item():.4f}, MAE: {final_mae_sparse.item():.4f}, RDE: {final_rde_sparse.item():.4f}")
    print(f"Final Scale: {final_scale.item():.4f}, Final Shift: {final_shift.item():.4f}")
    
    # print(f"\nSaved files:")
    # print(f"- Corrected depth tensor: {debug_dir}/corrected_depth.pt")
    # print(f"- Corrected depth video: {debug_dir}/corrected_depth.mp4")
    # print(f"- Optimized prompt: {debug_dir}/optimized_prompt.pt")
    # print(f"- Depth comparison: {debug_dir}/depth_comparison.mp4")
    # ===== END SAVE CORRECTED DEPTH =====
    