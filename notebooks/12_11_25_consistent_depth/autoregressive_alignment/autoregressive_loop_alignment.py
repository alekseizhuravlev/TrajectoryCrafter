import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/06_10_25_vggt')

from utils_autoregressive import save_video, load_video_frames, clean_single_mask_simple, sample_diffusion

sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/12_11_25_consistent_depth/depth_alignment')
from consistent_depth import denormalize_rgb


# given input frames + necessary inputs, estimate their depths and return point clouds
def video_to_pcs(
    frames_tensor,
    depths,
    intrinsics_torch,
    extrinsics_torch,
    funwarp,
):
    
    global_pcs = []
    global_colors = []
    
    for i in tqdm(range(frames_tensor.shape[0])):
        with torch.no_grad():
            points, colors, _ = funwarp.create_pointcloud_from_image(
                frames_tensor[i:i+1],
                None,
                depths[i:i+1],
                extrinsics_torch[i:i+1],
                intrinsics_torch[0:1],
                1,
            )
        global_pcs.append(points)
        global_colors.append(colors)
        
    return global_pcs, global_colors




def invert_depth_with_scale(
    depth,
    scale
):
    depth_inv = torch.zeros_like(depth)
    valid_mask = depth > 0
    depth_inv[valid_mask] = scale / depth[valid_mask]
    return depth_inv


# def estimate_depth_with_padding(
#     frames_tensor,
#     depth_trainer,
#     depth_scale,
#     sparse_depth=None, sparse_mask=None,
#     intrinsics_torch=None,
#     extrinsics_torch=None,
#     epochs=None,
#     multiple_of=14,
#     ):
#     """
#     Estimate depth with padding to ensure input is divisible by multiple_of.
    
#     Args:
#         frames_tensor: [T, C, H, W] input frames (ImageNet normalized)
#         video_depth_model: Video Depth Anything model
#         device: torch device
#         multiple_of: padding constraint (default: 14 for ViT)
    
#     Returns:
#         depths: [T, 1, H_orig, W_orig] depth predictions at original resolution
#     """
    
#     # TODO have trainer here and sparse depth
#     # TODO initialize global pcs during first run before loop
#     # TODO handle inverse depth
    
#     ################################################
#     # Calculate needed padding
#     ################################################
    
#     T, C, H_orig, W_orig = frames_tensor.shape
    
#     # Calculate padding needed
#     pad_h = (multiple_of - (H_orig % multiple_of)) % multiple_of
#     pad_w = (multiple_of - (W_orig % multiple_of)) % multiple_of
    
#     # Pad symmetrically (top/bottom, left/right)
#     pad_top = pad_h // 2
#     pad_bottom = pad_h - pad_top
#     pad_left = pad_w // 2
#     pad_right = pad_w - pad_left
    
    
#     ################################################
#     # Apply padding: (pad_left, pad_right, pad_top, pad_bottom)
#     ################################################
    
#     frames_padded = torch.nn.functional.pad(
#         frames_tensor, 
#         (pad_left, pad_right, pad_top, pad_bottom), 
#         mode='reflect'  # Use reflection padding to avoid edge artifacts
#     )
#     frames_padded = frames_padded.to(depth_trainer.device, dtype=torch.bfloat16)
    
#     H_padded, W_padded = frames_padded.shape[-2:]
#     print(f"Original size: {H_orig}x{W_orig}, Padded size: {H_padded}x{W_padded}")
#     print(f"Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
    
    
#     ################################################
#     # Depth estimation on padded frames
#     ################################################
    
#     # depths_padded, fps = video_depth_model.infer_video_depth(
#     #     frames_padded.permute(0, 2, 3, 1).numpy(), target_fps=10, input_size=max(H_padded, W_padded)
#     #     ) # [T, H, W]
#     # depths_padded = torch.tensor(depths_padded).unsqueeze(0)  # [1, T, H, W]
    
    
#     if sparse_depth is not None and sparse_mask is not None:
#         print("Optimizing depth alignment with sparse depth and mask...")
        
#         # Pad sparse depth and mask similarly, with 0 padding
#         sparse_depth = torch.nn.functional.pad(
#             sparse_depth, 
#             (pad_left, pad_right, pad_top, pad_bottom), 
#             mode='constant', value=0
#         )
#         sparse_mask = torch.nn.functional.pad(
#             sparse_mask, 
#             (pad_left, pad_right, pad_top, pad_bottom), 
#             mode='constant', value=0
#         )
        
#         sparse_depth = sparse_depth.to(depth_trainer.device, dtype=torch.bfloat16)
#         sparse_mask = sparse_mask.to(depth_trainer.device)
        
#         # convert sparse_depth to inverse depth, using scale, but preserve zeros
#         sparse_depth_inv = invert_depth_with_scale(sparse_depth, depth_scale)
    
#         # optimize depth alignment
#         depth_inv_consistent, visual_prompt, final_scale, final_shift = depth_trainer.train(
#             frames_padded.unsqueeze(0),  # [1, T, C, H, W]
#             sparse_depth_inv.unsqueeze(0),   # [1, T, H, W]
#             sparse_mask.unsqueeze(0),    # [1, T, H, W]
#             intrinsics_torch=intrinsics_torch,
#             extrinsics_torch=extrinsics_torch,
#             epochs=epochs,
#         )
#         # convert back to depth
#         depth_source_consistent = invert_depth_with_scale(depth_inv_consistent, depth_scale)
        
#     else:
#         print("Estimating depth without alignment...")
        
#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             depth_inv_consistent = depth_trainer.video_depth_model.forward(
#                 frames_padded.unsqueeze(0)
#                 )
#         depth_source_consistent = invert_depth_with_scale(depth_inv_consistent, depth_scale)
   
   
#     # Crop back to original size
#     depths_cropped = depth_source_consistent[
#         :, :, 
#         pad_top:H_padded-pad_bottom if pad_bottom > 0 else H_padded,
#         pad_left:W_padded-pad_right if pad_right > 0 else W_padded
#     ]
    
#     # Permute to [T, 1, H, W] format
#     depths_output = depths_cropped.permute(1, 0, 2, 3)
    
#     print(f"Output depth shape: {depths_output.shape}")
    
#     return depths_output


def estimate_depth_without_alignment(
    frames_tensor,
    depth_trainer,
    depth_scale,
    multiple_of=14,
):
    """
    Estimate depth without alignment, requires padding for ViT model.
    """
    T, C, H_orig, W_orig = frames_tensor.shape
    
    # Calculate padding needed
    pad_h = (multiple_of - (H_orig % multiple_of)) % multiple_of
    pad_w = (multiple_of - (W_orig % multiple_of)) % multiple_of
    
    # Pad symmetrically (top/bottom, left/right)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding
    frames_padded = torch.nn.functional.pad(
        frames_tensor, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode='reflect'
    )
    frames_padded = frames_padded.to(depth_trainer.device, dtype=torch.bfloat16)
    
    H_padded, W_padded = frames_padded.shape[-2:]
    print(f"Original size: {H_orig}x{W_orig}, Padded size: {H_padded}x{W_padded}")
    
    # Estimate depth
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        depth_inv_consistent = depth_trainer.video_depth_model.forward(
            frames_padded.unsqueeze(0)
        )
    depth_source_consistent = invert_depth_with_scale(depth_inv_consistent, depth_scale)
    
    # Crop back to original size
    depths_cropped = depth_source_consistent[
        :, :, 
        pad_top:H_padded-pad_bottom if pad_bottom > 0 else H_padded,
        pad_left:W_padded-pad_right if pad_right > 0 else W_padded
    ]
    
    # Permute to [T, 1, H, W] format
    depths_output = depths_cropped.permute(1, 0, 2, 3)
    
    return depths_output


def estimate_depth_with_alignment(
    frames_tensor,
    depth_trainer,
    depth_scale,
    sparse_depth,
    sparse_mask,
    intrinsics_torch,
    extrinsics_torch,
    epochs,
    resize_factor=2,
    multiple_of=14,
):
    """
    Estimate depth with sparse depth alignment. Resizes for memory efficiency.
    No padding needed as trainer handles size constraints internally.
    """
    T, C, H_orig, W_orig = frames_tensor.shape
    
    # Calculate resized dimensions (reduce by resize_factor, ensure multiple_of constraint)
    H_resized = ((H_orig // resize_factor) // multiple_of) * multiple_of
    W_resized = ((W_orig // resize_factor) // multiple_of) * multiple_of
    
    print(f"Resizing for alignment optimization: {H_orig}x{W_orig} -> {H_resized}x{W_resized}")
    
    print('frames_tensor.shape', frames_tensor.shape, frames_tensor.dtype, frames_tensor.min(), frames_tensor.max())
    print('sparse_depth.shape', sparse_depth.shape)
    print('sparse_mask.shape', sparse_mask.shape)
    
    
    # Resize frames (RGB) - use bilinear interpolation
    frames_resized = torch.nn.functional.interpolate(
        frames_tensor, 
        size=(H_resized, W_resized), 
        mode='bilinear', 
        align_corners=False
    ).to(depth_trainer.device, dtype=torch.bfloat16) # [32, 3, 182, 336]
    
    # Resize sparse depth - use bilinear for continuous values
    sparse_depth_resized = torch.nn.functional.interpolate(
        sparse_depth, 
        size=(H_resized, W_resized), 
        mode='bilinear', 
        align_corners=False
    ) # [32, 1, 182, 336]
    
    # Resize mask - use nearest neighbor to preserve binary nature
    sparse_mask_resized = torch.nn.functional.interpolate(
        sparse_mask.float(), 
        size=(H_resized, W_resized), 
        mode='nearest'
    ).bool() # [32, 1, 182, 336]
    
    # print shapes for debugging
    print('frames_resized.shape', frames_resized.shape)
    print('sparse_depth_resized.shape', sparse_depth_resized.shape)
    print('sparse_mask_resized.shape', sparse_mask_resized.shape)
    
    # Zero out depth values where mask is False after resizing
    sparse_depth_resized = sparse_depth_resized * sparse_mask_resized.float()
    
    # update the mask so it covers only valid depth pixels
    sparse_mask_resized = sparse_depth_resized > 0
    
    
    # Move to device
    sparse_depth_resized = sparse_depth_resized.to(depth_trainer.device, dtype=torch.bfloat16)
    sparse_mask_resized = sparse_mask_resized.to(depth_trainer.device)
    
    # Convert sparse_depth to inverse depth, using scale, but preserve zeros
    sparse_depth_inv_resized = invert_depth_with_scale(sparse_depth_resized, depth_scale)
    
    # Scale intrinsics for resized resolution
    scale_x = W_resized / W_orig
    scale_y = H_resized / H_orig
    intrinsics_resized = intrinsics_torch.clone()
    intrinsics_resized[0, 0] *= scale_x  # fx
    intrinsics_resized[1, 1] *= scale_y  # fy
    intrinsics_resized[0, 2] *= scale_x  # cx
    intrinsics_resized[1, 2] *= scale_y  # cy
    
    # print intrinsics for debugging
    print('intrinsics_orig', intrinsics_torch.shape)
    print('intrinsics_resized', intrinsics_resized.shape)
    print('extrinsics_torch', extrinsics_torch.shape)

    # Optimize depth alignment at reduced resolution
    depth_inv_consistent_resized, visual_prompt, final_scale, final_shift = depth_trainer.train(
        frames_resized.unsqueeze(0),  # [1, T, C, H_resized, W_resized]
        sparse_depth_inv_resized.permute(1, 0, 2, 3),   # [1, T, H_resized, W_resized]
        sparse_mask_resized.permute(1, 0, 2, 3),    # [1, T, H_resized, W_resized]
        intrinsics_torch=intrinsics_resized,
        extrinsics_torch=extrinsics_torch,
        epochs=epochs,
    )
    
    print('depth_inv_consistent_resized.shape', depth_inv_consistent_resized.shape)
    
    # Resize back to original resolution
    depth_inv_consistent = torch.nn.functional.interpolate(
        depth_inv_consistent_resized, 
        size=(H_orig, W_orig), 
        mode='bilinear', 
        align_corners=False
    )
    
    print('depth_inv_consistent.shape', depth_inv_consistent.shape)
    
    # Convert back to depth
    depth_source_consistent = invert_depth_with_scale(depth_inv_consistent, depth_scale)
    
    # Permute to [T, 1, H, W] format
    depths_output = depth_source_consistent.permute(1, 0, 2, 3)
    
    return depths_output




def save_warped_comparison_plot(warped_images, frames_target_tensor, segment_dir, frame_idx=10):
    """Save comparison plot between warped image and target frame."""
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.imshow((warped_images[frame_idx][0].permute(1, 2, 0).cpu().numpy() + 1) / 2)
    plt.axis('off')
    plt.title('Warped Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow((frames_target_tensor[frame_idx].permute(1, 2, 0).cpu().numpy() + 1) / 2)
    plt.axis('off')
    plt.title('Target Frame')
    
    plt.savefig(os.path.join(segment_dir, f'warped_vs_target_frame_{frame_idx:02d}.png'))
    plt.close()


def imagenet_to_0_1(frames_im):
    frames_denorm = denormalize_rgb(frames_im.cpu())
    frames_clamped = torch.clamp(frames_denorm, 0, 1)
    return frames_clamped


def imagenet_to_1_1(frames_im):
    frames_0_1 = imagenet_to_0_1(frames_im)
    frames_1_1 = frames_0_1 * 2.0 - 1.0
    return frames_1_1

def save_depth_comparison_plots(sparse_depth, consistent_depth, save_dir, step=5):
    """
    Save comparison plots between sparse depth and consistent depth.
    
    Args:
        sparse_depth: [T, 1, H, W] sparse depth tensor
        consistent_depth: [T, 1, H, W] consistent depth tensor
        save_dir: directory to save plots
        step: frame step for plotting (default: every 5th frame)
    """
    # Create subdirectory for depth comparisons
    depth_comparison_dir = os.path.join(save_dir, 'depth_comparisons')
    os.makedirs(depth_comparison_dir, exist_ok=True)
    
    print(f"Saving depth comparison plots to: {depth_comparison_dir}")
    
    # Plot sparse depth vs consistent depth for every step-th frame
    for i in range(0, sparse_depth.shape[0], step):
        plt.figure(figsize=(15, 5))
        
        # Get depth arrays
        sparse_depth_np = sparse_depth[i, 0].cpu().numpy()
        consistent_depth_np = consistent_depth[i, 0].cpu().numpy()
        
        # Calculate common colormap range based on valid (non-zero) values
        valid_sparse_mask = sparse_depth_np > 0
        valid_sparse_values = sparse_depth_np[valid_sparse_mask]
        valid_consistent_values = consistent_depth_np[valid_sparse_mask]  # Only compare where sparse is valid
        
        if len(valid_sparse_values) > 0 and len(valid_consistent_values) > 0:
            # Use same scale for both plots
            vmin = min(valid_sparse_values.min(), valid_consistent_values.min())
            # vmin = 0
            vmax = max(valid_sparse_values.max(), valid_consistent_values.max())
            print(f"Frame {i}: vmin={vmin}, vmax={vmax}")
        else:
            # Fallback if no valid values
            vmin, vmax = 0, 1
        
        # Sparse depth
        plt.subplot(1, 3, 1)
        plt.imshow(sparse_depth_np, cmap='plasma', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title('Sparse Depth (Warped)')
        plt.colorbar(shrink=0.4)
        
        # Consistent depth
        plt.subplot(1, 3, 2)
        plt.imshow(consistent_depth_np, cmap='plasma', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title('Consistent Depth (Aligned)')
        plt.colorbar(shrink=0.4)
        
        # Difference
        plt.subplot(1, 3, 3)
        # Only compute difference where sparse depth is valid (non-zero)
        diff = np.zeros_like(sparse_depth_np)
        diff[valid_sparse_mask] = consistent_depth_np[valid_sparse_mask] - sparse_depth_np[valid_sparse_mask]
        diff_max = np.abs(diff).max() if np.abs(diff).max() > 0 else 1
        plt.imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        plt.axis('off')
        plt.title('Difference (Aligned - Sparse)')
        plt.colorbar(shrink=0.4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(depth_comparison_dir, f'depth_comparison_frame_{i:02d}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"Saved depth comparison plots for frames: {list(range(0, sparse_depth.shape[0], step))}")



def align_video_to_pc(
    frames_source_im,
    frames_source_tensor,
    poses_source,
    global_pcs,
    global_colors,
    funwarp,
    K,
    depth_trainer,
    depth_scale,
    opts,
):
    # render existing depth
    # run depth estimator with depth optimization
    # unproject source frames
    # merge with global pcs (debug, later only inpainted regions)
    
    # render existing depth
    warped_images, warped_depths, masks = render_video_from_pc(
        global_pcs,
        global_colors,
        poses_source,
        K,
        (frames_source_im.shape[2], frames_source_im.shape[3]),
        funwarp,
    )
    
    print('warped_depths[0].shape', warped_depths[0].shape)
    
    warped_depths = torch.cat(warped_depths, dim=0)  # [T, 1, H, W]
    masks = torch.cat(masks, dim=0)
    
    print('warped_depths', warped_depths.shape, warped_depths.dtype)
        
        
    # run depth estimator with depth optimization
    # depth_source_consistent = depth_trainer.estimate_aligned_depth(
    #     frames_source_im,
    #     torch.stack(warped_depths, dim=0),
    #     torch.stack(masks, dim=0),
    # )
    
    # this needs padding
    # depth_source_consistent, visual_prompt, final_scale, final_shift = trainer.train(
    #     rgb, sparse_depth, sparse_mask,
    #     intrinsics_torch=K,
    #     extrinsics_torch=poses_source,
    #     epochs=50,
    # )
    
    depth_source_consistent = estimate_depth_with_alignment(
        frames_source_im,
        depth_trainer,
        depth_scale,
        sparse_depth=warped_depths, 
        sparse_mask=masks,
        intrinsics_torch=K[0],
        extrinsics_torch=poses_source,
        epochs=50,
    )
    
    # Create comparison plots between sparse depth and consistent depth
    save_depth_comparison_plots(
        sparse_depth=warped_depths,
        consistent_depth=depth_source_consistent,
        save_dir=opts.segment_dir,
        step=5
    )
    

    
    
    
    # unproject source frames
    # source_pcs = []
    # source_colors = []
    # for i in tqdm(range(frames_source_im.shape[0])):
    #     with torch.no_grad():
    #         points, colors, _ = funwarp.create_pointcloud_from_image(
    #             frames_source_tensor[i:i+1],
    #             None,
    #             depth_source_consistent[i:i+1],
    #             poses_source[i:i+1],
    #             K[0:1],
    #             1,
    #         )
    #     source_pcs.append(points)
    #     source_colors.append(colors)
    
    # run video_to_pcs
    source_pcs, source_colors = video_to_pcs(
        frames_source_tensor,
        depth_source_consistent,
        K,
        poses_source,
        funwarp,
    )
        
    # merge with global pcs (debug, later only inpainted regions)
    pcs_merged = []
    colors_merged = []
    for i in range(frames_source_im.shape[0]):

        pc = torch.cat((global_pcs[i], source_pcs[i]), dim=0)
        color = torch.cat((global_colors[i], source_colors[i]), dim=0)
        
        # downsample by factor of 2 randomly
        indices = torch.randperm(pc.shape[0], device=pc.device)[:len(pc)//2]
        pc = pc[indices]
        color = color[indices]
        
        pcs_merged.append(pc)
        colors_merged.append(color)
        
    return pcs_merged, colors_merged


def render_video_from_pc(
    global_pcs,
    global_colors,
    poses,
    K,
    dims,
    funwarp,
):
    
    # render warped images and masks
    warped_images = []
    warped_depths = []
    masks = []
    
    for i in tqdm(range(len(global_pcs))):
        warped_image, mask, warped_depth = funwarp.render_pointcloud_zbuffer_vectorized_point_size(
            global_pcs[i],
            global_colors[i],
            poses[i:i+1],
            K,
            (dims[0], dims[1]),
            point_size=2,
            return_depth=True,
        )
        
        cleaned_mask = clean_single_mask_simple(
            mask[0],
            kernel_size=9,
            n_erosion_steps=1,
            n_dilation_steps=1
            )
        
        cleaned_mask = cleaned_mask.unsqueeze(0)
        
        warped_image = warped_image * cleaned_mask
        warped_depth = warped_depth * cleaned_mask
        
        warped_images.append(warped_image)
        warped_depths.append(warped_depth)
        masks.append(cleaned_mask)
        
    # warped_images = torch.stack(warped_images, dim=0)  # [T, 1, C, H, W]
    # warped_depths = torch.stack(warped_depths, dim=0)  # [T, 1, 1, H, W]
    # masks = torch.stack(masks, dim=0)                  # [T, 1, 1, H, W]
        
    print('warped_images', len(warped_images), warped_images[0].shape)
    print('warped_depths', len(warped_depths), warped_depths[0].shape)
    print('masks', len(masks), masks[0].shape)
        
    return warped_images, warped_depths, masks
             
    

# with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#     depths_source_target_inv = video_depth_anything.forward(
#         frames_source_target_im.to(opts.device, dtype=torch.bfloat16).unsqueeze(0)
#         ).permute(1, 0, 2, 3)  # [1, T, H, W] -> [T, 1, H, W]   
    

def autoregressive_loop(
    frames_source_im, # torch.Size([32, 3, 266, 462]) torch.float32 tensor(-2.2437) tensor(2.6739)
    poses_source,
    poses_target,
    global_pcs,
    global_colors,
    radius,
    opts,                   # Configuration options
    vis_crafter,            # TrajCrafterAutoregressive instance
    funwarp,                # Point cloud warping utilities
    video_depth_anything,
    depth_trainer,
    args_vda,
    depth_scale,
    curr_stage,
):
    
    # --- determine output directory ---
    n_subdirs = len([
        name for name in os.listdir(opts.save_dir)
        if os.path.isdir(os.path.join(opts.save_dir, name))
    ])
    # curr_stage = n_subdirs
    
    segment_dir = os.path.join(opts.save_dir, f'stage_{curr_stage:02d}')
    os.makedirs(segment_dir, exist_ok=True)
    print(f"Saving to: {segment_dir}")
    
    opts.segment_dir = segment_dir    
    
    #########################################
    # Handle imagenet normalization
    #########################################
    
    # frames_source_np should be (T, H, W, 3) 0-1 numpy array
    frames_source_np = imagenet_to_0_1(frames_source_im).permute(0, 2, 3, 1).cpu().numpy()
    
    frames_source_tensor = (
        torch.from_numpy(frames_source_np).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
    )
    
    save_video(
        frames_source_np,
        os.path.join(segment_dir, 'frames_source.mp4'),
        fps=opts.fps,
    )
    
    #########################################

    # prompt
    if vis_crafter.prompt is None:
        vis_crafter.prompt = vis_crafter.get_caption(opts, frames_source_np[opts.video_length // 2])

    torch.save(poses_target.cpu(), os.path.join(segment_dir, 'c2ws_target.pt'))
    torch.save(poses_source.cpu(), os.path.join(segment_dir, 'c2ws_source.pt'))
    
    ##########################################
    # Update global pcs
    ##########################################
    
    if curr_stage > 0:
        global_pcs, global_colors = align_video_to_pc(
            frames_source_im,
            frames_source_tensor,
            poses_source,
            global_pcs,
            global_colors,
            funwarp,
            vis_crafter.K,
            depth_trainer,
            depth_scale,
            opts,
        )
    
    
    if curr_stage % 2 == 0:
        # reverse global pcs and colors
        print('reversing the global PC')
        global_pcs = global_pcs[::-1]
        global_colors = global_colors[::-1]
    
    
    # save global pcs and colors to stage directory
    os.makedirs(os.path.join(segment_dir, 'global_pcs'), exist_ok=True)
    os.makedirs(os.path.join(segment_dir, 'global_colors'), exist_ok=True)
    for i in range(opts.video_length):
        torch.save(global_pcs[i].cpu(), os.path.join(segment_dir, 'global_pcs', f'pc_{i:02d}.pt'))
        torch.save(global_colors[i].cpu(), os.path.join(segment_dir, 'global_colors', f'color_{i:02d}.pt'))    
    
    
    ##########################################
    # render warped images and masks for inpainting
    ##########################################
    
    warped_images, warped_depths, masks = render_video_from_pc(
        global_pcs,
        global_colors,
        poses_target,
        vis_crafter.K,
        (frames_source_im.shape[2], frames_source_im.shape[3]),
        funwarp,
    )
        
    # plot the result
    # get 4 evenly spaced frames, not 0
    idxs_plot = np.linspace(0, opts.video_length - 1, num=5, dtype=int)
    for k in idxs_plot:
        if k == 0:
            continue
        save_warped_comparison_plot(
            warped_images, frames_source_tensor, segment_dir,
            frame_idx=k
            )
    
        
    sample_autoreg, segment_dir_autoreg = sample_diffusion(
        vis_crafter,
        frames_tensor=frames_source_tensor,
        warped_images=warped_images,
        # frames 49 - 39
        frames_ref=frames_source_tensor[:10],
        masks=masks,
        opts=opts,
        segment_dir=segment_dir,
    )
    
    print('sample_autoreg', sample_autoreg.shape, sample_autoreg.dtype, sample_autoreg.min(), sample_autoreg.max())
    
    return segment_dir_autoreg, global_pcs, global_colors
