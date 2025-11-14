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


def estimate_depth_with_padding(frames_tensor, video_depth_model, device, multiple_of=14):
    """
    Estimate depth with padding to ensure input is divisible by multiple_of.
    
    Args:
        frames_tensor: [T, C, H, W] input frames (ImageNet normalized)
        video_depth_model: Video Depth Anything model
        device: torch device
        multiple_of: padding constraint (default: 14 for ViT)
    
    Returns:
        depths: [T, 1, H_orig, W_orig] depth predictions at original resolution
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
    
    # Apply padding: (pad_left, pad_right, pad_top, pad_bottom)
    frames_padded = torch.nn.functional.pad(
        frames_tensor, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode='reflect'  # Use reflection padding to avoid edge artifacts
    )
    
    H_padded, W_padded = frames_padded.shape[-2:]
    print(f"Original size: {H_orig}x{W_orig}, Padded size: {H_padded}x{W_padded}")
    print(f"Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
    
    # Estimate depth on padded frames
    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    #     depths_padded = video_depth_model.forward(
    #         frames_padded.to(device, dtype=torch.bfloat16).unsqueeze(0)
    #     )  # [1, T, H_padded, W_padded]

    #assert frames_padded is 0 to 255, and not 0 to 1 or imagenet normalized
    assert frames_padded.min() >= 0 and 2 <= frames_padded.max() <= 255 and frames_padded.mean() > 0.5

    depths_padded, fps = video_depth_model.infer_video_depth(
        frames_padded.permute(0, 2, 3, 1).numpy(), target_fps=10, input_size=max(H_padded, W_padded)
        ) # [T, H, W]
    depths_padded = torch.tensor(depths_padded).unsqueeze(0)  # [1, T, H, W]
    
    # Crop back to original size
    depths_cropped = depths_padded[
        :, :, 
        pad_top:H_padded-pad_bottom if pad_bottom > 0 else H_padded,
        pad_left:W_padded-pad_right if pad_right > 0 else W_padded
    ]
    
    # Permute to [T, 1, H, W] format
    depths_output = depths_cropped.permute(1, 0, 2, 3)
    
    print(f"Output depth shape: {depths_output.shape}")
    
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


def autoregressive_loop(
    frames_source_im, # torch.Size([32, 3, 266, 462]) torch.float32 tensor(-2.2437) tensor(2.6739)
    frames_target_im, # torch.Size([32, 3, 266, 462]) torch.float32 tensor(-2.2437) tensor(2.6739)
    poses_source,
    poses_target,
    radius,
    opts,                    # Configuration options
    vis_crafter,            # TrajCrafterAutoregressive instance
    funwarp,                # Point cloud warping utilities
    video_depth_anything,
    depth_trainer,
    args_vda,
    depth_scale,
):
    
    # --- determine output directory ---
    n_subdirs = len([
        name for name in os.listdir(opts.save_dir)
        if os.path.isdir(os.path.join(opts.save_dir, name))
    ])
    segment_dir = os.path.join(opts.save_dir, f'stage_{n_subdirs + 1}')
    os.makedirs(segment_dir, exist_ok=True)
    print(f"Saving to: {segment_dir}")
    
    
    #########################################
    # Handle imagenet normalization
    #########################################
    
    # frames_source_np should be (T, H, W, 3) 0-1 numpy array
    if frames_source_im is not None:
        frames_source_np = imagenet_to_0_1(frames_source_im).permute(0, 2, 3, 1).cpu().numpy()
    else:
        frames_source_np = None
        
    frames_target_np = imagenet_to_0_1(frames_target_im).permute(0, 2, 3, 1).cpu().numpy()

    #########################################
    # Input for depth estimation, imagenet normalized
    #########################################
    
    if frames_source_im is not None:
        frames_source_target_im = torch.cat([frames_source_im, frames_target_im], dim=0)
    else:
        frames_source_target_im = frames_target_im
    
    #########################################

    # prompt
    if vis_crafter.prompt is None:
        vis_crafter.prompt = vis_crafter.get_caption(opts, frames_target_np[opts.video_length // 2])
    
    # concatenate with source
    if frames_source_np is None:
        frames_source_target_np = frames_target_np
    else:
        frames_source_target_np = np.concatenate([frames_source_np, frames_target_np], axis=0)
       
    # convert to tensors 
    frames_source_target_tensor = (
        torch.from_numpy(frames_source_target_np).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
    )
    frames_target_tensor = (
        torch.from_numpy(frames_target_np).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
    )
    
    poses_source_target = torch.cat([poses_source, poses_target], dim=0)
    
    # save camera poses
    torch.save(poses_source_target.cpu(), os.path.join(segment_dir, 'c2ws_source_target.pt'))
    torch.save(poses_target.cpu(), os.path.join(segment_dir, 'c2ws_target.pt'))
    torch.save(poses_source.cpu(), os.path.join(segment_dir, 'c2ws_source.pt'))
    
    # also save frames_source_np, frames_target_np frames_source_target_np, name as variables
    
    save_video(
        frames_source_target_np,
        os.path.join(segment_dir, 'frames_source_target.mp4'),
        fps=opts.fps,
    )
    save_video(
        frames_target_np,
        os.path.join(segment_dir, 'frames_target.mp4'),
        fps=opts.fps,
    )
    if frames_source_np is not None:
        save_video(
            frames_source_np,
            os.path.join(segment_dir, 'frames_source.mp4'),
            fps=opts.fps,
        )
        
        
    # depth estimation
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    #     depths_source_target = vis_crafter.depth_estimater.infer(
    #         frames_source_target_np,
    #         opts.near,
    #         opts.far,
    #         opts.depth_inference_steps,
    #         opts.depth_guidance_scale,
    #         window_size=opts.window_size,
    #         overlap=opts.overlap,
    #     ).to(opts.device)
    
    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    #     depths_source_target_inv = video_depth_anything.forward(
    #         frames_source_target_im.to(opts.device, dtype=torch.bfloat16).unsqueeze(0)
    #         ).permute(1, 0, 2, 3)  # [1, T, H, W] -> [T, 1, H, W]
    
    depths_source_target_inv = estimate_depth_with_padding(
        # frames_source_target_im, 
        torch.from_numpy(frames_source_target_np).permute(0, 3, 1, 2) * 255.0,
        video_depth_anything, 
        opts.device, 
        multiple_of=14
    )
        
    depths_source_target = depth_scale / depths_source_target_inv

    
    
    
    
        
    # align the depth to match the target cameras
    radius_after = (
        depths_source_target[opts.video_length - 1, 0, depths_source_target.shape[-2] // 2, depths_source_target.shape[-1] // 2].cpu()
        * opts.radius_scale
    )
    depths_source_target = depths_source_target / (radius_after / radius)

    # assert that the radius of aligned depth is equal to radius before
    radius_after_2 = depths_source_target[opts.video_length - 1, 0, depths_source_target.shape[-2] // 2, depths_source_target.shape[-1] // 2].cpu() * opts.radius_scale
    assert abs(radius_after_2 - radius) < 0.1
    
    
    # visualize inverse depth
    depths_vis = 1/depths_source_target  # Invert for better visualization
    depths_vis_np = depths_vis.permute(0, 2, 3, 1).cpu().numpy()
    depths_vis_np = np.repeat(depths_vis_np, 3, axis=3)  # (T, H, W, 3)
    save_video(
        depths_vis_np,
        os.path.join(segment_dir, 'depths_all.mp4'),
        fps=opts.fps,
    )
    
    
    
    # extract point clouds
    point_clouds_source_target = []
    colors_list_source_target = []
    for i in tqdm(range(frames_source_target_np.shape[0])):
        with torch.no_grad():
            points, colors, _ = funwarp.create_pointcloud_from_image(
                frames_source_target_tensor[i:i+1],
                None,
                depths_source_target[i:i+1],
                poses_source_target[i:i+1],
                vis_crafter.K[0:1],
                1,
            )
        point_clouds_source_target.append(points)
        colors_list_source_target.append(colors)
    
    # concatenate and downsample the point clouds,
    # we should have opts.video_length point clouds at the end
    global_pc = []
    global_colors = []
    for i in range(opts.video_length):
        # select 0 * opts.video_length + i, 1 * opts.video_length + i, ... 
        # from the full point cloud list
        
        # print(f'i = {i}')
        
        # os.makedirs(os.path.join(segment_dir, f'local_pcs/{i:02d}'), exist_ok=True)
        # os.makedirs(os.path.join(segment_dir, f'local_colors/{i:02d}'), exist_ok=True)
        
        pcs_to_merge = []
        colors_to_merge = []
        for j in range(0, len(point_clouds_source_target) // opts.video_length):
            
            # print(f'  j = {j}')
            
            # even = reverse, odd = normal
            if j % 2 == 1:
                pc_j = point_clouds_source_target[j * opts.video_length + i]
                color_j = colors_list_source_target[j * opts.video_length + i]
            else:
                pc_j = point_clouds_source_target[(j + 1) * opts.video_length - i - 1]
                color_j = colors_list_source_target[(j + 1) * opts.video_length - i - 1]
                
            pcs_to_merge.append(pc_j)
            colors_to_merge.append(color_j)
            
            # save pc_j and color_j
            # torch.save(pc_j.cpu(), os.path.join(segment_dir, f'local_pcs/{i:02d}', f'pc_{j:02d}.pt'))
            # torch.save(color_j.cpu(), os.path.join(segment_dir, f'local_colors/{i:02d}', f'color_{j:02d}.pt'))

        pc_merged = torch.cat(pcs_to_merge, dim=0)
        color_merged = torch.cat(colors_to_merge, dim=0)
        
        # downsample by factor of len(pcs_to_merge) randomly
        indices = torch.randperm(pc_merged.shape[0], device=pc_merged.device)[:len(pc_merged)//len(pcs_to_merge)]
        pc_merged = pc_merged[indices]
        color_merged = color_merged[indices]
        
        global_pc.append(pc_merged)
        global_colors.append(color_merged)
        
        
    if (len(point_clouds_source_target) // opts.video_length) % 2 == 0:
        # reverse global pcs and colors
        print('reversing the global PC')
        global_pc = global_pc[::-1]
        global_colors = global_colors[::-1]
    
    # save global pcs and colors to stage directory
    os.makedirs(os.path.join(segment_dir, 'global_pc'), exist_ok=True)
    os.makedirs(os.path.join(segment_dir, 'global_colors'), exist_ok=True)
    for i in range(opts.video_length):
        torch.save(global_pc[i].cpu(), os.path.join(segment_dir, 'global_pc', f'pc_{i:02d}.pt'))
        torch.save(global_colors[i].cpu(), os.path.join(segment_dir, 'global_colors', f'color_{i:02d}.pt'))    
    
    
    # render warped images and masks
    warped_images = []
    masks = []
    
    for i in tqdm(range(opts.video_length)):
        warped_image, mask = funwarp.render_pointcloud_zbuffer_vectorized_point_size(
            global_pc[i],
            global_colors[i],
            poses_target[i:i+1],
            vis_crafter.K[0:1].to(opts.device),
            (frames_target_tensor.shape[2], frames_target_tensor.shape[3]),
            point_size=2,
        )
        
        cleaned_mask = clean_single_mask_simple(
            mask[0],
            kernel_size=9,
            n_erosion_steps=1,
            n_dilation_steps=1
            )
        
        cleaned_mask = cleaned_mask.unsqueeze(0)
        
        warped_image = warped_image * cleaned_mask
        
        warped_images.append(warped_image)
        masks.append(cleaned_mask)
             
        
    # plot the result
    # get 4 evenly spaced frames, not 0
    idxs_plot = np.linspace(0, opts.video_length - 1, num=5, dtype=int)
    for k in idxs_plot:
        if k == 0:
            continue
        save_warped_comparison_plot(
            warped_images, frames_target_tensor, segment_dir,
            frame_idx=k
            )
    
        
    sample_autoreg, segment_dir_autoreg = sample_diffusion(
        vis_crafter,
        frames_tensor=frames_target_tensor,
        warped_images=warped_images,
        # frames 49 - 39
        frames_ref=frames_source_target_tensor[:10],
        masks=masks,
        opts=opts,
        segment_dir=segment_dir,
    )
    
    print('sample_autoreg', sample_autoreg.shape, sample_autoreg.dtype, sample_autoreg.min(), sample_autoreg.max())
    
    return segment_dir_autoreg
