# Standard library imports
import sys
import os
import copy
import json
from datetime import datetime

# Third-party imports
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

# Add project paths
sys.path.append('/home/azhuravl/work')
sys.path.append('/home/azhuravl/work/TrajectoryCrafter')
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/15_10_25_depth')
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/06_10_25_vggt')

# Project imports
import stereoanyvideo.datasets.video_datasets_custom as video_datasets_custom
import collect_dataset
import models.utils as utils
from models.utils import save_video
from parsing import get_parser
import utils_autoregressive as utils_ar



def get_resize_strategy(original_height, original_width, target_height=384, target_width=672):
    """Determine the best resize strategy based on aspect ratios"""
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height  # 672/384 = 1.75
    
    if original_aspect > target_aspect * 1.2:
        return "crop_width"
    # If original is much taller, crop height  
    elif original_aspect < target_aspect * 0.8:
        return "crop_height"
    else:
        # For remaining cases, just use resize (may cause some distortion)
        return "resize"


def smart_video_resize(video, target_height=384, target_width=672, interpolation_mode='bilinear'):
    """
    Smart resize for videos with various aspect ratios
    
    Args:
        video: tensor of shape [T, C, H, W]
        target_height: target height (default 384)
        target_width: target width (default 672)
        interpolation_mode: 'bilinear' for RGB videos, 'nearest' for masks
        
    Returns:
        video: tensor of shape [T, C, target_height, target_width]
    """
    T, C, H, W = video.shape
    
    strategy = get_resize_strategy(H, W, target_height, target_width)
    
    print(f"resize strategy: '{strategy}' for original size ({H}, {W}), aspect ratio {W/H:.2f}, target ratio {target_width/target_height:.2f}")
    
    if strategy == "resize":
        # Simple resize when aspect ratios are similar or as fallback
        video = F.interpolate(
            video, 
            size=(target_height, target_width), 
            mode=interpolation_mode, 
            align_corners=False if interpolation_mode == 'bilinear' else None
        )
        
    elif strategy == "crop_width":
        # Video is too wide - crop width first, then resize
        target_aspect = target_width / target_height
        new_width = int(H * target_aspect)
        
        # Center crop width
        start_w = (W - new_width) // 2
        video = video[:, :, :, start_w:start_w + new_width]
        
        # Then resize to target
        video = F.interpolate(
            video,
            size=(target_height, target_width),
            mode=interpolation_mode,
            align_corners=False if interpolation_mode == 'bilinear' else None
        )
        
    elif strategy == "crop_height":
        # Video is too tall - crop height first, then resize
        target_aspect = target_width / target_height
        new_height = int(W / target_aspect)
        
        # Center crop height
        start_h = (H - new_height) // 2
        video = video[:, :, start_h:start_h + new_height, :]
        
        # Then resize to target
        video = F.interpolate(
            video,
            size=(target_height, target_width),
            mode=interpolation_mode,
            align_corners=False if interpolation_mode == 'bilinear' else None
        )
    
    return video




def make_dimensions_even(tensor):
    """Pad tensor to make height and width even numbers"""
    _, h, w, c = tensor.shape
    pad_h = h % 2
    pad_w = w % 2
    
    if pad_h > 0 or pad_w > 0:
        # Pad bottom and right if needed
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_w, 0, pad_h))
    
    return tensor


def apply_colormap_to_depth(depth_tensor, colormap='viridis', inverse=True):
    """
    Apply colormap to depth tensor for better visualization
    
    Input: [T, H, W] - Time/batch dimension, Height, Width (single channel depth)
    Output: [T, H, W, 3] - Same spatial dimensions but with RGB color channels
    """
    # Create mask for zero values
    zero_mask = (depth_tensor == 0)
    
    if inverse:
        # Compute inverse depth, avoid division by zero
        depth_processed = torch.where(depth_tensor > 0, 1.0 / depth_tensor, torch.zeros_like(depth_tensor))
    else:
        depth_processed = depth_tensor
    
    # Normalize non-zero values to [0, 1]
    if depth_processed[~zero_mask].numel() > 0 and not inverse:
        depth_norm = depth_processed / depth_processed[~zero_mask].max()
    else:
        depth_norm = depth_processed
        
        
    # print(f'min depth (after processing): {depth_norm[~zero_mask].min().item() if depth_norm[~zero_mask].numel() > 0 else 0.0}, max depth: {depth_norm[~zero_mask].max().item() if depth_norm[~zero_mask].numel() > 0 else 0.0}')
    depth_norm[0, 0, 0] = 0.0  # For consistent colormap scaling in case of single value
    depth_norm[0, 0, 1] = 1.0
    
    
    # Convert to numpy and apply colormap
    depth_np = depth_norm.cpu().numpy()
    colormap_func = matplotlib.colormaps.get_cmap(colormap)
    depth_colored = colormap_func(depth_np)  # Returns RGBA
    
    # Convert back to tensor, drop alpha channel
    depth_colored_tensor = torch.from_numpy(depth_colored[..., :3]).to(depth_tensor.device)
    
    # Set zero depth areas to black
    zero_mask_expanded = zero_mask.unsqueeze(-1).expand_as(depth_colored_tensor)
    depth_colored_tensor[zero_mask_expanded] = 0.0  # Black for zero depth
    
    return depth_colored_tensor


# def apply_colormap_to_depth(depth_tensor, colormap='viridis', inverse=True, vmin=None, vmax=None):
#     """
#     Apply colormap to depth tensor for better visualization
    
#     Args:
#         depth_tensor: Input depth tensor of shape [T, H, W]
#         colormap: Matplotlib colormap name (default: 'viridis')
#         inverse: If True, applies inverse depth transformation (default: True)
#         vmin: Minimum value for colormap normalization (default: None, auto-compute)
#         vmax: Maximum value for colormap normalization (default: None, auto-compute)
    
#     Returns:
#         depth_colored_tensor: RGB colored depth tensor of shape [T, H, W, 3]
#     """
#     # Create mask for zero values
#     zero_mask = (depth_tensor == 0)
    
#     if inverse:
#         # Compute inverse depth, avoid division by zero
#         depth_processed = torch.where(depth_tensor > 0, 1.0 / depth_tensor, torch.zeros_like(depth_tensor))
#     else:
#         depth_processed = depth_tensor
    
#     # Determine normalization range
#     if vmin is None or vmax is None:
#         if depth_processed[~zero_mask].numel() > 0:
#             computed_min = depth_processed[~zero_mask].min().item()
#             computed_max = depth_processed[~zero_mask].max().item()
#             if vmin is None:
#                 vmin = computed_min
#             if vmax is None:
#                 vmax = computed_max
#         else:
#             vmin = vmin or 0.0
#             vmax = vmax or 1.0
    
#     # Normalize to [0, 1] using specified range
#     if vmax > vmin:
#         depth_norm = torch.clamp((depth_processed - vmin) / (vmax - vmin), 0.0, 1.0)
#     else:
#         depth_norm = torch.zeros_like(depth_processed)
    
#     # Convert to numpy and apply colormap
#     depth_np = depth_norm.cpu().numpy()
#     colormap_func = matplotlib.colormaps.get_cmap(colormap)
#     depth_colored = colormap_func(depth_np)  # Returns RGBA
    
#     # Convert back to tensor, drop alpha channel
#     depth_colored_tensor = torch.from_numpy(depth_colored[..., :3]).to(depth_tensor.device)
    
#     # Set zero depth areas to black
#     zero_mask_expanded = zero_mask.unsqueeze(-1).expand_as(depth_colored_tensor)
#     depth_colored_tensor[zero_mask_expanded] = 0.0  # Black for zero depth
    
#     return depth_colored_tensor


def encode_inputs_to_latents(
    pipeline,
    video=None,
    reference=None,
    mask_video=None,
    masked_video_latents=None,
    prompt=None,
    negative_prompt=None,
    height=480,
    width=720,
    device="cuda",
    batch_size=1,
    noise_aug_strength=0.0563,
    max_sequence_length=226,
    do_classifier_free_guidance=True,
    # New parameters for training
    ground_truth_video=None,  # GT video for training
    encode_for_training=False  # Flag to indicate training vs inference
):
    """
    Encode all inputs (video, reference, mask, prompts) to latents for training dataset preparation.
    
    Args:
        video: Conditioning video (warped/masked video for inference)
        ground_truth_video: Ground truth video for training targets
        encode_for_training: If True, encodes GT video as training targets
    """
    
    results = {}
    
    with torch.no_grad():
        # 1. Encode text prompts (same for both training and inference)
        if prompt is not None:
            prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            
            results['prompt_embeds'] = prompt_embeds.cpu()
            if negative_prompt_embeds is not None:
                results['negative_prompt_embeds'] = negative_prompt_embeds.cpu()
        
        # 2. Process reference video (same for both training and inference)
        if reference is not None:
            ref_length = reference.shape[2]
            ref_video = pipeline.image_processor.preprocess(
                rearrange(reference, "b c f h w -> (b f) c h w"), 
                height=height, 
                width=width
            )
            ref_video = rearrange(ref_video, "(b f) c h w -> b c f h w", f=ref_length)
            ref_video = ref_video.to(device=device, dtype=pipeline.vae.dtype)
            
            # Encode reference video
            bs = 1
            new_ref_video = []
            for i in range(0, ref_video.shape[0], bs):
                video_bs = ref_video[i : i + bs]
                video_bs = pipeline.vae.encode(video_bs)[0]
                video_bs = video_bs.sample()
                new_ref_video.append(video_bs)
            new_ref_video = torch.cat(new_ref_video, dim=0)
            new_ref_video = new_ref_video * pipeline.vae.config.scaling_factor
            ref_latents = new_ref_video.repeat(batch_size // new_ref_video.shape[0], 1, 1, 1, 1)
            
            # Rearrange ONLY for final storage
            ref_latents_final = rearrange(ref_latents, "b c f h w -> b f c h w")
            results['ref_latents'] = ref_latents_final.cpu()
        
        # 3. Encode ground truth video for training targets
        if encode_for_training and ground_truth_video is not None:
            gt_video_length = ground_truth_video.shape[2]
            gt_processed = pipeline.image_processor.preprocess(
                rearrange(ground_truth_video, "b c f h w -> (b f) c h w"), 
                height=height, 
                width=width
            )
            gt_processed = rearrange(gt_processed, "(b f) c h w -> b c f h w", f=gt_video_length)
            gt_processed = gt_processed.to(device=device, dtype=pipeline.vae.dtype)
            
            # Encode ground truth video
            bs = 1
            new_gt_video = []
            for i in range(0, gt_processed.shape[0], bs):
                video_bs = gt_processed[i : i + bs]
                video_bs = pipeline.vae.encode(video_bs)[0]
                video_bs = video_bs.sample()
                new_gt_video.append(video_bs)
            gt_encoded = torch.cat(new_gt_video, dim=0)
            gt_encoded = gt_encoded * pipeline.vae.config.scaling_factor
            
            # Store GT latents for training (in transformer format)
            gt_latents_final = rearrange(gt_encoded, "b c f h w -> b f c h w")
            results['gt_video_latents'] = gt_latents_final.cpu()
        
        # 4. Process conditioning video (if provided)
        init_video = None
        video_latents_bcfhw = None  # Keep in original format for mask processing
        
        if video is not None:
            video_length = video.shape[2]
            init_video = pipeline.image_processor.preprocess(
                rearrange(video, "b c f h w -> (b f) c h w"), 
                height=height, 
                width=width
            )
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
            init_video = init_video.to(device=device, dtype=pipeline.vae.dtype)
            
            # Encode conditioning video
            bs = 1
            new_video = []
            for i in range(0, init_video.shape[0], bs):
                video_bs = init_video[i : i + bs]
                video_bs = pipeline.vae.encode(video_bs)[0]
                video_bs = video_bs.sample()
                new_video.append(video_bs)
            video_encoded = torch.cat(new_video, dim=0)
            video_encoded = video_encoded * pipeline.vae.config.scaling_factor
            
            # Keep in [b, c, f, h, w] format for mask processing
            video_latents_bcfhw = video_encoded
            
            # Rearrange ONLY for final storage
            video_latents_final = rearrange(video_encoded, "b c f h w -> b f c h w")
            results['cond_video_latents'] = video_latents_final.cpu()  # Renamed for clarity
        
        # 5. Get model configuration
        num_channels_transformer = pipeline.transformer.config.in_channels
        num_channels_latents = pipeline.vae.config.latent_channels
        
        # 6. Process mask video (same as before but use video_latents_bcfhw or gt if available)
        reference_latents_bcfhw = video_latents_bcfhw
        if encode_for_training and ground_truth_video is not None:
            # For training, use GT video shape as reference
            reference_latents_bcfhw = rearrange(gt_encoded, "b f c h w -> b c f h w")
        
        if mask_video is not None and reference_latents_bcfhw is not None:
            video_length = mask_video.shape[2]
            
            if (mask_video == 255).all():
                # All mask case
                mask_latents = torch.zeros_like(reference_latents_bcfhw)[:, :, :1]
                masked_video_latents = torch.zeros_like(reference_latents_bcfhw)
                
                results['mask_latents'] = rearrange(mask_latents, "b c f h w -> b f c h w").cpu()
                results['masked_video_latents'] = rearrange(masked_video_latents, "b c f h w -> b f c h w").cpu()
                results['mask'] = None
            else:
                # Process mask condition
                mask_condition = pipeline.mask_processor.preprocess(
                    rearrange(mask_video, "b c f h w -> (b f) c h w"),
                    height=height,
                    width=width,
                )
                mask_condition = mask_condition.to(dtype=torch.float32)
                mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=video_length)
                
                if num_channels_transformer != num_channels_latents:
                    # Inpainting model case
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    
                    # Create masked video (use init_video as base)
                    if masked_video_latents is None and init_video is not None:
                        
                        # masked_video = init_video  # Use the conditioning video
                                # Apply the mask properly: preserve known regions, mask unknown regions
                        masked_video = (
                            init_video * (mask_condition_tile < 0.5)
                            + torch.ones_like(init_video) * (mask_condition_tile > 0.5) * -1
                        )
                    else:
                        masked_video = masked_video_latents or init_video
                    
                    # Encode masked video using prepare_mask_latents
                    _, masked_video_latents = pipeline.prepare_mask_latents(
                        None,
                        masked_video,
                        batch_size,
                        height,
                        width,
                        pipeline.vae.dtype,
                        device,
                        None,  # generator
                        do_classifier_free_guidance=False,
                        noise_aug_strength=noise_aug_strength,
                    )
                    
                    # Resize mask to latent size
                    mask_latents = resize_mask(1 - mask_condition, masked_video_latents)
                    mask_latents = mask_latents * pipeline.vae.config.scaling_factor
                    
                    # FIX: Ensure mask_latents matches the dtype of other latents
                    mask_latents = mask_latents.to(dtype=masked_video_latents.dtype)

                    
                    # Create mask at latent resolution
                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(
                        mask,
                        size=reference_latents_bcfhw.size()[-3:],
                        mode='trilinear', 
                        align_corners=True
                    )
                    
                    # FIX: Ensure mask matches the dtype of other latents  
                    mask = mask.to(dtype=reference_latents_bcfhw.dtype)

                    
                    # Rearrange for final storage
                    mask_input = rearrange(mask_latents, "b c f h w -> b f c h w")
                    masked_video_latents_input = rearrange(masked_video_latents, "b c f h w -> b f c h w")
                    mask_final = rearrange(mask, "b c f h w -> b f c h w")
                    
                    results['mask_latents'] = mask_input.cpu()
                    results['masked_video_latents'] = masked_video_latents_input.cpu()
                    results['mask'] = mask_final.cpu()
                else:
                    # Non-inpainting model case
                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(
                        mask,
                        size=reference_latents_bcfhw.size()[-3:],
                        mode='trilinear',
                        align_corners=True
                    )
                    mask_final = rearrange(mask, "b c f h w -> b f c h w")
                    results['mask'] = mask_final.cpu()
                    results['mask_latents'] = None
                    results['masked_video_latents'] = None
    
    # for all latents, squeeze the batch dimension
    for key in results:
        if results[key] is not None and results[key].shape[0] == 1:
            print(f"squeezing {key} from shape {results[key].shape}")
            results[key] = results[key].squeeze(0)
    
    return results


def resize_mask(mask, latent, process_first_frame_only=True):
    """Helper function from the pipeline - resize mask to match latent dimensions"""
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False,
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False,
            )
            resized_mask = torch.cat(
                [first_frame_resized, remaining_frames_resized], dim=2
            )
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask, size=target_size, mode='trilinear', align_corners=False
        )
    return resized_mask


def prepare_encoded_inputs_for_inference(encoded_data, do_classifier_free_guidance=True):
    """
    Helper function to prepare pre-encoded inputs for inference.
    """
    prepared_inputs = {}
    
    # Prepare prompt embeddings with CFG
    if 'prompt_embeds' in encoded_data:
        prompt_embeds = encoded_data['prompt_embeds']
        if do_classifier_free_guidance and 'negative_prompt_embeds' in encoded_data:
            negative_prompt_embeds = encoded_data['negative_prompt_embeds']
            # Concatenate for CFG as done in the pipeline
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prepared_inputs['prompt_embeds'] = prompt_embeds
    
    # Prepare reference latents with CFG
    if 'ref_latents' in encoded_data:
        ref_latents = encoded_data['ref_latents']
        if do_classifier_free_guidance:
            ref_input = torch.cat([ref_latents] * 2)
        else:
            ref_input = ref_latents
        prepared_inputs['ref_input'] = ref_input
    
    # Prepare inpaint latents with CFG
    if 'mask_latents' in encoded_data and 'masked_video_latents' in encoded_data:
        mask_latents = encoded_data['mask_latents']
        masked_video_latents = encoded_data['masked_video_latents']
        
        if mask_latents is not None and masked_video_latents is not None:
            if do_classifier_free_guidance:
                mask_input = torch.cat([mask_latents] * 2)
                masked_video_latents_input = torch.cat([masked_video_latents] * 2)
            else:
                mask_input = mask_latents
                masked_video_latents_input = masked_video_latents
            
            # Channel concatenation for inpainting
            inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2)
            prepared_inputs['inpaint_latents'] = inpaint_latents
    
    # Load other encoded inputs
    for key in ['video_latents', 'mask']:
        if key in encoded_data and encoded_data[key] is not None:
            prepared_inputs[key] = encoded_data[key]
    
    return prepared_inputs



def load_trajcrafter():

    sys.argv = [
        "",
        "--video_path", "/home/azhuravl/nobackup/DAVIS_testing/trainval/vkitti2.mp4",
        "--n_splits", "4",
        "--overlap_frames", "0",
        "--radius", "0",
        "--mode", "gradual",
    ]

    parser = get_parser()
    opts_base = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    video_basename = os.path.splitext(os.path.basename(opts_base.video_path))[0]

    # Setup
    opts_base.weight_dtype = torch.bfloat16
    opts_base.exp_name = f"{video_basename}_{timestamp}_autoregressive"
    opts_base.save_dir = os.path.join(opts_base.out_dir, opts_base.exp_name)

    # Create TrajCrafterVisualization instance for autoregressive generation
    radius = opts_base.radius

    variants = [
        ("right_90", [0, 90, radius, 0, 0]),
    ]

    pose = [90, 0, 0, 0, 1]
    name = f"{pose[0]}_{pose[1]}_{pose[2]}_{pose[3]}_{pose[4]}"

    opts = copy.deepcopy(opts_base)
    opts.exp_name = f"{video_basename}_{timestamp}_{name}_auto_s{opts_base.n_splits}"
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    opts.camera = "target"
    opts.target_pose = pose
    opts.traj_txt = 'test/trajs/loop2.txt'

    # Make directories
    # os.makedirs(opts.save_dir, exist_ok=True)

    trajcrafter = utils_ar.TrajCrafterAutoregressive(opts)
    
    return trajcrafter, opts






if __name__ == "__main__":
    
    dataset_driving = video_datasets_custom.SequenceSceneFlowDatasetCamera(
        aug_params=None,
        root="/home/azhuravl/scratch/SceneFlow",
        dstype="frames_cleanpass",
        sample_len=59,
        things_test=False,
        add_things=False,
        add_monkaa=False,
        add_driving=True,
        split="test",
        stride=5,
    )
    # print(len(dataset_driving))
    print(f"Dataset length: {len(dataset_driving)}")
    
    warper_old = utils.Warper(device='cuda')
    
    trajcrafter, opts = load_trajcrafter()


    # Initialize motion filter
    filter_motion = collect_dataset.CameraMotionFilter(
        device='cuda',
        min_total_translation=5,      # More permissive
        max_total_translation=120,    # More permissive
        min_total_rotation=0.05,      # More permissive
        max_total_rotation=0.7        # More permissive (~40 degrees)
    )
    
    # Track processed samples and current index
    samples_processed = 0
    current_idx = 0
    target_samples = 1000
    
    dataset_name = 'driving_1000'

    while samples_processed < target_samples and current_idx < len(dataset_driving):
        
        print(f"Processing sample index: {current_idx}")
        print(f"Samples processed: {samples_processed}/{target_samples}")

        data_0 = dataset_driving[current_idx]        

        frames_tensor, depths, poses_tensor, K_tensor = collect_dataset.extract_video_data(
            data_0,
            )
        
        # Check motion criteria
        low_motion, metrics = filter_motion.is_low_motion(poses_tensor)
        print(f"Low motion: {low_motion}, Metrics: {metrics}")
        
        if not low_motion:
            print("Skipping due to high motion.")
            current_idx += 1
            continue

        # warping
        warped_images = []
        masks = []
        warped_depths = []

        warped_images_once = []

        for i in tqdm(range(10, frames_tensor.shape[0])):
            
            transformation_1 = poses_tensor[i:i+1].clone()
            transformation_2 = poses_tensor[10:11].clone()
            
            warped_frame2, mask2, warped_depth2, flow12 = warper_old.forward_warp(
                frame1=frames_tensor[i:i+1],
                mask1=None,
                depth1=depths[i:i+1],
                transformation1=transformation_1,
                transformation2=transformation_2,
                intrinsic1=K_tensor[0].unsqueeze(0),
                intrinsic2=K_tensor[0].unsqueeze(0),
                mask=False,
                twice=True,
            )
            # depth returned is incorrect, multiply input depth by mask
            warped_depth2 = depths[i:i+1] * mask2

            warped_frame_once, _, _, _ = warper_old.forward_warp(
                frame1=frames_tensor[i:i+1],
                mask1=None,
                depth1=depths[i:i+1],
                transformation1=transformation_1,
                transformation2=transformation_2,
                intrinsic1=K_tensor[0].unsqueeze(0),
                intrinsic2=K_tensor[0].unsqueeze(0),
                mask=False,
                twice=False,
            )
            
            warped_images.append(warped_frame2)
            masks.append(mask2)
            warped_depths.append(warped_depth2)
            
            warped_images_once.append(warped_frame_once)    
            
            
        #####################################################    
        # SAVE VIDEOS AND DEPTHS
        #####################################################

        # Create save directory for this sample
        save_dir = f'/home/azhuravl/scratch/datasets_latents/{dataset_name}/{samples_processed:03d}'
        os.makedirs(f'{save_dir}/videos', exist_ok=True)


        # for saving: [T, H, W, C] in [0,1]
        # for diffusion: [C, T, H, W] in [0,1]


        # input and ref videos
        frames_tensor_resized = (frames_tensor + 1.0) / 2.0  # [T, 3, H, W] in [0,1]
        frames_tensor_resized = smart_video_resize(frames_tensor_resized)
        save_video(
            frames_tensor_resized[10:].permute(0, 2, 3, 1),
            f'{save_dir}/videos/input_video.mp4',
            fps=10,
        )
        save_video(
            frames_tensor_resized[:10].permute(0, 2, 3, 1),
            f'{save_dir}/videos/ref_video.mp4',
            fps=10,
        )

        # masks
        masks_tensor = torch.cat(masks)
        masks_tensor_resized = smart_video_resize(masks_tensor, interpolation_mode='nearest')
        save_video(
            masks_tensor_resized.permute(0, 2, 3, 1).repeat(1, 1, 1, 3),
            f'{save_dir}/videos/masks.mp4',
            fps=10,
        )

        cond_video_twice_resized = frames_tensor_resized[10:] * masks_tensor_resized
        save_video(
            cond_video_twice_resized.permute(0, 2, 3, 1),
            f'{save_dir}/videos/warped_video_twice.mp4',
            fps=10,
        )

        # once warped
        cond_video_once = (torch.cat(warped_images_once) + 1.0) / 2.0  # [T, 3, H, W] in [0,1]
        cond_video_once_resized = smart_video_resize(
            cond_video_once,
            interpolation_mode='nearest'
        )
        save_video(
            cond_video_once_resized.permute(0, 2, 3, 1),
            f'{save_dir}/videos/warped_video_once.mp4',
            fps=10,
        )


        ### Save depths

        # input and ref depths
        depths_resized = smart_video_resize(depths)
        depths_colored = apply_colormap_to_depth(depths_resized.squeeze(1), inverse=True)
        save_video(
            depths_colored[10:],
            f'{save_dir}/videos/input_depths.mp4',
            fps=10,
        )
        save_video(
            depths_colored[:10],
            f'{save_dir}/videos/ref_depths.mp4',
            fps=10,
        )


        # warped depths
        # warped_depths_tensor = torch.cat(warped_depths)
        # warped_depths_tensor_resized = smart_video_resize(
        #     warped_depths_tensor,
        #     interpolation_mode='nearest'
        # )
        warped_depths_tensor_resized = depths_resized[10:] * masks_tensor_resized
        warped_depths_colored = apply_colormap_to_depth(warped_depths_tensor_resized.squeeze(1), inverse=True)
        save_video(
            warped_depths_colored,
            f'{save_dir}/videos/warped_depths.mp4',
            fps=10,
        )
        
        
        #########################################################
        # SAVE CAMERA PARAMETERS
        #########################################################
        
        
        # extrinsics
        with open(f"{save_dir}/videos/extrinsics.json", "w") as f:
            json.dump(poses_tensor.cpu().tolist(), f, indent=2)

        # intrinsics
        with open(f"{save_dir}/videos/intrinsics.json", "w") as f:
            json.dump(K_tensor[0].cpu().tolist(), f, indent=2)


        #####################################################
        # GENERATE CAPTION
        #####################################################

        frames_np = ((frames_tensor.cpu().permute(0, 2, 3, 1).numpy() + 1.0) / 2.0).astype(np.float32)

        caption = trajcrafter.get_caption(opts, frames_np[opts.video_length // 2])
        with open(f"{save_dir}/videos/caption.txt", "w") as f:
            f.write(caption)
        print("Caption:", caption)
        
        
        #####################################################
        # Save video latents
        #####################################################
        
        mask_video = (1.0 - masks_tensor_resized.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0


        latents_dict = encode_inputs_to_latents(
            trajcrafter.pipeline,
            video=cond_video_twice_resized.permute(1, 0, 2, 3).unsqueeze(0).to('cuda'),
            reference=frames_tensor_resized[:10].permute(1, 0, 2, 3).unsqueeze(0).to('cuda'),
            mask_video=mask_video.to('cuda'),
            masked_video_latents=None,
            prompt=caption,
            negative_prompt=opts.negative_prompt,
            height=384,
            width=672,
            device="cuda",
            batch_size=1,
            noise_aug_strength=0.0563,
            max_sequence_length=226,
            do_classifier_free_guidance=True,
            ground_truth_video=frames_tensor_resized[10:].permute(1, 0, 2, 3).unsqueeze(0).to('cuda'),  # GT video for training
            encode_for_training=True  # Flag to indicate training vs inference
        )
        
        
        torch.save(
            latents_dict,
            f'{save_dir}/video_latents.pt'
        )
        
        
        #####################################################
        # Save depth latents
        #####################################################
        
        
        depths_min = depths_resized.min()
        depths_max = depths_resized.max()

        depths_resized_norm = (depths_resized - depths_min) / (depths_max - depths_min + 1e-8)
        warped_depths_tensor_resized_norm = depths_resized_norm[10:] * masks_tensor_resized
        
        # save resized and normalized depths + warped depths
        torch.save(
            depths_resized_norm[10:],
            f'{save_dir}/videos/input_depths.pt'
        )
        torch.save(
            depths_resized_norm[:10],
            f'{save_dir}/videos/ref_depths.pt'
        )
        torch.save(
            warped_depths_tensor_resized_norm,
            f'{save_dir}/videos/warped_depths.pt'
        )

        # permute, repeat, unsqueeze, to cuda


        latents_dict_depths = encode_inputs_to_latents(
            trajcrafter.pipeline,
            video=warped_depths_tensor_resized_norm.permute(1, 0, 2, 3).repeat(3, 1, 1, 1).unsqueeze(0).to('cuda'),
            reference=depths_resized_norm[:10].permute(1, 0, 2, 3).repeat(3, 1, 1, 1).unsqueeze(0).to('cuda'),
            mask_video=mask_video.to('cuda'),
            masked_video_latents=None,
            prompt=caption,
            negative_prompt=opts.negative_prompt,
            height=384,
            width=672,
            device="cuda",
            batch_size=1,
            noise_aug_strength=0.0563,
            max_sequence_length=226,
            do_classifier_free_guidance=True,
            ground_truth_video=depths_resized_norm[10:].permute(1, 0, 2, 3).repeat(3, 1, 1, 1).unsqueeze(0).to('cuda'),  # GT video for training
            encode_for_training=True  # Flag to indicate training vs inference
        )
        
        torch.save(
            latents_dict_depths,
            f'{save_dir}/depth_latents.pt'
        )
        
        samples_processed += 1
        current_idx += 1