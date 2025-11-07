# validation.py (updated)
import gc
import os
import torch
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from diffusers import DDIMScheduler
from models.crosstransformer3d import CrossTransformer3DModel
from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.utils import save_videos_grid
from transformers import T5EncoderModel, T5Tokenizer
from dataset_videos import SimpleValidationDataset
import json

import sys
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/22_10_25_scaling_up')
from generate_sceneflow import apply_colormap_to_depth
import numpy as np

logger = get_logger(__name__, log_level="INFO")

# metrics we need for depth
# relative depth error for all, inpainted, and non-inpainted regions
# for rgb, psnr, ssim, lpips


def convert_depth_single_channel(tensor_bcthw):
    return tensor_bcthw.mean(dim=1, keepdim=False)[0]  # [T, H, W]


def unnormalize_depth(depth_tensor, depth_min=1.0, depth_max=100.0):
    depth_unnorm = depth_tensor * (depth_max - depth_min) + depth_min
    depth_unnorm = torch.where(depth_tensor > 0, depth_unnorm, torch.zeros_like(depth_unnorm))
    return depth_unnorm


def calculate_relative_error(pred, gt, mask):
    """Calculate relative error: |pred - gt| / gt"""
    if mask.sum() == 0:
        return float('nan')
    
    
    # print(pred.shape, gt.shape, mask.shape)
    
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    
    # Avoid division by zero
    valid_gt_mask = gt_masked > 1e-6
    if valid_gt_mask.sum() == 0:
        return float('nan')
        
    pred_valid = pred_masked[valid_gt_mask]
    gt_valid = gt_masked[valid_gt_mask]
    
    relative_error = torch.abs(pred_valid - gt_valid) / gt_valid
    return relative_error.mean().item()


def convert_depth_sample_to_rgb(depth_tensor):
    # sample (1, C, T, H, W)
    # input [T, H, W]
    # output [T, H, W, 3]
    
    # depth_thw = depth_tensor.mean(dim=1, keepdim=False)[0]  # [T, H, W]
    depth_thw = convert_depth_single_channel(depth_tensor)  # [T, H, W]
    
    # unnormalize depth to min 1, max 100, 
    # set values where depth_thw is zero to zero
    # depth_thw_unnorm = depth_thw * (100.0 - 1.0) + 1.0
    # depth_thw_unnorm = torch.where(depth_thw > 0, depth_thw_unnorm, torch.zeros_like(depth_thw_unnorm))
    
    depth_thw_unnorm = unnormalize_depth(depth_thw, depth_min=1.0, depth_max=100.0)
    
    depth_colormap = apply_colormap_to_depth(
        depth_thw_unnorm,
        inverse=True,
        # vmin=1.0,
        # vmax=100.0
    )  # [T, H, W, 3]
    depth_bcthw = depth_colormap.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
    return depth_bcthw
    

def calculate_depth_errors(sample, gt, warped_video, masks):
    """
    Calculate relative depth errors between predicted and ground truth depth maps.
    
    Args:
        sample: Generated depth tensor [1, C, T, H, W] - prediction 
        gt: Ground truth depth tensor [1, C, T, H, W] - input_video
        warped_video: Warped conditioning tensor [1, C, T, H, W] 
        masks: Mask tensor [1, C, T, H, W] where >0 indicates inpainted regions
        
    Returns:
        dict: Dictionary containing error metrics
    """
    
    # Convert all tensors to depth format
    pred_depth = convert_depth_single_channel(sample)  # [T, H, W]
    gt_depth = convert_depth_single_channel(gt)          # [T, H, W]
    warped_depth = convert_depth_single_channel(warped_video)  # [T, H, W]
    masks = masks[0, 0]  # [T, H, W]

    # Unnormalize depths to actual values (assuming they were normalized to [0,1])
    pred_depth_real = unnormalize_depth(pred_depth, depth_min=1.0, depth_max=100.0)
    gt_depth_real = unnormalize_depth(gt_depth, depth_min=1.0, depth_max=100.0)
    warped_depth_real = unnormalize_depth(warped_depth, depth_min=1.0, depth_max=100.0)


    # Create mask for inpainted regions (where mask > threshold)
    # Assuming masks are in [0, 255] range
    mask_threshold = 127.5  # Adjust based on your mask encoding
    inpainted_mask = masks > mask_threshold
    non_inpainted_mask = ~inpainted_mask


    # Calculate errors for all three cases
    errors = {
        'overall_rel_error': calculate_relative_error(pred_depth_real, gt_depth_real, torch.ones_like(gt_depth_real, dtype=torch.bool)),
        'inpainted_rel_error': calculate_relative_error(pred_depth_real, gt_depth_real, inpainted_mask),
        'non_inpainted_rel_error': calculate_relative_error(pred_depth_real, gt_depth_real, non_inpainted_mask),

        # Pixel counts for context
        'total_valid_pixels': torch.ones_like(gt_depth_real).sum().item(),
        'inpainted_pixels': inpainted_mask.sum().item(), 
        'non_inpainted_pixels': non_inpainted_mask.sum().item(),
    }
    
    return errors



def log_validation(vae, text_encoder, tokenizer, transformer3d, network, args, accelerator, weight_dtype, global_step, val_dataloader=None):
    try:
        logger.info("Running validation... ")

        # Use the passed validation dataloader or skip validation
        if val_dataloader is None:
            logger.warning("No validation dataloader provided, skipping validation")
            return

        logger.info(f"Using validation dataset with {len(val_dataloader.dataset)} samples")


        # Load the pruned CrossTransformer3DModel
        # transformer3d_val = CrossTransformer3DModel.from_pretrained(
        #     '/home/azhuravl/scratch/checkpoints/TrajectoryCrafter'
        # ).to(weight_dtype)
        
        # Apply the same pruning as in training
        # num_layers_to_keep = 16
        # num_cross_layers_to_keep = 2
        # transformer3d_val.transformer_blocks = transformer3d_val.transformer_blocks[:num_layers_to_keep]
        # transformer3d_val.perceiver_cross_attention = transformer3d_val.perceiver_cross_attention[:num_cross_layers_to_keep]
        
        # Load the trained state
        # transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer3d).state_dict())
        
        transformer3d_val = accelerator.unwrap_model(transformer3d)
        
        # Use DDIMScheduler
        scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        
        # Create TrajCrafter_Pipeline
        pipeline = TrajCrafter_Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae).to(weight_dtype),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            transformer=transformer3d_val,
            scheduler=scheduler,
            torch_dtype=weight_dtype,
        )

        pipeline = pipeline.to(accelerator.device)
        
        # Merge LoRA weights
        # pipeline = merge_lora(
        #     pipeline, None, 1, accelerator.device, 
        #     state_dict=accelerator.unwrap_model(network).state_dict(), 
        #     transformer_only=True,
        #     sub_transformer_name="transformer"
        # )

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            
        # Run validation with real data
        validation_output_dir = os.path.join(args.output_dir, "validation_results")
        os.makedirs(validation_output_dir, exist_ok=True)
        
        
        error_accumulator = {
            'train': {
                'overall_rel_error': [],
                'inpainted_rel_error': [],
                'non_inpainted_rel_error': [],
            },
            'val': {
                'overall_rel_error': [],
                'inpainted_rel_error': [],
                'non_inpainted_rel_error': [],
            }
        }

        
        for i, batch in enumerate(val_dataloader):
            sample_name = batch['sample_name'][0]
            logger.info(f"Processing validation sample {i+1}/{len(val_dataloader)}: {sample_name}")
            
            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    caption = batch['caption'][0]  # Get string from batch

                    # Move data to device
                    ref_video = batch['ref_video'].to(accelerator.device)
                    warped_video = batch['warped_video'].to(accelerator.device) 
                    masks = batch['masks'].to(accelerator.device)
                    input_video = batch['input_video'].to(accelerator.device)
                    
                    # Get video dimensions
                    num_frames = warped_video.shape[2]
                    height, width = warped_video.shape[-2:]
                    
                    logger.info(f"  Caption: {caption}")
                    logger.info(f"  Video shape: {num_frames} frames, {height}x{width}")
                    
                    # Generate sample
                    sample = pipeline(
                        prompt=caption,
                        num_frames=num_frames,
                        negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                        height=height,
                        width=width,
                        guidance_scale=6.0,
                        generator=generator,
                        num_inference_steps=50,
                        # Real TrajectoryCrafter inputs from dataset
                        video=warped_video,
                        mask_video=masks,
                        reference=ref_video,
                    ).videos
                    
                    # sample = torch.randn_like(input_video)  # Placeholder for generated sample
                    
                    # Move tensors to CPU before saving
                    sample_cpu = sample.cpu()
                    input_video_cpu = input_video.cpu()
                    warped_video_cpu = warped_video.cpu()
                    masks_cpu = masks.cpu()
                    
                    if args.use_depth:
                        # Calculate depth errors
                        depth_errors = calculate_depth_errors(sample_cpu, input_video_cpu, warped_video_cpu, masks_cpu)
                        
                        # Accumulate errors (only non-NaN values)
                        
                        split = 'train' if i < len(val_dataloader.dataset) // 2 else 'val'
                        for key in ['overall_rel_error', 'inpainted_rel_error', 'non_inpainted_rel_error']:
                            
                            if key in depth_errors and not np.isnan(depth_errors[key]):
                                error_accumulator[split][key].append(depth_errors[key])
                       
                        
                        
                        # Log the errors
                        logger.info(f"  Depth Error Metrics for {sample_name}:")
                        logger.info(f"    Split: {split}")
                        logger.info(f"    Overall - Rel Error: {depth_errors['overall_rel_error']:.4f}")
                        logger.info(f"    Inpainted - Rel Error: {depth_errors['inpainted_rel_error']:.4f}")
                        logger.info(f"    Non-inpainted - Rel Error: {depth_errors['non_inpainted_rel_error']:.4f}")
                        logger.info(f"    Pixel counts - Total: {depth_errors['total_valid_pixels']}, Inpainted: {depth_errors['inpainted_pixels']}, Non-inpainted: {depth_errors['non_inpainted_pixels']}")
                        
                        # Save errors to file for later analysis
                        error_file = os.path.join(validation_output_dir, f"step_{global_step:06d}_sample_{i:02d}_{sample_name}_errors.json")
                        with open(error_file, 'w') as f:
                            json.dump(depth_errors, f, indent=2)
                            
                        # save depth sample as pt 
                        depth_sample_file = os.path.join(validation_output_dir, f"step_{global_step:06d}_sample_{i:02d}_{sample_name}_gen_depth.pt")
                        torch.save(sample_cpu, depth_sample_file)
                        
                        # Apply colormap for visualization
                        sample_cpu = convert_depth_sample_to_rgb(sample_cpu)
                        input_video_cpu = convert_depth_sample_to_rgb(input_video_cpu) 
                        warped_video_cpu = convert_depth_sample_to_rgb(warped_video_cpu)
                        
                    
                    # Save generated video as MP4
                    output_filename = f"step_{global_step:06d}_sample_{i:02d}_{sample_name}_gen.mp4"
                    save_videos_grid(
                        sample_cpu, 
                        os.path.join(validation_output_dir, output_filename)
                    )
                    
                    # Save input video for comparison as MP4
                    input_filename = f"step_{global_step:06d}_sample_{i:02d}_{sample_name}_input.mp4"
                    save_videos_grid(
                        input_video_cpu, 
                        os.path.join(validation_output_dir, input_filename)
                    )
                    
                    # Save warped conditioning video as MP4
                    warped_filename = f"step_{global_step:06d}_sample_{i:02d}_{sample_name}_warped.mp4"
                    save_videos_grid(
                        warped_video_cpu, 
                        os.path.join(validation_output_dir, warped_filename)
                    )
                    
                    # Save masks as MP4
                    mask_filename = f"step_{global_step:06d}_sample_{i:02d}_{sample_name}_masks.mp4"
                    # Convert masks to 3-channel for visualization
                    masks_vis = masks_cpu.repeat(1, 3, 1, 1, 1) / 255.0
                    save_videos_grid(
                        masks_vis, 
                        os.path.join(validation_output_dir, mask_filename)
                    )
                    
                    logger.info(f"  Saved validation outputs for {sample_name}")

        
        
        # save error metrics summary
        if args.use_depth:
            # Calculate mean errors for both splits
            mean_errors = {}
            
            for split in ['train', 'val']:
                for error_type in ['overall_rel_error', 'inpainted_rel_error', 'non_inpainted_rel_error']:
                    values = error_accumulator[split][error_type]
                    if values:  # Only if we have valid values
                        mean_errors[f'{split}_mean_{error_type}'] = np.mean(values)
                    else:
                        mean_errors[f'{split}_mean_{error_type}'] = float('nan')
            
            # Log mean errors for both splits
            logger.info("=" * 70)
            logger.info(f"DATASET-WIDE DEPTH ERROR STATISTICS (Step {global_step})")
            logger.info("=" * 70)
            
            for split in ['train', 'val']:
                logger.info(f"{split.upper()} SPLIT:")
                logger.info(f"  Mean Overall Rel Error: {mean_errors.get(f'{split}_mean_overall_rel_error', 'N/A'):.4f}")
                logger.info(f"  Mean Inpainted Rel Error: {mean_errors.get(f'{split}_mean_inpainted_rel_error', 'N/A'):.4f}")
                logger.info(f"  Mean Non-inpainted Rel Error: {mean_errors.get(f'{split}_mean_non_inpainted_rel_error', 'N/A'):.4f}")
                logger.info("")
            
            logger.info("=" * 70)
            
            # Save mean errors to file
            mean_errors_file = os.path.join(validation_output_dir, f"step_{global_step:06d}_mean_errors.json")
            with open(mean_errors_file, 'w') as f:
                json.dump(mean_errors, f, indent=2, default=str)  # default=str handles NaN values
            
            # Log to accelerator (tensorboard/wandb) if available
            if accelerator.is_main_process:
                log_dict = {}
                for split in ['train', 'val']:
                    for error_type in ['overall_rel_error', 'inpainted_rel_error', 'non_inpainted_rel_error']:
                        mean_key = f'{split}_mean_{error_type}'
                        if mean_key in mean_errors and not np.isnan(mean_errors[mean_key]):
                            log_dict[f"validation/{split}_{error_type}"] = mean_errors[mean_key]
                
                # Log if we have any valid metrics
                if log_dict and hasattr(accelerator, 'log'):
                    accelerator.log(log_dict, step=global_step)
                    
            logger.info(f"Mean validation statistics saved to {mean_errors_file}")

        logger.info(f"Validation complete! Results saved to {validation_output_dir}/")

        # Clean up
        del pipeline
        del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache() 
        torch.cuda.ipc_collect()
        logger.error(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        raise
        return None