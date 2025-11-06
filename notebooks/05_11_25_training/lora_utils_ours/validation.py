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

import sys
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/22_10_25_scaling_up')
from generate_sceneflow import apply_colormap_to_depth

logger = get_logger(__name__, log_level="INFO")


def convert_depth_sample_to_rgb(depth_tensor):
    # sample (1, C, T, H, W)
    # input [T, H, W]
    # output [T, H, W, 3]
    
    depth_thw = depth_tensor.mean(dim=1, keepdim=False)[0]  # [T, H, W]
    
    # unnormalize depth to min 1, max 100, 
    # set values where depth_thw is zero to zero
    depth_thw_unnorm = depth_thw * (100.0 - 1.0) + 1.0
    depth_thw_unnorm = torch.where(depth_thw > 0, depth_thw_unnorm, torch.zeros_like(depth_thw_unnorm))
    
    depth_colormap = apply_colormap_to_depth(depth_thw_unnorm, inverse=True)  # [T, H, W, 3]
    depth_bcthw = depth_colormap.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
    return depth_bcthw
    


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
                    
                    # Move tensors to CPU before saving
                    sample_cpu = sample.cpu()
                    input_video_cpu = input_video.cpu()
                    warped_video_cpu = warped_video.cpu()
                    masks_cpu = masks.cpu()
                    
                    if args.use_depth:
                        # Apply colormap to depth videos for better visualization
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