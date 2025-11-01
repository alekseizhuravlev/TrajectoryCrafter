import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange
from videox_fun.utils.utils import save_videos_grid
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.pipeline.pipeline_cogvideox_fun_inpaint import add_noise_to_reference_video

import sys
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/lora_utils')

from training_utils import (
    resize_mask, prepare_rotary_positional_embeddings,
    apply_frame_cropping, apply_token_length_adjustment,
    process_inpaint_mode, batch_encode_vae
)

def run_training_loop(
    args, accelerator, train_dataloader, network, optimizer, lr_scheduler,
    vae, text_encoder, tokenizer, transformer3d, noise_scheduler,
    weight_dtype, global_step, first_epoch, batch_sampler, 
    save_model_fn, log_validation_fn, rng, torch_rng, index_rng
):
    """Main training loop"""
    
    # Setup progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Setup CUDA streams for inpainting if needed
    if args.multi_stream and args.train_mode != "normal":
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    # Setup discrete sampling
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # Main training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                _perform_sanity_check(args, batch, global_step)

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)

                # Apply batch size tiling if needed
                pixel_values, batch = _apply_batch_tiling(args, pixel_values, batch)

                # Handle inpaint mode
                mask_pixel_values, mask, t2v_flag = None, None, None
                if args.train_mode != "normal":
                    mask_pixel_values, mask, t2v_flag = process_inpaint_mode(
                        args, batch, weight_dtype, accelerator
                    )

                # Apply frame cropping if enabled
                if args.random_frame_crop:
                    pixel_values, mask_pixel_values, mask = apply_frame_cropping(
                        args, pixel_values, mask_pixel_values, mask, 
                        vae.config.temporal_compression_ratio, accelerator, 
                        transformer3d, rng
                    )

                # Apply token length adjustment if enabled
                if args.keep_all_node_same_token_length:
                    pixel_values, mask_pixel_values, mask = apply_token_length_adjustment(
                        args, pixel_values, mask_pixel_values, mask,
                        vae.config.temporal_compression_ratio, accelerator,
                        transformer3d, index_rng
                    )

                # Handle low VRAM mode
                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                # Encode to latents
                with torch.no_grad():
                    latents, inpaint_latents = _encode_to_latents(
                        args, pixel_values, mask_pixel_values, mask, t2v_flag,
                        vae, transformer3d, accelerator, weight_dtype,
                        vae_stream_1, vae_stream_2
                    )

                # Handle low VRAM cleanup
                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                # Get text embeddings
                prompt_embeds = _get_text_embeddings(
                    args, batch, tokenizer, text_encoder, latents
                )

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                # Prepare for training step
                bsz = latents.shape[0]
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)
                timesteps = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                timesteps = timesteps.long()

                # Prepare rotary embeddings
                height, width = batch["pixel_values"].size()[-2], batch["pixel_values"].size()[-1]
                image_rotary_emb = prepare_rotary_positional_embeddings(
                    height, width, latents.size(1), latents.device,
                    vae, transformer3d, accelerator
                )

                prompt_embeds = prompt_embeds.to(device=latents.device)

                # Add noise and get target
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Forward pass
                noise_pred = transformer3d(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    inpaint_latents=inpaint_latents if args.train_mode != "normal" else None,
                )[0]
                
                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # Apply motion sub-loss if enabled
                if args.motion_sub_loss and noise_pred.size()[1] > 2:
                    gt_sub_noise = noise_pred[:, 1:, :].float() - noise_pred[:, :-1, :].float()
                    pre_sub_noise = target[:, 1:, :].float() - target[:, :-1, :].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather losses and backpropagate
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                

            # Handle synchronization and checkpointing
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    _handle_checkpointing(args, accelerator, global_step, save_model_fn, network)

                if accelerator.is_main_process:
                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        log_validation_fn(
                            vae, text_encoder, tokenizer, transformer3d, network,
                            args, accelerator, weight_dtype, global_step
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # progress_bar.set_postfix(**logs)
            
            progress_bar.set_postfix(
                step_loss=f"{loss.detach().item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.4f}"
            )

            if global_step >= args.max_train_steps:
                break

        # Validation at epoch end
        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                log_validation_fn(
                    vae, text_encoder, tokenizer, transformer3d, network,
                    args, accelerator, weight_dtype, global_step
                )

    return global_step

def _perform_sanity_check(args, batch, global_step):
    """Perform sanity check on first batch"""
    pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
        pixel_value = pixel_value[None, ...]
        gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
        save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
    
    if args.train_mode != "normal":
        mask_pixel_values, texts = batch['mask_pixel_values'].cpu(), batch['text']
        mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
        for idx, (pixel_value, text) in enumerate(zip(mask_pixel_values, texts)):
            pixel_value = pixel_value[None, ...]
            save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.gif", rescale=True)

def _apply_batch_tiling(args, pixel_values, batch):
    """Apply batch size tiling when needed"""
    if args.auto_tile_batch_size and args.training_with_video_token_length:
        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
            pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
            if args.enable_text_encoder_in_dataloader:
                batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
            else:
                batch['text'] = batch['text'] * 4
        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
            pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
            if args.enable_text_encoder_in_dataloader:
                batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
            else:
                batch['text'] = batch['text'] * 2
    return pixel_values, batch

def _encode_to_latents(args, pixel_values, mask_pixel_values, mask, t2v_flag, vae, transformer3d, accelerator, weight_dtype, vae_stream_1, vae_stream_2):
    """Encode pixel values to latents"""
    if vae_stream_1 is not None:
        vae_stream_1.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(vae_stream_1):
            latents = batch_encode_vae(pixel_values, vae, args.vae_mini_batch)
    else:
        latents = batch_encode_vae(pixel_values, vae, args.vae_mini_batch)
    latents = latents * vae.config.scaling_factor

    inpaint_latents = None
    if args.train_mode != "normal":
        mask = rearrange(mask, "b f c h w -> b c f h w")
        mask = 1 - mask
        mask = resize_mask(mask, latents)

        unwrap_model = accelerator.unwrap_model
        if unwrap_model(transformer3d).config.add_noise_in_inpaint_model:
            mask_pixel_values = add_noise_to_reference_video(mask_pixel_values)
        
        mask_latents = batch_encode_vae(mask_pixel_values, vae, args.vae_mini_batch)
        if vae_stream_2 is not None:
            torch.cuda.current_stream().wait_stream(vae_stream_2) 

        inpaint_latents = torch.concat([mask, mask_latents], dim=1)
        inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents
        inpaint_latents = inpaint_latents * vae.config.scaling_factor
        inpaint_latents = rearrange(inpaint_latents, "b c f h w -> b f c h w")

    latents = rearrange(latents, "b c f h w -> b f c h w")
    
    # Wait for latents encoding to complete
    if vae_stream_1 is not None:
        torch.cuda.current_stream().wait_stream(vae_stream_1)

    return latents, inpaint_latents

def _get_text_embeddings(args, batch, tokenizer, text_encoder, latents):
    """Get text embeddings"""
    if args.enable_text_encoder_in_dataloader:
        prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
    else:
        with torch.no_grad():
            prompt_ids = tokenizer(
                batch['text'], 
                max_length=args.tokenizer_max_length, 
                padding="max_length", 
                add_special_tokens=True, 
                truncation=True, 
                return_tensors="pt"
            )
            prompt_embeds = text_encoder(
                prompt_ids.input_ids.to(latents.device),
                return_dict=False
            )[0]
    return prompt_embeds

def _handle_checkpointing(args, accelerator, global_step, save_model_fn, network):
    """Handle checkpointing logic"""
    if args.use_deepspeed or accelerator.is_main_process:
        # Handle checkpoint limit
        if args.checkpoints_total_limit is not None:
            import shutil
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        if not args.save_state:
            safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
            save_model_fn(safetensor_save_path, accelerator.unwrap_model(network))
        else:
            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(accelerator_save_path)