# TODO
# training fully in latent space
# dataset for latents
# make cropped checkpoints for crosstransformer
# load these checkpoints, add lora to last 8 layers only

# TODO: check which latents should be used for input

import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange
from videox_fun.utils.utils import save_videos_grid
from videox_fun.utils.discrete_sampler import DiscreteSampling

import sys
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/lora_utils_ours')

from training_utils import (
    prepare_rotary_positional_embeddings
)
from checkpoint_utils import handle_checkpointing


def run_trajectorycrafter_training_loop(
    args, accelerator, train_dataloader, network, optimizer, lr_scheduler,
    transformer3d, noise_scheduler,
    weight_dtype, global_step, first_epoch, batch_sampler, 
    save_model_fn, log_validation_fn, rng, torch_rng, index_rng
):
    """Main training loop for TrajectoryCrafter with pre-encoded latents and CFG support"""
    
    # Setup progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Setup discrete sampling
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # CFG dropout probabilities
    text_dropout_prob = getattr(args, 'text_dropout_prob', 0.1)
    reference_dropout_prob = getattr(args, 'reference_dropout_prob', 0.1)
    inpaint_dropout_prob = getattr(args, 'inpaint_dropout_prob', 0.1)

    # Main training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                _perform_sanity_check_trajectorycrafter_latents(args, batch, global_step)

            with accelerator.accumulate(transformer3d):
                # Get all pre-encoded latents from dataset
                target_latents = batch["gt_video_latents"].to(weight_dtype)
                reference_latents = batch["ref_latents"].to(weight_dtype)
                
                # Handle inpaint latents - concatenate mask and masked video if separate
                if "mask_latents" in batch and "masked_video_latents" in batch:
                    # Concatenate mask and masked video latents for inpainting
                    mask_latents = batch["mask_latents"].to(weight_dtype)
                    masked_video_latents = batch["masked_video_latents"].to(weight_dtype)
                    inpaint_latents = torch.cat([mask_latents, masked_video_latents], dim=2)  # Concat on channel dim
                elif "cond_video_latents" in batch:
                    inpaint_latents = batch["cond_video_latents"].to(weight_dtype)
                else:
                    inpaint_latents = None
                
                text_embeddings = batch["prompt_embeds"].to(weight_dtype)

                # Apply classifier-free guidance dropout during training
                bsz = target_latents.shape[0]
                
                # Text dropout
                if text_dropout_prob > 0:
                    text_mask = torch.rand(bsz, device=text_embeddings.device) > text_dropout_prob
                    # Create unconditional text embeddings (zeros or special null embedding)
                    uncond_text_embeds = torch.zeros_like(text_embeddings)
                    text_embeddings = torch.where(
                        text_mask.unsqueeze(1).unsqueeze(2), 
                        text_embeddings, 
                        uncond_text_embeds
                    )
                
                # Reference video dropout
                if reference_dropout_prob > 0:
                    ref_mask = torch.rand(bsz, device=reference_latents.device) > reference_dropout_prob
                    uncond_ref_latents = torch.zeros_like(reference_latents)
                    reference_latents = torch.where(
                        ref_mask.view(bsz, 1, 1, 1, 1), 
                        reference_latents, 
                        uncond_ref_latents
                    )
                
                # Inpaint latents dropout
                if inpaint_latents is not None and inpaint_dropout_prob > 0:
                    inpaint_mask = torch.rand(bsz, device=inpaint_latents.device) > inpaint_dropout_prob
                    uncond_inpaint_latents = torch.zeros_like(inpaint_latents)
                    inpaint_latents = torch.where(
                        inpaint_mask.view(bsz, 1, 1, 1, 1), 
                        inpaint_latents, 
                        uncond_inpaint_latents
                    )

                # Prepare for training step
                noise = torch.randn(target_latents.size(), device=target_latents.device, generator=torch_rng, dtype=weight_dtype)
                timesteps = idx_sampling(bsz, generator=torch_rng, device=target_latents.device)
                timesteps = timesteps.long()

                # Prepare rotary embeddings from latent dimensions
                latent_height, latent_width = target_latents.shape[-2], target_latents.shape[-1]
                frames = target_latents.shape[1]
                
                image_rotary_emb = prepare_rotary_positional_embeddings(
                    latent_height * 8, latent_width * 8, frames, target_latents.device,
                    None, transformer3d, accelerator
                )

                # Add noise to target and get prediction target
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # TrajectoryCrafter forward pass
                model_kwargs = {
                    # == warped (conditioning) video latents
                    "hidden_states": noisy_latents,
                    "encoder_hidden_states": text_embeddings,
                    "timestep": timesteps,
                    "image_rotary_emb": image_rotary_emb,
                    "return_dict": False,
                    # == ref video latents
                    "cross_latents": reference_latents,
                }
                
                # Add inpaint latents if available
                # == mask latents
                if inpaint_latents is not None:
                    model_kwargs["inpaint_latents"] = inpaint_latents
                
                noise_pred = transformer3d(**model_kwargs)[0]
                
                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # Apply motion sub-loss if enabled
                if hasattr(args, 'motion_sub_loss') and args.motion_sub_loss and noise_pred.size()[1] > 2:
                    gt_sub_noise = noise_pred[:, 1:, :].float() - noise_pred[:, :-1, :].float()
                    pre_sub_noise = target[:, 1:, :].float() - target[:, :-1, :].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    motion_sub_loss_ratio = getattr(args, 'motion_sub_loss_ratio', 0.25)
                    loss = loss * (1 - motion_sub_loss_ratio) + sub_loss * motion_sub_loss_ratio

                # Debug prints for first few steps
                if step < 3:
                    print(f"DEBUG Step {step}:")
                    print(f"  Target latents: {target_latents.shape}, mean: {target_latents.mean():.6f}")
                    print(f"  Reference latents: {reference_latents.shape}, mean: {reference_latents.mean():.6f}")
                    if inpaint_latents is not None:
                        print(f"  Inpaint latents: {inpaint_latents.shape}, mean: {inpaint_latents.mean():.6f}")
                    print(f"  Text embeddings: {text_embeddings.shape}, mean: {text_embeddings.mean():.6f}")
                    print(f"  Noise pred: mean={noise_pred.mean():.6f}, std={noise_pred.std():.6f}")
                    print(f"  Target: mean={target.mean():.6f}, std={target.std():.6f}")
                    print(f"  Loss: {loss.item():.6f}")

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
                    handle_checkpointing(args, accelerator, global_step, save_model_fn, network)

                if accelerator.is_main_process:
                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        log_validation_fn(
                            transformer3d, network, args, accelerator, weight_dtype, global_step
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # Validation at epoch end
        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                log_validation_fn(
                    transformer3d, network, args, accelerator, weight_dtype, global_step
                )

    return global_step


def _perform_sanity_check_trajectorycrafter_latents(args, batch, global_step):
    """Perform sanity check on first batch - latent space version"""
    print(f"Sanity check - Batch keys: {list(batch.keys())}")
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, mean: {value.mean():.6f}, std: {value.std():.6f}")
        else:
            print(f"  {key}: {type(value)}")
    