import os
import pickle
import shutil
import torch
from packaging import version
import accelerate
from accelerate.logging import get_logger

import sys
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/lora_utils')
from utils import resize_mask

logger = get_logger(__name__, log_level="INFO")

def setup_checkpoint_hooks(args, accelerator, network):
    """Setup checkpoint saving and loading hooks"""
    batch_sampler = None
    first_epoch = 0
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                save_model(safetensor_save_path, accelerator.unwrap_model(models[-1]))
                if not args.use_deepspeed:
                    for _ in range(len(weights)):
                        weights.pop()

                with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                    pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

        def load_model_hook(models, input_dir):
            pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    loaded_number, _ = pickle.load(file)
                    batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    return batch_sampler, first_epoch

def save_model(ckpt_file, unwrapped_nw):
    """Save model function - placeholder that needs to be defined in main"""
    pass

def load_from_checkpoint(args, accelerator, network, optimizer, lr_scheduler, batch_sampler, num_update_steps_per_epoch):
    """Load from checkpoint if specified"""
    global_step = 0
    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])
            initial_global_step = global_step

            checkpoint_folder_path = os.path.join(args.output_dir, path)
            pkl_path = os.path.join(checkpoint_folder_path, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(checkpoint_folder_path, "lora_diffusion_pytorch_model.safetensors"), device=str(accelerator.device))
            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

            optimizer_file_pt = os.path.join(checkpoint_folder_path, "optimizer.pt")
            optimizer_file_bin = os.path.join(checkpoint_folder_path, "optimizer.bin")
            optimizer_file_to_load = None

            if os.path.exists(optimizer_file_pt):
                optimizer_file_to_load = optimizer_file_pt
            elif os.path.exists(optimizer_file_bin):
                optimizer_file_to_load = optimizer_file_bin

            if optimizer_file_to_load:
                try:
                    accelerator.print(f"Loading optimizer state from {optimizer_file_to_load}")
                    optimizer_state = torch.load(optimizer_file_to_load, map_location=accelerator.device)
                    optimizer.load_state_dict(optimizer_state)
                    accelerator.print("Optimizer state loaded successfully.")
                except Exception as e:
                    accelerator.print(f"Failed to load optimizer state from {optimizer_file_to_load}: {e}")

            scheduler_file_pt = os.path.join(checkpoint_folder_path, "scheduler.pt")
            scheduler_file_bin = os.path.join(checkpoint_folder_path, "scheduler.bin")
            scheduler_file_to_load = None

            if os.path.exists(scheduler_file_pt):
                scheduler_file_to_load = scheduler_file_pt
            elif os.path.exists(scheduler_file_bin):
                scheduler_file_to_load = scheduler_file_bin

            if scheduler_file_to_load:
                try:
                    accelerator.print(f"Loading scheduler state from {scheduler_file_to_load}")
                    scheduler_state = torch.load(scheduler_file_to_load, map_location=accelerator.device)
                    lr_scheduler.load_state_dict(scheduler_state)
                    accelerator.print("Scheduler state loaded successfully.")
                except Exception as e:
                    accelerator.print(f"Failed to load scheduler state from {scheduler_file_to_load}: {e}")

            if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                scaler_file = os.path.join(checkpoint_folder_path, "scaler.pt")
                if os.path.exists(scaler_file):
                    try:
                        accelerator.print(f"Loading GradScaler state from {scaler_file}")
                        scaler_state = torch.load(scaler_file, map_location=accelerator.device)
                        accelerator.scaler.load_state_dict(scaler_state)
                        accelerator.print("GradScaler state loaded successfully.")
                    except Exception as e:
                        accelerator.print(f"Failed to load GradScaler state: {e}")
            else:
                accelerator.load_state(checkpoint_folder_path)
                accelerator.print("accelerator.load_state() completed for zero_stage 3.")

    else:
        initial_global_step = 0

    return global_step, first_epoch

def handle_checkpointing(args, accelerator, global_step, save_model_fn, network):
    """Handle checkpointing during training"""
    if args.use_deepspeed or accelerator.is_main_process:
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        if not args.save_state:
            safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
            save_model_fn(safetensor_save_path, accelerator.unwrap_model(network))
            logger.info(f"Saved safetensor to {safetensor_save_path}")
        else:
            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(accelerator_save_path)
            logger.info(f"Saved state to {accelerator_save_path}")
