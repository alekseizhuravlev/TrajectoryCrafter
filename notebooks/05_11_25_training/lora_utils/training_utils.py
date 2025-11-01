import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from videox_fun.pipeline.pipeline_cogvideox_fun_inpaint import (
    get_3d_rotary_pos_embed, get_resize_crop_region_for_grid
)

def resize_mask(mask, latent, process_first_frame_only=True):
    """Resize mask to match latent dimensions - extracted from original code"""

    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

def prepare_rotary_positional_embeddings(height, width, num_frames, device, vae, transformer3d, accelerator):
    """Prepare rotary positional embeddings"""
    unwrap_model = accelerator.unwrap_model
    
    vae_scale_factor_spatial = (
        2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
    )

    p = unwrap_model(transformer3d).config.patch_size
    p_t = unwrap_model(transformer3d).config.patch_size_t

    grid_height = height // (vae_scale_factor_spatial * p)
    grid_width = width // (vae_scale_factor_spatial * p)
    base_size_height = unwrap_model(transformer3d).config.sample_height // p
    base_size_width = unwrap_model(transformer3d).config.sample_width // p

    if p_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=unwrap_model(transformer3d).config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + p_t - 1) // p_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=unwrap_model(transformer3d).config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )
    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def apply_frame_cropping(args, pixel_values, mask_pixel_values, mask, sample_n_frames_bucket_interval, accelerator, transformer3d, rng):
    """Apply random frame cropping"""
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            last_element = 0.90
            remaining_sum = 1.0 - last_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [other_elements_value] * (length - 1) + [last_element]
            return special_list
    
    select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
    select_frames_prob = np.array(_create_special_list(len(select_frames)))

    if len(select_frames) != 0:
        if rng is None:
            temp_n_frames = np.random.choice(select_frames, p=select_frames_prob)
        else:
            temp_n_frames = rng.choice(select_frames, p=select_frames_prob)
    else:
        temp_n_frames = 1

    # Magvae needs the number of frames to be 4n + 1.
    local_latent_length = (temp_n_frames - 1) // sample_n_frames_bucket_interval + 1
    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = accelerator.unwrap_model(transformer3d).config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and local_latent_length % patch_size_t != 0:
        additional_frames = local_latent_length % patch_size_t
        temp_n_frames -= additional_frames * sample_n_frames_bucket_interval
    if temp_n_frames <= 0:
        temp_n_frames = 1

    pixel_values = pixel_values[:, :temp_n_frames, :, :]

    if args.train_mode != "normal":
        mask_pixel_values = mask_pixel_values[:, :temp_n_frames, :, :]
        mask = mask[:, :temp_n_frames, :, :]
    
    return pixel_values, mask_pixel_values, mask

def apply_token_length_adjustment(args, pixel_values, mask_pixel_values, mask, sample_n_frames_bucket_interval, accelerator, transformer3d, index_rng):
    """Apply token length adjustment"""
    if args.token_sample_size > 256:
        numbers_list = list(range(256, args.token_sample_size + 1, 128))
        if numbers_list[-1] != args.token_sample_size:
            numbers_list.append(args.token_sample_size)
    else:
        numbers_list = [256]
    numbers_list = [_number * _number * args.video_sample_n_frames for _number in numbers_list]

    actual_token_length = index_rng.choice(numbers_list)
    actual_video_length = (min(
            actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
    ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
    actual_video_length = int(max(actual_video_length, 1))

    # Magvae needs the number of frames to be 4n + 1.
    local_latent_length = (actual_video_length - 1) // sample_n_frames_bucket_interval + 1
    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = accelerator.unwrap_model(transformer3d).config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and local_latent_length % patch_size_t != 0:
        additional_frames = local_latent_length % patch_size_t
        actual_video_length -= additional_frames * sample_n_frames_bucket_interval
    if actual_video_length <= 0:
        actual_video_length = 1

    pixel_values = pixel_values[:, :actual_video_length, :, :]
    if args.train_mode != "normal":
        mask_pixel_values = mask_pixel_values[:, :actual_video_length, :, :]
        mask = mask[:, :actual_video_length, :, :]

    return pixel_values, mask_pixel_values, mask

def process_inpaint_mode(args, batch, weight_dtype, accelerator):
    """Process inpaint mode logic"""
    mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
    mask = batch["mask"].to(weight_dtype)
    
    # Apply batch tiling for inpaint data
    if args.auto_tile_batch_size and args.training_with_video_token_length:
        pixel_values = batch["pixel_values"].to(weight_dtype)
        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
            mask_pixel_values = torch.tile(mask_pixel_values, (4, 1, 1, 1, 1))
            mask = torch.tile(mask, (4, 1, 1, 1, 1))
        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
            mask_pixel_values = torch.tile(mask_pixel_values, (2, 1, 1, 1, 1))
            mask = torch.tile(mask, (2, 1, 1, 1, 1))

    # Create t2v flag
    t2v_flag = [(_mask == 1).all() for _mask in mask]
    new_t2v_flag = []
    for _mask in t2v_flag:
        if _mask and np.random.rand() < 0.90:
            new_t2v_flag.append(0)
        else:
            new_t2v_flag.append(1)
    t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)

    return mask_pixel_values, mask, t2v_flag

def batch_encode_vae(pixel_values, vae, vae_mini_batch):
    """Encode pixel values using VAE in batches"""
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    bs = vae_mini_batch
    new_pixel_values = []
    for i in range(0, pixel_values.shape[0], bs):
        pixel_values_bs = pixel_values[i : i + bs]
        pixel_values_bs = vae.encode(pixel_values_bs)[0]
        pixel_values_bs = pixel_values_bs.sample()
        new_pixel_values.append(pixel_values_bs)
    return torch.cat(new_pixel_values, dim=0)