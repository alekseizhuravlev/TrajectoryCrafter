import numpy as np
import torch
from torchvision import transforms
from einops import rearrange
from videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                           ASPECT_RATIO_RANDOM_CROP_512,
                                           ASPECT_RATIO_RANDOM_CROP_PROB,
                                           get_closest_ratio)
from videox_fun.data.dataset_image_video import get_random_mask

import sys
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/05_11_25_training/lora_utils')
from utils import get_random_downsample_ratio

def create_collate_fn(args, vae, text_encoder, tokenizer, rng=None):
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio

    def get_length_to_frame_num(token_length):
        if args.image_sample_size > args.video_sample_size:
            sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

            if sample_sizes[-1] != args.image_sample_size:
                sample_sizes.append(args.image_sample_size)
        else:
            sample_sizes = [args.image_sample_size]
        
        length_to_frame_num = {
            sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
        }

        return length_to_frame_num

    def collate_fn(examples):
        # Get token length
        target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
        length_to_frame_num = get_length_to_frame_num(target_token_length)

        # Create new output
        new_examples                 = {}
        new_examples["target_token_length"] = target_token_length
        new_examples["pixel_values"] = []
        new_examples["text"]         = []
        # Used in Inpaint mode
        if args.train_mode != "normal":
            new_examples["mask_pixel_values"] = []
            new_examples["mask"] = []

        # Get downsample ratio in image and videos
        pixel_value     = examples[0]["pixel_values"]
        data_type       = examples[0]["data_type"]
        f, h, w, c      = np.shape(pixel_value)
        if data_type == 'image':
            random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size], rng=rng)

            aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
            aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
            
            batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
        else:
            if args.random_hw_adapt:
                if args.training_with_video_token_length:
                    local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))
                    # The video will be resized to a lower resolution than its own.
                    choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                    if len(choice_list) == 0:
                        choice_list = list(length_to_frame_num.keys())
                    if rng is None:
                        local_video_sample_size = np.random.choice(choice_list)
                    else:
                        local_video_sample_size = rng.choice(choice_list)
                    batch_video_length = length_to_frame_num[local_video_sample_size]
                    random_downsample_ratio = args.video_sample_size / local_video_sample_size
                else:
                    random_downsample_ratio = get_random_downsample_ratio(
                            args.video_sample_size, rng=rng)
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                random_downsample_ratio = 1
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

            aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
            aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

        closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
        closest_size = [int(x / 16) * 16 for x in closest_size]
        if args.random_ratio_crop:
            if rng is None:
                random_sample_size = aspect_ratio_random_crop_sample_size[
                    np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                ]
            else:
                random_sample_size = aspect_ratio_random_crop_sample_size[
                    rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                ]
            random_sample_size = [int(x / 16) * 16 for x in random_sample_size]

        for example in examples:
            if args.random_ratio_crop:
                # To 0~1
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.

                # Get adapt hw for resize
                b, c, h, w = pixel_values.size()
                th, tw = random_sample_size
                if th / tw > h / w:
                    nh = int(th)
                    nw = int(w / h * nh)
                else:
                    nw = int(tw)
                    nh = int(h / w * nw)
                
                transform = transforms.Compose([
                    transforms.Resize([nh, nw]),
                    transforms.CenterCrop([int(x) for x in random_sample_size]),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ])
            else:
                # To 0~1
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.

                # Get adapt hw for resize
                closest_size = list(map(lambda x: int(x), closest_size))
                if closest_size[0] / h > closest_size[1] / w:
                    resize_size = closest_size[0], int(w * closest_size[0] / h)
                else:
                    resize_size = int(h * closest_size[1] / w), closest_size[1]
                
                transform = transforms.Compose([
                    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                    transforms.CenterCrop(closest_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ])
            new_examples["pixel_values"].append(transform(pixel_values))
            new_examples["text"].append(example["text"])

            batch_video_length = int(min(batch_video_length, len(pixel_values)))

            # Magvae needs the number of frames to be 4n + 1.
            local_latent_length = (batch_video_length - 1) // sample_n_frames_bucket_interval + 1
            local_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

            # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
            additional_frames = 0
            patch_size_t = getattr(args, 'patch_size_t', None)  # This would need to be passed from transformer
            if patch_size_t is not None and local_latent_length % patch_size_t != 0:
                additional_frames = local_latent_length % patch_size_t
                local_video_length -= additional_frames * sample_n_frames_bucket_interval
            batch_video_length = local_video_length

            if batch_video_length <= 0:
                batch_video_length = 1

            if args.train_mode != "normal":
                mask = get_random_mask(new_examples["pixel_values"][-1].size())
                mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                new_examples["mask_pixel_values"].append(mask_pixel_values)
                new_examples["mask"].append(mask)

        # Limit the number of frames to the same
        new_examples["pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["pixel_values"]])
        if args.train_mode != "normal":
            new_examples["mask_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["mask_pixel_values"]])
            new_examples["mask"] = torch.stack([example[:batch_video_length] for example in new_examples["mask"]])

        # Encode prompts when enable_text_encoder_in_dataloader=True
        if args.enable_text_encoder_in_dataloader:
            prompt_ids = tokenizer(
                new_examples['text'], 
                max_length=args.tokenizer_max_length, 
                padding="max_length", 
                add_special_tokens=True, 
                truncation=True, 
                return_tensors="pt"
            )
            encoder_hidden_states = text_encoder(
                prompt_ids.input_ids,
                return_dict=False
            )[0]
            new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
            new_examples['encoder_hidden_states'] = encoder_hidden_states

        return new_examples
    
    return collate_fn