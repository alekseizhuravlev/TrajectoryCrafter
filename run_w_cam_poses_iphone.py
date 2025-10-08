import torch
import numpy as np
from demo import TrajCrafter
import json
import os
from datetime import datetime
import cv2
import sys
import math

# Add your dataset path
sys.path.insert(0, '/home/azhuravl/work/TrajectoryCrafter/notebooks/09_09_25_multiview')
from iphone_original_dataset import iPhoneDataset

class CameraPoseiPhoneTraCrafter(TrajCrafter):
    def __init__(self, opts, gradio=False):
        super().__init__(opts, gradio)
        # Don't initialize depth_estimater since we get depth from dataset
        self.depth_estimater = None
    
    def convert_camera_format_from_iphone(self, camera_data):
        """Convert iPhone camera format to format expected by TrajCrafter"""
        # Extract camera parameters from your dataset format
        K = camera_data['K']  # [3, 3] numpy array
        c2w = camera_data['c2w']  # [4, 4] numpy array
        
        c2w = np.linalg.inv(c2w)
        
        # Convert to torch tensors
        K_tensor = torch.from_numpy(K).float()
        c2w_tensor = torch.from_numpy(c2w).float()
        
        return c2w_tensor, K_tensor
    
    def prepare_iphone_data(self, sequence_data, target_size=(576, 1024)):
        """Prepare data from iPhone dataset sequence"""
        print(f"==> preparing iPhone data from sequence of {len(sequence_data)} frames")
        
        # Extract frames and depths
        source_frames = []
        depths = []
        
        for frame_data in sequence_data:
            # Get camera 0 data (source camera with depth)
            cam0_data = frame_data['cameras'][0]
            
            # RGB frame: [H, W, 3] numpy array, values in [0, 255]
            rgb = cam0_data['rgb'].astype(np.float32) / 255.0  # Normalize to [0, 1]
            source_frames.append(rgb)
            
            # Depth: [H, W] numpy array
            depth = cam0_data['depth']
            if depth is None:
                raise ValueError(f"No depth data for frame {frame_data['frame_id']}")
            depths.append(depth.astype(np.float32))
        
        source_frames = np.stack(source_frames)  # [T, H, W, 3]
        depths = np.stack(depths)  # [T, H, W]
        
        print(f"==> source frames shape: {source_frames.shape}")
        print(f"==> depths shape: {depths.shape}")
        print(f"==> source frames range: [{source_frames.min():.3f}, {source_frames.max():.3f}]")
        print(f"==> depths range: [{depths.min():.3f}, {depths.max():.3f}]")
        
        # Get camera parameters from first frame
        source_c2w, source_K = self.convert_camera_format_from_iphone(sequence_data[0]['cameras'][0])
        
        # For now, use same camera for target (you can modify this to use camera 1 or 2)
        target_c2w, target_K = source_c2w.clone(), source_K.clone()
        
        # Resize frames and update intrinsics
        resized_source_frames, updated_source_K = self.resize_frames_and_update_intrinsics(
            source_frames, source_K.numpy(), target_size
        )
        
        # Resize depth maps
        resized_depths = self.resize_depth_maps(depths, target_size)
        
        return {
            'frames': resized_source_frames,
            'depths': resized_depths,
            'source_intrs': updated_source_K,
            'source_c2w': source_c2w.numpy(),
            'target_intrs': updated_source_K,  # Same for now
            'target_c2w': target_c2w.numpy(),  # Same for now
            'seq_name': f"iphone_frames_{sequence_data[0]['frame_id']}_to_{sequence_data[-1]['frame_id']}"
        }
    
    def prepare_iphone_multiview_data(self, sequence_data, source_cam_id=0, target_cam_id=1, target_size=(576, 1024)):
        """Prepare data from iPhone dataset sequence with different source/target cameras"""
        print(f"==> preparing iPhone multi-view data: camera {source_cam_id} -> camera {target_cam_id}")
        
        # Extract frames from source and target cameras
        source_frames = []
        target_frames = []
        depths = []
        
        for frame_data in sequence_data:
            # Source camera data
            source_cam_data = frame_data['cameras'][source_cam_id]
            target_cam_data = frame_data['cameras'][target_cam_id]
            
            # RGB frames: [H, W, 3] numpy array, values in [0, 255]
            source_rgb = source_cam_data['rgb'].astype(np.float32) / 255.0
            target_rgb = target_cam_data['rgb'].astype(np.float32) / 255.0
            
            source_frames.append(source_rgb)
            target_frames.append(target_rgb)
            
            # Depth from source camera (only camera 0 has depth)
            if source_cam_id == 0:
                depth = source_cam_data['depth']
                if depth is None:
                    raise ValueError(f"No depth data for frame {frame_data['frame_id']}")
                depths.append(depth.astype(np.float32))
            else:
                raise ValueError(f"Depth only available for camera 0, but source camera is {source_cam_id}")
        
        source_frames = np.stack(source_frames)  # [T, H, W, 3]
        target_frames = np.stack(target_frames)  # [T, H, W, 3] 
        depths = np.stack(depths)  # [T, H, W]
        
        print(f"==> source frames shape: {source_frames.shape}")
        print(f"==> target frames shape: {target_frames.shape}")
        print(f"==> depths shape: {depths.shape}")
        
        # Get camera parameters
        source_c2w, source_K = self.convert_camera_format_from_iphone(sequence_data[0]['cameras'][source_cam_id])
        target_c2w, target_K = self.convert_camera_format_from_iphone(sequence_data[0]['cameras'][target_cam_id])
        
        # Resize frames and update intrinsics
        resized_source_frames, updated_source_K = self.resize_frames_and_update_intrinsics(
            source_frames, source_K.numpy(), target_size
        )
        
        resized_target_frames, updated_target_K = self.resize_frames_and_update_intrinsics(
            target_frames, target_K.numpy(), target_size
        )
        
        # Resize depth maps
        resized_depths = self.resize_depth_maps(depths, target_size)
        
        return {
            'frames': resized_source_frames,
            'target_frames': resized_target_frames,
            'depths': resized_depths,
            'source_intrs': updated_source_K,
            'source_c2w': source_c2w.numpy(),
            'target_intrs': updated_target_K,
            'target_c2w': target_c2w.numpy(),
            'seq_name': f"iphone_cam{source_cam_id}_to_cam{target_cam_id}_frames_{sequence_data[0]['frame_id']}_to_{sequence_data[-1]['frame_id']}"
        }
    
    def resize_frames_and_update_intrinsics(self, frames, original_K, target_size=(576, 1024)):
        """Resize frames and update camera intrinsics accordingly"""
        # frames: [T, H, W, C] numpy array
        original_height, original_width = frames.shape[1:3]
        target_height, target_width = target_size
        
        print(f"==> resizing from {original_width}x{original_height} to {target_width}x{target_height}")
        
        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Resize frames
        resized_frames = []
        for frame in frames:
            # Ensure frame is in [0, 1] range
            if frame.max() <= 1.0:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                frame_uint8 = frame.astype(np.uint8)
            
            resized_frame = cv2.resize(frame_uint8, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame.astype(np.float32) / 255.0)
        
        resized_frames = np.stack(resized_frames)
        
        # Update intrinsic matrix
        updated_K = original_K.copy()
        updated_K[0, 0] *= scale_x  # fx
        updated_K[1, 1] *= scale_y  # fy
        updated_K[0, 2] *= scale_x  # cx
        updated_K[1, 2] *= scale_y  # cy
        
        print(f"==> intrinsics updated:")
        print(f"    Original K:\n{original_K}")
        print(f"    Resized K:\n{updated_K}")
        print(f"    Scale factors: sx={scale_x:.3f}, sy={scale_y:.3f}")
        
        return resized_frames, updated_K
    
    def resize_depth_maps(self, depths, target_size=(576, 1024)):
        """Resize depth maps to target size"""
        target_height, target_width = target_size
        
        print(f"==> resizing depth maps to {target_width}x{target_height}")
        
        resized_depths = []
        for depth in depths:
            resized_depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            resized_depths.append(resized_depth)
        
        return np.stack(resized_depths)
    
    def infer_iphone_sequence(self, opts, sequence_data, source_cam_id=0, target_cam_id=1):
        """Inference using iPhone dataset sequence"""
        
        print(f"==> running inference with iPhone sequence of {len(sequence_data)} frames")
        print(f"==> source camera: {source_cam_id}, target camera: {target_cam_id}")
        
        # Prepare data
        target_size = (576, 1024)  # Height, Width
        if target_cam_id == source_cam_id:
            # Single camera mode
            prepared_data = self.prepare_iphone_data(sequence_data, target_size)
        else:
            # Multi-view mode
            prepared_data = self.prepare_iphone_multiview_data(
                sequence_data, source_cam_id, target_cam_id, target_size
            )
        
        # Limit to opts.video_length frames
        max_frames = min(opts.video_length, len(prepared_data['frames']))
        frames = prepared_data['frames'][:max_frames]
        depths = prepared_data['depths'][:max_frames]
        
        # Handle target frames
        if 'target_frames' in prepared_data:
            target_frames = prepared_data['target_frames'][:max_frames]
        else:
            target_frames = frames.copy()  # Use same frames for single camera mode
        
        print(f"==> using {max_frames} frames")
        
        # Pad if necessary
        if len(frames) < opts.video_length:
            last_frame = frames[-1:]
            last_target_frame = target_frames[-1:]
            last_depth = depths[-1:]
            num_pad = opts.video_length - len(frames)
            pad_frames = np.repeat(last_frame, num_pad, axis=0)
            pad_target_frames = np.repeat(last_target_frame, num_pad, axis=0)
            pad_depths = np.repeat(last_depth, num_pad, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
            target_frames = np.concatenate([target_frames, pad_target_frames], axis=0)
            depths = np.concatenate([depths, pad_depths], axis=0)
            print(f"==> padded to {opts.video_length} frames")
        
        # Get caption
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        
        # Convert to torch tensors
        depths_tensor = torch.from_numpy(depths).unsqueeze(1).to(opts.device).float()  # [T, 1, H, W]
        
        # depths_tensor = 1 / (torch.clamp(depths_tensor, min=1e-6))  # Convert to disparity
        
        frames_tensor = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        target_frames_tensor = (
            torch.from_numpy(target_frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        
        # Convert camera formats
        source_c2w = torch.from_numpy(prepared_data['source_c2w']).float().to(opts.device)
        target_c2w = torch.from_numpy(prepared_data['target_c2w']).float().to(opts.device)
        source_K = torch.from_numpy(prepared_data['source_intrs']).float().to(opts.device)
        target_K = torch.from_numpy(prepared_data['target_intrs']).float().to(opts.device)
        
        # Create pose matrices for all frames
        pose_s = source_c2w.unsqueeze(0).repeat(opts.video_length, 1, 1)
        pose_t = target_c2w.unsqueeze(0).repeat(opts.video_length, 1, 1)
        K_matrices_s = source_K.unsqueeze(0).repeat(opts.video_length, 1, 1)
        K_matrices_t = target_K.unsqueeze(0).repeat(opts.video_length, 1, 1)
        
        print(f"==> using iPhone camera data for warping")
        
        # Warp images
        warped_images = []
        masks = []
        
        for i in range(opts.video_length):
            warped_frame, mask, _, _ = self.funwarp.forward_warp(
                frames_tensor[i : i + 1],
                None,
                depths_tensor[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K_matrices_s[i : i + 1],
                K_matrices_t[i : i + 1],
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame)
            masks.append(mask)
        
        # Continue with existing pipeline
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        
        # Resize for processing
        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        target_frames_tensor = torch.nn.functional.interpolate(
            target_frames_tensor, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = torch.nn.functional.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = torch.nn.functional.interpolate(
            cond_masks, size=opts.sample_size, mode='nearest'
        )
        
        # Save intermediate videos
        self.save_intermediate_videos(frames_tensor, target_frames_tensor, cond_video, cond_masks, opts, prepared_data['seq_name'])
        print("Intermediate videos saved.")
    
        # Save camera and depth data for reference
        self.save_iphone_data(prepared_data, opts.save_dir)
        
        # For now, exit after saving intermediates (uncomment below for full pipeline)
        # exit(0)
        
        # Prepare for diffusion
        frames_for_diffusion = (frames_tensor.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames_for_diffusion[:, :, :10, :, :]
        cond_video_for_diffusion = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks_for_diffusion = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)
        
        # Run diffusion
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video_for_diffusion.to(opts.device),
                mask_video=cond_masks_for_diffusion.to(opts.device),
                reference=frames_ref,
            ).videos
        
        # Save final video
        self.save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )
        
        return sample
    
    def save_iphone_data(self, data, save_dir):
        """Save iPhone data for reference"""
        iphone_data = {
            "seq_name": data['seq_name'],
            "source_intrs": data['source_intrs'].tolist(),
            "target_intrs": data['target_intrs'].tolist(),
            "source_c2w": data['source_c2w'].tolist(),
            "target_c2w": data['target_c2w'].tolist(),
            "frame_shape": list(data['frames'].shape),
            "depth_shape": list(data['depths'].shape),
        }
        
        if 'target_frames' in data:
            iphone_data["target_frame_shape"] = list(data['target_frames'].shape)
        
        with open(os.path.join(save_dir, 'iphone_data.json'), 'w') as f:
            json.dump(iphone_data, f, indent=2)
        
        print(f"iPhone data saved to {os.path.join(save_dir, 'iphone_data.json')}")
    
    def save_intermediate_videos(self, frames, target_frames, cond_video, cond_masks, opts, seq_name="iphone"):
        """Save intermediate videos for debugging"""
        # Source frames (input)
        self.save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, f'input_source_{seq_name}.mp4'),
            fps=opts.fps,
        )
        
        # Target frames (ground truth)
        self.save_video(
            (target_frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, f'target_gt_{seq_name}.mp4'),
            fps=opts.fps,
        )
        
        # Warped/rendered frames
        self.save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, f'render_{seq_name}.mp4'),
            fps=opts.fps,
        )
        
        # Masks
        self.save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, f'mask_{seq_name}.mp4'),
            fps=opts.fps,
        )
        
        # Create side-by-side comparison video (render vs ground truth)
        self.save_comparison_video(cond_video, target_frames, opts, seq_name)
    
    def save_comparison_video(self, rendered_frames, target_frames, opts, seq_name="iphone"):
        """Save side-by-side comparison of rendered vs ground truth"""
        rendered_normalized = rendered_frames.permute(0, 2, 3, 1)
        target_normalized = (target_frames.permute(0, 2, 3, 1) + 1.0) / 2.0
        
        device = rendered_normalized.device
        target_normalized = target_normalized.to(device)
        
        comparison_frames = torch.cat([rendered_normalized, target_normalized], dim=2)
        
        self.save_video(
            comparison_frames,
            os.path.join(opts.save_dir, f'comparison_render_vs_gt_{seq_name}.mp4'),
            fps=opts.fps,
        )

    def save_video(self, tensor, path, fps):
        """Save video tensor to file"""
        from models.utils import save_video
        save_video(tensor, path, fps=fps)

# Usage function
def run_iphone_inference(source_cam_id=0, target_cam_id=1, sequence_length=49, output_dir="./experiments/"):
    """Run inference using iPhone dataset"""
    
    # Create minimal opts
    class Opts:
        def __init__(self, source_cam_id, target_cam_id):
            # Video settings
            self.video_length = sequence_length
            self.fps = 10
            self.stride = 1
            
            # Device
            self.device = 'cuda:0'
            self.weight_dtype = torch.bfloat16
            
            # Output
            timestamp = datetime.now().strftime("%H%M")
            date = datetime.now().strftime("%d-%m-%Y")
            self.exp_name = f"{timestamp}_iphone_cam{source_cam_id}_to_cam{target_cam_id}"
            self.save_dir = f"./experiments/{date}/{self.exp_name}"
            
            # Diffusion
            self.model_name = 'checkpoints/CogVideoX-Fun-V1.1-5b-InP'
            self.transformer_path = 'checkpoints/TrajectoryCrafter'
            self.sampler_name = 'DDIM_Origin'
            self.sample_size = [384, 672]
            self.diffusion_guidance_scale = 6.0
            self.diffusion_inference_steps = 50
            self.negative_prompt = "The video is not of a high quality, it has a low resolution."
            self.low_gpu_memory_mode = False
            
            # Camera
            self.mask = False
            self.seed = 43
            
            # Paths
            self.unet_path = "checkpoints/DepthCrafter"
            self.pre_train_path = "checkpoints/stable-video-diffusion-img2vid"
            self.blip_path = "checkpoints/blip2-opt-2.7b"
            self.cpu_offload = 'model'
            self.refine_prompt = ". High quality, masterpiece, best quality."
    
    opts = Opts(source_cam_id, target_cam_id)
    os.makedirs(opts.save_dir, exist_ok=True)
    
    # Initialize iPhone dataset
    dataset = iPhoneDataset(
        root_dir='/home/azhuravl/nobackup/iphone',
        sequence_name='paper-windmill',
        scale='2x',
        camera_ids=[source_cam_id, target_cam_id],
        min_sequence_length=sequence_length
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Available contiguous sequences: {len(dataset.contiguous_sequences)}")
    
    # Get a sequence of the desired length
    subsequences = dataset.get_contiguous_subsequences(sequence_length)
    
    if not subsequences:
        print(f"No sequences of length {sequence_length} available")
        return None
    
    # Use the first available subsequence
    selected_frames = subsequences[0]
    sequence_data = dataset.get_sequence_data(selected_frames)
    
    print(f"Using sequence: frames {selected_frames[0]}-{selected_frames[-1]}")
    
    # Initialize model
    crafter = CameraPoseiPhoneTraCrafter(opts)
    
    # Run inference
    result = crafter.infer_iphone_sequence(opts, sequence_data, source_cam_id, target_cam_id)
    
    print(f"Generated video saved to: {opts.save_dir}")
    return result

if __name__ == "__main__":
    # Run with camera 0 -> camera 1
    result = run_iphone_inference(
        source_cam_id=0,
        target_cam_id=2,
        sequence_length=49
        )