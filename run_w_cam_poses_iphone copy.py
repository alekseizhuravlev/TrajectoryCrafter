import torch
import numpy as np
from demo import TrajCrafter
import json
import os
from datetime import datetime
import cv2
import sys
import math

# Add dataset path
sys.path.append('/home/azhuravl/work/MonoDyGauBench_code')
from src.data import NerfiesDataModule

class CameraPoseNerfiesTraCrafter(TrajCrafter):
    def __init__(self, opts, gradio=False):
        super().__init__(opts, gradio)
        # Don't initialize depth_estimater since we get depth from dataset
        self.depth_estimater = None
    
    def convert_camera_format_from_nerfies(self, camera_data):
        """Convert Nerfies camera format to format expected by TrajCrafter"""
        # Extract camera parameters
        world_view_transform = camera_data['world_view_transform']  # [4, 4]
        full_proj_transform = camera_data['full_proj_transform']    # [4, 4] 
        camera_center = camera_data['camera_center']               # [3]
        FoVx = camera_data['FoVx']
        FoVy = camera_data['FoVy']
        image_height = camera_data['image_height']
        image_width = camera_data['image_width']
        
        # Convert FoV to focal lengths (using numpy since FoV values are floats)
        fx = image_width / (2.0 * math.tan(FoVx / 2.0))
        fy = image_height / (2.0 * math.tan(FoVy / 2.0))
        
        # Create intrinsic matrix
        K = torch.zeros(3, 3, dtype=torch.float32)
        K[0, 0] = fx  # focal length x
        K[1, 1] = fy  # focal length y
        # K[0, 0] = fy  # focal length x
        # K[1, 1] = fx  # focal length y
        K[0, 2] = image_width / 2.0   # principal point x
        K[1, 2] = image_height / 2.0  # principal point y
        K[2, 2] = 1.0
        

        # world_view_transform is typically world-to-camera, so invert for camera-to-world
        c2w = torch.inverse(world_view_transform.T)        
        # c2w = torch.inverse(world_view_transform).T
        # c2w = world_view_transform.T
        
        # c2w = world_view_transform
        
        # c2w = torch.zeros_like(world_view_transform)
        # c2w[:3, :3] = world_view_transform[:3, :3]
        # c2w[:3, 3] = world_view_transform[3, :3]
        # c2w[3, 3] = 1.0
        
        # c2w = torch.inverse(c2w)

        # K = K.T
        
        # increase camera scale
        # scale = 100
        # c2w[:3, 3] = c2w[:3, 3] * scale
        
        
        return c2w, K
    
    def prepare_nerfies_data(self, train_sequence, test_sequence, target_size=(576, 1024)):
        """Prepare data from Nerfies dataset"""
        print(f"==> preparing Nerfies data (train=source, test=target)")
        
        # Extract frames (convert RGBA to RGB)
        source_frames = []
        target_frames = []
        source_depths = []
        
        for i, (train_data, test_data) in enumerate(zip(train_sequence, test_sequence)):
            # Convert RGBA to RGB and from [C, H, W] to [H, W, C]
            train_rgba = train_data['original_image'].permute(1, 2, 0).cpu().numpy()  # [H, W, 4]
            test_rgba = test_data['original_image'].permute(1, 2, 0).cpu().numpy()   # [H, W, 4]
            
            # Take RGB channels only (first 3)
            train_rgb = train_rgba[:, :, :3]
            test_rgb = test_rgba[:, :, :3]
            
            source_frames.append(train_rgb)
            target_frames.append(test_rgb)
            
            # Extract depth
            depth = train_data['depth'].cpu().numpy().astype(np.float32)
            depth = 1 / (np.clip(depth, 1e-6, None))  # Convert disparity to depth
            source_depths.append(depth)
        
        source_frames = np.stack(source_frames)  # [T, H, W, 3]
        target_frames = np.stack(target_frames)  # [T, H, W, 3]
        source_depths = np.stack(source_depths)  # [T, H, W]
        
        print(f"==> source frames shape: {source_frames.shape}")
        print(f"==> target frames shape: {target_frames.shape}")
        print(f"==> source depths shape: {source_depths.shape}")
        print(f"==> source frames range: [{source_frames.min():.3f}, {source_frames.max():.3f}]")
        print(f"==> target frames range: [{target_frames.min():.3f}, {target_frames.max():.3f}]")
        
        # Get camera parameters from first frame
        source_c2w, source_K = self.convert_camera_format_from_nerfies(train_sequence[0])
        target_c2w, target_K = self.convert_camera_format_from_nerfies(test_sequence[0])
        
        # Resize frames and update intrinsics
        resized_source_frames, updated_source_K = self.resize_frames_and_update_intrinsics(
            source_frames, source_K.numpy(), target_size
        )
        
        resized_target_frames, updated_target_K = self.resize_frames_and_update_intrinsics(
            target_frames, target_K.numpy(), target_size
        )
        
        # Resize depth maps
        resized_depths = self.resize_depth_maps(source_depths, target_size)
        
        return {
            'frames': resized_source_frames,
            'target_frames': resized_target_frames,
            'depths': resized_depths,
            'source_intrs': updated_source_K,
            'source_c2w': source_c2w.numpy(),
            'target_intrs': updated_target_K,  
            'target_c2w': target_c2w.numpy(),
            'seq_name': f"train_{train_sequence[0]['image_name']}_to_test_{test_sequence[0]['image_name']}"
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
    
    def infer_camera_poses_nerfies(self, opts, train_sequence, test_sequence):
        """Inference using Nerfies dataset"""
        
        print(f"==> running inference with {len(train_sequence)} train frames and {len(test_sequence)} test frames")
        
        # Prepare data
        target_size = (576, 1024)  # Height, Width
        prepared_data = self.prepare_nerfies_data(train_sequence, test_sequence, target_size)
        
        # Limit to opts.video_length frames
        max_frames = min(opts.video_length, len(prepared_data['frames']))
        frames = prepared_data['frames'][:max_frames]
        target_frames = prepared_data['target_frames'][:max_frames]
        depths = prepared_data['depths'][:max_frames]
        
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
        
        print(f"==> using Nerfies camera data for warping")
        
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
        
        # Save intermediate videos INCLUDING target ground truth
        self.save_intermediate_videos(frames_tensor, target_frames_tensor, cond_video, cond_masks, opts, prepared_data['seq_name'])
        print("Intermediate videos saved.")
    
        
        # Save camera and depth data for reference
        # self.save_nerfies_data(prepared_data, opts.save_dir)
        exit(0)
        
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
    
    def save_nerfies_data(self, data, save_dir):
        """Save Nerfies data for reference"""
        nerfies_data = {
            "seq_name": data['seq_name'],
            "source_intrs": data['source_intrs'].tolist(),
            "target_intrs": data['target_intrs'].tolist(),
            "source_c2w": data['source_c2w'].tolist(),
            "target_c2w": data['target_c2w'].tolist(),
            "frame_shape": list(data['frames'].shape),
            "target_frame_shape": list(data['target_frames'].shape),
            "depth_shape": list(data['depths'].shape),
        }
        
        with open(os.path.join(save_dir, 'nerfies_data.json'), 'w') as f:
            json.dump(nerfies_data, f, indent=2)
        
        print(f"Nerfies data saved to {os.path.join(save_dir, 'nerfies_data.json')}")
    
    def save_intermediate_videos(self, frames, target_frames, cond_video, cond_masks, opts, seq_name="nerfies"):
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
    
    def save_comparison_video(self, rendered_frames, target_frames, opts, seq_name="nerfies"):
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



def find_synchronized_sequence_from_ids(train_ids, test_ids, test_camera_id=1, sequence_length=49):
    """Find synchronized consecutive sequences using ID lists directly"""
    
    # Extract frame numbers from IDs (format: "camera_frame")
    train_frame_map = {}  # frame_num -> index in train_ids
    test_frame_map = {}   # frame_num -> index in test_ids
    
    # Process train IDs
    for idx, train_id in enumerate(train_ids):
        frame_num = int(train_id.split('_')[1])
        train_frame_map[frame_num] = idx
    
    # Process test IDs - filter by specified camera
    for idx, test_id in enumerate(test_ids):
        camera_id, frame_num = test_id.split('_')
        camera_id = int(camera_id)
        frame_num = int(frame_num)
        
        # Only include frames from the specified test camera
        if camera_id == test_camera_id:
            test_frame_map[frame_num] = idx
    
    print(f"Train camera frames: {len(train_frame_map)}")
    print(f"Test camera {test_camera_id} frames: {len(test_frame_map)}")
    
    # Find common frame numbers
    common_frames = set(train_frame_map.keys()) & set(test_frame_map.keys())
    common_frames = sorted(list(common_frames))
    
    print(f"Found {len(common_frames)} common frame numbers")
    if common_frames:
        print(f"Frame range: {min(common_frames)} to {max(common_frames)}")
    
    # Find consecutive sequences
    sequences = []
    
    for start_idx in range(len(common_frames) - sequence_length + 1):
        frame_sequence = common_frames[start_idx:start_idx + sequence_length]
        
        # Check if consecutive
        is_consecutive = all(frame_sequence[i+1] == frame_sequence[i] + 1 
                           for i in range(len(frame_sequence) - 1))
        
        if is_consecutive:
            # Get dataset indices for these frame numbers
            train_indices = [train_frame_map[frame_num] for frame_num in frame_sequence]
            test_indices = [test_frame_map[frame_num] for frame_num in frame_sequence]
            
            # Get the actual IDs
            train_ids_seq = [train_ids[idx] for idx in train_indices]
            test_ids_seq = [test_ids[idx] for idx in test_indices]
            
            sequences.append({
                'frame_numbers': frame_sequence,
                'train_dataset_indices': train_indices,
                'test_dataset_indices': test_indices,
                'train_ids': train_ids_seq,
                'test_ids': test_ids_seq,
                'test_camera_id': test_camera_id,
                'start_frame': frame_sequence[0],
                'end_frame': frame_sequence[-1],
                'length': len(frame_sequence)
            })
    
    return sequences


def load_synchronized_sequence(sequence, train_dataset, test_dataset):
    """Load actual data only when needed"""
    train_data = [train_dataset[idx] for idx in sequence['train_dataset_indices']]
    test_data = [test_dataset[idx] for idx in sequence['test_dataset_indices']]
    return train_data, test_data



# Usage function
def run_nerfies_inference(test_camera_id=1, output_dir="./experiments/"):
    """Run inference using Nerfies dataset"""
    
    # Create minimal opts
    class Opts:
        def __init__(self, test_camera_id):
            # Video settings
            self.video_length = 49
            self.fps = 10
            self.stride = 1
            
            # Device
            self.device = 'cuda:0'
            self.weight_dtype = torch.bfloat16
            
            # Output
            timestamp = datetime.now().strftime("%H%M")
            date = datetime.now().strftime("%d-%m-%Y")
            self.exp_name = f"{timestamp}_nerfies_cam{test_camera_id}"
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
    
    opts = Opts(test_camera_id)
    os.makedirs(opts.save_dir, exist_ok=True)
    
    # Initialize dataset
    all_dataset = NerfiesDataModule(
        datadir='/home/azhuravl/nobackup/iphone/paper-windmill',
        eval=True,
        ratio=0.5,
        white_background=True,
        num_pts_ratio=0.,
        num_pts=0,
        load_flow=False
    )
    all_dataset.setup('test')
    
    # Find synchronized sequence
    train_ids = all_dataset.train_cam_infos.train_id
    test_ids = all_dataset.train_cam_infos.val_id
    
    sequences = find_synchronized_sequence_from_ids(train_ids, test_ids, test_camera_id, 49)
    
    if not sequences:
        print(f"No synchronized sequences found for camera {test_camera_id}")
        return None
    
    # Load the sequence
    best_seq = sequences[0]
    
    # print('changed order of sequences!!!')
    train_sequence, test_sequence = load_synchronized_sequence(best_seq, all_dataset.train_cameras, all_dataset.test_cameras)
    # test_sequence, train_sequence = load_synchronized_sequence(best_seq, all_dataset.train_cameras, all_dataset.test_cameras)
    
    # Initialize model
    crafter = CameraPoseNerfiesTraCrafter(opts)
    
    # Run inference
    result = crafter.infer_camera_poses_nerfies(opts, train_sequence, test_sequence)
    
    print(f"Generated video saved to: {opts.save_dir}")
    return result


if __name__ == "__main__":
    # Run with camera 1
    result = run_nerfies_inference(test_camera_id=2)