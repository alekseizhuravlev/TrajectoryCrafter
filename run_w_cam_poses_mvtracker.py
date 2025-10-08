import torch
import numpy as np
from demo import TrajCrafter
import json
import os
from datetime import datetime
import cv2
import sys

# Add MVTracker to path
sys.path.append('/home/azhuravl/work/mvtracker')
from mvtracker.datasets.panoptic_studio_multiview_dataset import PanopticStudioMultiViewDataset
from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset

class CameraPoseTrajCrafter(TrajCrafter):
    def __init__(self, opts, gradio=False):
        super().__init__(opts, gradio)
        # Don't initialize depth_estimater since we get depth from dataset
        self.depth_estimater = None
    
    def convert_camera_format_from_mvtracker(self, intrs, extrs):
        """Convert MVTracker camera format to c2w matrix and intrinsics"""
        # intrs: [3, 3] intrinsic matrix
        # extrs: [3, 4] extrinsic matrix [R|t]
        
        K = intrs.numpy() if torch.is_tensor(intrs) else intrs
        extr = extrs.numpy() if torch.is_tensor(extrs) else extrs
        
        # Create 4x4 extrinsic matrix
        RT = np.vstack([extr, [0, 0, 0, 1]])
        
        # Convert to c2w (camera to world)
        c2w = RT  # Use as is, or np.linalg.inv(RT) depending on your convention
        
        return torch.tensor(c2w, dtype=torch.float32), torch.tensor(K, dtype=torch.float32)
    
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
            # Convert to uint8 for cv2 if needed
            if frame.dtype == np.float32 and frame.max() <= 1.0:
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
        # depths: [T, H, W] numpy array
        target_height, target_width = target_size
        
        print(f"==> resizing depth maps to {target_width}x{target_height}")
        
        resized_depths = []
        for depth in depths:
            # Use nearest interpolation for depth to avoid artifacts
            resized_depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            resized_depths.append(resized_depth)
        
        return np.stack(resized_depths)
    
    def prepare_mvtracker_data(self, data, target_size=(576, 1024)):
        """Prepare data from MVTracker dataset - assumes data only has 2 views [source, target]"""
        print(f"==> preparing MVTracker data (source=view[0], target=view[1])")
        
        # Since views_to_return=[source_view_idx, target_view_idx], 
        # the returned data always has source at index 0, target at index 1
        source_frames = data.video[0].permute(0, 2, 3, 1).cpu().numpy()  # Always 0
        source_depths = data.videodepth[0].squeeze(1).cpu().numpy()      # Always 0
        source_intrs = data.intrs[0][0].cpu().numpy()  # Always 0
        source_extrs = data.extrs[0][0].cpu().numpy()  # Always 0
        
        # Extract target camera data
        target_frames = data.video[1].permute(0, 2, 3, 1).cpu().numpy()  # Always 1 - ADDED
        target_intrs = data.intrs[1][0].cpu().numpy()  # Always 1
        target_extrs = data.extrs[1][0].cpu().numpy()  # Always 1
        
        print(f"==> source frames shape: {source_frames.shape}")
        print(f"==> source depths shape: {source_depths.shape}")
        print(f"==> target frames shape: {target_frames.shape}")  # ADDED
        print(f"==> source frames range: [{source_frames.min():.3f}, {source_frames.max():.3f}]")
        print(f"==> target frames range: [{target_frames.min():.3f}, {target_frames.max():.3f}]")  # ADDED
        
        # Resize source frames and update intrinsics
        resized_source_frames, updated_source_K = self.resize_frames_and_update_intrinsics(
            source_frames, source_intrs, target_size
        )
        
        # Resize target frames (ground truth)
        resized_target_frames, updated_target_K = self.resize_frames_and_update_intrinsics(
            target_frames, target_intrs, target_size
        )
        
        # Resize depth maps
        resized_depths = self.resize_depth_maps(source_depths, target_size)
        
        return {
            'frames': resized_source_frames,
            'target_frames': resized_target_frames,  # ADDED
            'depths': resized_depths,
            'source_intrs': updated_source_K,
            'source_extrs': source_extrs,
            'target_intrs': updated_target_K,
            'target_extrs': target_extrs,
            'seq_name': data.seq_name
        }
    
    def infer_camera_poses_mvtracker(self, opts, source_view_idx=0, target_view_idx=1):
        """Inference using MVTracker dataset"""
        
        # Initialize MVTracker dataset with specific views
        print("==> initializing MVTracker dataset...")
        # dataset = PanopticStudioMultiViewDataset(
        #     '/home/azhuravl/nobackup/mvtracker_data/datasets/panoptic-multiview',
        #     traj_per_sample=384, 
        #     seed=72,
        #     max_videos=100, 
        #     perform_sanity_checks=False, 
        #     views_to_return=[source_view_idx, target_view_idx],  # Select specific views
        #     use_duster_depths=False, 
        #     clean_duster_depths=False,
        # )
        
        dataset = KubricMultiViewDataset(
            data_root = '/home/azhuravl/nobackup/mvtracker_data/datasets/kubric-multiview/test',
            seq_len = 24,
            traj_per_sample = 200,
            seed = 72,
            sample_vis_1st_frame = True,
            tune_per_scene = False,
            max_videos = 100,
            use_duster_depths = False,
            duster_views = None,
            clean_duster_depths = False,
            views_to_return = [source_view_idx, target_view_idx],  # Select specific views
            # novel_views = [20, 21],
            num_views = -1,
            depth_noise_std = 0,
        )
        
        # Get first sample
        data = dataset[0][0]
        print(f"==> loaded sequence: {data.seq_name}")
        print(f"==> video shape: {data.video.shape}")  # Should be [2, T, C, H, W]
        print(f"==> depth shape: {data.videodepth.shape}")
        print(f"==> requested views: source={source_view_idx}, target={target_view_idx}")
        
        # Prepare data (no need to pass view indices, they're always 0,1 now)
        target_size = (576, 1024)  # Height, Width
        prepared_data = self.prepare_mvtracker_data(data, target_size)
        
        # Add the original view indices to prepared data for reference
        prepared_data['source_view_idx'] = source_view_idx
        prepared_data['target_view_idx'] = target_view_idx
        
        # Limit to opts.video_length frames
        max_frames = min(opts.video_length, len(prepared_data['frames']))
        frames = prepared_data['frames'][:max_frames]
        target_frames = prepared_data['target_frames'][:max_frames]  # ADDED
        depths = prepared_data['depths'][:max_frames]
        
        print(f"==> using {max_frames} frames")
        
        # Pad if necessary
        if len(frames) < opts.video_length:
            last_frame = frames[-1:]
            last_target_frame = target_frames[-1:]  # ADDED
            last_depth = depths[-1:]
            num_pad = opts.video_length - len(frames)
            pad_frames = np.repeat(last_frame, num_pad, axis=0)
            pad_target_frames = np.repeat(last_target_frame, num_pad, axis=0)  # ADDED
            pad_depths = np.repeat(last_depth, num_pad, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
            target_frames = np.concatenate([target_frames, pad_target_frames], axis=0)  # ADDED
            depths = np.concatenate([depths, pad_depths], axis=0)
            print(f"==> padded to {opts.video_length} frames")
        
        # Get caption
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        
        # Convert depths to torch tensor and add batch dimension
        depths_tensor = torch.from_numpy(depths).unsqueeze(1).to(opts.device).float()  # [T, 1, H, W]
        
        # Convert frames to tensor
        frames_tensor = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        
        # Convert target frames to tensor (for saving ground truth)
        target_frames_tensor = (
            torch.from_numpy(target_frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        
        # Convert camera formats
        source_c2w, source_K = self.convert_camera_format_from_mvtracker(
            prepared_data['source_intrs'], prepared_data['source_extrs']
        )
        target_c2w, target_K = self.convert_camera_format_from_mvtracker(
            prepared_data['target_intrs'], prepared_data['target_extrs']
        )
        
        # Create pose matrices for all frames
        pose_s = source_c2w.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)
        pose_t = target_c2w.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)
        
        K_matrices_s = source_K.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)
        K_matrices_t = target_K.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)
        
        print(f"==> using MVTracker camera data for warping")
        
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
        self.save_mvtracker_data(prepared_data, opts.save_dir)
        
        # exit(0)  # Remove this when you want to continue with diffusion
        
                # Prepare for diffusion
        # Prepare for diffusion - FIX: use frames_tensor not frames
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
        
        # Save final comparison: generated vs ground truth vs render
        # self.save_final_comparison_video(sample[0], target_frames_tensor, cond_video, opts, prepared_data['seq_name'])
        
        return sample
        
    def save_mvtracker_data(self, data, save_dir):
        """Save MVTracker data for reference"""
        mvtracker_data = {
            "seq_name": data['seq_name'],
            "source_view_idx": data.get('source_view_idx', 0),
            "target_view_idx": data.get('target_view_idx', 1),
            "source_intrs": data['source_intrs'].tolist(),
            "target_intrs": data['target_intrs'].tolist(),
            "source_extrs": data['source_extrs'].tolist(),
            "target_extrs": data['target_extrs'].tolist(),
            "frame_shape": list(data['frames'].shape),
            "target_frame_shape": list(data['target_frames'].shape),  # ADDED
            "depth_shape": list(data['depths'].shape),
        }
        
        with open(os.path.join(save_dir, 'mvtracker_data.json'), 'w') as f:
            json.dump(mvtracker_data, f, indent=2)
        
        print(f"MVTracker data saved to {os.path.join(save_dir, 'mvtracker_data.json')}")
    
    def save_intermediate_videos(self, frames, target_frames, cond_video, cond_masks, opts, seq_name="mvtracker"):
        """Save intermediate videos for debugging"""
        # Source frames (input)
        self.save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, f'input_source_{seq_name}.mp4'),
            fps=opts.fps,
        )
        
        # Target frames (ground truth) - ADDED
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
    
    def save_comparison_video(self, rendered_frames, target_frames, opts, seq_name="mvtracker"):
        """Save side-by-side comparison of rendered vs ground truth"""
        # Ensure both videos are in [0, 1] range and on the same device
        rendered_normalized = rendered_frames.permute(0, 2, 3, 1)  # Already [0, 1]
        target_normalized = (target_frames.permute(0, 2, 3, 1) + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        
        # Ensure both tensors are on the same device
        device = rendered_normalized.device
        target_normalized = target_normalized.to(device)
        
        # Create side-by-side comparison
        comparison_frames = torch.cat([rendered_normalized, target_normalized], dim=2)  # Concatenate along width
        
        self.save_video(
            comparison_frames,
            os.path.join(opts.save_dir, f'comparison_render_vs_gt_{seq_name}.mp4'),
            fps=opts.fps,
        )

    def save_video(self, tensor, path, fps):
        """Save video tensor to file"""
        from models.utils import save_video
        save_video(tensor, path, fps=fps)


# Updated usage function
def run_mvtracker_inference(source_view_idx=0, target_view_idx=1, output_dir="./experiments/"):
    """Run inference using MVTracker dataset"""
    
    # Create minimal opts
    class Opts:
        def __init__(self, source_view_idx, target_view_idx):
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
            self.exp_name = f"{timestamp}_mvtracker_{source_view_idx}to{target_view_idx}"
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
    
    opts = Opts(source_view_idx, target_view_idx)
    os.makedirs(opts.save_dir, exist_ok=True)
    
    # Initialize model (without depth estimator)
    crafter = CameraPoseTrajCrafter(opts)
    
    # Run inference with MVTracker
    result = crafter.infer_camera_poses_mvtracker(opts, source_view_idx, target_view_idx)
    
    print(f"Generated video saved to: {opts.save_dir}")
    return result


if __name__ == "__main__":
    # Run with different view combinations
    result = run_mvtracker_inference(
        source_view_idx=4,
        target_view_idx=5
        )