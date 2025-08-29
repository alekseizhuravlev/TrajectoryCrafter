import torch
import numpy as np
import copy

from demo import TrajCrafter
from models.utils import Warper, read_video_frames
from models.infer import DepthCrafterDemo
import inference_orbits


# Cell 3: Visualization Classes
class VisualizationWarper(Warper):
    """Extended Warper class for 3D visualization"""
    
    def extract_3d_points_with_colors(
        self,
        frame1: torch.Tensor,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        intrinsic1: torch.Tensor,
        subsample_step: int = 10
    ):
        """Extract 3D world points and their corresponding colors for visualization"""
        b, c, h, w = frame1.shape
        
        # Move tensors to device
        frame1 = frame1.to(self.device).to(self.dtype)
        depth1 = depth1.to(self.device).to(self.dtype)
        transformation1 = transformation1.to(self.device).to(self.dtype)
        intrinsic1 = intrinsic1.to(self.device).to(self.dtype)
        
        # Create subsampled pixel coordinates for performance
        x_coords = torch.arange(0, w, subsample_step, dtype=torch.float32)
        y_coords = torch.arange(0, h, subsample_step, dtype=torch.float32)
        x2d, y2d = torch.meshgrid(x_coords, y_coords, indexing='xy')
        x2d = x2d.to(depth1.device)
        y2d = y2d.to(depth1.device)
        ones_2d = torch.ones_like(x2d)
        
        # Stack into homogeneous coordinates
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]
        
        # Subsample depth and colors
        depth_sub = depth1[:, 0, ::subsample_step, ::subsample_step]
        colors_sub = frame1[:, :, ::subsample_step, ::subsample_step]
        
        # Unproject to 3D camera coordinates
        intrinsic1_inv = torch.linalg.inv(intrinsic1)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]
        depth_4d = depth_sub[:, :, :, None, None]
        
        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)
        camera_points = depth_4d * unnormalized_pos
        
        # Transform to world coordinates
        ones_4d = torch.ones(b, camera_points.shape[1], camera_points.shape[2], 1, 1).to(depth1)
        world_points_homo = torch.cat([camera_points, ones_4d], dim=3)
        trans_4d = transformation1[:, None, None]
        world_points_homo = torch.matmul(trans_4d, world_points_homo)
        world_points = world_points_homo[:, :, :, :3, 0]  # (b, h_sub, w_sub, 3)
        
        # Prepare colors
        colors = colors_sub.permute(0, 2, 3, 1)  # (b, h_sub, w_sub, 3)
        
        # Filter valid points (positive depth)
        valid_mask = depth_sub > 0  # (b, h_sub, w_sub)
        
        # Flatten and filter
        points_3d = world_points[valid_mask]  # (N, 3)
        colors_rgb = colors[valid_mask]       # (N, 3)
        
        return points_3d, colors_rgb


class TrajCrafterVisualization(TrajCrafter):
    """Lightweight TrajCrafter subclass for camera trajectory visualization"""
    
    def __init__(self, opts):
        # Only initialize what we need for pose generation and depth estimation
        self.device = opts.device
        self.depth_estimater = DepthCrafterDemo(
            unet_path=opts.unet_path,
            pre_train_path=opts.pre_train_path,
            cpu_offload=opts.cpu_offload,
            device=opts.device,
        )
        print("TrajCrafterVisualization initialized (diffusion pipeline skipped)")
    
    def extract_scene_data(self, opts):
        """Extract all data needed for 3D visualization"""
        print("Reading video frames...")
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        
        print("Estimating depth...")
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        
        print("Converting frames to tensors...")
        frames_tensor = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        
        print("Generating camera poses...")
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)
        
        # Calculate scene radius
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        
        return {
            'frames_numpy': frames,
            'frames_tensor': frames_tensor,
            'depths': depths,
            'pose_source': pose_s,
            'pose_target': pose_t,
            'intrinsics': K,
            'radius': radius,
            'trajectory_params': opts.target_pose if hasattr(opts, 'target_pose') else None
        }
        
    
    def infer_gradual(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)
        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[i : i + 1],
                None,
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        return cond_video, cond_masks
        
   