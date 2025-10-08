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
        # x2d, y2d = torch.meshgrid(x_coords, y_coords, indexing='ij')
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
        
        # print('depth_4d', depth_4d.shape)
        # print('unnormalized_pos', unnormalized_pos.shape)
        
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
    
    
    def render_pointcloud_native(
        self,
        points_3d: torch.Tensor,
        colors_3d: torch.Tensor,
        transformation_target: torch.Tensor,
        intrinsic_target: torch.Tensor,
        image_size: tuple = (576, 1024),
        mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Native point cloud rendering using scatter operations, designed for sparse point data.
        
        Args:
            points_3d: (N, 3) 3D points in world coordinates
            colors_3d: (N, 3) RGB colors for each point [-1, 1] range
            transformation_target: (b, 4, 4) target camera extrinsic matrix
            intrinsic_target: (b, 3, 3) target camera intrinsic matrix
            image_size: (height, width) output image size
            mask: whether to clean rendered points
            
        Returns:
            rendered_frame: (b, 3, h, w) rendered image in [-1, 1] range
            rendered_mask: (b, 1, h, w) validity mask
        """
        h, w = image_size
        b = transformation_target.shape[0]
        device = self.device
        
        # Ensure tensors are on correct device and dtype
        points_3d = points_3d.to(device).to(self.dtype)
        colors_3d = colors_3d.to(device).to(self.dtype)
        transformation_target = transformation_target.to(device).to(self.dtype)
        intrinsic_target = intrinsic_target.to(device).to(self.dtype)
        
        # print(points_3d.device)
        # print(colors_3d.device)
        
        # Step 1: Transform world points to target camera coordinates
        # Use the same transformation logic as compute_transformed_points
        transformation = transformation_target  # Already in target camera frame
        transformation_inv = torch.linalg.inv(transformation)  # World to camera
        
        # Convert to homogeneous coordinates
        ones = torch.ones(points_3d.shape[0], 1, device=device, dtype=self.dtype)
        points_homo = torch.cat([points_3d, ones], dim=1)  # (N, 4)
        
        # Transform to camera coordinates (same logic as compute_transformed_points)
        camera_points_homo = torch.matmul(transformation_inv[0], points_homo.T).T  # (N, 4)
        camera_points = camera_points_homo[:, :3]  # (N, 3)
        
        # Step 2: Filter points behind camera (same as original)
        valid_depth_mask = camera_points[:, 2] > 0.01
        valid_camera_points = camera_points[valid_depth_mask]
        valid_colors = colors_3d[valid_depth_mask]
        
        if valid_camera_points.shape[0] == 0:
            return (torch.full((b, 3, h, w), -1.0, device=device, dtype=self.dtype),
                    torch.zeros(b, 1, h, w, device=device, dtype=self.dtype))
        
        # Step 3: Project to 2D (same as compute_transformed_points)
        projected_homo = torch.matmul(intrinsic_target[0], valid_camera_points.T).T  # (N, 3)
        pixel_coords = projected_homo[:, :2] / projected_homo[:, 2:3]  # (N, 2)
        depths = valid_camera_points[:, 2]  # (N,)
        
        # Step 4: Filter points within image bounds
        x_coords = pixel_coords[:, 0]
        y_coords = pixel_coords[:, 1]
        in_bounds_mask = ((x_coords >= 0) & (x_coords < w) & 
                         (y_coords >= 0) & (y_coords < h))
        
        final_coords = pixel_coords[in_bounds_mask]  # (M, 2)
        final_colors = valid_colors[in_bounds_mask]  # (M, 3)
        final_depths = depths[in_bounds_mask]  # (M,)
        
        if final_coords.shape[0] == 0:
            return (torch.full((b, 3, h, w), -1.0, device=device, dtype=self.dtype),
                    torch.zeros(b, 1, h, w, device=device, dtype=self.dtype))
        
        # Step 5: Native point splatting with depth-based weighting
        # Use the same depth weighting logic as bilinear_splatting
        sat_depths = torch.clamp(final_depths, min=0, max=1000)
        log_depths = torch.log(1 + sat_depths)
        depth_weights = torch.exp(log_depths / log_depths.max() * 50)
        
        # Initialize output buffers
        rendered_frame = torch.full((h, w, 3), -1.0, device=device, dtype=self.dtype)
        rendered_weights = torch.zeros(h, w, device=device, dtype=self.dtype)
        
        # Step 6: Scatter points using bilinear interpolation
        # Get integer coordinates for bilinear interpolation
        x_floor = torch.floor(final_coords[:, 0]).long()
        y_floor = torch.floor(final_coords[:, 1]).long()
        x_ceil = x_floor + 1
        y_ceil = y_floor + 1
        
        # Calculate fractional parts for interpolation weights
        x_frac = final_coords[:, 0] - x_floor.float()
        y_frac = final_coords[:, 1] - y_floor.float()
        
        # Bilinear interpolation weights (same logic as prox_weight in original)
        weight_nw = (1 - x_frac) * (1 - y_frac) / depth_weights
        weight_ne = x_frac * (1 - y_frac) / depth_weights
        weight_sw = (1 - x_frac) * y_frac / depth_weights
        weight_se = x_frac * y_frac / depth_weights
        
        # Clamp coordinates to image bounds
        x_floor = torch.clamp(x_floor, 0, w - 1)
        y_floor = torch.clamp(y_floor, 0, h - 1)
        x_ceil = torch.clamp(x_ceil, 0, w - 1)
        y_ceil = torch.clamp(y_ceil, 0, h - 1)
        
        # Scatter colors to four corners (same accumulation logic as original)
        # for i in range(final_coords.shape[0]):
        #     color = final_colors[i]
            
        #     # NW corner
        #     rendered_frame[y_floor[i], x_floor[i]] += color * weight_nw[i]
        #     rendered_weights[y_floor[i], x_floor[i]] += weight_nw[i]
            
        #     # NE corner  
        #     rendered_frame[y_floor[i], x_ceil[i]] += color * weight_ne[i]
        #     rendered_weights[y_floor[i], x_ceil[i]] += weight_ne[i]
            
        #     # SW corner
        #     rendered_frame[y_ceil[i], x_floor[i]] += color * weight_sw[i]
        #     rendered_weights[y_ceil[i], x_floor[i]] += weight_sw[i]
            
        #     # SE corner
        #     rendered_frame[y_ceil[i], x_ceil[i]] += color * weight_se[i]
        #     rendered_weights[y_ceil[i], x_ceil[i]] += weight_se[i]
        
        flat_indices_nw = y_floor * w + x_floor
        flat_indices_ne = y_floor * w + x_ceil
        flat_indices_sw = y_ceil * w + x_floor  
        flat_indices_se = y_ceil * w + x_ceil
        
        # Flatten buffers
        flat_frame = rendered_frame.view(-1, 3)  # (H*W, 3)
        flat_weights = rendered_weights.view(-1)  # (H*W,)
        
        # Scatter add all corners at once
        flat_frame.index_add_(0, flat_indices_nw, final_colors * weight_nw.unsqueeze(-1))
        flat_frame.index_add_(0, flat_indices_ne, final_colors * weight_ne.unsqueeze(-1))
        flat_frame.index_add_(0, flat_indices_sw, final_colors * weight_sw.unsqueeze(-1))
        flat_frame.index_add_(0, flat_indices_se, final_colors * weight_se.unsqueeze(-1))
        
        flat_weights.index_add_(0, flat_indices_nw, weight_nw)
        flat_weights.index_add_(0, flat_indices_ne, weight_ne)
        flat_weights.index_add_(0, flat_indices_sw, weight_sw)
        flat_weights.index_add_(0, flat_indices_se, weight_se)
        
        # Reshape back
        rendered_frame = flat_frame.view(h, w, 3)
        rendered_weights = flat_weights.view(h, w)
        
        # Step 7: Normalize by weights (same as original)
        valid_mask = rendered_weights > 0
        normalized_frame = torch.full((h, w, 3), -1.0, device=device, dtype=self.dtype)
        normalized_frame[valid_mask] = rendered_frame[valid_mask] / rendered_weights[valid_mask].unsqueeze(-1)
        
        # Clamp to valid range (same as original is_image=True logic)
        normalized_frame = torch.clamp(normalized_frame, min=-1, max=1)
        
        # Convert to batch format (b, c, h, w)
        output_frame = normalized_frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, h, w)
        output_mask = valid_mask.unsqueeze(0).unsqueeze(0).to(self.dtype)  # (1, 1, h, w)
        
        # Expand to batch size if needed
        if b > 1:
            output_frame = output_frame.repeat(b, 1, 1, 1)
            output_mask = output_mask.repeat(b, 1, 1, 1)
        
        # Apply cleaning if requested (same as original)
        if mask:
            output_frame, output_mask = self.clean_points(output_frame, output_mask)
        
        return output_frame, output_mask



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
        

