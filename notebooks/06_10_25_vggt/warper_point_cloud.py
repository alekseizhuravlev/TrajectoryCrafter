from typing import Optional, Tuple, List
import torch

class GlobalPointCloudWarper:
    def __init__(self, resolution: tuple = None, device: str = 'cuda', max_points: int = 1000000):
        # super().__init__(resolution, device)
        self.max_points = max_points
        self.device = device
        self.dtype = torch.float32
        self.resolution = resolution
    
    def lift_to_3d_pointcloud(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        intrinsic1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lift 2D image points to 3D world coordinates with colors.
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w), device=self.device, dtype=self.dtype)

        # Move tensors to device and convert to proper dtype
        frame1 = frame1.to(self.device).to(self.dtype)
        mask1 = mask1.to(self.device).to(self.dtype)
        depth1 = depth1.to(self.device).to(self.dtype)
        transformation1 = transformation1.to(self.device).to(self.dtype)
        intrinsic1 = intrinsic1.to(self.device).to(self.dtype)

        # Create pixel coordinates directly on device
        x1d = torch.arange(0, w, device=self.device, dtype=self.dtype)[None]
        y1d = torch.arange(0, h, device=self.device, dtype=self.dtype)[:, None]
        x2d = x1d.repeat([h, 1])  # (h, w)
        y2d = y1d.repeat([1, w])  # (h, w)
        ones_2d = torch.ones(size=(h, w), device=self.device, dtype=self.dtype)  # (h, w)
        
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[
            None, :, :, :, None
        ]  # (1, h, w, 3, 1)

        # Rest of the function remains the same...
        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)

        unnormalized_pos = torch.matmul(
            intrinsic1_inv_4d, pos_vectors_homo
        )  # (b, h, w, 3, 1)
        
        # Get 3D points in camera coordinate system
        camera_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        
        # Transform to world coordinates
        ones_4d = torch.ones(size=(b, h, w, 1, 1), device=self.device, dtype=self.dtype)
        camera_points_homo = torch.cat([camera_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        
        # Apply inverse transformation to get world coordinates
        transformation1_inv = torch.linalg.inv(transformation1)  # (b, 4, 4)
        transformation1_inv_4d = transformation1_inv[:, None, None]  # (b, 1, 1, 4, 4)
        world_points_homo = torch.matmul(transformation1_inv_4d, camera_points_homo)  # (b, h, w, 4, 1)
        world_points = world_points_homo[:, :, :, :3, 0]  # (b, h, w, 3)
        
        # Get colors (convert from channel-first to spatial layout)
        colors = frame1.permute(0, 2, 3, 1)  # (b, h, w, 3)
        
        # Apply mask to filter out invalid points
        valid_mask = mask1[:, 0, :, :].unsqueeze(-1)  # (b, h, w, 1)
        world_points = world_points * valid_mask
        colors = colors * valid_mask
        
        return world_points, colors

    def create_pointcloud_from_image(
        self,
        frame: torch.Tensor,
        mask: Optional[torch.Tensor],
        depth: torch.Tensor,
        transformation: torch.Tensor,
        intrinsic: torch.Tensor,
        confidence_weight: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a point cloud from a single image"""
        # Ensure all inputs are on the correct device
        frame = frame.to(self.device).to(self.dtype)
        depth = depth.to(self.device).to(self.dtype)
        transformation = transformation.to(self.device).to(self.dtype)
        intrinsic = intrinsic.to(self.device).to(self.dtype)
        
        if mask is not None:
            mask = mask.to(self.device).to(self.dtype)
        
        # Lift 2D points to 3D world coordinates
        world_points, colors = self.lift_to_3d_pointcloud(
            frame, mask, depth, transformation, intrinsic
        )
        
        # Flatten to point cloud format
        b, h, w, _ = world_points.shape
        if mask is None:
            mask = torch.ones(b, 1, h, w, device=self.device, dtype=self.dtype)
            
        # Only keep valid points
        valid_mask = mask[:, 0, :, :].bool()  # (b, h, w)
        
        points = world_points[valid_mask]  # (N_valid, 3)
        point_colors = colors[valid_mask]  # (N_valid, 3)
        weights = torch.full(
            (points.shape[0], 1), 
            confidence_weight, 
            device=self.device, 
            dtype=self.dtype
        )
        
        return points, point_colors, weights
    
    def merge_pointclouds(
        self,
        point_clouds: List[torch.Tensor],
        colors_list: List[torch.Tensor], 
        weights_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge multiple point clouds into one"""
        if not point_clouds:
            return None, None, None
            
        merged_points = torch.cat(point_clouds, dim=0)
        merged_colors = torch.cat(colors_list, dim=0)
        merged_weights = torch.cat(weights_list, dim=0)
        
        return merged_points, merged_colors, merged_weights
    
    def downsample_pointcloud(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        weights: torch.Tensor,
        max_points: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Downsample a point cloud to max_points"""
        if max_points is None:
            max_points = self.max_points
            
        if points.shape[0] <= max_points:
            return points, colors, weights
            
        # Random sampling weighted by confidence
        probabilities = (weights / weights.sum()).squeeze()
        indices = torch.multinomial(probabilities, max_points, replacement=False)
        
        return points[indices], colors[indices], weights[indices]
    
    def render_from_camera(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        weights: torch.Tensor,
        transformation: torch.Tensor,
        intrinsic: torch.Tensor,
        target_height: int,
        target_width: int,
        depth_threshold: float = 100000.0,
        point_size: float = 2.0,  # Add this parameter
        use_bilinear: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render point cloud from a target camera viewpoint
        
        :param points: (N, 3) world coordinates  
        :param colors: (N, 3) RGB colors
        :param weights: (N, 1) confidence weights
        :param transformation: (b, 4, 4) target camera extrinsic matrix
        :param intrinsic: (b, 3, 3) target camera intrinsic matrix
        :param target_height: output image height
        :param target_width: output image width
        :param depth_threshold: maximum depth to render
        :return: rendered_image: (b, 3, h, w), mask: (b, 1, h, w)
        """
        if points is None or points.shape[0] == 0:
            b = transformation.shape[0]
            return (
                torch.zeros(b, 3, target_height, target_width, device=self.device),
                torch.zeros(b, 1, target_height, target_width, device=self.device)
            )
        
        b = transformation.shape[0]
        point_count = points.shape[0]
        
        # Transform world points to camera coordinates
        points_homo = torch.cat([
            points, 
            torch.ones(point_count, 1, device=self.device)
        ], dim=1)  # (N, 4)
        
        # Apply transformation for each batch item
        rendered_images = []
        rendered_masks = []
        
        for batch_idx in range(b):
            # Transform to camera space
            camera_points_homo = torch.matmul(
                transformation[batch_idx], points_homo.T
            ).T  # (N, 4)
            camera_points = camera_points_homo[:, :3]  # (N, 3)
            
            # Filter points behind camera and too far
            valid_depth = (camera_points[:, 2] > 0.1) & (camera_points[:, 2] < depth_threshold)
            valid_points = camera_points[valid_depth]
            valid_colors = colors[valid_depth]
            valid_weights = weights[valid_depth]
            
            if valid_points.shape[0] == 0:
                rendered_images.append(torch.zeros(3, target_height, target_width, device=self.device))
                rendered_masks.append(torch.zeros(1, target_height, target_width, device=self.device))
                continue
            
            # Project to 2D
            projected = torch.matmul(intrinsic[batch_idx], valid_points.T).T  # (N, 3)
            pixel_coords = projected[:, :2] / projected[:, 2:3]  # (N, 2)
            depths = projected[:, 2]  # (N,)
            
            # Filter points outside image bounds
            in_bounds = (
                (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < target_width) &
                (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < target_height)
            )
            
            final_coords = pixel_coords[in_bounds]
            final_colors = valid_colors[in_bounds]
            final_depths = depths[in_bounds]
            final_weights = valid_weights[in_bounds]
            
            # Choose splatting method
            if use_bilinear:
                rendered_image, rendered_mask = self._splat_points_bilinear_efficient(
                    final_coords, final_colors, final_depths, final_weights,
                    target_height, target_width
                )
            else:
                rendered_image, rendered_mask = self._splat_points_to_image_with_size(
                    final_coords, final_colors, final_depths, final_weights,
                    target_height, target_width, point_size=point_size
                )
            
            rendered_images.append(rendered_image)
            rendered_masks.append(rendered_mask)
        
        return (
            torch.stack(rendered_images, dim=0),
            torch.stack(rendered_masks, dim=0)
        )
    
    def _splat_points_to_image_with_size(
        self,
        coords: torch.Tensor,  # (N, 2)
        colors: torch.Tensor,  # (N, 3)
        depths: torch.Tensor,  # (N,)
        weights: torch.Tensor,  # (N, 1)
        height: int,
        width: int,
        point_size: float = 2.0  # Controllable point size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render points with controllable size to avoid holes"""
        
        if coords.shape[0] == 0:
            return (
                torch.zeros(3, height, width, device=self.device),
                torch.zeros(1, height, width, device=self.device)
            )
        
        # Depth weighting
        normalized_depths = depths / (depths.max() + 1e-8)
        depth_weights = torch.exp(-normalized_depths * 2)
        total_weights = weights.squeeze() * depth_weights
        
        # Create splat pattern based on point size
        radius = int(point_size / 2)
        offsets = []
        splat_weights = []
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist = (dx**2 + dy**2)**0.5
                if dist <= point_size / 2:
                    offsets.append([dx, dy])
                    # Gaussian falloff
                    weight = torch.exp(torch.tensor(-dist**2 / (2 * (point_size/4)**2)))
                    splat_weights.append(weight)
        
        offsets = torch.tensor(offsets, device=self.device)
        splat_weights = torch.tensor(splat_weights, device=self.device)
        
        # Expand coordinates for all splat positions
        n_points = coords.shape[0]
        n_splats = len(offsets)
        
        # Repeat coordinates for each splat offset
        expanded_coords = coords[:, None, :] + offsets[None, :, :]  # (N, n_splats, 2)
        expanded_coords = expanded_coords.view(-1, 2)  # (N*n_splats, 2)
        
        # Repeat colors and weights
        expanded_colors = colors[:, None, :].repeat(1, n_splats, 1).view(-1, 3)
        expanded_weights = total_weights[:, None].repeat(1, n_splats).view(-1)
        expanded_splat_weights = splat_weights[None, :].repeat(n_points, 1).view(-1)
        
        # Apply splat weights
        final_weights = expanded_weights * expanded_splat_weights
        
        # Filter out-of-bounds
        pixel_coords = torch.round(expanded_coords).long()
        valid_mask = (
            (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
        )
        
        if not valid_mask.any():
            return (
                torch.zeros(3, height, width, device=self.device),
                torch.zeros(1, height, width, device=self.device)
            )
        
        valid_coords = pixel_coords[valid_mask]
        valid_colors = expanded_colors[valid_mask]
        valid_weights = final_weights[valid_mask]
        
        # Render using scatter
        linear_indices = valid_coords[:, 1] * width + valid_coords[:, 0]
        
        color_buffer = torch.zeros(3, height * width, device=self.device)
        weight_buffer = torch.zeros(height * width, device=self.device)
        
        weighted_colors = valid_colors.T * valid_weights
        
        for c in range(3):
            color_buffer[c].scatter_add_(0, linear_indices, weighted_colors[c])
        
        weight_buffer.scatter_add_(0, linear_indices, valid_weights)
        
        # Normalize
        valid_pixels = weight_buffer > 1e-6
        color_buffer[:, valid_pixels] /= weight_buffer[valid_pixels]
        
        return color_buffer.view(3, height, width), valid_pixels.float().view(1, height, width)
    
    
    def _splat_points_bilinear_efficient(
        self,
        coords: torch.Tensor,  # (N, 2)
        colors: torch.Tensor,  # (N, 3)
        depths: torch.Tensor,  # (N,)
        weights: torch.Tensor,  # (N, 1)
        height: int,
        width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficient bilinear splatting matching original warper behavior"""
        
        if coords.shape[0] == 0:
            return (
                torch.zeros(3, height, width, device=self.device),
                torch.zeros(1, height, width, device=self.device)
            )
        
        # Match original depth weighting
        # sat_depth = torch.clamp(depths, min=0.1, max=1000)
        # log_depth = torch.log(1 + sat_depth)
        # depth_weights = torch.exp(log_depth / log_depth.max() * 50)
        # point_weights = weights.squeeze() / depth_weights
        
        # Alternative: gentler depth weighting that preserves distant points
        normalized_depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
        depth_weights = torch.exp(-normalized_depths * 2)  # Gentler falloff
        point_weights = weights.squeeze() * depth_weights
        
        
        # Bilinear interpolation coordinates (matches original)
        coords_floor = torch.floor(coords)
        coords_frac = coords - coords_floor
        
        # Four corner coordinates
        x0, y0 = coords_floor[:, 0].long(), coords_floor[:, 1].long()
        x1, y1 = x0 + 1, y0 + 1
        
        # Bilinear weights (exactly matches original implementation)
        dx, dy = coords_frac[:, 0], coords_frac[:, 1]
        w_nw = (1 - dx) * (1 - dy) * point_weights  # top-left
        w_ne = dx * (1 - dy) * point_weights        # top-right  
        w_sw = (1 - dx) * dy * point_weights        # bottom-left
        w_se = dx * dy * point_weights              # bottom-right
        
        # Vectorized approach: collect all coordinates and weights
        all_x = torch.cat([x0, x1, x0, x1])
        all_y = torch.cat([y0, y0, y1, y1])
        all_weights = torch.cat([w_nw, w_ne, w_sw, w_se])
        all_colors = torch.cat([colors, colors, colors, colors], dim=0)
        
        # Filter valid coordinates (within bounds and non-zero weight)
        valid_mask = (
            (all_x >= 0) & (all_x < width) &
            (all_y >= 0) & (all_y < height) &
            (all_weights > 1e-8)
        )
        
        if not valid_mask.any():
            return (
                torch.zeros(3, height, width, device=self.device),
                torch.zeros(1, height, width, device=self.device)
            )
        
        # Extract valid data
        valid_x = all_x[valid_mask]
        valid_y = all_y[valid_mask]
        valid_weights = all_weights[valid_mask]
        valid_colors = all_colors[valid_mask]
        
        # Convert to linear indices for scatter operations
        linear_indices = valid_y * width + valid_x
        
        # GPU-efficient scatter accumulation
        color_buffer = torch.zeros(3, height * width, device=self.device)
        weight_buffer = torch.zeros(height * width, device=self.device)
        
        # Vectorized color accumulation
        weighted_colors = valid_colors.T * valid_weights  # (3, N_valid)
        
        for c in range(3):
            color_buffer[c].scatter_add_(0, linear_indices, weighted_colors[c])
        
        weight_buffer.scatter_add_(0, linear_indices, valid_weights)
        
        # Normalize colors by accumulated weights
        valid_pixels = weight_buffer > 1e-8
        color_buffer[:, valid_pixels] /= weight_buffer[valid_pixels]
        
        # Reshape back to 2D
        color_image = color_buffer.view(3, height, width)
        mask_image = valid_pixels.float().view(1, height, width)
        
        return color_image, mask_image
        
    def render_pointcloud_zbuffer_vectorized(
        self,
        points_3d: torch.Tensor,
        colors_3d: torch.Tensor,
        transformation_target: torch.Tensor,
        intrinsic_target: torch.Tensor,
        image_size: tuple = (576, 1024)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fully vectorized Z-buffer rendering - no loops!
        """
        
        h, w = image_size
        device = self.device
        
        # Transform and project (same as before)
        ones = torch.ones(points_3d.shape[0], 1, device=device, dtype=self.dtype)
        world_points_homo = torch.cat([points_3d, ones], dim=1)
        camera_points_homo = torch.matmul(transformation_target[0], world_points_homo.T).T
        camera_points = camera_points_homo[:, :3]
        
        projected_homo = torch.matmul(intrinsic_target[0], camera_points.T).T
        pixel_coords = projected_homo[:, :2] / projected_homo[:, 2:3]
        depths_vals = camera_points[:, 2]
        
        # Filter valid points
        valid_mask = (depths_vals > 0.01) & \
                    (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) & \
                    (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h)
        
        if valid_mask.sum() == 0:
            return (torch.full((1, 3, h, w), -1.0, device=device),
                    torch.zeros(1, 1, h, w, device=device))
        
        valid_coords = pixel_coords[valid_mask]
        valid_colors = colors_3d[valid_mask]
        valid_depths = depths_vals[valid_mask]
        
        # Convert to integer coordinates
        x_int = torch.clamp(torch.round(valid_coords[:, 0]).long(), 0, w-1)
        y_int = torch.clamp(torch.round(valid_coords[:, 1]).long(), 0, h-1)
        linear_indices = y_int * w + x_int
        
        # Vectorized Z-buffer using scatter_reduce (PyTorch 1.12+)
        # This finds the minimum depth per pixel and corresponding colors
        
        # Method 1: Use unique + scatter for closest point per pixel
        unique_indices, inverse_indices = torch.unique(linear_indices, return_inverse=True)
        
        # For each unique pixel, find the point with minimum depth
        min_depths = torch.full((len(unique_indices),), float('inf'), device=device)
        min_depths.scatter_reduce_(0, inverse_indices, valid_depths, reduce='amin')
        
        # Create mask for points that have minimum depth at their pixel
        expanded_min_depths = min_depths[inverse_indices]
        closest_mask = (valid_depths == expanded_min_depths)
        
        # Keep only the closest points
        final_indices = linear_indices[closest_mask]
        final_colors = valid_colors[closest_mask]
        final_depths = valid_depths[closest_mask]
        
        # Render final result
        color_buffer = torch.full((3, h * w), -1.0, device=device)
        depth_buffer = torch.full((h * w,), 0.0, device=device)
        
        # Scatter the closest colors (no conflicts now since we filtered to closest only)
        color_buffer[:, final_indices] = final_colors.T
        depth_buffer[final_indices] = final_depths
        
        # Reshape
        final_frame = color_buffer.view(3, h, w).unsqueeze(0)
        final_mask = (depth_buffer > 0).float().view(1, 1, h, w)
        
        return final_frame, final_mask
    
    
    def render_pointcloud_zbuffer_vectorized_point_size(
        self,
        points_3d: torch.Tensor,
        colors_3d: torch.Tensor,
        transformation_target: torch.Tensor,
        intrinsic_target: torch.Tensor,
        image_size: tuple = (576, 1024),
        point_size: int = 1,
        return_depth: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized Z-buffer rendering with proper color handling for larger points
        """
        
        h, w = image_size
        device = self.device
        
        # Transform and project (same as before)
        ones = torch.ones(points_3d.shape[0], 1, device=device, dtype=self.dtype)
        world_points_homo = torch.cat([points_3d, ones], dim=1)
        camera_points_homo = torch.matmul(transformation_target[0], world_points_homo.T).T
        camera_points = camera_points_homo[:, :3]
        
        projected_homo = torch.matmul(intrinsic_target[0], camera_points.T).T
        pixel_coords = projected_homo[:, :2] / projected_homo[:, 2:3]
        depths_vals = camera_points[:, 2]
        
        # Filter valid points
        valid_mask = (depths_vals > 0.01) & \
                    (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) & \
                    (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h)
        
        if valid_mask.sum() == 0:
            return (torch.full((1, 3, h, w), -1.0, device=device),
                    torch.zeros(1, 1, h, w, device=device))
        
        valid_coords = pixel_coords[valid_mask]
        valid_colors = colors_3d[valid_mask]
        valid_depths = depths_vals[valid_mask]
        
        # Generate splat pattern - FIXED VERSION
        if point_size == 1:
            splat_offsets = torch.tensor([[0, 0]], device=device)
            splat_weights = torch.tensor([1.0], device=device)
        else:
            radius = point_size // 2
            offsets = []
            weights = []
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    # Use square pattern with uniform weights
                    if abs(dx) <= radius and abs(dy) <= radius:
                        offsets.append([dx, dy])
                        
                        # OPTION 1: Uniform weights (prevents graying)
                        weights.append(1.0)
                        
                        # OPTION 2: Gentle falloff that preserves brightness
                        # dist = max(abs(dx), abs(dy))  # Square distance
                        # weight = 1.0 - (dist / (radius + 1)) * 0.3  # Only 30% falloff
                        # weights.append(weight)
            
            splat_offsets = torch.tensor(offsets, device=device)
            splat_weights = torch.tensor(weights, device=device)
        
        n_points = len(valid_coords)
        n_splats = len(splat_offsets)
        
        # Expand points for each splat offset
        expanded_coords = valid_coords[:, None, :] + splat_offsets[None, :, :]
        expanded_coords = expanded_coords.reshape(-1, 2)
        
        # Repeat colors and depths
        expanded_colors = valid_colors[:, None, :].repeat(1, n_splats, 1).reshape(-1, 3)
        expanded_depths = valid_depths[:, None].repeat(1, n_splats).reshape(-1)
        
        # DON'T multiply colors by splat weights - keep original intensity
        # expanded_colors = expanded_colors * expanded_splat_weights[:, None]  # REMOVE THIS
        
        # Convert to integer coordinates
        x_int = torch.round(expanded_coords[:, 0]).long()
        y_int = torch.round(expanded_coords[:, 1]).long()
        
        bounds_mask = (x_int >= 0) & (x_int < w) & (y_int >= 0) & (y_int < h)
        
        if bounds_mask.sum() == 0:
            return (torch.full((1, 3, h, w), -1.0, device=device),
                    torch.zeros(1, 1, h, w, device=device))
        
        final_x = x_int[bounds_mask]
        final_y = y_int[bounds_mask]
        final_colors = expanded_colors[bounds_mask]
        final_depths = expanded_depths[bounds_mask]
        
        linear_indices = final_y * w + final_x
        
        # Z-buffer logic (same as before)
        unique_indices, inverse_indices = torch.unique(linear_indices, return_inverse=True)
        
        min_depths = torch.full((len(unique_indices),), float('inf'), device=device)
        min_depths.scatter_reduce_(0, inverse_indices, final_depths, reduce='amin')
        
        expanded_min_depths = min_depths[inverse_indices]
        closest_mask = (final_depths == expanded_min_depths)
        
        winner_indices = linear_indices[closest_mask]
        winner_colors = final_colors[closest_mask]
        winner_depths = final_depths[closest_mask]
        
        # Render buffers
        color_buffer = torch.full((3, h * w), -1.0, device=device)
        depth_buffer = torch.full((h * w,), 0.0, device=device)
        
        color_buffer[:, winner_indices] = winner_colors.T
        depth_buffer[winner_indices] = winner_depths
        
        # Reshape
        final_frame = color_buffer.view(3, h, w).unsqueeze(0)
        final_mask = (depth_buffer > 0).float().view(1, 1, h, w)
        
        if return_depth:
            return final_frame, final_mask, depth_buffer.view(1, 1, h, w)
        else:
            return final_frame, final_mask



    def render_pointcloud_zbuffer_vectorized_fixed(
        self,
        points_3d: torch.Tensor,
        colors_3d: torch.Tensor,
        transformation_target: torch.Tensor,
        intrinsic_target: torch.Tensor,
        image_size: tuple = (576, 1024),
        point_size: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fixed vectorized Z-buffer rendering that works for all point sizes
        """
        
        h, w = image_size
        device = self.device
        
        # Transform and project
        ones = torch.ones(points_3d.shape[0], 1, device=device, dtype=self.dtype)
        world_points_homo = torch.cat([points_3d, ones], dim=1)
        camera_points_homo = torch.matmul(transformation_target[0], world_points_homo.T).T
        camera_points = camera_points_homo[:, :3]
        
        projected_homo = torch.matmul(intrinsic_target[0], camera_points.T).T
        pixel_coords = projected_homo[:, :2] / projected_homo[:, 2:3]
        depths_vals = camera_points[:, 2]
        
        # Filter valid points - ONLY filter points behind camera and in bounds
        valid_mask = (depths_vals > 0.01) & \
                    (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) & \
                    (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h)
        
        if valid_mask.sum() == 0:
            return (torch.full((1, 3, h, w), -1.0, device=device),
                    torch.zeros(1, 1, h, w, device=device))
        
        valid_coords = pixel_coords[valid_mask]
        valid_colors = colors_3d[valid_mask]
        valid_depths = depths_vals[valid_mask]
        
        # Handle point splatting
        if point_size <= 1:
            # Simple case - no splatting needed
            x_int = torch.clamp(torch.round(valid_coords[:, 0]).long(), 0, w-1)
            y_int = torch.clamp(torch.round(valid_coords[:, 1]).long(), 0, h-1)
            linear_indices = y_int * w + x_int
            
            final_indices = linear_indices
            final_colors = valid_colors
            final_depths = valid_depths
            
        else:
            # Generate splat pattern
            radius = point_size // 2
            offsets = []
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) <= radius and abs(dy) <= radius:
                        offsets.append([dx, dy])
            
            splat_offsets = torch.tensor(offsets, device=device)
            n_points = len(valid_coords)
            n_splats = len(splat_offsets)
            
            # Expand points for each splat offset
            expanded_coords = valid_coords[:, None, :] + splat_offsets[None, :, :]
            expanded_coords = expanded_coords.reshape(-1, 2)
            
            # Repeat colors and depths (keep original colors - no graying)
            expanded_colors = valid_colors[:, None, :].repeat(1, n_splats, 1).reshape(-1, 3)
            expanded_depths = valid_depths[:, None].repeat(1, n_splats).reshape(-1)
            
            # Convert to integer coordinates and filter bounds
            x_int = torch.round(expanded_coords[:, 0]).long()
            y_int = torch.round(expanded_coords[:, 1]).long()
            
            bounds_mask = (x_int >= 0) & (x_int < w) & (y_int >= 0) & (y_int < h)
            
            if bounds_mask.sum() == 0:
                return (torch.full((1, 3, h, w), -1.0, device=device),
                        torch.zeros(1, 1, h, w, device=device))
            
            x_int = x_int[bounds_mask]
            y_int = y_int[bounds_mask]
            final_colors = expanded_colors[bounds_mask]
            final_depths = expanded_depths[bounds_mask]
            linear_indices = y_int * w + x_int
            
            final_indices = linear_indices
        
        # FIXED Z-buffer logic: proper depth testing without losing points
        # Initialize buffers with far depth values
        color_buffer = torch.full((3, h * w), -1.0, device=device)
        depth_buffer = torch.full((h * w,), float('inf'), device=device)
        
        # Process all points and keep only the closest at each pixel
        for i in range(len(final_indices)):
            pixel_idx = final_indices[i]
            point_depth = final_depths[i]
            point_color = final_colors[i]
            
            # Only update if this point is closer
            if point_depth < depth_buffer[pixel_idx]:
                depth_buffer[pixel_idx] = point_depth
                color_buffer[:, pixel_idx] = point_color
        
        # Create mask for pixels that were rendered (finite depth)
        rendered_mask = (depth_buffer < float('inf'))
        
        # Set unrendered pixels to have 0 depth in final output
        depth_buffer[~rendered_mask] = 0.0
        
        # Reshape to final format
        final_frame = color_buffer.view(3, h, w).unsqueeze(0)
        final_mask = rendered_mask.float().view(1, 1, h, w)
        
        return final_frame, final_mask