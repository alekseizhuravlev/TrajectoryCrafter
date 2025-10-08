import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
import sys

# Add your dataset path
sys.path.insert(0, '/home/azhuravl/work/TrajectoryCrafter/notebooks/09_09_25_multiview')
from iphone_original_dataset import iPhoneDataset

class CameraProjectionDebugger:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def project_image_to_3d(self, rgb_image, depth_map, K, c2w):
        """Project RGB-D image to 3D point cloud"""
        H, W = depth_map.shape
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten()
        v = v.flatten()
        depth = depth_map.flatten()
        
        # Remove invalid depth values
        valid_mask = (depth > 0) & (depth < 1000)  # Adjust thresholds as needed
        u = u[valid_mask]
        v = v[valid_mask] 
        depth = depth[valid_mask]
        
        print(f"Valid pixels: {len(u)}/{H*W} ({100*len(u)/(H*W):.1f}%)")
        print(f"Depth range: {depth.min():.3f} - {depth.max():.3f}")
        
        # Unproject to camera coordinates
        x_cam = (u - K[0, 2]) * depth / K[0, 0]  # (u - cx) * z / fx
        y_cam = (v - K[1, 2]) * depth / K[1, 1]  # (v - cy) * z / fy
        z_cam = depth
        
        # Stack to homogeneous coordinates
        points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=0)  # [4, N]
        
        # Transform to world coordinates
        points_world = c2w @ points_cam  # [4, N]
        points_world = points_world[:3]  # [3, N] - remove homogeneous coordinate
        
        # Get corresponding colors
        rgb_flat = rgb_image.reshape(-1, 3)[valid_mask]  # [N, 3]
        
        return points_world.T, rgb_flat  # [N, 3], [N, 3]
    
    def render_from_camera(self, points_3d, colors, K, c2w, image_size):
        """Render 3D points from given camera viewpoint"""
        H, W = image_size
        
        # Transform world points to camera coordinates
        w2c = np.linalg.inv(c2w)
        points_cam = w2c @ np.column_stack([points_3d, np.ones(len(points_3d))]).T  # [4, N]
        points_cam = points_cam[:3]  # [3, N]
        
        # Project to image plane
        x_proj = K[0, 0] * points_cam[0] / points_cam[2] + K[0, 2]  # fx * x/z + cx
        y_proj = K[1, 1] * points_cam[1] / points_cam[2] + K[1, 2]  # fy * y/z + cy
        z_proj = points_cam[2]
        
        # Filter points behind camera and outside image bounds
        valid_mask = (z_proj > 0) & (x_proj >= 0) & (x_proj < W) & (y_proj >= 0) & (y_proj < H)
        
        print(f"Projectable points: {valid_mask.sum()}/{len(points_3d)} ({100*valid_mask.sum()/len(points_3d):.1f}%)")
        
        x_proj = x_proj[valid_mask]
        y_proj = y_proj[valid_mask]
        z_proj = z_proj[valid_mask]
        colors_valid = colors[valid_mask]
        
        # Create rendered image using Z-buffer
        rendered_image = np.zeros((H, W, 3), dtype=np.uint8)
        depth_buffer = np.full((H, W), np.inf)
        
        # Convert to integer pixel coordinates
        x_int = np.round(x_proj).astype(int)
        y_int = np.round(y_proj).astype(int)
        
        # Render points (simple nearest neighbor)
        for i in range(len(x_int)):
            x, y, z = x_int[i], y_int[i], z_proj[i]
            if 0 <= x < W and 0 <= y < H and z < depth_buffer[y, x]:
                depth_buffer[y, x] = z
                rendered_image[y, x] = colors_valid[i]
        
        return rendered_image
    
    def debug_camera_projection(self, frame_id, visualize=True):
        """Debug camera projection for a specific frame"""
        print(f"\n=== Debugging Frame {frame_id} ===")
        
        # Get frame data
        frame_data = self.dataset.get_frame_by_id(frame_id)
        
        # Source camera (camera 0 with depth)
        source_cam = frame_data['cameras'][0]
        source_rgb = source_cam['rgb']
        source_depth = source_cam['depth']
        source_K = source_cam['K']
        source_c2w = source_cam['c2w']
        
        print(f"Source RGB shape: {source_rgb.shape}")
        print(f"Source depth shape: {source_depth.shape}")
        print(f"Source K:\n{source_K}")
        print(f"Source position: {source_c2w[:3, 3]}")
        
        # Project to 3D
        points_3d, colors = self.project_image_to_3d(
            source_rgb, source_depth, source_K, source_c2w
        )
        
        print(f"Generated {len(points_3d)} 3D points")
        print(f"3D points range:")
        print(f"  X: {points_3d[:, 0].min():.3f} - {points_3d[:, 0].max():.3f}")
        print(f"  Y: {points_3d[:, 1].min():.3f} - {points_3d[:, 1].max():.3f}")
        print(f"  Z: {points_3d[:, 2].min():.3f} - {points_3d[:, 2].max():.3f}")
        
        # Render from target cameras
        results = {'source': source_rgb}
        
        for target_cam_id in [1, 2]:
            if target_cam_id in frame_data['cameras']:
                target_cam = frame_data['cameras'][target_cam_id]
                target_rgb_gt = target_cam['rgb']
                target_K = target_cam['K'] 
                target_c2w = target_cam['c2w']
                
                print(f"\nTarget camera {target_cam_id}:")
                print(f"  Position: {target_c2w[:3, 3]}")
                print(f"  GT RGB shape: {target_rgb_gt.shape}")
                
                # Render from target viewpoint
                rendered = self.render_from_camera(
                    points_3d, colors, target_K, target_c2w, target_rgb_gt.shape[:2]
                )
                
                results[f'target_{target_cam_id}_gt'] = target_rgb_gt
                results[f'target_{target_cam_id}_rendered'] = rendered
                
                # Calculate distance between cameras
                camera_distance = np.linalg.norm(target_c2w[:3, 3] - source_c2w[:3, 3])
                print(f"  Distance from source: {camera_distance:.3f}")
        
        if visualize:
            self.visualize_results(results, frame_id)
        
        return results
    
    def visualize_results(self, results, frame_id):
        """Visualize rendering results"""
        n_views = len([k for k in results.keys() if 'target' in k and 'gt' in k])
        
        fig, axes = plt.subplots(2, n_views + 1, figsize=(5 * (n_views + 1), 10))
        if n_views == 1:
            axes = axes.reshape(2, -1)
        
        # Source image
        axes[0, 0].imshow(results['source'])
        axes[0, 0].set_title('Source Camera (Camera 0)')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')  # Empty bottom-left
        
        # Target views
        col_idx = 1
        for target_cam_id in [1, 2]:
            gt_key = f'target_{target_cam_id}_gt'
            rendered_key = f'target_{target_cam_id}_rendered'
            
            if gt_key in results and rendered_key in results:
                # Ground truth
                axes[0, col_idx].imshow(results[gt_key])
                axes[0, col_idx].set_title(f'Ground Truth Camera {target_cam_id}')
                axes[0, col_idx].axis('off')
                
                # Rendered
                axes[1, col_idx].imshow(results[rendered_key])
                axes[1, col_idx].set_title(f'Rendered Camera {target_cam_id}')
                axes[1, col_idx].axis('off')
                
                col_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'camera_debug_frame_{frame_id:05d}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved as 'camera_debug_frame_{frame_id:05d}.png'")
    
    def test_camera_conventions(self, frame_id):
        """Test different camera matrix conventions"""
        print(f"\n=== Testing Camera Conventions for Frame {frame_id} ===")
        
        frame_data = self.dataset.get_frame_by_id(frame_id)
        
        # Get raw camera parameters
        source_params = frame_data['cameras'][0]['camera_params']
        target_params = frame_data['cameras'][1]['camera_params']
        
        print("Raw camera parameters:")
        print(f"Source position: {source_params['position']}")
        print(f"Target position: {target_params['position']}")
        
        # Test different parsing methods
        conventions = {
            'current': self._parse_current_convention,
            'direct_c2w': self._parse_direct_c2w,
            'w2c_format': self._parse_w2c_format,
        }
        
        for name, parse_func in conventions.items():
            print(f"\n--- Convention: {name} ---")
            
            source_K, source_c2w = parse_func(source_params)
            target_K, target_c2w = parse_func(target_params)
            
            print(f"Source c2w position: {source_c2w[:3, 3]}")
            print(f"Target c2w position: {target_c2w[:3, 3]}")
            
            distance = np.linalg.norm(target_c2w[:3, 3] - source_c2w[:3, 3])
            print(f"Camera distance: {distance:.3f}")
            
            # Check if cameras are looking in reasonable directions
            source_forward = source_c2w[:3, 2]  # Camera forward direction
            target_forward = target_c2w[:3, 2]
            print(f"Source forward: {source_forward}")
            print(f"Target forward: {target_forward}")
    
    def _parse_current_convention(self, camera_params):
        """Current parsing from dataset"""
        focal_length = camera_params['focal_length']
        principal_point = camera_params['principal_point']
        position = np.array(camera_params['position'])
        orientation = np.array(camera_params['orientation'])
        
        K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        R = orientation
        t = position
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        c2w = np.linalg.inv(c2w)
        
        return K, c2w
    
    def _parse_direct_c2w(self, camera_params):
        """Direct c2w interpretation"""
        focal_length = camera_params['focal_length']
        principal_point = camera_params['principal_point']
        position = np.array(camera_params['position'])
        orientation = np.array(camera_params['orientation'])
        
        K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = orientation
        c2w[:3, 3] = position
        
        return K, c2w
    
    def _parse_w2c_format(self, camera_params):
        """World-to-camera format interpretation"""
        focal_length = camera_params['focal_length']
        principal_point = camera_params['principal_point']
        position = np.array(camera_params['position'])
        orientation = np.array(camera_params['orientation'])
        
        K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assume iPhone gives w2c, convert to c2w
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = orientation
        w2c[:3, 3] = position
        c2w = np.linalg.inv(w2c)
        
        return K, c2w
    
    def debug_vertical_alignment(self, frame_id):
        """Specifically debug vertical alignment issues"""
        frame_data = self.dataset.get_frame_by_id(frame_id)
        
        source_cam = frame_data['cameras'][0]
        target_cam = frame_data['cameras'][1]
        
        # Get 3D points
        points_3d, colors = self.project_image_to_3d(
            source_cam['rgb'], source_cam['depth'], 
            source_cam['K'], source_cam['c2w']
        )
        
        # Test different vertical adjustments
        vertical_offsets = [-300, -200, -100, 0, 100, 200, 300]
        
        results = {}
        for offset in vertical_offsets:
            # Modify target camera K matrix
            K_adjusted = target_cam['K'].copy()
            K_adjusted[1, 2] += offset  # Adjust cy
            
            rendered = self.render_from_camera(
                points_3d, colors, K_adjusted, target_cam['c2w'],
                target_cam['rgb'].shape[:2]
            )
            
            results[f'offset_{offset}'] = rendered
        
        # Visualize all offsets
        fig, axes = plt.subplots(2, len(vertical_offsets)//2 + 1, figsize=(20, 8))
        axes = axes.flatten()
        
        # Ground truth
        axes[0].imshow(target_cam['rgb'])
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Different offsets
        for i, offset in enumerate(vertical_offsets):
            axes[i+1].imshow(results[f'offset_{offset}'])
            axes[i+1].set_title(f'Offset: {offset}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'vertical_alignment_debug_{frame_id}.png', dpi=150)
        plt.show()
        
        return results

    def debug_coordinate_systems(self, frame_id):
        """Test different coordinate system conventions"""
        print(f"\n=== Testing Coordinate Systems for Frame {frame_id} ===")
        
        frame_data = self.dataset.get_frame_by_id(frame_id)
        source_cam = frame_data['cameras'][0]
        target_cam = frame_data['cameras'][1]
        
        # Get 3D points from source
        points_3d, colors = self.project_image_to_3d(
            source_cam['rgb'], source_cam['depth'], 
            source_cam['K'], source_cam['c2w']
        )
        
        # Test different coordinate system transformations
        coord_systems = {
            'original': self._get_original_matrices,
            'flip_y': self._get_flip_y_matrices,
            'flip_z': self._get_flip_z_matrices,
            'flip_yz': self._get_flip_yz_matrices,
            'opengl_to_cv': self._get_opengl_to_cv_matrices,
            'principal_point_flip': self._get_principal_point_flip_matrices,
        }
        
        results = {'ground_truth': target_cam['rgb']}
        
        for name, matrix_func in coord_systems.items():
            print(f"\nTesting {name}...")
            
            try:
                # Get modified matrices
                source_K_mod, source_c2w_mod, target_K_mod, target_c2w_mod = matrix_func(
                    source_cam, target_cam
                )
                
                # Re-project points with modified source camera
                points_3d_mod, colors_mod = self.project_image_to_3d(
                    source_cam['rgb'], source_cam['depth'], 
                    source_K_mod, source_c2w_mod
                )
                
                # Render with modified target camera
                rendered = self.render_from_camera(
                    points_3d_mod, colors_mod, target_K_mod, target_c2w_mod,
                    target_cam['rgb'].shape[:2]
                )
                
                results[name] = rendered
                
                # Print camera info
                print(f"  Source pos: {source_c2w_mod[:3, 3]}")
                print(f"  Target pos: {target_c2w_mod[:3, 3]}")
                print(f"  Distance: {np.linalg.norm(target_c2w_mod[:3, 3] - source_c2w_mod[:3, 3]):.3f}")
                
            except Exception as e:
                print(f"  Error with {name}: {e}")
                results[name] = np.zeros_like(target_cam['rgb'])
        
        # Visualize all coordinate systems
        self._visualize_coordinate_systems(results, frame_id)
        
        return results
    
    def _get_original_matrices(self, source_cam, target_cam):
        """Original matrices (no modification)"""
        return (source_cam['K'], source_cam['c2w'], 
                target_cam['K'], target_cam['c2w'])
    
    def _get_flip_y_matrices(self, source_cam, target_cam):
        """Flip Y coordinate system"""
        # Transformation matrix to flip Y
        flip_y = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],  # Flip Y
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ])
        
        # Apply to camera matrices
        source_c2w_mod = flip_y @ source_cam['c2w'] @ np.linalg.inv(flip_y)
        target_c2w_mod = flip_y @ target_cam['c2w'] @ np.linalg.inv(flip_y)
        
        # Modify intrinsics - flip cy
        source_K_mod = source_cam['K'].copy()
        target_K_mod = target_cam['K'].copy()
        H = source_cam['rgb'].shape[0]
        source_K_mod[1, 2] = H - source_K_mod[1, 2]  # cy_new = H - cy_old
        target_K_mod[1, 2] = H - target_K_mod[1, 2]
        
        return source_K_mod, source_c2w_mod, target_K_mod, target_c2w_mod
    
    def _get_flip_z_matrices(self, source_cam, target_cam):
        """Flip Z coordinate system"""
        flip_z = np.array([
            [1,  0,  0, 0],
            [0,  1,  0, 0],
            [0,  0, -1, 0],  # Flip Z
            [0,  0,  0, 1]
        ])
        
        source_c2w_mod = flip_z @ source_cam['c2w'] @ np.linalg.inv(flip_z)
        target_c2w_mod = flip_z @ target_cam['c2w'] @ np.linalg.inv(flip_z)
        
        return (source_cam['K'], source_c2w_mod, 
                target_cam['K'], target_c2w_mod)
    
    def _get_flip_yz_matrices(self, source_cam, target_cam):
        """Flip both Y and Z coordinate systems"""
        flip_yz = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],  # Flip Y
            [0,  0, -1, 0],  # Flip Z
            [0,  0,  0, 1]
        ])
        
        source_c2w_mod = flip_yz @ source_cam['c2w'] @ np.linalg.inv(flip_yz)
        target_c2w_mod = flip_yz @ target_cam['c2w'] @ np.linalg.inv(flip_yz)
        
        # Also flip cy in intrinsics
        source_K_mod = source_cam['K'].copy()
        target_K_mod = target_cam['K'].copy()
        H = source_cam['rgb'].shape[0]
        source_K_mod[1, 2] = H - source_K_mod[1, 2]
        target_K_mod[1, 2] = H - target_K_mod[1, 2]
        
        return source_K_mod, source_c2w_mod, target_K_mod, target_c2w_mod
    
    def _get_opengl_to_cv_matrices(self, source_cam, target_cam):
        """Convert from OpenGL to Computer Vision convention"""
        # OpenGL: Y-up, Z towards viewer
        # CV: Y-down, Z away from viewer
        opengl_to_cv = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],  # Flip Y
            [0,  0, -1, 0],  # Flip Z
            [0,  0,  0, 1]
        ])
        
        source_c2w_mod = opengl_to_cv @ source_cam['c2w']
        target_c2w_mod = opengl_to_cv @ target_cam['c2w']
        
        # Flip cy in intrinsics
        source_K_mod = source_cam['K'].copy()
        target_K_mod = target_cam['K'].copy()
        H = source_cam['rgb'].shape[0]
        source_K_mod[1, 2] = H - source_K_mod[1, 2]
        target_K_mod[1, 2] = H - target_K_mod[1, 2]
        
        return source_K_mod, source_c2w_mod, target_K_mod, target_c2w_mod
    
    def _get_principal_point_flip_matrices(self, source_cam, target_cam):
        """Just flip the principal point Y coordinate"""
        source_K_mod = source_cam['K'].copy()
        target_K_mod = target_cam['K'].copy()
        
        H_source = source_cam['rgb'].shape[0]
        H_target = target_cam['rgb'].shape[0]
        
        # Flip cy: cy_new = H - cy_old
        source_K_mod[1, 2] = H_source - source_K_mod[1, 2]
        target_K_mod[1, 2] = H_target - target_K_mod[1, 2]
        
        return (source_K_mod, source_cam['c2w'], 
                target_K_mod, target_cam['c2w'])
    
    def _visualize_coordinate_systems(self, results, frame_id):
        """Visualize all coordinate system tests"""
        n_systems = len(results) - 1  # Exclude ground truth
        cols = 3
        rows = (n_systems + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows + 1, cols, figsize=(cols * 5, (rows + 1) * 4))
        if rows == 0:
            axes = axes.reshape(1, -1)
        
        # Ground truth (first row, center)
        gt_col = cols // 2
        axes[0, gt_col].imshow(results['ground_truth'])
        axes[0, gt_col].set_title('Ground Truth', fontsize=12, weight='bold')
        axes[0, gt_col].axis('off')
        
        # Hide unused subplots in first row
        for col in range(cols):
            if col != gt_col:
                axes[0, col].axis('off')
        
        # Plot coordinate system results
        system_names = [k for k in results.keys() if k != 'ground_truth']
        
        for i, system_name in enumerate(system_names):
            row = (i // cols) + 1
            col = i % cols
            
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].imshow(results[system_name])
                axes[row, col].set_title(system_name.replace('_', ' ').title(), fontsize=10)
                axes[row, col].axis('off')
        
        # Hide unused subplots
        total_plots = len(system_names)
        for i in range(total_plots, rows * cols):
            row = (i // cols) + 1
            col = i % cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'coordinate_systems_debug_{frame_id}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Coordinate systems comparison saved as 'coordinate_systems_debug_{frame_id}.png'")
    
    def debug_raw_camera_parameters(self, frame_id):
        """Debug raw camera parameters from iPhone"""
        print(f"\n=== Raw Camera Parameters Analysis ===")
        
        frame_data = self.dataset.get_frame_by_id(frame_id)
        
        for cam_id in [0, 1, 2]:
            if cam_id in frame_data['cameras']:
                cam_params = frame_data['cameras'][cam_id]['camera_params']
                
                print(f"\nCamera {cam_id} raw parameters:")
                print(f"  Focal length: {cam_params['focal_length']}")
                print(f"  Principal point: {cam_params['principal_point']}")
                print(f"  Position: {cam_params['position']}")
                print(f"  Orientation shape: {np.array(cam_params['orientation']).shape}")
                print(f"  Orientation matrix:\n{np.array(cam_params['orientation'])}")
                
                # Check if orientation is orthogonal
                R = np.array(cam_params['orientation'])
                should_be_identity = R @ R.T
                print(f"  R @ R.T (should be identity):\n{should_be_identity}")
                print(f"  Det(R) (should be 1): {np.linalg.det(R):.6f}")
                
                # Analyze orientation vectors
                right = R[:, 0]    # X axis
                up = R[:, 1]       # Y axis  
                forward = R[:, 2]  # Z axis
                
                print(f"  Right vector: {right}")
                print(f"  Up vector: {up}")
                print(f"  Forward vector: {forward}")
                
                # Check if iPhone uses different conventions
                print(f"  Image size from RGB: {frame_data['cameras'][cam_id]['rgb'].shape[:2]}")


    def debug_principal_point_origins(self, frame_id):
        """Test different principal point origin conventions"""
        print(f"\n=== Testing Principal Point Origins ===")
        
        frame_data = self.dataset.get_frame_by_id(frame_id)
        source_cam = frame_data['cameras'][0]
        target_cam = frame_data['cameras'][1]
        
        # Get image dimensions
        H, W = target_cam['rgb'].shape[:2]
        
        # Get 3D points from source
        points_3d, colors = self.project_image_to_3d(
            source_cam['rgb'], source_cam['depth'], 
            source_cam['K'], source_cam['c2w']
        )
        
        # Test different principal point interpretations
        pp_conventions = {
            'original': lambda cx, cy: (cx, cy),
            'top_left_origin': lambda cx, cy: (cx, cy),  # Standard CV convention
            'center_origin': lambda cx, cy: (cx + W/2, cy + H/2),  # Center-based
            'bottom_left_origin': lambda cx, cy: (cx, H - cy),  # Graphics convention
            'normalized_coords': lambda cx, cy: (cx * W, cy * H),  # [0,1] to pixels
            'offset_half_pixel': lambda cx, cy: (cx + 0.5, cy + 0.5),  # Half-pixel offset
        }
        
        results = {'ground_truth': target_cam['rgb']}
        
        orig_cx, orig_cy = target_cam['K'][0, 2], target_cam['K'][1, 2]
        print(f"Original principal point: ({orig_cx:.2f}, {orig_cy:.2f})")
        print(f"Image size: {W} x {H}")
        print(f"Image center: ({W/2:.2f}, {H/2:.2f})")
        
        for name, pp_func in pp_conventions.items():
            try:
                # Modify target camera K matrix
                K_modified = target_cam['K'].copy()
                new_cx, new_cy = pp_func(orig_cx, orig_cy)
                K_modified[0, 2] = new_cx
                K_modified[1, 2] = new_cy
                
                print(f"{name}: ({new_cx:.2f}, {new_cy:.2f})")
                
                # Render with modified principal point
                rendered = self.render_from_camera(
                    points_3d, colors, K_modified, target_cam['c2w'],
                    target_cam['rgb'].shape[:2]
                )
                
                results[name] = rendered
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                results[name] = np.zeros_like(target_cam['rgb'])
        
        # Visualize
        self._visualize_principal_point_tests(results, frame_id, 'origins')
        return results

    def _visualize_principal_point_tests(self, results, frame_id, test_type):
        """Visualize principal point test results"""
        n_tests = len(results) - 1  # Exclude ground truth
        cols = 4
        rows = (n_tests + cols - 1) // cols
        
        fig, axes = plt.subplots(rows + 1, cols, figsize=(cols * 4, (rows + 1) * 3))
        if rows == 0:
            axes = axes.reshape(1, -1)
        
        # Ground truth in center of first row
        gt_col = cols // 2
        axes[0, gt_col].imshow(results['ground_truth'])
        axes[0, gt_col].set_title('Ground Truth', fontsize=12, weight='bold')
        axes[0, gt_col].axis('off')
        
        # Hide other first row subplots
        for col in range(cols):
            if col != gt_col:
                axes[0, col].axis('off')
        
        # Show test results
        test_names = [k for k in results.keys() if k != 'ground_truth']
        for i, test_name in enumerate(test_names):
            row = (i // cols) + 1
            col = i % cols
            
            if row < axes.shape[0]:
                axes[row, col].imshow(results[test_name])
                axes[row, col].set_title(test_name.replace('_', ' ').title(), fontsize=9)
                axes[row, col].axis('off')
        
        # Hide unused subplots
        total_used = len(test_names)
        for i in range(total_used, rows * cols):
            row = (i // cols) + 1
            col = i % cols
            if row < axes.shape[0]:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'principal_point_{test_type}_debug_{frame_id}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Principal point {test_type} tests saved as 'principal_point_{test_type}_debug_{frame_id}.png'")

# Usage
def debug_iphone_cameras():
    """Main debugging function"""
    
    # Initialize dataset
    dataset = iPhoneDataset(
        root_dir='/home/azhuravl/nobackup/iphone',
        sequence_name='paper-windmill',
        scale='1x',  # Use 1x for faster debugging
        camera_ids=[0, 1, 2],
        min_sequence_length=10  # Lower for debugging
    )
    
    print(f"Dataset loaded with {len(dataset)} frames")
    print(f"Available sequences: {len(dataset.contiguous_sequences)}")
    
    # Create debugger
    debugger = CameraProjectionDebugger(dataset)
    
    # Test with first available frame
    if len(dataset.frame_ids) > 0:
        test_frame_id = dataset.frame_ids[10]  # Use frame 10 for example
        print(f"Testing with frame ID: {test_frame_id}")
        
        # NEW: Debug raw parameters first
        debugger.debug_raw_camera_parameters(test_frame_id)
        
        # Test camera conventions
        debugger.test_camera_conventions(test_frame_id)
        
        # NEW: Test coordinate systems
        coord_results = debugger.debug_coordinate_systems(test_frame_id)
        
        # Original projection test
        results = debugger.debug_camera_projection(test_frame_id, visualize=True)
        
        # Vertical alignment test
        debugger.debug_vertical_alignment(test_frame_id)
        
        debugger.debug_principal_point_origins(test_frame_id)
        
        return results, coord_results
    else:
        print("No frames available for debugging!")
        return None, None

def quick_camera_check():
    """Quick check of camera parameters"""
    dataset = iPhoneDataset(
        root_dir='/home/azhuravl/nobackup/iphone',
        sequence_name='paper-windmill',
        scale='1x',
        camera_ids=[0, 1, 2],
        min_sequence_length=5
    )
    
    if len(dataset.frame_ids) > 0:
        frame_data = dataset[0]
        
        print("Quick camera parameter check:")
        for cam_id in [0, 1, 2]:
            if cam_id in frame_data['cameras']:
                cam_data = frame_data['cameras'][cam_id]
                pos = cam_data['c2w'][:3, 3]
                forward = cam_data['c2w'][:3, 2]
                print(f"Camera {cam_id}: pos={pos}, forward={forward}")
                print(f"  K diagonal: {np.diag(cam_data['K'])}")

if __name__ == "__main__":
    # Quick check first
    print("=== Quick Camera Check ===")
    quick_camera_check()
    
    print("\n" + "="*50)
    print("=== Full Debug ===")
    
    # Full debugging
    results = debug_iphone_cameras()
    
    # In your main debug function, add:
