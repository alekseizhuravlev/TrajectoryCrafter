import viser
import numpy as np
import torch
from pathlib import Path
import time
import threading
from typing import List, Tuple, Optional
import argparse
from tqdm import tqdm
import os


def load_poses_torch(filepath):
    """Load camera poses from PyTorch tensor file"""
    return torch.load(filepath, map_location='cpu').numpy()


def load_point_clouds_torch(dirpath):
    """Load point cloud sequence from PyTorch tensor files (individual files)"""
    dirpath = Path(dirpath)
    
    # Find all point files
    point_files = sorted(list(dirpath.glob('points_*.pth')))
    color_files = sorted(list(dirpath.glob('colors_*.pth')))
    
    if len(point_files) != len(color_files):
        raise ValueError(f"Mismatch: {len(point_files)} point files vs {len(color_files)} color files")
    
    point_clouds = []
    color_clouds = []
    
    print(f"Loading {len(point_files)} point clouds from PyTorch tensors...")
    
    for pc_file, color_file in tqdm(zip(point_files, color_files), total=len(point_files)):
        pc = torch.load(pc_file, map_location='cpu').numpy()
        color = torch.load(color_file, map_location='cpu').numpy()
        
        # Ensure colors are in [0, 255] range for visualization
        if color.max() <= 1.0:
            color = (color * 255).astype(np.uint8)
        
        point_clouds.append(pc)
        color_clouds.append(color)
    
    return point_clouds, color_clouds


def load_point_cloud_sequence_auto(stage_dir, subsample_method='uniform', subsample_factor=1):
    """
    Auto-detect and load point clouds from PyTorch tensor directories
    Priority: merged -> inpainted -> input
    """
    stage_path = Path(stage_dir)
    
    # Try different point cloud sources in order of preference
    pc_dirs = [
        # 'point_cloud_merged',
        # 'point_cloud_inpainted', 
        'point_cloud_input',
    ]
    
    for pc_dir_name in pc_dirs:
        pc_dir = stage_path / pc_dir_name
        if pc_dir.exists():
            try:
                print(f"Loading from PyTorch tensors: {pc_dir_name}")
                points, colors = load_point_clouds_torch(pc_dir)
                
                # Apply subsampling
                if subsample_factor > 1:
                    subsampled_points = []
                    subsampled_colors = []
                    for pc, color in zip(points, colors):
                        pc_sub, color_sub = subsample_point_cloud(pc, color, subsample_method, subsample_factor)
                        subsampled_points.append(pc_sub)
                        subsampled_colors.append(color_sub)
                    points, colors = subsampled_points, subsampled_colors
                
                return points, colors
                
            except Exception as e:
                print(f"Failed to load from {pc_dir}: {e}")
                continue
    
    print("No point cloud data found!")
    return [], []


def subsample_point_cloud(points, colors, method='random', factor=1):
    """
    Subsample point cloud using different methods
    
    Args:
        points: (N, 3) numpy array of 3D points
        colors: (N, 3) numpy array of RGB colors
        method: 'random', 'uniform', 'grid' - subsampling method
        factor: subsampling factor (keep every factor-th point or 1/factor fraction)
        
    Returns:
        subsampled_points, subsampled_colors
    """
    if factor <= 1:
        return points, colors
        
    n_points = len(points)
    
    if method == 'random':
        # Random subsampling - keep 1/factor of points
        n_keep = max(1, n_points // factor)
        indices = np.random.choice(n_points, size=n_keep, replace=False)
        
    elif method == 'uniform':
        # Uniform subsampling - keep every factor-th point
        indices = np.arange(0, n_points, factor)
        
    elif method == 'grid':
        # Grid-based subsampling
        if len(points) < 1000:  # For small point clouds, use random
            return subsample_point_cloud(points, colors, 'random', factor)
            
        # Simple grid approach: divide space and keep one point per cell
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        range_coords = max_coords - min_coords
        
        # Create grid resolution based on factor
        grid_size = int(np.ceil(np.cbrt(n_points / factor)))
        cell_size = range_coords / grid_size
        
        # Assign points to grid cells
        grid_indices = np.floor((points - min_coords) / cell_size).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)
        
        # Create unique grid keys
        grid_keys = grid_indices[:, 0] * (grid_size ** 2) + grid_indices[:, 1] * grid_size + grid_indices[:, 2]
        
        # Keep one point per grid cell
        unique_keys, unique_indices = np.unique(grid_keys, return_index=True)
        indices = unique_indices
        
    else:
        raise ValueError(f"Unknown subsampling method: {method}")
    
    return points[indices], colors[indices]


def create_camera_frustum(pose, scale=0.1, color=(1.0, 0.0, 0.0)):
    """Create camera frustum geometry for visualization"""
    # Define camera frustum vertices (in camera coordinates)
    vertices = np.array([
        [0, 0, 0],      # Camera center
        [-1, -1, 2],    # Bottom left
        [1, -1, 2],     # Bottom right  
        [1, 1, 2],      # Top right
        [-1, 1, 2],     # Top left
    ]) * scale
    
    # Transform vertices to world coordinates
    vertices_homo = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    vertices_world = (pose @ vertices_homo.T).T[:, :3]
    
    # Define edges of the frustum
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle at far plane
    ]
    
    return vertices_world, edges


class PointCloudAnimator:
    def __init__(self, server_port=8080):
        self.port = server_port
        self.server = viser.ViserServer(port=server_port)
        self.playing = False
        self.current_frame = 0
        self.point_clouds = []
        self.color_clouds = []
        self.poses = []
        self.fps = 10
        
        # Subsampling parameters
        self.subsample_method = 'uniform'
        self.subsample_factor = 1
        
        # Track scene objects for cleanup
        self.scene_objects = {}  # Store references to scene objects
        
        # GUI elements
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI controls"""
        # Animation controls
        self.play_button = self.server.gui.add_button("Play")
        self.pause_button = self.server.gui.add_button("Pause")
        self.reset_button = self.server.gui.add_button("Reset")
        
        # Frame control
        self.frame_slider = self.server.gui.add_slider(
            "Frame", min=0, max=100, step=1, initial_value=0
        )
        
        # Speed control  
        self.fps_slider = self.server.gui.add_slider(
            "FPS", min=1, max=30, step=1, initial_value=10
        )
        
        # Display options
        self.show_cameras = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.show_points = self.server.gui.add_checkbox("Show Points", initial_value=True)
        self.camera_scale = self.server.gui.add_slider(
            "Camera Scale", min=0.01, max=1.0, step=0.01, initial_value=0.1
        )
        
        # Point cloud options
        self.point_size = self.server.gui.add_slider(
            "Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
        )
        
        # Subsampling controls
        self.subsample_dropdown = self.server.gui.add_dropdown(
            "Subsample Method", 
            options=['uniform', 'random', 'grid'],
            initial_value='uniform'
        )
        
        self.subsample_slider = self.server.gui.add_slider(
            "Subsample Factor", min=1, max=50, step=1, initial_value=1
        )
        
        # Real-time subsampling controls
        self.realtime_subsample = self.server.gui.add_checkbox("Real-time Subsample", initial_value=False)
        
        # Point cloud statistics display
        self.point_count_text = self.server.gui.add_text("Points", initial_value="Points: 0")
        
        # Bind callbacks
        self.play_button.on_click(self._on_play)
        self.pause_button.on_click(self._on_pause) 
        self.reset_button.on_click(self._on_reset)
        self.frame_slider.on_update(self._on_frame_change)
        self.fps_slider.on_update(self._on_fps_change)
        self.show_cameras.on_update(self._on_display_update)
        self.show_points.on_update(self._on_display_update)
        self.camera_scale.on_update(self._on_camera_scale_change)
        self.point_size.on_update(self._on_point_size_change)
        self.subsample_dropdown.on_update(self._on_subsample_change)
        self.subsample_slider.on_update(self._on_subsample_change)
        self.realtime_subsample.on_update(self._on_subsample_change)
        
    def load_data(self, stage_dir, subsample_method='uniform', subsample_factor=1):
        """Load point clouds and camera poses from stage directory"""
        stage_path = Path(stage_dir)
        
        # Store subsampling parameters
        self.subsample_method = subsample_method
        self.subsample_factor = subsample_factor
        
        # Load camera poses from PyTorch tensor
        camera_file_pth = stage_path / "cameras_target.pth"
        if camera_file_pth.exists():
            self.poses = load_poses_torch(camera_file_pth)
            print(f"Loaded {len(self.poses)} camera poses from PyTorch tensor")
        else:
            self.poses = []
            print("No camera poses found (cameras_target.pth)")
        
        # Load point clouds
        self.point_clouds, self.color_clouds = load_point_cloud_sequence_auto(
            stage_dir, subsample_method, subsample_factor
        )
        
        # Update GUI controls
        self.subsample_dropdown.value = subsample_method
        self.subsample_slider.value = subsample_factor
        
        # Update slider range
        max_frames = max(len(self.point_clouds), len(self.poses)) - 1 if (self.point_clouds or self.poses) else 0
        if max_frames > 0:
            self.frame_slider.max = max_frames
        
        # Report loading results
        print(f"Loaded {len(self.point_clouds)} point cloud frames")
        if self.point_clouds:
            avg_points = np.mean([len(pc) for pc in self.point_clouds])
            total_points = sum(len(pc) for pc in self.point_clouds)
            print(f"Average points per frame: {avg_points:.0f}")
            print(f"Total points: {total_points:,}")
            
    def clear_scene_objects(self, prefix):
        """Clear scene objects with given prefix"""
        objects_to_remove = []
        for name in self.scene_objects.keys():
            if name.startswith(prefix):
                objects_to_remove.append(name)
        
        for name in objects_to_remove:
            try:
                self.scene_objects[name].remove()
                del self.scene_objects[name]
            except:
                pass  # Object might already be removed
                
    def _on_play(self, _):
        """Start animation"""
        if not self.playing:
            self.playing = True
            threading.Thread(target=self._animation_loop, daemon=True).start()
            
    def _on_pause(self, _):
        """Pause animation"""
        self.playing = False
        
    def _on_reset(self, _):
        """Reset to first frame"""
        self.playing = False
        self.current_frame = 0
        self.frame_slider.value = 0
        self.update_display()
        
    def _on_frame_change(self, _):
        """Handle manual frame change"""
        self.current_frame = self.frame_slider.value
        self.update_display()
        
    def _on_fps_change(self, _):
        """Handle FPS change"""
        self.fps = self.fps_slider.value
        
    def _on_display_update(self, _):
        """Handle display option changes"""
        self.update_display()
        
    def _on_camera_scale_change(self, _):
        """Handle camera scale change"""
        self.update_cameras()
        
    def _on_point_size_change(self, _):
        """Handle point size change"""
        self.update_point_cloud()
        
    def _on_subsample_change(self, _):
        """Handle subsampling parameter changes"""
        if self.realtime_subsample.value:
            # Apply real-time subsampling
            self.update_point_cloud()
        
    def _animation_loop(self):
        """Main animation loop"""
        while self.playing:
            # Update frame
            max_frames = max(len(self.point_clouds), len(self.poses)) if (self.point_clouds or self.poses) else 0
            if max_frames > 0:
                self.current_frame = (self.current_frame + 1) % max_frames
                self.frame_slider.value = self.current_frame
                self.update_display()
                
            # Sleep based on FPS
            time.sleep(1.0 / self.fps)
            
    def update_display(self):
        """Update the complete display"""
        self.update_point_cloud()
        self.update_cameras()
        
    def update_point_cloud(self):
        """Update point cloud display"""
        # Clear existing point clouds
        self.clear_scene_objects("points_")
                
        if not self.show_points.value or not self.point_clouds:
            self.point_count_text.value = "Points: 0"
            return
            
        # Display current frame point cloud
        if self.current_frame < len(self.point_clouds):
            points = self.point_clouds[self.current_frame]
            colors = self.color_clouds[self.current_frame]
            
            # Apply real-time subsampling if enabled
            if self.realtime_subsample.value and self.subsample_slider.value > 1:
                method = self.subsample_dropdown.value
                factor = self.subsample_slider.value
                points, colors = subsample_point_cloud(points, colors, method, factor)
            
            if len(points) > 0:
                # Convert colors to float [0, 1]
                colors_float = colors.astype(np.float32) / 255.0
                
                # Add point cloud and store reference
                name = f"points_{self.current_frame}"
                point_cloud_obj = self.server.scene.add_point_cloud(
                    name,
                    points=points,
                    colors=colors_float,
                    point_size=self.point_size.value,
                )
                self.scene_objects[name] = point_cloud_obj
                
                # Update point count display
                self.point_count_text.value = f"Points: {len(points):,}"
            else:
                self.point_count_text.value = "Points: 0"
                
    def update_cameras(self):
        """Update camera display"""
        # Clear existing cameras
        self.clear_scene_objects("camera_")
                
        if not self.show_cameras.value or len(self.poses) == 0:
            return
            
        # Display all camera poses
        for i, pose in enumerate(self.poses):
            # Highlight current camera
            if i == self.current_frame:
                color = (1.0, 0.0, 0.0)  # Red for current
                scale = self.camera_scale.value * 1.5
            else:
                color = (0.0, 0.0, 1.0)  # Blue for others
                scale = self.camera_scale.value
                
            vertices, edges = create_camera_frustum(pose, scale=scale, color=color)
            
            # Add camera frustum lines and store references
            for j, (start_idx, end_idx) in enumerate(edges):
                start_pos = vertices[start_idx]
                end_pos = vertices[end_idx]
                
                name = f"camera_{i}_edge_{j}"
                line_obj = self.server.scene.add_spline_catmull_rom(
                    name,
                    positions=np.array([start_pos, end_pos]),
                    color=color,
                    line_width=2.0 if i == self.current_frame else 1.0,
                )
                self.scene_objects[name] = line_obj
                
    def run(self):
        """Start the server"""
        print("Starting Viser server...")
        print("Open your browser and go to:")
        print(f"http://localhost:{self.port}")
        print("\nLoading from PyTorch tensor files:")
        print("- Point clouds: point_cloud_*/points_*.pth + colors_*.pth")
        print("- Camera poses: cameras_target.pth")
        print("\nSubsampling options:")
        print("- uniform: Keep every N-th point")
        print("- random: Randomly select points")
        print("- grid: Grid-based spatial subsampling")
        print("- Real-time: Apply subsampling interactively")
        
        # Keep server running
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutting down server...")


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud sequence and camera poses from PyTorch tensors")
    parser.add_argument("--stage_dir", type=str, required=True, 
                       help="Path to stage directory containing PyTorch tensor files")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for Viser server")
    parser.add_argument("--subsample_method", type=str, default='uniform',
                       choices=['uniform', 'random', 'grid'],
                       help="Point cloud subsampling method")
    parser.add_argument("--subsample_factor", type=int, default=100,
                       help="Subsampling factor (1=no subsampling, 10=keep 1/10 points)")
    
    args = parser.parse_args()
    
    # Create animator
    animator = PointCloudAnimator(server_port=args.port)
    
    # Load data with subsampling
    animator.load_data(args.stage_dir, args.subsample_method, args.subsample_factor)
    
    # Start visualization
    animator.run()


if __name__ == "__main__":
    main()