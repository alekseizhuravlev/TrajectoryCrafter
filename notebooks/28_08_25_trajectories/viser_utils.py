import viser
import numpy as np
import torch
import threading
import time
import math

# append /home/azhuravl/work/TrajectoryCrafter/notebooks/28_08_25_trajectories
import sys
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/28_08_25_trajectories')

import trajectory_generation


# Cell 2: Animated Viser Content
def setup_viser_scene(server, scene_data):
    """Setup static scene elements (trajectory and camera poses)"""

    poses_np_c2w = scene_data['pose_target'].cpu().numpy()
    
    poses_np = np.linalg.inv(poses_np_c2w)  # Convert to world-to-camera
    
    positions = poses_np[:, :3, 3]
    
    # Add trajectory (static)
    server.scene.add_spline_catmull_rom(
        "/trajectory", 
        positions=positions, 
        color=(1.0, 0.0, 0.0), 
        line_width=3.0
    )
    
    # Add all camera poses (static)
    for i, pose in enumerate(poses_np[::2]):  # Every 2nd pose to reduce clutter

        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        
        # print(position)
        
        # flip_z = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        # corrected_rotation = rotation_matrix @ flip_z
        
        corrected_rotation = rotation_matrix  # No correction
        wxyz = viser.transforms.SO3.from_matrix(corrected_rotation).wxyz
        
        server.scene.add_camera_frustum(
            f"/camera_{i}",
            fov=60, aspect=16/9, scale=0.15,
            position=position, wxyz=wxyz,
            color=(0.8, 0.2, 0.2)
        )
    
    # Add start/end markers
    server.scene.add_icosphere("/start", radius=0.1, position=positions[0], color=(0.0, 1.0, 0.0))
    server.scene.add_icosphere("/end", radius=0.1, position=positions[-1], color=(1.0, 0.0, 1.0))
    server.scene.add_frame("/world", axes_length=0.5, position=(0, 0, 0), wxyz=(1, 0, 0, 0))


def animate_frame(server, vis_warper, scene_data, frame_idx, max_points=10000):
    """Update only the point cloud for given frame"""
    # Clear previous frame
    try:
        server.scene.remove_by_name("/current_frame")
        server.scene.remove_by_name("/current_camera")
    except:
        pass
    
    # Extract points for this frame
    frame_data = {
        'frame': scene_data['frames_tensor'][frame_idx:frame_idx+1],
        'depth': scene_data['depths'][frame_idx:frame_idx+1], 
        'pose_source': scene_data['pose_source'][frame_idx:frame_idx+1],
        'intrinsics': scene_data['intrinsics'][frame_idx:frame_idx+1],
    }
    
    points_3d, colors_rgb = vis_warper.extract_3d_points_with_colors(
        frame_data['frame'], frame_data['depth'], 
        frame_data['pose_source'], frame_data['intrinsics'],
        subsample_step=5
    )
    
    if points_3d.shape[0] > 0:
        points_np = points_3d.cpu().numpy()
        colors_np = colors_rgb.cpu().numpy()
        
        # Limit points
        if len(points_np) > max_points:
            indices = np.random.choice(len(points_np), max_points, replace=False)
            points_np = points_np[indices]
            colors_np = colors_np[indices]
        
        if colors_np.min() < 0:
            colors_np = (colors_np + 1) / 2
            
        # Update point cloud
        server.scene.add_point_cloud(
            "/current_frame", 
            points=points_np, 
            colors=colors_np, 
            point_size=0.05
        )
        
        # Highlight current camera
        # this is w2c, get c2w
        # pos = scene_data['pose_target'][frame_idx, :3, 3].cpu().numpy()
        
        pose_c2w = np.linalg.inv(scene_data['pose_target'][frame_idx].cpu().numpy())
        pos = pose_c2w[:3, 3]
        
        server.scene.add_icosphere(
            "/current_camera", 
            radius=0.08, 
            position=pos, 
            color=(1.0, 1.0, 0.0)  # Yellow
        )


def add_animation_controls(server, vis_warper, scene_data):
    """Add animation controls to viser server"""
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Animation controls
        play_button = client.gui.add_button("Play/Pause")
        frame_slider = client.gui.add_slider(
            "Frame", 
            min=0, 
            max=scene_data['frames_tensor'].shape[0]-1, 
            step=1, 
            initial_value=0
        )
        speed_slider = client.gui.add_slider(
            "Speed", 
            min=1, 
            max=10.0, 
            step=0.1, 
            initial_value=3.0
        )
        
        # Animation state
        is_playing = [False]
        
        @play_button.on_click
        def _(_):
            is_playing[0] = not is_playing[0]
            
        @frame_slider.on_update
        def _(_):
            animate_frame(server, vis_warper, scene_data, frame_slider.value)
        
        # Animation loop
        def animation_loop():
            while True:
                if is_playing[0]:
                    current_frame = frame_slider.value
                    next_frame = (current_frame + 1) % scene_data['frames_tensor'].shape[0]
                    frame_slider.value = next_frame
                    animate_frame(server, vis_warper, scene_data, next_frame)
                time.sleep(0.5 / speed_slider.value)
        
        # Start animation thread
        animation_thread = threading.Thread(target=animation_loop, daemon=True)
        animation_thread.start()


def add_point_size_control(server, points_3d, colors_rgb):
    """Add point size control to viser server"""
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Add simple point size slider
        point_size_slider = client.gui.add_slider(
            "Point Size",
            min=0.005,
            max=0.1,
            step=0.005,
            initial_value=0.015,
        )
        
        # Update point size when slider changes
        @point_size_slider.on_update
        def _(_) -> None:
            if points_3d is not None:
                points_np = points_3d.cpu().numpy()
                colors_np = colors_rgb.cpu().numpy()
                if colors_np.min() < 0:
                    colors_np = (colors_np + 1) / 2
                server.scene.add_point_cloud(
                    "/scene_points",
                    points=points_np,
                    colors=colors_np,
                    point_size=point_size_slider.value
                )



def add_camera_controls(viser_server):
    """Add sliders to control camera position and orientation"""
    
    # Set initial camera position
    initial_theta = 0
    initial_phi = 75
    initial_roll = -90
    initial_radius = 10

    # Add sliders for camera control
    theta_slider = viser_server.gui.add_slider(
        "Camera Theta (deg)",
        min=0, max=360, step=1, initial_value=initial_theta,
    )

    phi_slider = viser_server.gui.add_slider(
        "Camera Phi (deg)", 
        min=-90, max=270, step=1, initial_value=initial_phi,
    )

    roll_slider = viser_server.gui.add_slider(
        "Camera Roll (deg)",
        min=-180, max=180, step=1, initial_value=initial_roll,
    )

    radius_slider = viser_server.gui.add_slider(
        "Camera Distance",
        min=1, max=20, step=0.1, initial_value=initial_radius,
    )

    def update_camera_position():
        """Update camera position based on slider values"""
        theta = math.radians(theta_slider.value)
        phi = math.radians(phi_slider.value)
        r = radius_slider.value
        roll = math.radians(roll_slider.value)
        
        # Convert spherical to cartesian
        x = r * math.cos(phi) * math.cos(theta)
        y = r * math.cos(phi) * math.sin(theta) 
        z = r * math.sin(phi)
        
        position = np.array([x, y, z])
        look_at = np.array([0, 0, 0])
        
        # Calculate camera orientation
        forward = (look_at - position) / np.linalg.norm(look_at - position)
        world_up = np.array([0, 0, 1]) if abs(phi) < math.pi/2 else np.array([0, 0, -1])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-6 else np.array([1, 0, 0])
        up = np.cos(roll) * np.cross(right, forward) + np.sin(roll) * right
        
        # Update all clients
        for client in viser_server.get_clients().values():
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.up_direction = up

    # Connect the sliders - THIS GOES INSIDE THE FUNCTION
    @theta_slider.on_update
    def _(_):
        update_camera_position()
        
    @phi_slider.on_update 
    def _(_):
        update_camera_position()
        
    @radius_slider.on_update
    def _(_):
        update_camera_position()

    @roll_slider.on_update
    def _(_):
        update_camera_position()

    @viser_server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        update_camera_position()
        
        
        
        
        

def update_trajectory_visualization(server, new_scene_data):
    """Update only the trajectory visualization, keep point clouds"""
    
    # Clear existing trajectory elements
    try:
        server.scene.remove_by_name("/trajectory")
        for i in range(50):  # Clear up to 50 camera poses
            try:
                server.scene.remove_by_name(f"/camera_{i}")
            except:
                print(f"Camera /camera_{i} not found, stopping removal.")
        server.scene.remove_by_name("/start")
        server.scene.remove_by_name("/end")
    except:
        print("Some trajectory elements not found, skipping removal.")
    
    # Add new trajectory
    poses_np_c2w = new_scene_data['pose_target'].cpu().numpy()
    
    # invert poses to get world-to-camera
    poses_np = np.linalg.inv(poses_np_c2w)
    
    positions = poses_np[:, :3, 3]
    
    # Add trajectory spline
    server.scene.add_spline_catmull_rom(
        "/trajectory", 
        positions=positions, 
        color=(1.0, 0.0, 0.0), 
        line_width=3.0
    )
    
    # Add camera poses (every 2nd to reduce clutter)
    for i, pose in enumerate(poses_np[::2]):
        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        
        # flip_z = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        # corrected_rotation = rotation_matrix @ flip_z
        
        corrected_rotation = rotation_matrix  # No correction
        wxyz = viser.transforms.SO3.from_matrix(corrected_rotation).wxyz
        
        server.scene.add_camera_frustum(
            f"/camera_{i}",
            fov=60, aspect=16/9, scale=0.15,
            position=position, wxyz=wxyz,
            color=(0.8, 0.2, 0.2)
        )
    
    # Add new start/end markers
    server.scene.add_icosphere("/start", radius=0.1, position=positions[0], color=(0.0, 1.0, 0.0))
    server.scene.add_icosphere("/end", radius=0.1, position=positions[-1], color=(1.0, 0.0, 1.0))


# def update_trajectory_visualization(server, new_scene_data):
#     """Update trajectory visualization with relative transformations - no coordinate conversion"""
    
#     # Clear existing trajectory elements
#     try:
#         server.scene.remove("/trajectory")
#         for i in range(50):
#             try:
#                 server.scene.remove(f"/camera_{i}")
#             except:
#                 break
#         server.scene.remove("/start")
#         server.scene.remove("/end")
#     except:
#         pass
    
#     # Get poses from scene data
#     pose_source = new_scene_data['pose_source'].cpu().numpy()  # Reference/anchor poses
#     pose_target = new_scene_data['pose_target'].cpu().numpy()  # Target trajectory poses
    
#     # Compute relative transformations: pose_source^(-1) * pose_target
#     relative_poses = []
    
#     for i in range(len(pose_target)):
#         # Invert source pose and multiply with target pose
#         pose_src_inv = np.linalg.inv(pose_source[i])
#         relative_pose = pose_src_inv @ pose_target[i]
#         relative_poses.append(relative_pose)
    
#     relative_poses = np.array(relative_poses)
    
#     # Extract positions from relative transformations (no coordinate conversion)
#     positions = relative_poses[:, :3, 3]
    
#     # Add trajectory spline using raw relative positions
#     server.scene.add_spline_catmull_rom(
#         "/trajectory", 
#         positions=positions, 
#         color=(1.0, 0.0, 0.0), 
#         line_width=3.0
#     )
    
#     # Add camera frustums for relative poses (every 2nd to reduce clutter)
#     for i, relative_pose in enumerate(relative_poses[::2]):
#         position = relative_pose[:3, 3]
#         rotation_matrix = relative_pose[:3, :3]
        
#         # Use raw rotation matrix without coordinate transformation
#         wxyz = viser.transforms.SO3.from_matrix(rotation_matrix).wxyz
        
#         try:
#             server.scene.add_camera_frustum(
#                 f"/camera_{i}",
#                 fov=60, aspect=16/9, scale=0.15,
#                 position=position, wxyz=wxyz,
#                 color=(0.8, 0.2, 0.2)
#             )
#         except:
#             # If rotation fails, just show position as sphere
#             server.scene.add_icosphere(
#                 f"/camera_{i}",
#                 radius=0.05,
#                 position=position,
#                 color=(0.8, 0.2, 0.2)
#             )
    
#     # Add start/end markers using raw positions
#     server.scene.add_icosphere("/start", radius=0.1, position=positions[0], color=(0.0, 1.0, 0.0))
#     server.scene.add_icosphere("/end", radius=0.1, position=positions[-1], color=(1.0, 0.0, 1.0))
    
#     print(f"Relative trajectory updated with {len(positions)} points")
#     print(f"Relative start position: {positions[0]}")
#     print(f"Relative end position: {positions[-1]}")
#     print(f"Relative movement range:")
#     print(f"  X: {positions[:, 0].min():.3f} to {positions[:, 0].max():.3f}")
#     print(f"  Y: {positions[:, 1].min():.3f} to {positions[:, 1].max():.3f}")
#     print(f"  Z: {positions[:, 2].min():.3f} to {positions[:, 2].max():.3f}")






# Predefined trajectory presets
TRAJECTORY_PRESETS = {
    "Original Right 90°": [0, 90, 1, 0, 0],
    "Left 90°": [0, -90, 1, 0, 0], 
    "Full Circle": [0, 360, 1, 0, 0],
    "Up and Right": [45, 45, 0.5, 0, 1],
    "Pull Back": [0, 0, 3, 0, 0],
    "Orbit Up": [30, 180, 0, 0, 0],
    "Dolly Forward": [0, 0, -2, 0, 0],
    "Rise and Turn": [60, 120, 1, 0, 2],
}


def add_trajectory_controls(server, scene_data, vis_crafter, opts_base):
    """Add trajectory selection and animation controls"""
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        
        # Trajectory dropdown
        trajectory_dropdown = client.gui.add_dropdown(
            "Trajectory Preset",
            options=list(TRAJECTORY_PRESETS.keys()),
            initial_value="Original Right 90°"
        )
        
        # Custom controls
        with client.gui.add_folder("Custom Trajectory"):
            theta_input = client.gui.add_slider("Theta", min=-180, max=180, step=1, initial_value=0)
            phi_input = client.gui.add_slider("Phi", min=-360, max=360, step=5, initial_value=90)
            dr_input = client.gui.add_slider("Distance", min=-3, max=3, step=0.1, initial_value=1.0)
            dx_input = client.gui.add_slider("X offset", min=-2, max=2, step=0.1, initial_value=0.0)
            dy_input = client.gui.add_slider("Y offset", min=-2, max=2, step=0.1, initial_value=0.0)
            generate_button = client.gui.add_button("Generate Custom")
        
        # # Animation controls
        # with client.gui.add_folder("Animation"):
        #     play_button = client.gui.add_button("Play/Pause")
        #     frame_slider = client.gui.add_slider("Frame", min=0, max=scene_data['frames_tensor'].shape[0]-1, step=1, initial_value=0)
        #     speed_slider = client.gui.add_slider("Speed", min=0.5, max=5.0, step=0.1, initial_value=1.0)
        
        # State
        current_scene_data = [scene_data]
        is_playing = [False]
        
        def update_trajectory(target_pose):
            new_scene_data = trajectory_generation.generate_new_trajectory(vis_crafter, opts_base, target_pose, scene_data)
            current_scene_data[0] = new_scene_data
            update_trajectory_visualization(server, new_scene_data)
        
        # Callbacks
        @trajectory_dropdown.on_update
        def _(_):
            target_pose = TRAJECTORY_PRESETS[trajectory_dropdown.value]
            update_trajectory(target_pose)
            # Update sliders to match preset
            theta_input.value, phi_input.value, dr_input.value, dx_input.value, dy_input.value = target_pose
        
        @generate_button.on_click
        def _(_):
            custom_pose = [theta_input.value, phi_input.value, dr_input.value, dx_input.value, dy_input.value]
            update_trajectory(custom_pose)
        
        # @play_button.on_click
        # def _(_):
        #     is_playing[0] = not is_playing[0]
        
        # @frame_slider.on_update
        # def _(_):
        #     animate_frame(server, current_scene_data[0], frame_slider.value)
        
        # Animation loop
        # def animation_loop():
        #     while True:
        #         if is_playing[0]:
        #             current_frame = frame_slider.value
        #             next_frame = (current_frame + 1) % current_scene_data[0]['frames_tensor'].shape[0]
        #             frame_slider.value = next_frame
        #         time.sleep(1.0 / speed_slider.value)
        
        # import threading
        # animation_thread = threading.Thread(target=animation_loop, daemon=True)
        # animation_thread.start()