import torch
import numpy as np
import copy     



def generate_new_trajectory(vis_crafter, opts_base, target_pose_params, scene_data):
    """Generate new trajectory without recomputing point clouds"""
    print(f"Generating trajectory for pose: {target_pose_params}")
    
    # Create new opts with different target pose
    opts_new = copy.deepcopy(opts_base)
    opts_new.target_pose = target_pose_params
    
    # Only regenerate poses, reuse existing depths
    pose_s, pose_t, K = vis_crafter.get_poses(opts_new, scene_data['depths'], num_frames=opts_new.video_length)
    # pose_s, pose_t, K = get_poses_circular(
    #     opts_new, 
    #     scene_data['depths'], 
    #     num_frames=opts_new.video_length,
    #     circle_type='horizontal'  # or other types based on params
    # )
    
    # Create new scene data with same point clouds but new trajectory
    new_scene_data = {
        'frames_numpy': scene_data['frames_numpy'],
        'frames_tensor': scene_data['frames_tensor'], 
        'depths': scene_data['depths'],
        'pose_source': pose_s,
        'pose_target': pose_t,
        'intrinsics': K,
        'radius': scene_data['radius'],
        'trajectory_params': target_pose_params
    }
    
    return new_scene_data

#######################################
# rest doesn't work
##################33    #############




def generate_circular_trajectory(c2ws_anchor, circle_type, radius, num_frames, device, center_offset=(0,0,0)):
    """
    Generate circular camera trajectory
    
    Args:
        c2ws_anchor: Initial camera pose tensor
        circle_type: 'horizontal', 'vertical_xz', 'vertical_yz', or 'tilted'
        radius: Circle radius
        num_frames: Number of frames
        device: torch device
        center_offset: (x,y,z) offset from origin
    """
    angles = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    c2ws_list = []
    
    for angle in angles:
        c2w = copy.deepcopy(c2ws_anchor)
        
        if circle_type == 'horizontal':
            # Circle in XY plane (horizontal orbit)
            x = radius * np.cos(angle) + center_offset[0]
            y = radius * np.sin(angle) + center_offset[1]
            z = center_offset[2]
            
            # Set camera position
            c2w[0, 0, 3] = x
            c2w[0, 1, 3] = y  
            c2w[0, 2, 3] = z
            
            # Point camera toward center (optional - creates look-at behavior)
            # For now, just translate without rotation
            
        elif circle_type == 'vertical_xz':
            # Circle in XZ plane (side view)
            x = radius * np.cos(angle) + center_offset[0]
            y = center_offset[1]
            z = radius * np.sin(angle) + center_offset[2]
            
            c2w[0, 0, 3] = x
            c2w[0, 1, 3] = y
            c2w[0, 2, 3] = z
            
        elif circle_type == 'vertical_yz':
            # Circle in YZ plane (front view)
            x = center_offset[0]
            y = radius * np.cos(angle) + center_offset[1]
            z = radius * np.sin(angle) + center_offset[2]
            
            c2w[0, 0, 3] = x
            c2w[0, 1, 3] = y
            c2w[0, 2, 3] = z
            
        elif circle_type == 'tilted':
            # 45° tilted circle
            x = radius * np.cos(angle) + center_offset[0]
            y = radius * np.sin(angle) * np.cos(np.pi/4) + center_offset[1]
            z = radius * np.sin(angle) * np.sin(np.pi/4) + center_offset[2]
            
            c2w[0, 0, 3] = x
            c2w[0, 1, 3] = y
            c2w[0, 2, 3] = z
        
        c2ws_list.append(c2w)
    
    return torch.cat(c2ws_list, dim=0)


def get_poses_circular(opts, depths, num_frames, circle_type='horizontal', custom_radius=None):
    """
    Generate circular camera poses instead of linear interpolation
    
    Args:
        opts: Options object
        depths: Depth maps
        num_frames: Number of frames
        circle_type: Type of circular motion
        custom_radius: Override default radius calculation
    """
    # Calculate radius (same as original)
    if custom_radius is None:
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
    else:
        radius = custom_radius
        
    # Camera intrinsics (same as original)
    cx = 512.0
    cy = 288.0  
    f = 500
    K = (
        torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        .repeat(num_frames, 1, 1)
        .to(opts.device)
    )
    
    # Initial camera pose (same as original)
    c2w_init = (
        torch.tensor([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], 
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .to(opts.device)
        .unsqueeze(0)
    )
    
    # Generate circular trajectory
    poses = generate_circular_trajectory(
        c2w_init, circle_type, radius, num_frames, opts.device
    )
    
    # Apply Z offset (same as original)
    poses[:, 2, 3] = poses[:, 2, 3] + radius
    
    # Source and target poses
    pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
    pose_t = poses
    
    return pose_s, pose_t, K
    
 
 # Utility function for easy use
def create_circular_preset_params(preset_name):
    """Create parameters for common circular motion presets"""
    presets = {
        "horizontal_circle": {
            "circle_type": "horizontal",
            "description": "Horizontal circular orbit around scene"
        },
        "vertical_orbit_xz": {
            "circle_type": "vertical_xz", 
            "description": "Vertical orbit in XZ plane (side view)"
        },
        "vertical_orbit_yz": {
            "circle_type": "vertical_yz",
            "description": "Vertical orbit in YZ plane (front view)" 
        },
        "tilted_orbit": {
            "circle_type": "tilted",
            "description": "45° tilted circular orbit"
        }
    }
    return presets.get(preset_name, presets["horizontal_circle"])       
        


# Generate CIRCULAR trajectory instead of linear
def generate_circular_scene_data(crafter, opts, scene_data, circle_type='horizontal'):
    """Generate new scene data with circular motion"""
    
    # Reuse existing depths and frames
    pose_s, pose_t, K = crafter.get_poses_circular(
        opts, 
        scene_data['depths'], 
        num_frames=opts.video_length,
        circle_type=circle_type
    )
    
    return {
        'frames_numpy': scene_data['frames_numpy'],
        'frames_tensor': scene_data['frames_tensor'],
        'depths': scene_data['depths'], 
        'pose_source': pose_s,
        'pose_target': pose_t,
        'intrinsics': K,
        'radius': scene_data['radius'],
        'trajectory_params': f"circular_{circle_type}"
    }
    
    
