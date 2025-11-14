import torch




import viser
import numpy as np

import os


def add_points(
    server,
    points: np.ndarray,  # (N, 3)
    colors: np.ndarray,  # (N, 3)
    name: str,
):
    # ensure colors are in [0, 1]
    if colors.min() < -0.1:
        colors = (colors + 1.0) / 2.0
    
    server.scene.add_point_cloud(
        name=name,
        points=points,
        colors=colors,
        point_size=0.01
    )

def add_camera(
    server,
    pose: np.ndarray,  # (4, 4)
    name: str,
    color: tuple = (0.2, 0.8, 0.2),
):
    pose = np.linalg.inv(pose)
    
    position = pose[:3, 3]
    rotation_matrix = pose[:3, :3]
    
    # Convert rotation to quaternion
    wxyz = viser.transforms.SO3.from_matrix(rotation_matrix).wxyz
    
    server.scene.add_camera_frustum(
        name,
        fov=60, aspect=4/3, scale=0.1,
        position=position, wxyz=wxyz,
        color=color
    )
    
if __name__ == '__main__':
    
    server = viser.ViserServer()
    # print slurm node name

    node_name=os.environ.get('SLURM_NODELIST', 'localhost')

    print(f'http://{node_name}:{server.get_port()}')

    
    # Add this after creating your server and adding point clouds
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        yaw_slider = client.gui.add_slider("Camera Yaw", min=-180, max=180, step=1, initial_value=0)
        
        @yaw_slider.on_update
        def _(_):
            angle_rad = np.deg2rad(yaw_slider.value)
            radius = 8.0
            
            position = np.array([
                radius * np.sin(angle_rad),
                0,  # height
                radius * np.cos(angle_rad)
            ])
            
            client.camera.position = position
            client.camera.look_at = np.array([0, 0, 0])
            client.camera.up_direction = np.array([0, -1, 0])



    server.scene.reset()

    # add original point cloud #0, and all cameras from first segment

    stage_i = 3
    step_j = 16


    base_dir = '/home/azhuravl/work/TrajectoryCrafter/experiments/14-11-2025/rhino_20251114_1753_90_0_0_0_1_auto_s4/'\
                f'stage_{stage_i}'

    for k in range(3):
        print('adding', f'{base_dir}/global_pc/pc_{step_j:02d}.pt')
        add_points(
            server,
            torch.load(f'{base_dir}/global_pc/pc_{step_j:02d}.pt', weights_only=True).numpy(),
            torch.load(f'{base_dir}/global_colors/color_{step_j:02d}.pt', weights_only=True).numpy(),
            name=f'pc_{k:02d}'
        )
        
    print("Viser server is running. Press Enter to exit.")
    input()  # Wait for user input