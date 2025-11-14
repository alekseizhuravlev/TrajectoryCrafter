import sys
import os
import copy
from datetime import datetime
import torch
import numpy as np

sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/06_10_25_vggt')
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/12_11_25_consistent_depth')

from utils_autoregressive import load_video_frames, generate_traj_specified, TrajCrafterAutoregressive
from parsing import get_parser
from autoregressive_loop import autoregressive_loop
from warper_point_cloud import GlobalPointCloudWarper

def setup_opts():
    sys.argv = [
        "",
        "--video_path", "/home/azhuravl/nobackup/DAVIS_testing/trainval/rhino.mp4",
        "--n_splits", "4",
        "--overlap_frames", "0",
        "--radius", "0",
        "--mode", "gradual",
    ]

    parser = get_parser()
    opts_base = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    video_basename = os.path.splitext(os.path.basename(opts_base.video_path))[0]

    # Setup
    opts_base.weight_dtype = torch.bfloat16
    opts_base.exp_name = f"{video_basename}_{timestamp}_autoregressive"
    opts_base.save_dir = os.path.join(opts_base.out_dir, opts_base.exp_name)

    # Create TrajCrafterVisualization instance for autoregressive generation
    radius = opts_base.radius

    pose = [90, 0, 0, 0, 1]
    # name = '120_0_0_0_3', make it infer values from pose
    name = f"{pose[0]}_{pose[1]}_{pose[2]}_{pose[3]}_{pose[4]}"


    print(f"\n=== Running Autoregressive {name} ===")
    opts = copy.deepcopy(opts_base)
    opts.exp_name = f"{video_basename}_{timestamp}_{name}_auto_s{opts_base.n_splits}"
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    opts.camera = "target"
    opts.target_pose = pose
    opts.traj_txt = 'test/trajs/loop2.txt'

    # Make directories
    os.makedirs(opts.save_dir, exist_ok=True)

    return opts



if __name__ == '__main__':
    
    opts = setup_opts()
    
    vis_crafter = TrajCrafterAutoregressive(opts)
    funwarp = GlobalPointCloudWarper(device=opts.device, max_points=2000000)
    
    # TODO: depth estimator + alignment
    
    
    
    
    
    


    # Load target frames with optional reversal
    frames_input_np, _ = load_video_frames(
        opts.video_path, 
        opts.video_length, 
        opts.stride, 
        opts.max_res,
        opts.device,
        reverse=True
    )
    # depth estimation
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        depths_input = vis_crafter.depth_estimater.infer(
            frames_input_np,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        

    ##########################################
    # Cameras
    ##########################################

    radius = (
        depths_input[0, 0, depths_input.shape[-2] // 2, depths_input.shape[-1] // 2].cpu()
        * opts.radius_scale
    )
    radius = min(radius, 5)

    # radius = 10


    c2ws_anchor = torch.tensor([ 
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
        ]).unsqueeze(0).to(opts.device)

    c2ws_target = generate_traj_specified(
        c2ws_anchor, 
        opts.target_pose, 
        opts.video_length * opts.n_splits, 
        opts.device
    )
    c2ws_target[:, 2, 3] += radius

    c2ws_init = c2ws_target[0].repeat(opts.video_length, 1, 1)


    traj_segments = c2ws_target.view(opts.n_splits, opts.video_length, 4, 4)
    
    ##########################
    
    frames_source_np = None
    frames_target_np = frames_input_np

    poses_source = c2ws_init
    poses_target = traj_segments[0]
        
    for i in range(opts.n_splits):
        segment_dir_autoreg = autoregressive_loop(
            frames_source_np,
            frames_target_np,
            poses_source,
            poses_target,
            radius,
            opts,  
            vis_crafter,
            funwarp,
        )
        
        # concatenate frames_source_np and frames_target_np, handling None initially
        if frames_source_np is None:
            frames_source_np = frames_target_np
        else:
            frames_source_np = np.concatenate([frames_source_np, frames_target_np], axis=0)
        
        frames_target_np, _ = load_video_frames(
            segment_dir_autoreg + '/gen.mp4', 
            opts.video_length, 
            opts.stride, 
            opts.max_res,
            opts.device,
            reverse=False
        )
        
        poses_source = torch.cat([poses_source, poses_target], dim=0)
        poses_target = traj_segments[i+1] if i + 1 < opts.n_splits else None
        
        # break
            