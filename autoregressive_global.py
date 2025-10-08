import torch
import numpy as np
import copy
import os
from pathlib import Path
import cv2
import shutil

from demo import TrajCrafter
from models.utils import Warper, read_video_frames, sphere2pose, save_video
import torch.nn.functional as F
from tqdm import tqdm

from models.infer import DepthCrafterDemo

import os
from datetime import datetime
import torch
import copy
import time
import sys
import tempfile
from pathlib import Path

# Add core.py to path if needed
sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/28_08_25_trajectories')
from core import VisualizationWarper

sys.path.append('/home/azhuravl/work/TrajectoryCrafter/notebooks/06_10_25_vggt')
from parsing import get_parser


def pad_video(frames, target_length):
    if frames.shape[0] < target_length:
        last_frame = frames[-1:]
        num_pad = target_length - frames.shape[0]
        pad_frames = np.repeat(last_frame, num_pad, axis=0)
        frames = np.concatenate([frames, pad_frames], axis=0)
    return frames


class TrajCrafterAutoregressive(TrajCrafter):
    def __init__(self, opts):
        super().__init__(opts)

        self.funwarp = VisualizationWarper(device=opts.device)
        self.prompt = None
        
        self.K = torch.tensor(
            [[500, 0.0, 512.], [0.0, 500, 288.], [0.0, 0.0, 1.0]]
            ).repeat(opts.video_length, 1, 1).to(opts.device)
        
        
    def extract_point_cloud(self, frames, c2ws, opts):
        
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        
        frames = torch.from_numpy(frames).to(opts.device)
        
        pc_list = []
        color_list = []
        for i in range(opts.video_length):
            pc, color = self.funwarp.extract_3d_points_with_colors(
                frames[i:i+1],
                depths[i:i+1],
                c2ws[i:i+1],
                self.K[i:i+1],
                subsample_step=1
            )
            pc_list.append(pc)
            color_list.append(color)
        
        return pc_list, color_list
    
    
    def generate_traj_specified(self, c2ws_anchor, target_pose, n_frames, device):
        theta, phi, d_r, d_x, d_y = target_pose
        
        thetas = np.linspace(0, theta, n_frames)  
        phis = np.linspace(0, phi, n_frames)          
        rs = np.linspace(0, d_r, n_frames)            
        xs = np.linspace(0, d_x, n_frames)            
        ys = np.linspace(0, d_y, n_frames)            
        
        c2ws_list = []
        for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
            c2w_new = sphere2pose(
                c2ws_anchor,
                np.float32(th),
                np.float32(ph),
                np.float32(r),
                device,
                np.float32(x),
                np.float32(y),
            )
            c2ws_list.append(c2w_new)
        c2ws = torch.cat(c2ws_list, dim=0)
        return c2ws
    
    
    def save_poses_colmap(self, c2ws, filepath):
        c2ws = c2ws.cpu().numpy()
        with open(filepath, 'w') as f:
            f.write('# Camera poses in COLMAP format\n')
            f.write(f'{c2ws.shape[0]}\n')
            for i, c2w in enumerate(c2ws):
                r = c2w[:3, :3].flatten()
                t = c2w[:3, 3]
                f.write(f'{i+1} {r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]} {r[7]} {r[8]} {t[0]} {t[1]} {t[2]}\n')
                
    
    def save_point_clouds_colmap(self, pc_list, color_list, dirpath):
        # save point cloud in COLMAP format, 1 file per point cloud
        # they are from a video, not multi-view
        os.makedirs(dirpath, exist_ok=True)
        
        for idx, (pc, color) in enumerate(zip(pc_list, color_list)):
            point_id = 1
            pc = pc.cpu().numpy()
            color = (color.cpu().numpy() * 255).astype(np.int32)
            with open(os.path.join(dirpath, f'point_cloud_{idx:03d}.txt'), 'w') as f:
                f.write('# Point cloud in COLMAP format\n')
                f.write(f'{pc.shape[0]} 0\n')
                for p, c in zip(pc, color):
                    x, y, z = p
                    r, g, b = c
                    f.write(f'{point_id} {x} {y} {z} {r} {g} {b}\n')
                    point_id += 1
                    
    
    def setup_exp_directory(self, opts, frames, c2ws_init, pc_input, colors, prompt, traj_segments):
        # create directories
        exp_dir = Path(opts.save_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        input_dir = exp_dir / 'input'
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # save input video
        save_video(
            (torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0).permute(0, 2, 3, 1) / 2.0,
            input_dir / 'input.mp4',
            fps=opts.fps,
        )
        
        # save camera poses in COLMAP format
        c2ws_init_np = c2ws_init.cpu().numpy()
        np.save(input_dir / 'cameras_input.npy', c2ws_init_np)
        self.save_poses_colmap(c2ws_init, input_dir / 'cameras_input.txt')
        
        # save point cloud in COLMAP format
        self.save_point_clouds_colmap(pc_input, colors, input_dir / 'point_clouds')
        
        # save prompt
        with open(input_dir / 'prompt.txt', 'w') as f:
            f.write(prompt)
        
        # create stage directories
        for i in range(opts.n_splits):
            stage_dir = exp_dir / f'stage_{i+1}'
            stage_dir.mkdir(parents=True, exist_ok=True)
        
            # save target camera poses for each segment
            c2ws_target_np = traj_segments[i].cpu().numpy()
            np.save(stage_dir / 'cameras_target.npy', c2ws_target_np)
            self.save_poses_colmap(traj_segments[i], stage_dir / 'cameras_target.txt')
        
        
    def generate_segment(self, frames, pc_input, color_input, traj_segment, segment_dir, opts):
        
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length


        # render the point clouds
        warped_images = []
        masks = []        
        for i in tqdm(range(opts.video_length)):
            # warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
            #     frames[i : i + 1],
            #     None,
            #     depths[i : i + 1],
            #     pose_s[i : i + 1],
            #     pose_t[i : i + 1],
            #     K[i : i + 1],
            #     None,
            #     opts.mask,
            #     twice=False,
            # )
            # warped_images.append(warped_frame2)
            # masks.append(mask2)

            output_frame, output_mask = self.funwarp.render_pointcloud_native(
                pc_input[i].unsqueeze(0),
                color_input[i].unsqueeze(0),
                traj_segment[i:i+1],
                self.K[i:i+1],
                image_size=(576, 1024),
                mask=opts.mask,
            )
            warped_images.append(output_frame)
            masks.append(output_mask)
            
            
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
            os.path.join(segment_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(segment_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(segment_dir, 'mask.mp4'),
            fps=opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        with torch.no_grad():            
            sample = self.pipeline(
                self.prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video.to(opts.device),
                mask_video=cond_masks.to(opts.device),
                reference=frames_ref,
            ).videos
            
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(segment_dir, 'gen.mp4'),
            fps=opts.fps,
        )
        
        return sample[0].permute(0, 2, 3, 1)

        
        
    def infer_autoregressive(self, opts):
        
        # read input video
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        # frames = torch.from_numpy(frames).to(opts.device)
        
        # pad if too short
        frames = pad_video(frames, opts.video_length)
        
        # prompt
        self.prompt = self.get_caption(opts, frames[opts.video_length // 2])
        
        ########################################################
        # Geometric FM
        ########################################################
        
        c2ws_init = torch.tensor([
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
            ]).repeat(opts.video_length, 1, 1).to(opts.device)
        
        pc_input, color_input = self.extract_point_cloud(frames, c2ws_init, opts)
        
        ########################################################
        # Camera Pose Planning
        ########################################################
             
        c2ws_target = self.generate_traj_specified(
            c2ws_init[0:1], 
            opts.target_pose, 
            opts.video_length * opts.n_splits, 
            opts.device
        )
        
        # split into segments
        traj_segments = c2ws_target.view(opts.n_splits, opts.video_length, 4, 4)
        
        ########################################################
        # Autoregressive Generation
        ########################################################
        
        for i in range(opts.n_splits):
            
            segment_dir = os.path.join(opts.save_dir, f'stage_{i+1}')
            os.makedirs(segment_dir, exist_ok=True)
            
            inpainted_video = self.generate_segment(
                frames, pc_input, color_input, traj_segments[i], segment_dir, opts
                )
            
            pc_inpainted, color_inpainted = self.extract_point_cloud(inpainted_video, traj_segments[i], opts)
            
            # pc_merged, color_merged = merge_point_clouds(pc_input, color_input, pc_inpainted, color_inpainted)
            pc_merged, color_merged = pc_inpainted, color_inpainted
                        
            save_segment_results(
                pc_input,
                color_input,
                pc_inpainted,
                color_inpainted,            
                pc_merged,
                color_merged,
                traj_segments[i],
                opts, segment_idx=i)
            
            frames = inpainted_video
            pc_input = pc_merged
            color_input = color_merged
            
            
            
        
        
        
        
    # get global cam_pos of input video + n=video_length 3D point clouds
    # poses_input, pc_input = geometric_fm(frames, opts)
    
    # # plan trajectory
    # traj_segments = plan_trajectory(
    #     poses_input[0], opts.target_pose, opts.n_splits
    # )
    
    # save everything
    # Directory Structure
    # - exp_name/
    
    #   - input/
    #     + input.mp4
    #     + cameras_input.npy
    #     + point_cloud_input.txt in COLMAP format
    #     + prompt.txt
    
    #   - stage_1/
    #     - input.mp4 mask.mp4 render.mp4 gen.mp4
    #     - point_cloud_input.ply in COLMAP format
    #     + cameras_target.npy
    # setup_exp_directory(opts, frames, poses_input, pc_input, prompt, traj_segments)
    
    # pc_global = pc_input
    
    # # 1: autoregressive generation
    # for i in range(opts.n_splits):
        
    #     # TODO: video reversal for even segments
        
    #     inpainted_video = generate_segment(frames, pc_global, traj_segments[i], opts)
        
    #     pc_inpainted = geometric_fm(inpainted_video, opts)
    #     pc_global = merge_point_clouds(pc_global, pc_inpainted)
        
    #     save_segment_results(inpainted_video, pc_global, traj_segments[i], opts, segment_idx=i)    
        


# opts_base.video_path = "/home/azhuravl/nobackup/DAVIS_testing/trainval/rhino.mp4"
# opts_base.n_splits = 4
# opts_base.overlap_frames = 0
# opts_base.radius = 0
# opts_base.mode = "gradual"
# put above in args
# sys.argv = ["", "--epochs", "10", "--lr", "0.001"]

if __name__ == "__main__":

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
    vis_crafter = TrajCrafterAutoregressive(opts_base)

    radius = opts_base.radius

    variants = [
        ("right_90", [0, 90, radius, 0, 0]),
    ]
    name = "right_90"
    pose = [0, 90, radius, 0, 0]

    print(f"\n=== Running Autoregressive {name} ===")
    opts = copy.deepcopy(opts_base)
    opts.exp_name = f"{video_basename}_{timestamp}_{name}_auto_s{opts_base.n_splits}"
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    opts.camera = "target"
    opts.target_pose = pose
    opts.traj_txt = 'test/trajs/loop2.txt'

    # Make directories
    os.makedirs(opts.save_dir, exist_ok=True)

    start_time = time.time()

    # Use autoregressive generation for large trajectories
    final_video = vis_crafter.infer_autoregressive(
        opts, 
        # n_splits=opts_base.n_splits,
        # overlap_frames=opts_base.overlap_frames
    )

    elapsed = time.time() - start_time
    print(f"Finished {name} in {elapsed:.2f} seconds")