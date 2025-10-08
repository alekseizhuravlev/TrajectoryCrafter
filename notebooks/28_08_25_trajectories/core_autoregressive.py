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


class TrajCrafterAutoregressive(TrajCrafter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_pose = [0.0, 0.0, 0.0, 0.0, 0.0]  # Track cumulative state
        self.pose_s = None
        self.pose_t = None


    def infer_autoregressive(self, opts, n_splits=3, overlap_frames=5):
        """Combined autoregressive generation with pose recording"""
        
        # Setup directories and trajectory splits
        intermediate_dir = Path(opts.save_dir) / "autoregressive_segments"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        original_target_pose = opts.target_pose.copy()
        dtheta, dphi, dr, dx, dy = original_target_pose
        
        # Calculate trajectory deltas
        theta_deltas = [dtheta * (i + 1) / n_splits for i in range(n_splits)]
        phi_deltas = [dphi * (i + 1) / n_splits for i in range(n_splits)]
        dr_deltas = [dr * (i + 1) / n_splits for i in range(n_splits)]
        dx_deltas = [dx * (i + 1) / n_splits for i in range(n_splits)]
        dy_deltas = [dy * (i + 1) / n_splits for i in range(n_splits)]
        
        theta_deltas = [theta_deltas[0]] + [theta_deltas[i] - theta_deltas[i-1] for i in range(1, n_splits)]
        phi_deltas = [phi_deltas[0]] + [phi_deltas[i] - phi_deltas[i-1] for i in range(1, n_splits)]
        dr_deltas = [dr_deltas[0]] + [dr_deltas[i] - dr_deltas[i-1] for i in range(1, n_splits)]
        dx_deltas = [dx_deltas[0]] + [dx_deltas[i] - dx_deltas[i-1] for i in range(1, n_splits)]
        dy_deltas = [dy_deltas[0]] + [dy_deltas[i] - dy_deltas[i-1] for i in range(1, n_splits)]
        
        # Initialize pose tracking
        current_poses = None
        generated_videos = []
        render_videos = []  
        
        for segment_idx in range(n_splits):
            print(f"\n=== Segment {segment_idx + 1}/{n_splits} ===")
            
            # Set segment directory
            segment_dir = intermediate_dir / f"segment_{segment_idx:02d}"
            segment_dir.mkdir(exist_ok=True)
            opts.save_dir = str(segment_dir)
            
            # Set trajectory for this segment
            opts.target_pose = [theta_deltas[segment_idx], phi_deltas[segment_idx], 
                               dr_deltas[segment_idx], dx_deltas[segment_idx], dy_deltas[segment_idx]]
            
            # Generate segment with pose continuation
            current_poses = self._infer_segment(opts, current_poses, segment_idx)
            
            # Track generated video
            gen_path = segment_dir / "gen.mp4"
            if gen_path.exists():
                generated_videos.append(str(gen_path))
                opts.video_path = str(gen_path)  # Use for next segment
                
            # Track render video
            render_path = segment_dir / "render.mp4"
            if render_path.exists():
                render_videos.append(str(render_path))
        
        # Create final result
        final_path = Path(opts.save_dir).parent / "autoregressive_result.mp4"
        render_final_path = Path(opts.save_dir).parent / "autoregressive_render.mp4"
        
        # concatenate all generated videos
        self._concatenate_videos(generated_videos, final_path, overlap_frames)
        self._concatenate_videos(render_videos, render_final_path, overlap_frames)
        
        return str(final_path)
    
    def _infer_segment(self, opts, previous_poses, segment_idx):
        """Generate single segment with pose continuation"""
        
        # Read and prepare frames
        
        print(f"Reading frames from {opts.video_path}")
        
        frames = read_video_frames(opts.video_path, opts.video_length, opts.stride, opts.max_res)
        if frames.shape[0] < opts.video_length:
            last_frame = frames[-1:]
            num_pad = opts.video_length - frames.shape[0]
            pad_frames = np.repeat(last_frame, num_pad, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
        
        # Get depth and poses
        depths = self.depth_estimater.infer(frames, opts.near, opts.far, 
                                           opts.depth_inference_steps, opts.depth_guidance_scale,
                                           window_size=opts.window_size, overlap=opts.overlap).to(opts.device)
        
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        
        # Get poses with continuation
        pose_s, pose_t = self._get_poses_with_continuation(opts, depths, previous_poses, segment_idx)
        
        # if opt.direct == True, only use the last target pose
        if opts.mode == "direct":
            pose_t = pose_t[-1:].repeat(opts.video_length, 1, 1)
            
        # print the shape of poses_t
        # print(f"Pose_t shape: {pose_t.shape}")
        
        # Continue with standard warping and generation...
        K = self._get_intrinsics(opts, depths)
        warped_images, masks = self._warp_frames(frames, depths, pose_s, pose_t, K, opts)
        
        # Generate video
        # Convert tensor back to numpy array with proper format for get_caption
        frame_tensor = frames[opts.video_length // 2]  # Shape: (C, H, W)
        frame_numpy = frame_tensor.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C)
        frame_numpy = ((frame_numpy + 1.0) / 2.0 * 255).astype(np.uint8)  # Convert from [-1,1] to [0,255]
        prompt = self.get_caption(opts, frame_numpy)

        self._generate_video(frames, warped_images, masks, prompt, opts)
        
        return pose_t  # Return poses for next segment
    
    # def _get_poses_with_continuation(self, opts, depths, previous_poses, segment_idx):
    #     """Get poses with autoregressive continuation"""
        
    #     # Calculate target poses for this segment
    #     radius = (depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu() * opts.radius_scale)
    #     radius = min(radius, 5)
        
        
    #     ################
    #     # is this the problem?
    #     ###############
    #     c2w_init = torch.tensor([[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], 
    #                             [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]).to(opts.device).unsqueeze(0)
        
    #     # Calculate cumulative input pose
    #     current_theta = sum([opts.target_pose[0] * i for i in range(segment_idx)])
    #     current_phi = sum([opts.target_pose[1] * i for i in range(segment_idx)])
    #     input_pose = [current_theta, current_phi, 0, 0, 0]
        
    #     # Generate target poses
    #     dtheta, dphi, dr, dx, dy = opts.target_pose
    #     pose_t = self.generate_traj_specified(c2w_init, (dtheta, dphi, dr * radius, dx, dy), 
    #                                          input_pose, opts.video_length, opts.device)
    #     pose_t[:, 2, 3] = pose_t[:, 2, 3] + radius
        
    #     # Set source poses
    #     if segment_idx == 0:
    #         # First segment: use anchor
    #         pose_s = pose_t[0:1].repeat(opts.video_length, 1, 1)
    #     else:
    #         # Subsequent segments: use last pose from previous segment
    #         # pose_s = previous_poses[-1:].repeat(opts.video_length, 1, 1)
    #         pose_s = previous_poses
        
    #     return pose_s, pose_t
    
    
    def _get_poses_with_continuation(self, opts, depths, previous_poses, segment_idx):
        """Get poses with autoregressive continuation"""
        
        # Calculate target poses for this segment
        radius = (depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu() * opts.radius_scale)
        radius = min(radius, 5)
        
        # Define c2w_init (this was missing!)
        c2w_init = torch.tensor([
            [-1.0, 0.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0, 0.0], 
            [0.0, 0.0, -1.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0]
        ]).to(opts.device).unsqueeze(0)
        
        # Use the tracked cumulative pose as input
        input_pose = self.cumulative_pose.copy()
        
        # Generate segment poses
        pose_t = self.generate_traj_specified(
            c2w_init, 
            opts.target_pose, 
            input_pose, 
            opts.video_length, 
            opts.device
        )
        
        print()
        
        # Apply radius offset
        pose_t[:, 2, 3] = pose_t[:, 2, 3] + radius
        
        # Set source poses
        if segment_idx == 0:
            # First segment: use anchor from first pose
            pose_s = pose_t[0:1].repeat(opts.video_length, 1, 1)
        else:
            # Subsequent segments: use last pose from previous segment
            if previous_poses is not None:
                # pose_s = previous_poses[-1:].repeat(opts.video_length, 1, 1)
                pose_s = previous_poses
                # pose_s[:, 2, 3] = pose_s[:, 2, 3] + radius  # Apply radius offset
            else:
                pose_s = pose_t[0:1].repeat(opts.video_length, 1, 1)
        
        if segment_idx == 1:
            self.pose_s = pose_s
            self.pose_t = pose_t
        elif segment_idx > 1:
            pose_s = self.pose_s
            pose_t = self.pose_t
            
        
        
        # Update cumulative state for next segment
        delta_theta, delta_phi, delta_dr, delta_dx, delta_dy = opts.target_pose
        self.cumulative_pose[0] += delta_theta
        self.cumulative_pose[1] += delta_phi
        self.cumulative_pose[2] += delta_dr
        self.cumulative_pose[3] += delta_dx
        self.cumulative_pose[4] += delta_dy
        
        return pose_s, pose_t
    
    def _get_intrinsics(self, opts, depths):
        """Get camera intrinsics"""
        cx, cy, f = 512.0, 288.0, 500
        return torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]).repeat(opts.video_length, 1, 1).to(opts.device)
    
    def _warp_frames(self, frames, depths, pose_s, pose_t, K, opts):
        """Warp frames using poses"""
        warped_images, masks = [], []
        for i in range(opts.video_length):
            warped_frame, mask, _, _ = self.funwarp.forward_warp(
                frames[i:i+1], None, depths[i:i+1], pose_s[i:i+1], pose_t[i:i+1], K[i:i+1], None, opts.mask, twice=False)
            warped_images.append(warped_frame)
            masks.append(mask)
        return warped_images, masks
    
    def _generate_video(self, frames, warped_images, masks, prompt, opts):
        """Generate final video"""
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        
        frames = F.interpolate(frames, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_video = F.interpolate(cond_video, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        
        # Save intermediate videos
        save_video((frames.permute(0, 2, 3, 1) + 1.0) / 2.0, os.path.join(opts.save_dir, 'input.mp4'), fps=opts.fps)
        save_video(cond_video.permute(0, 2, 3, 1), os.path.join(opts.save_dir, 'render.mp4'), fps=opts.fps)
        save_video(cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1), os.path.join(opts.save_dir, 'mask.mp4'), fps=opts.fps)
        
        # Generate with pipeline
        frames_prepared = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames_prepared[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)
        with torch.no_grad():
            sample = self.pipeline(prompt, num_frames=opts.video_length, negative_prompt=opts.negative_prompt,
                                  height=opts.sample_size[0], width=opts.sample_size[1], generator=generator,
                                  guidance_scale=opts.diffusion_guidance_scale, num_inference_steps=opts.diffusion_inference_steps,
                                  video=cond_video.to(opts.device), mask_video=cond_masks.to(opts.device), reference=frames_ref).videos
        
        # # use cond video instead of generated, i want to skip generation part
        # sample = cond_video
        
        save_video(sample[0].permute(1, 2, 3, 0), os.path.join(opts.save_dir, 'gen.mp4'), fps=opts.fps)



    def _create_progression_video(self, video_paths, output_path):
        """
        Create a video showing the progression through all segments
        """
        import cv2
        import numpy as np
        
        print(f"Creating progression video with {len(video_paths)} segments...")
        
        if not video_paths:
            return
        
        # Read all videos
        all_video_frames = []
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            all_video_frames.append(frames)
        
        if not all_video_frames or not all_video_frames[0]:
            return
        
        # All videos should have the same number of frames
        num_frames = len(all_video_frames[0])
        h, w = all_video_frames[0][0].shape[:2]
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 8.0, (w, h))
        
        # Create progression: play each segment in sequence
        for segment_idx, frames in enumerate(all_video_frames):
            print(f"  Adding segment {segment_idx + 1} to progression...")
            for frame_idx, frame in enumerate(frames):
                # Add segment label
                labeled_frame = frame.copy()
                cv2.putText(labeled_frame, f"Segment {segment_idx + 1}/{len(video_paths)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(labeled_frame, f"Frame {frame_idx + 1}/{len(frames)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                out.write(labeled_frame)
        
        out.release()
        print(f"✅ Progression video created: {output_path}")


    def _prepare_next_input_video(self, generated_video_path, output_path, overlap_frames):
        """
        Prepare input video for next segment by taking the last part of generated video
        """
        import cv2
        import numpy as np
        
        print(f"Preparing next input from {generated_video_path}")
        
        # Read generated video
        cap = cv2.VideoCapture(generated_video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) < overlap_frames:
            print(f"Warning: Video has only {len(frames)} frames, using all")
            selected_frames = frames
        else:
            # Take last overlap_frames + some extra for the next segment
            start_idx = len(frames) - overlap_frames - 10  # Extra frames for context
            start_idx = max(0, start_idx)
            selected_frames = frames[start_idx:]
        
        # Write new video
        if len(selected_frames) > 0:
            height, width = selected_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 8.0, (width, height))
            
            for frame in selected_frames:
                out.write(frame)
            
            out.release()
            
            print(f"Next input video created: {output_path} ({len(selected_frames)} frames)")
            return output_path
        else:
            raise ValueError("No frames to create next input video")
    
    def _concatenate_videos(self, video_paths, output_path, overlap_frames):
        """
        Concatenate videos with overlap handling
        """
        import cv2
        import numpy as np
        
        print(f"Concatenating {len(video_paths)} videos...")
        
        all_frames = []
        
        for i, video_path in enumerate(video_paths):
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if i == 0:
                # First video: take all frames
                all_frames.extend(frames)
            else:
                # Subsequent videos: skip overlap frames at beginning
                skip_frames = min(overlap_frames, len(frames) // 2)
                all_frames.extend(frames[skip_frames:])
            
            print(f"  Video {i+1}: {len(frames)} frames, added {len(frames) if i == 0 else len(frames) - skip_frames}")
        
        # Write concatenated video
        if all_frames:
            height, width = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 8.0, (width, height))
            
            for frame in all_frames:
                out.write(frame)
            
            out.release()
            print(f"✅ Concatenated video created: {output_path} ({len(all_frames)} frames)")
        else:
            raise ValueError("No frames to concatenate")
    
    def _create_comparison_video(self, original_path, generated_path, output_path):
        """
        Create side-by-side comparison video
        """
        import cv2
        import numpy as np
        
        print("Creating comparison video...")
        
        # Read both videos
        cap1 = cv2.VideoCapture(original_path)
        cap2 = cv2.VideoCapture(generated_path)
        
        frames1, frames2 = [], []
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 and not ret2:
                break
                
            if ret1:
                frames1.append(frame1)
            if ret2:
                frames2.append(frame2)
        
        cap1.release()
        cap2.release()
        
        # Create side-by-side comparison
        min_frames = min(len(frames1), len(frames2))
        if min_frames > 0:
            h1, w1 = frames1[0].shape[:2]
            h2, w2 = frames2[0].shape[:2]
            
            # Resize to same height
            target_height = min(h1, h2)
            target_width1 = int(w1 * target_height / h1)
            target_width2 = int(w2 * target_height / h2)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 8.0, (target_width1 + target_width2, target_height))
            
            for i in range(min_frames):
                frame1 = cv2.resize(frames1[i], (target_width1, target_height))
                frame2 = cv2.resize(frames2[i], (target_width2, target_height))
                
                # Add labels
                cv2.putText(frame1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame2, "Generated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                combined = np.concatenate([frame1, frame2], axis=1)
                out.write(combined)
            
            out.release()
            print(f"✅ Comparison video created: {output_path}")
            
            
    def get_poses(self, opts, depths, num_frames):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = self.generate_traj_specified(
                c2w_init, 
                (dtheta, dphi, dr * radius, dx, dy), 
                opts.input_pose,
                num_frames, opts.device
            )
        # elif opts.camera == 'traj':
        #     with open(opts.traj_txt, 'r') as file:
        #         lines = file.readlines()
        #         theta = [float(i) for i in lines[0].split()]
        #         phi = [float(i) for i in lines[1].split()]
        #         r = [float(i) * radius for i in lines[2].split()]
        #     poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
            
            
        # print('get_poses(), poses', poses)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        
        # print('get_poses(), poses[:, 2, 3] + radius', poses)
        
        # pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        
        pose_s = opts.source_poses
        pose_t = poses
        return pose_s, pose_t, K
    
    
    def generate_traj_specified(self, c2ws_anchor, target_pose, input_pose, frame, device):
        theta, phi, d_r, d_x, d_y = target_pose
        theta_0, phi_0, d_r_0, d_x_0, d_y_0 = input_pose        
        
        # Generate trajectories from input_pose to target_pose (RELATIVE movement)
        thetas = np.linspace(theta_0, theta_0 + theta, frame)  # Add relative movement
        phis = np.linspace(phi_0, phi_0 + phi, frame)          # Add relative movement  
        rs = np.linspace(d_r_0, d_r_0 + d_r, frame)            # Add relative movement
        xs = np.linspace(d_x_0, d_x_0 + d_x, frame)            # Add relative movement
        ys = np.linspace(d_y_0, d_y_0 + d_y, frame)            # Add relative movement
        
        print("Generating trajectories")       
        print(f"  From: θ={theta_0:.1f}°, φ={phi_0:.1f}°, r={d_r_0:.2f}, x={d_x_0:.2f}, y={d_y_0:.2f}")
        print(f"  To:   θ={theta_0 + theta:.1f}°, φ={phi_0 + phi:.1f}°, r={d_r_0 + d_r:.2f}, x={d_x_0 + d_x:.2f}, y={d_y_0 + d_y:.2f}")
        print(f"  Delta: θ={theta:.1f}°, φ={phi:.1f}°, r={d_r:.2f}, x={d_x:.2f}, y={d_y:.2f}")
        
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
            print('w2c_new', np.linalg.inv(c2w_new.cpu().numpy()))
            c2ws_list.append(c2w_new)
        c2ws = torch.cat(c2ws_list, dim=0)
        return c2ws
    
    
