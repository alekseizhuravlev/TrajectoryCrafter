import torch
import numpy as np
from demo import TrajCrafter
import json
import os
from datetime import datetime
import cv2
import sys

# Add evaluation metrics
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

# Add MVTracker to path
sys.path.append('/home/azhuravl/work/mvtracker')
from mvtracker.datasets.panoptic_studio_multiview_dataset import PanopticStudioMultiViewDataset
from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset

class CameraPoseTrajCrafter(TrajCrafter):
    def __init__(self, opts, gradio=False):
        super().__init__(opts, gradio)
        # Don't initialize depth_estimater since we get depth from dataset
        self.depth_estimater = None
        
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(opts.device)
        print("==> LPIPS model initialized")
    
    def calculate_metrics(self, generated_frame, target_frame, save_dir, seq_name):
        """
        Calculate PSNR, SSIM, and LPIPS between generated and target frames
        
        Args:
            generated_frame: [H, W, 3] tensor in range [0, 1]
            target_frame: [H, W, 3] tensor in range [0, 1] 
            save_dir: directory to save metrics
            seq_name: sequence name for naming
        """
        print("==> calculating evaluation metrics for last frame")
        
        # Ensure frames are on CPU and in numpy format for PSNR/SSIM
        if torch.is_tensor(generated_frame):
            gen_np = generated_frame.detach().cpu().numpy()
        else:
            gen_np = generated_frame
            
        if torch.is_tensor(target_frame):
            tgt_np = target_frame.detach().cpu().numpy()
        else:
            tgt_np = target_frame
            
        # Ensure values are in [0, 1] range
        gen_np = np.clip(gen_np, 0, 1)
        tgt_np = np.clip(tgt_np, 0, 1)
        
        # Calculate PSNR
        psnr_value = psnr(tgt_np, gen_np, data_range=1.0)
        
        # Calculate SSIM (convert to grayscale for SSIM or use multichannel)
        ssim_value = ssim(tgt_np, gen_np, multichannel=True, channel_axis=2, data_range=1.0)
        
        # Calculate LPIPS (requires tensors in [-1, 1] range and [C, H, W] format)
        if torch.is_tensor(generated_frame):
            gen_tensor = generated_frame.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        else:
            gen_tensor = torch.from_numpy(generated_frame).permute(2, 0, 1).unsqueeze(0)
            
        if torch.is_tensor(target_frame):
            tgt_tensor = target_frame.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        else:
            tgt_tensor = torch.from_numpy(target_frame).permute(2, 0, 1).unsqueeze(0)
        
        # Convert to [-1, 1] range for LPIPS
        gen_tensor = gen_tensor * 2.0 - 1.0
        tgt_tensor = tgt_tensor * 2.0 - 1.0
        
        # Move to device and calculate LPIPS
        gen_tensor = gen_tensor.to(self.lpips_model.net.parameters().__next__().device)
        tgt_tensor = tgt_tensor.to(self.lpips_model.net.parameters().__next__().device)
        
        with torch.no_grad():
            lpips_value = self.lpips_model(gen_tensor, tgt_tensor).item()
        
        # Create metrics dictionary
        metrics = {
            "sequence_name": seq_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": {
                "PSNR": float(psnr_value),
                "SSIM": float(ssim_value), 
                "LPIPS": float(lpips_value)
            },
            "frame_info": {
                "frame_shape": list(gen_np.shape),
                "value_range": [0.0, 1.0],
                "compared_frames": "last_frame_generated_vs_target"
            }
        }
        
        # Save metrics to JSON file
        metrics_path = os.path.join(save_dir, f'{psnr_value:.2f}_{ssim_value:.2f}_{lpips_value:.2f}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also save a summary metrics file
        summary_path = os.path.join(save_dir, 'metrics_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Evaluation Metrics for {seq_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"PSNR:  {psnr_value:.4f} dB\n")
            f.write(f"SSIM:  {ssim_value:.4f}\n")
            f.write(f"LPIPS: {lpips_value:.4f}\n")
            f.write(f"\nTimestamp: {datetime.now()}\n")
        
        print(f"==> Evaluation metrics:")
        print(f"    PSNR:  {psnr_value:.4f} dB")
        print(f"    SSIM:  {ssim_value:.4f}")
        print(f"    LPIPS: {lpips_value:.4f}")
        print(f"==> Metrics saved to: {metrics_path}")
        print(f"==> Summary saved to: {summary_path}")
        
        return metrics

    def save_comparison_images(self, generated_frame, target_frame, save_dir, seq_name):
        """Save last frame comparison as images"""
        
        # Ensure frames are in [0, 1] range and numpy format
        if torch.is_tensor(generated_frame):
            gen_img = (generated_frame.detach().cpu().numpy() * 255).astype(np.uint8)
        else:
            gen_img = (np.clip(generated_frame, 0, 1) * 255).astype(np.uint8)
            
        if torch.is_tensor(target_frame):
            tgt_img = (target_frame.detach().cpu().numpy() * 255).astype(np.uint8)
        else:
            tgt_img = (np.clip(target_frame, 0, 1) * 255).astype(np.uint8)
        
        # Save individual images
        gen_path = os.path.join(save_dir, f'last_frame_generated_{seq_name}.png')
        tgt_path = os.path.join(save_dir, f'last_frame_target_{seq_name}.png')
        
        # Convert RGB to BGR for cv2
        cv2.imwrite(gen_path, cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tgt_path, cv2.cvtColor(tgt_img, cv2.COLOR_RGB2BGR))
        
        # Create side-by-side comparison
        comparison_img = np.concatenate([gen_img, tgt_img], axis=1)
        comparison_path = os.path.join(save_dir, f'last_frame_comparison_{seq_name}.png')
        cv2.imwrite(comparison_path, cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))
        
        print(f"==> Comparison images saved:")
        print(f"    Generated: {gen_path}")
        print(f"    Target: {tgt_path}")
        print(f"    Side-by-side: {comparison_path}")

    # ... [keep all existing methods unchanged until infer_camera_poses_mvtracker] ...

    def convert_camera_format_from_mvtracker(self, intrs, extrs):
        """Convert MVTracker camera format to c2w matrix and intrinsics"""
        # intrs: [3, 3] intrinsic matrix
        # extrs: [3, 4] extrinsic matrix [R|t]
        
        K = intrs.numpy() if torch.is_tensor(intrs) else intrs
        extr = extrs.numpy() if torch.is_tensor(extrs) else extrs
        
        # Create 4x4 extrinsic matrix
        RT = np.vstack([extr, [0, 0, 0, 1]])
        
        # Convert to c2w (camera to world)
        c2w = RT  # Use as is, or np.linalg.inv(RT) depending on your convention
        
        return torch.tensor(c2w, dtype=torch.float32), torch.tensor(K, dtype=torch.float32)
    
    def interpolate_camera_poses(self, source_c2w, target_c2w, num_steps):
        """
        Interpolate camera poses from source to target over num_steps
        
        Args:
            source_c2w: [4, 4] tensor - source camera-to-world matrix
            target_c2w: [4, 4] tensor - target camera-to-world matrix  
            num_steps: int - number of interpolation steps
            
        Returns:
            interpolated_poses: [num_steps, 4, 4] tensor - interpolated poses
        """
        print(f"==> interpolating camera poses over {num_steps} steps")
        
        # Extract rotation matrices and translation vectors
        R_src = source_c2w[:3, :3]  # [3, 3]
        t_src = source_c2w[:3, 3]   # [3]
        R_tgt = target_c2w[:3, :3]  # [3, 3]
        t_tgt = target_c2w[:3, 3]   # [3]
        
        interpolated_poses = []
        
        for i in range(num_steps):
            # Linear interpolation parameter [0, 1]
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            
            # Interpolate translation (simple linear interpolation)
            t_interp = (1 - alpha) * t_src + alpha * t_tgt
            
            # Interpolate rotation using SLERP (Spherical Linear Interpolation)
            R_interp = self.slerp_rotation_matrices(R_src, R_tgt, alpha)
            
            # Construct interpolated pose matrix
            pose_interp = torch.zeros(4, 4, dtype=source_c2w.dtype, device=source_c2w.device)
            pose_interp[:3, :3] = R_interp
            pose_interp[:3, 3] = t_interp
            pose_interp[3, 3] = 1.0
            
            interpolated_poses.append(pose_interp)
            
            # Debug info for first, middle, and last frames
            if i == 0:
                print(f"  Frame {i:2d} (source): t={t_interp.numpy()}")
            elif i == num_steps // 2:
                print(f"  Frame {i:2d} (middle): t={t_interp.numpy()}")
            elif i == num_steps - 1:
                print(f"  Frame {i:2d} (target): t={t_interp.numpy()}")
        
        return torch.stack(interpolated_poses)
    
    def interpolate_intrinsics(self, source_K, target_K, num_steps):
        """
        Interpolate camera intrinsics from source to target over num_steps
        
        Args:
            source_K: [3, 3] tensor - source intrinsics
            target_K: [3, 3] tensor - target intrinsics
            num_steps: int - number of interpolation steps
            
        Returns:
            interpolated_K: [num_steps, 3, 3] tensor - interpolated intrinsics
        """
        print(f"==> interpolating camera intrinsics over {num_steps} steps")
        
        interpolated_K = []
        
        for i in range(num_steps):
            # Linear interpolation parameter [0, 1]
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            
            # Linear interpolation of intrinsic parameters
            K_interp = (1 - alpha) * source_K + alpha * target_K
            interpolated_K.append(K_interp)
            
            # Debug info for first and last frames
            if i == 0:
                print(f"  Frame {i:2d} (source): fx={K_interp[0,0]:.1f}, fy={K_interp[1,1]:.1f}, cx={K_interp[0,2]:.1f}, cy={K_interp[1,2]:.1f}")
            elif i == num_steps - 1:
                print(f"  Frame {i:2d} (target): fx={K_interp[0,0]:.1f}, fy={K_interp[1,1]:.1f}, cx={K_interp[0,2]:.1f}, cy={K_interp[1,2]:.1f}")
        
        return torch.stack(interpolated_K)
    
    def slerp_rotation_matrices(self, R1, R2, t):
        """
        Spherical linear interpolation between two rotation matrices
        
        Args:
            R1: [3, 3] tensor - first rotation matrix
            R2: [3, 3] tensor - second rotation matrix
            t: float - interpolation parameter [0, 1]
            
        Returns:
            R_interp: [3, 3] tensor - interpolated rotation matrix
        """
        if t == 0:
            return R1
        elif t == 1:
            return R2
        
        # Convert rotation matrices to quaternions
        q1 = self.rotation_matrix_to_quaternion(R1)
        q2 = self.rotation_matrix_to_quaternion(R2)
        
        # Perform SLERP on quaternions
        q_interp = self.slerp_quaternions(q1, q2, t)
        
        # Convert back to rotation matrix
        R_interp = self.quaternion_to_rotation_matrix(q_interp)
        
        return R_interp
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2  # s = 4 * w
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return torch.stack([w, x, y, z])
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix"""
        w, x, y, z = q
        
        # Normalize quaternion
        norm = torch.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = torch.zeros(3, 3, dtype=q.dtype, device=q.device)
        
        R[0, 0] = 1 - 2*(y*y + z*z)
        R[0, 1] = 2*(x*y - w*z)
        R[0, 2] = 2*(x*z + w*y)
        
        R[1, 0] = 2*(x*y + w*z)
        R[1, 1] = 1 - 2*(x*x + z*z)
        R[1, 2] = 2*(y*z - w*x)
        
        R[2, 0] = 2*(x*z - w*y)
        R[2, 1] = 2*(y*z + w*x)
        R[2, 2] = 1 - 2*(x*x + y*y)
        
        return R
    
    def slerp_quaternions(self, q1, q2, t):
        """Spherical linear interpolation between two quaternions"""
        # Ensure we take the shortest path
        dot = torch.sum(q1 * q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / torch.norm(result)
        
        # Calculate angle between quaternions
        theta_0 = torch.acos(torch.clamp(torch.abs(dot), 0, 1))
        sin_theta_0 = torch.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        
        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def resize_frames_and_update_intrinsics(self, frames, original_K, target_size=(576, 1024)):
        """Resize frames and update camera intrinsics accordingly"""
        # frames: [T, H, W, C] numpy array
        original_height, original_width = frames.shape[1:3]
        target_height, target_width = target_size
        
        print(f"==> resizing from {original_width}x{original_height} to {target_width}x{target_height}")
        
        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Resize frames
        resized_frames = []
        for frame in frames:
            # Convert to uint8 for cv2 if needed
            if frame.dtype == np.float32 and frame.max() <= 1.0:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                frame_uint8 = frame.astype(np.uint8)
            
            resized_frame = cv2.resize(frame_uint8, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized_frame.astype(np.float32) / 255.0)
        
        resized_frames = np.stack(resized_frames)
        
        # Update intrinsic matrix
        updated_K = original_K.copy()
        updated_K[0, 0] *= scale_x  # fx
        updated_K[1, 1] *= scale_y  # fy
        updated_K[0, 2] *= scale_x  # cx
        updated_K[1, 2] *= scale_y  # cy
        
        print(f"==> intrinsics updated:")
        print(f"    Original K:\n{original_K}")
        print(f"    Resized K:\n{updated_K}")
        print(f"    Scale factors: sx={scale_x:.3f}, sy={scale_y:.3f}")
        
        return resized_frames, updated_K
    
    def resize_depth_maps(self, depths, target_size=(576, 1024)):
        """Resize depth maps to target size"""
        # depths: [T, H, W] numpy array
        target_height, target_width = target_size
        
        print(f"==> resizing depth maps to {target_width}x{target_height}")
        
        resized_depths = []
        for depth in depths:
            # Use nearest interpolation for depth to avoid artifacts
            resized_depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            resized_depths.append(resized_depth)
        
        return np.stack(resized_depths)
    
    def prepare_mvtracker_data(self, data, target_size=(576, 1024)):
        """Prepare data from MVTracker dataset - assumes data only has 2 views [source, target]"""
        print(f"==> preparing MVTracker data (source=view[0], target=view[1])")
        
        # Since views_to_return=[source_view_idx, target_view_idx], 
        # the returned data always has source at index 0, target at index 1
        source_frames = data.video[0].permute(0, 2, 3, 1).cpu().numpy()  # Always 0
        source_depths = data.videodepth[0].squeeze(1).cpu().numpy()      # Always 0
        source_intrs = data.intrs[0][0].cpu().numpy()  # Always 0
        source_extrs = data.extrs[0][0].cpu().numpy()  # Always 0
        
        # Extract target camera data
        target_frames = data.video[1].permute(0, 2, 3, 1).cpu().numpy()  # Always 1 - ADDED
        target_intrs = data.intrs[1][0].cpu().numpy()  # Always 1
        target_extrs = data.extrs[1][0].cpu().numpy()  # Always 1
        
        print(f"==> source frames shape: {source_frames.shape}")
        print(f"==> source depths shape: {source_depths.shape}")
        print(f"==> target frames shape: {target_frames.shape}")  # ADDED
        print(f"==> source frames range: [{source_frames.min():.3f}, {source_frames.max():.3f}]")
        print(f"==> target frames range: [{target_frames.min():.3f}, {target_frames.max():.3f}]")  # ADDED
        
        # Resize source frames and update intrinsics
        resized_source_frames, updated_source_K = self.resize_frames_and_update_intrinsics(
            source_frames, source_intrs, target_size
        )
        
        # Resize target frames (ground truth)
        resized_target_frames, updated_target_K = self.resize_frames_and_update_intrinsics(
            target_frames, target_intrs, target_size
        )
        
        # Resize depth maps
        resized_depths = self.resize_depth_maps(source_depths, target_size)
        
        return {
            'frames': resized_source_frames,
            'target_frames': resized_target_frames,  # ADDED
            'depths': resized_depths,
            'source_intrs': updated_source_K,
            'source_extrs': source_extrs,
            'target_intrs': updated_target_K,
            'target_extrs': target_extrs,
            'seq_name': data.seq_name
        }
    
    def infer_camera_poses_mvtracker(self, opts, source_view_idx=0, target_view_idx=1):
        """Inference using MVTracker dataset with camera pose interpolation"""
        
        # Initialize MVTracker dataset with specific views
        print("==> initializing MVTracker dataset...")
        
        # dataset = KubricMultiViewDataset(
        #     data_root = '/home/azhuravl/nobackup/mvtracker_data/datasets/kubric-multiview/test',
        #     seq_len = 24,
        #     traj_per_sample = 200,
        #     seed = 72,
        #     sample_vis_1st_frame = True,
        #     tune_per_scene = False,
        #     max_videos = 100,
        #     use_duster_depths = False,
        #     duster_views = None,
        #     clean_duster_depths = False,
        #     views_to_return = [source_view_idx, target_view_idx],  # Select specific views
        #     num_views = -1,
        #     depth_noise_std = 0,
        # )
        
        dataset = PanopticStudioMultiViewDataset(
            '/home/azhuravl/nobackup/mvtracker_data/datasets/panoptic-multiview',
            traj_per_sample=384, 
            seed=72,
            max_videos=100, 
            perform_sanity_checks=False, 
            views_to_return=[source_view_idx, target_view_idx],  # Select specific views
            use_duster_depths=False, 
            clean_duster_depths=False,
        )
        
        # Get first sample
        data = dataset[0][0]
        print(f"==> loaded sequence: {data.seq_name}")
        print(f"==> video shape: {data.video.shape}")  # Should be [2, T, C, H, W]
        print(f"==> depth shape: {data.videodepth.shape}")
        print(f"==> requested views: source={source_view_idx}, target={target_view_idx}")
        
        # Prepare data (no need to pass view indices, they're always 0,1 now)
        target_size = (576, 1024)  # Height, Width
        prepared_data = self.prepare_mvtracker_data(data, target_size)
        
        # Add the original view indices to prepared data for reference
        prepared_data['source_view_idx'] = source_view_idx
        prepared_data['target_view_idx'] = target_view_idx
        
        # Limit to opts.video_length frames
        max_frames = min(opts.video_length, len(prepared_data['frames']))
        frames = prepared_data['frames'][:max_frames]
        target_frames = prepared_data['target_frames'][:max_frames]  # ADDED
        depths = prepared_data['depths'][:max_frames]
        
        print(f"==> using {max_frames} frames")
        
        # Pad if necessary
        if len(frames) < opts.video_length:
            last_frame = frames[-1:]
            last_target_frame = target_frames[-1:]  # ADDED
            last_depth = depths[-1:]
            num_pad = opts.video_length - len(frames)
            pad_frames = np.repeat(last_frame, num_pad, axis=0)
            pad_target_frames = np.repeat(last_target_frame, num_pad, axis=0)  # ADDED
            pad_depths = np.repeat(last_depth, num_pad, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
            target_frames = np.concatenate([target_frames, pad_target_frames], axis=0)  # ADDED
            depths = np.concatenate([depths, pad_depths], axis=0)
            print(f"==> padded to {opts.video_length} frames")
        
        # Get caption
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        
        # Convert depths to torch tensor and add batch dimension
        depths_tensor = torch.from_numpy(depths).unsqueeze(1).to(opts.device).float()  # [T, 1, H, W]
        
        # Convert frames to tensor
        frames_tensor = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        
        # Convert target frames to tensor (for saving ground truth)
        target_frames_tensor = (
            torch.from_numpy(target_frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        
        # Convert camera formats
        source_c2w, source_K = self.convert_camera_format_from_mvtracker(
            prepared_data['source_intrs'], prepared_data['source_extrs']
        )
        target_c2w, target_K = self.convert_camera_format_from_mvtracker(
            prepared_data['target_intrs'], prepared_data['target_extrs']
        )
        
        # NEW: Interpolate camera poses from source to target
        interpolated_poses = self.interpolate_camera_poses(
            source_c2w, target_c2w, opts.video_length
        ).to(opts.device)  # [video_length, 4, 4]
        
        # NEW: Interpolate camera intrinsics from source to target
        interpolated_intrinsics = self.interpolate_intrinsics(
            source_K, target_K, opts.video_length
        ).to(opts.device)  # [video_length, 3, 3]
        
        # Create pose matrices for all frames (SOURCE CAMERA STAYS SAME, TARGET INTERPOLATES)
        pose_s = source_c2w.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)  # Source stays fixed
        pose_t = interpolated_poses  # Target interpolates from source to target
        
        K_matrices_s = source_K.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)  # Source K stays fixed
        K_matrices_t = interpolated_intrinsics  # Target K interpolates
        
        print(f"==> using interpolated camera poses for trajectory generation")
        print(f"    Source pose (fixed): translation = {source_c2w[:3, 3].numpy()}")
        print(f"    Target pose (final): translation = {target_c2w[:3, 3].numpy()}")
        
        # Warp images
        warped_images = []
        masks = []
        
        for i in range(opts.video_length):
            warped_frame, mask, _, _ = self.funwarp.forward_warp(
                frames_tensor[i : i + 1],
                None,
                depths_tensor[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],  # This now changes over time!
                K_matrices_s[i : i + 1],
                K_matrices_t[i : i + 1],  # This now changes over time!
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame)
            masks.append(mask)
        
        # Continue with existing pipeline
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        
        # Resize for processing
        frames_tensor = torch.nn.functional.interpolate(
            frames_tensor, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        target_frames_tensor = torch.nn.functional.interpolate(
            target_frames_tensor, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = torch.nn.functional.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = torch.nn.functional.interpolate(
            cond_masks, size=opts.sample_size, mode='nearest'
        )
        
        # Save intermediate videos INCLUDING target ground truth
        self.save_intermediate_videos(frames_tensor, target_frames_tensor, cond_video, cond_masks, opts, prepared_data['seq_name'])
        print("Intermediate videos saved.")
        
        # Save camera and depth data for reference (including interpolation info)
        prepared_data['interpolated_poses'] = interpolated_poses.cpu().numpy()
        prepared_data['interpolated_intrinsics'] = interpolated_intrinsics.cpu().numpy()
        self.save_mvtracker_data(prepared_data, opts.save_dir)
        
        # Prepare for diffusion
        frames_for_diffusion = (frames_tensor.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames_for_diffusion[:, :, :10, :, :]
        cond_video_for_diffusion = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks_for_diffusion = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)
        
        # Run diffusion
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video_for_diffusion.to(opts.device),
                mask_video=cond_masks_for_diffusion.to(opts.device),
                reference=frames_ref,
            ).videos
        
        # Save final video
        self.save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )
        
        # NEW: Calculate evaluation metrics between last frames
        print("\n" + "="*60)
        print("EVALUATION METRICS CALCULATION")
        print("="*60)
        
        # Get last frame from generated video (convert from [-1, 1] to [0, 1])
        last_generated_frame = (sample[0][-1].permute(1, 2, 0) + 1.0) / 2.0  # [H, W, 3] in [0, 1]
        
        # Get last frame from target video (convert from [-1, 1] to [0, 1])
        last_target_frame = (target_frames_tensor[-1].permute(1, 2, 0) + 1.0) / 2.0  # [H, W, 3] in [0, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            last_generated_frame, 
            last_target_frame, 
            opts.save_dir, 
            prepared_data['seq_name']
        )
        
        # Save comparison images
        self.save_comparison_images(
            last_generated_frame,
            last_target_frame, 
            opts.save_dir,
            prepared_data['seq_name']
        )
        
        print("="*60)
        
        return sample, metrics
        
    def save_mvtracker_data(self, data, save_dir):
        """Save MVTracker data for reference"""
        mvtracker_data = {
            "seq_name": data['seq_name'],
            "source_view_idx": data.get('source_view_idx', 0),
            "target_view_idx": data.get('target_view_idx', 1),
            "source_intrs": data['source_intrs'].tolist(),
            "target_intrs": data['target_intrs'].tolist(),
            "source_extrs": data['source_extrs'].tolist(),
            "target_extrs": data['target_extrs'].tolist(),
            "frame_shape": list(data['frames'].shape),
            "target_frame_shape": list(data['target_frames'].shape),
            "depth_shape": list(data['depths'].shape),
        }
        
        # Add interpolation data if available
        if 'interpolated_poses' in data:
            mvtracker_data["interpolated_poses_shape"] = list(data['interpolated_poses'].shape)
            mvtracker_data["interpolated_intrinsics_shape"] = list(data['interpolated_intrinsics'].shape)
            # Save first, middle, and last poses for reference
            poses = data['interpolated_poses']
            mvtracker_data["pose_trajectory"] = {
                "first_frame": poses[0].tolist(),
                "middle_frame": poses[len(poses)//2].tolist(),
                "last_frame": poses[-1].tolist()
            }
        
        with open(os.path.join(save_dir, 'mvtracker_data.json'), 'w') as f:
            json.dump(mvtracker_data, f, indent=2)
        
        print(f"MVTracker data saved to {os.path.join(save_dir, 'mvtracker_data.json')}")
    
    def save_intermediate_videos(self, frames, target_frames, cond_video, cond_masks, opts, seq_name="mvtracker"):
        """Save intermediate videos for debugging"""
        # Source frames (input)
        self.save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, f'input_source_{seq_name}.mp4'),
            fps=opts.fps,
        )
        
        # Target frames (ground truth) - ADDED
        self.save_video(
            (target_frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, f'target_gt_{seq_name}.mp4'),
            fps=opts.fps,
        )
        
        # Warped/rendered frames (now shows camera trajectory!)
        self.save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, f'render_trajectory_{seq_name}.mp4'),  # Renamed to reflect trajectory
            fps=opts.fps,
        )
        
        # Masks
        self.save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, f'mask_trajectory_{seq_name}.mp4'),  # Renamed to reflect trajectory
            fps=opts.fps,
        )
        
        # Create side-by-side comparison video (trajectory render vs final target ground truth)
        self.save_comparison_video(cond_video, target_frames, opts, seq_name)
    
    def save_comparison_video(self, rendered_frames, target_frames, opts, seq_name="mvtracker"):
        """Save side-by-side comparison of trajectory render vs target ground truth"""
        # Ensure both videos are in [0, 1] range and on the same device
        rendered_normalized = rendered_frames.permute(0, 2, 3, 1)  # Already [0, 1]
        target_normalized = (target_frames.permute(0, 2, 3, 1) + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
        
        # Ensure both tensors are on the same device
        device = rendered_normalized.device
        target_normalized = target_normalized.to(device)
        
        # Create side-by-side comparison
        comparison_frames = torch.cat([rendered_normalized, target_normalized], dim=2)  # Concatenate along width
        
        self.save_video(
            comparison_frames,
            os.path.join(opts.save_dir, f'comparison_trajectory_vs_target_{seq_name}.mp4'),  # Updated name
            fps=opts.fps,
        )

    def save_video(self, tensor, path, fps):
        """Save video tensor to file"""
        from models.utils import save_video
        save_video(tensor, path, fps=fps)


# Updated usage function
def run_mvtracker_inference(source_view_idx=0, target_view_idx=1, output_dir="./experiments/"):
    """Run inference using MVTracker dataset with camera trajectory interpolation"""
    
    # Create minimal opts
    class Opts:
        def __init__(self, source_view_idx, target_view_idx):
            # Video settings
            self.video_length = 49
            self.fps = 10
            self.stride = 1
            
            # Device
            self.device = 'cuda:0'
            self.weight_dtype = torch.bfloat16
            
            # Output
            timestamp = datetime.now().strftime("%H%M")
            date = datetime.now().strftime("%d-%m-%Y")
            self.exp_name = f"{timestamp}_mvtracker_traj_{source_view_idx}to{target_view_idx}"  # Updated name
            self.save_dir = f"./experiments/{date}/{self.exp_name}"
            
            # Diffusion
            self.model_name = 'checkpoints/CogVideoX-Fun-V1.1-5b-InP'
            self.transformer_path = 'checkpoints/TrajectoryCrafter'
            self.sampler_name = 'DDIM_Origin'
            self.sample_size = [384, 672]
            self.diffusion_guidance_scale = 6.0
            self.diffusion_inference_steps = 50
            self.negative_prompt = "The video is not of a high quality, it has a low resolution."
            self.low_gpu_memory_mode = False
            
            # Camera
            self.mask = False
            self.seed = 43
            
            # Paths
            self.unet_path = "checkpoints/DepthCrafter"
            self.pre_train_path = "checkpoints/stable-video-diffusion-img2vid"
            self.blip_path = "checkpoints/blip2-opt-2.7b"
            self.cpu_offload = 'model'
            self.refine_prompt = ". High quality, masterpiece, best quality."
    
    opts = Opts(source_view_idx, target_view_idx)
    os.makedirs(opts.save_dir, exist_ok=True)
    
    # Initialize model (without depth estimator)
    crafter = CameraPoseTrajCrafter(opts)
    
    # Run inference with MVTracker
    result, metrics = crafter.infer_camera_poses_mvtracker(opts, source_view_idx, target_view_idx)
    
    print(f"Generated trajectory video saved to: {opts.save_dir}")
    print(f"Evaluation metrics: PSNR={metrics['metrics']['PSNR']:.4f}dB, SSIM={metrics['metrics']['SSIM']:.4f}, LPIPS={metrics['metrics']['LPIPS']:.4f}")
    
    return result, metrics


if __name__ == "__main__":
    # Run with different view combinations to create camera trajectory
    result, metrics = run_mvtracker_inference(
        source_view_idx=0,
        target_view_idx=10
    )