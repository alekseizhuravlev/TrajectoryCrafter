import torch
import numpy as np
from demo import TrajCrafter
import json
import os
from datetime import datetime
import cv2

class CameraPoseTrajCrafter(TrajCrafter):
    def __init__(self, opts, gradio=False):
        super().__init__(opts, gradio)
    
    def convert_camera_format(self, camera_data):
        """Convert camera data from your format to c2w matrix"""
        K = np.array(camera_data["K"])
        R = np.array(camera_data["R"])
        t = np.array(camera_data["t"]).reshape(3, 1)
        
        # Create extrinsic matrix [R|t]
        RT = np.hstack([R, t])
        RT = np.vstack([RT, [0, 0, 0, 1]])  # Make 4x4
        
        # Convert to c2w (camera to world)
        # c2w = np.linalg.inv(RT)
        c2w = RT
        
        return torch.tensor(c2w, dtype=torch.float32), torch.tensor(K, dtype=torch.float32)
    
    def interpolate_poses(self, source_cam, target_cam, num_frames):
        """Use source pose for first frame and target pose for all other frames"""
        source_c2w, source_K = self.convert_camera_format(source_cam)
        target_c2w, target_K = self.convert_camera_format(target_cam)
        
        poses = []
        K_matrices = []
        
        for i in range(num_frames):
            if i == 0:
                # First frame uses source camera
                poses.append(source_c2w)
                K_matrices.append(source_K)
            else:
                # All other frames use target camera
                poses.append(target_c2w)
                K_matrices.append(target_K)
        
        return torch.stack(poses), torch.stack(K_matrices)
    
    
    def read_video_frames_original_size(self, video_path, video_length, stride):
        """Read video frames at original resolution"""
        from decord import VideoReader, cpu
        
        print("==> processing video at original size: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        original_shape = vid.get_batch([0]).shape[1:]
        print("==> original video shape: ", (len(vid), *original_shape))
        
        frames_idx = list(range(0, len(vid), stride))
        print(f"==> frame indices: {len(frames_idx)} frames with stride: {stride}")
        
        if video_length != -1 and video_length < len(frames_idx):
            frames_idx = frames_idx[:video_length]
        
        print(f"==> reading {len(frames_idx)} frames at original resolution")
        frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
        
        return frames
    
    
    def resize_frames_and_update_intrinsics(self, frames, original_K, target_size=(576, 1024)):
        """Resize frames and update camera intrinsics accordingly"""
        original_height, original_width = frames.shape[1:3]
        target_height, target_width = target_size
        
        print(f"==> resizing from {original_width}x{original_height} to {target_width}x{target_height}")
        
        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Resize frames
        resized_frames = []
        for frame in frames:
            # Convert to uint8 for cv2
            frame_uint8 = (frame * 255).astype(np.uint8)
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
    
    
    def undistort_and_resize_workflow(self, frames, camera_data, target_size=(576, 1024)):
        """Complete workflow: undistort at original size, then resize and update intrinsics"""
        K = np.array(camera_data["K"])
        dist_coeffs = np.array(camera_data["distCoef"])
        
        # Get original dimensions
        h, w = frames[0].shape[:2]
        print(f"==> starting undistortion and resize workflow: {w}x{h} -> {target_size[1]}x{target_size[0]}")
        
        # Step 1: Undistort at original resolution
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)
        
        undistorted_frames = []
        print(f"==> undistorting {len(frames)} frames at original resolution...")
        
        for i, frame in enumerate(frames):
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                frame_uint8 = frame
            
            # Apply undistortion
            undistorted_frame = cv2.remap(frame_uint8, map1, map2, cv2.INTER_LINEAR)
            
            # Convert back to original dtype
            if frame.dtype != np.uint8:
                undistorted_frame = undistorted_frame.astype(np.float32) / 255.0
            
            undistorted_frames.append(undistorted_frame)
        
        undistorted_frames = np.stack(undistorted_frames)
        print(f"==> undistortion complete at original size: {undistorted_frames.shape}")
        
        # Step 2: Resize and update intrinsics
        resized_frames, final_K = self.resize_frames_and_update_intrinsics(
            undistorted_frames, new_K, target_size
        )
        
        print(f"==> final shape: {resized_frames.shape}")
        
        return resized_frames, final_K
    
    
    # Update the main method
    def infer_camera_poses(self, opts, source_camera, target_camera):
        """Direct inference with source and target camera poses"""
        # Read frames at original resolution
        frames = self.read_video_frames_original_size(
            opts.video_path, opts.video_length, opts.stride
        )
        
        # Pad frames if necessary
        if frames.shape[0] < opts.video_length:
            last_frame = frames[-1:]
            num_pad = opts.video_length - frames.shape[0]
            pad_frames = np.repeat(last_frame, num_pad, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
            print(f"==> padding video from {frames.shape[0]} to {opts.video_length} frames")
            
        # Complete workflow: undistort at original size, then resize with proper intrinsic scaling
        print("==> applying undistortion and resize workflow...")
        target_size = (576, 1024)  # Height, Width - matching the hardcoded values
        frames, final_K = self.undistort_and_resize_workflow(frames, source_camera, target_size)
        
        # Update source camera with final intrinsics
        source_camera_processed = source_camera.copy()
        source_camera_processed["K"] = final_K.tolist()
        source_camera_processed["distCoef"] = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion after undistortion
        
        # Get caption
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        
        # Get depth estimation (now with properly scaled frames)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        
        # Convert frames to tensor
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        
        # Use processed source camera and original target camera
        source_c2w, source_K = self.convert_camera_format(source_camera_processed)
        target_c2w, target_K = self.convert_camera_format(target_camera)
    
        pose_s = source_c2w.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)
        pose_t = target_c2w.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)
        
        # Use processed source intrinsics for warping
        K_matrices_s = source_K.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)
        K_matrices_t = target_K.to(opts.device).unsqueeze(0).repeat(opts.video_length, 1, 1)
        print(f"==> using processed source camera intrinsics for warping")
        
        # Rest of the method remains the same...
        # Warp images
        warped_images = []
        masks = []
        
        for i in range(opts.video_length):
            warped_frame, mask, _, _ = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K_matrices_s[i : i + 1],
                K_matrices_t[i : i + 1],
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame)
            masks.append(mask)
        
        # Continue with the rest of your existing code...
        # (cond_video creation, resizing, saving, etc.)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        
        # Resize for processing
        frames = torch.nn.functional.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = torch.nn.functional.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = torch.nn.functional.interpolate(
            cond_masks, size=opts.sample_size, mode='nearest'
        )
        
        # Save intermediate videos (including undistorted input)
        self.save_intermediate_videos(frames, cond_video, cond_masks, opts)
        print("Intermediate videos saved.")
        
        exit(0)
        
        # Prepare for diffusion
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
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
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        
        # Save final video
        self.save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )
        
        # Save camera parameters for reference
        # self.save_camera_params(source_camera_undistorted, target_camera, opts.save_dir)
        
        return sample

    def save_camera_params(self, source_camera, target_camera, save_dir):
        """Save camera parameters to JSON file"""
        camera_params = {
            "source_camera": source_camera,
            "target_camera": target_camera,
            "note": "Source camera has been undistorted (distCoef set to zeros)"
        }
        
        with open(os.path.join(save_dir, 'camera_params.json'), 'w') as f:
            json.dump(camera_params, f, indent=2)
        
        print(f"Camera parameters saved to {os.path.join(save_dir, 'camera_params.json')}")

    def save_intermediate_videos(self, frames, cond_video, cond_masks, opts):
        """Save intermediate videos for debugging"""
        self.save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input_undistorted.mp4'),  # Updated name to reflect undistortion
            fps=opts.fps,
        )
        self.save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        self.save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )

    def save_video(self, tensor, path, fps):
        """Save video tensor to file"""
        from models.utils import save_video
        save_video(tensor, path, fps=fps)



# Rest of the code remains the same...
def run_camera_pose_inference(video_path, source_camera, target_camera, output_dir="./experiments/"):
    """Run inference with camera poses"""
    import argparse
    
    # Create minimal opts
    class Opts:
        def __init__(self):
            # Video settings
            self.video_path = video_path
            self.video_length = 49
            self.fps = 10
            self.stride = 1
            self.max_res = 1024
            
            # Device
            self.device = 'cuda:0'
            self.weight_dtype = torch.bfloat16
            
            # Output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            date = datetime.now().strftime("%d-%m-%Y")
            video_basename = os.path.splitext(os.path.basename(video_path))[0]

            # Setup
            self.exp_name = f"{video_basename}_{timestamp}_cam_pose_undistorted"  # Updated name
            self.save_dir = f"./experiments/{date}/{self.exp_name}"

            # Depth
            self.near = 0.0001
            self.far = 10000.0
            self.depth_inference_steps = 5
            self.depth_guidance_scale = 1.0
            self.window_size = 110
            self.overlap = 25
            
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
            self.blip_path = "checkpoints/blip2-opt-2.7b"
            self.unet_path = "checkpoints/DepthCrafter"
            self.pre_train_path = "checkpoints/stable-video-diffusion-img2vid"
            self.cpu_offload = 'model'
            self.refine_prompt = ". High quality, masterpiece, best quality."
    
    opts = Opts()
    os.makedirs(opts.save_dir, exist_ok=True)
    
    # Initialize model
    crafter = CameraPoseTrajCrafter(opts)
    
    # Run inference
    result = crafter.infer_camera_poses(opts, source_camera, target_camera)
    
    print(f"Generated video saved to: {opts.save_dir}")
    return result


# Example usage remains the same...
if __name__ == "__main__":
    # Example camera data
    source_camera = {
        "name": "05_08",
        "type": "vga",
        "resolution": [640,480],
        "panel": 5,
        "node": 8,
        "K": [
            [748.194573,0.403304,388.156644],
            [0,747.455308,257.075025],
            [0,0,1]
        ],
        "distCoef": [-0.352118,0.186737,0,0,-0.119772],
        "R": [
            [-0.871497831,-0.004279560553,0.4903806847],
            [0.09322575792,0.980281239,0.1742344701],
            [-0.4814566321,0.1975610738,-0.8539140083]
        ],
        "t": [
            [-0.03934843898], #[-39.34843898],
            [0.09250008112], #[92.50008112],
            [0.3049007109] #[304.9007109]
        ]
    }
    
    target_camera = {
        "name": "01_01",
        "type": "vga",
        "resolution": [640,480],
        "panel": 1,
        "node": 1,
        "K": [
            [748.561374,0.083459,378.041653],
            [0,748.351299,223.336713],
            [0,0,1]
        ],
        "distCoef": [-0.32211,0.02854,0,0,0.101902],
        "R": [
            [-0.9610410199,0.02955079861,-0.2748215937],
            [0.005847346208,0.9962196747,0.08667276504],
            [0.2763439281,0.08168910551,-0.957580766]
        ],
        "t": [
            [-0.04625903829], #[-46.25903829],
            [0.1435237551], #[143.5237551],
            [0.2871962273] #[287.1962273]
        ]
    }
    
    # source_camera = {
	# 		"name": "00_00",
	# 		"type": "hd",
	# 		"resolution": [1920,1080],
	# 		"panel": 0,
	# 		"node": 0,
	# 		"K": [
	# 			[1399.965713,0.667095,943.129998],
	# 			[0,1394.609791,554.244042],
	# 			[0,0,1]
	# 		],
	# 		"distCoef": [-0.289299,0.19293,-9.4e-05,0.000344,-0.057798],
	# 		"R": [
	# 			[-0.6456051742,0.03473953627,0.7628808057],
	# 			[0.09040156866,0.9954173139,0.03117575436],
	# 			[-0.7583017311,0.08909284986,-0.6457870769]
	# 		],
	# 		"t": [
	# 			[20.06419515],
	# 			[126.2317869],
	# 			[289.6604271]
	# 		]
	# 	}
    
    # target_camera = {
	# 		"name": "00_01",
	# 		"type": "hd",
	# 		"resolution": [1920,1080],
	# 		"panel": 0,
	# 		"node": 1,
	# 		"K": [
	# 			[1400.7621,-1.153539,951.448218],
	# 			[0,1395.989371,558.927928],
	# 			[0,0,1]
	# 		],
	# 		"distCoef": [-0.285946,0.183494,-2.7e-05,0.000752,-0.046428],
	# 		"R": [
	# 			[0.4877691809,0.02238665679,-0.8726855469],
	# 			[-0.563972061,0.7711365189,-0.2954386291],
	# 			[0.6663458116,0.6362761246,0.3887620771]
	# 		],
	# 		"t": [
	# 			[9.050916471],
	# 			[110.4408308],
	# 			[356.5172629]
	# 		]
	# 	}
    
    camera_scale = 15  # Scale factor for translation vectors
    
    source_camera["t"] = (np.array(source_camera["t"]) * camera_scale).tolist()
    target_camera["t"] = (np.array(target_camera["t"]) * camera_scale).tolist()
    
    # Run inference
    # video_path = "/home/azhuravl/nobackup/DAVIS_testing/trainval/judo.mp4"
    video_path = "/home/azhuravl/work/panoptic-toolbox/150821_dance4/vgaVideos/vga_05_08.mp4"
    # video_path = "/home/azhuravl/work/panoptic-toolbox/150821_dance2/hdVideos/hd_00_00.mp4"
    result = run_camera_pose_inference(video_path, source_camera, target_camera)