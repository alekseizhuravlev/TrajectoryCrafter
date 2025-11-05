# dataset_videos.py (fixed)
import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange

class SimpleValidationDataset(Dataset):
    """
    Simple validation dataset for TrajectoryCrafter that reads preprocessed data.
    Each data point is a folder containing:
    - caption.txt: Text prompt
    - ref_video.mp4: Reference video frames (RGB mode)
    - input_video.mp4: Input video frames (RGB mode)
    - warped_video_twice.mp4: Warped conditioning video (RGB mode)
    - masks.mp4: Inpainting masks (RGB mode)
    OR (depth mode):
    - ref_depths.pt: Reference depth frames
    - input_depths.pt: Input depth frames
    - warped_depths.pt: Warped conditioning depths
    """
    
    def __init__(self, validation_dir, max_samples=None, use_depth=False):
        """
        Args:
            validation_dir: Directory containing validation sample folders
            max_samples: Maximum number of samples to load (None for all)
            use_depth: If True, load depth tensors instead of RGB videos
        """
        self.validation_dir = validation_dir
        self.use_depth = use_depth
        
        # Get all sample folders
        self.sample_folders = []
        for item in sorted(os.listdir(validation_dir)):
            folder_path = os.path.join(validation_dir, item)
            if os.path.isdir(folder_path):
                # Check if required files exist in videos/ subdirectory
                videos_path = os.path.join(folder_path, 'videos')
                if os.path.exists(videos_path):
                    if use_depth:
                        # Check for depth files
                        required_files = [
                            'caption.txt', 'ref_depths.pt', 'input_depths.pt', 
                            'warped_depths.pt', 'masks.mp4'  # Still need RGB masks
                        ]
                    else:
                        # Check for RGB video files
                        required_files = [
                            'caption.txt', 'ref_video.mp4', 'input_video.mp4', 
                            'warped_video_twice.mp4', 'masks.mp4'
                        ]
                    
                    if all(os.path.exists(os.path.join(videos_path, f)) for f in required_files):
                        self.sample_folders.append(folder_path)
        
        if max_samples is not None:
            self.sample_folders = self.sample_folders[:max_samples]
        
        mode = "depth" if use_depth else "RGB"
        print(f"Found {len(self.sample_folders)} validation samples ({mode} mode)")
    
    def __len__(self):
        return len(self.sample_folders)
    
    def __getitem__(self, idx):
        folder_path = self.sample_folders[idx]
        videos_path = os.path.join(folder_path, 'videos')
        folder_name = os.path.basename(folder_path)
        
        # Read caption
        caption_path = os.path.join(videos_path, 'caption.txt')
        with open(caption_path, 'r') as f:
            caption = f.read().strip()
        
        if self.use_depth:
            # Load depth tensors
            ref_depths = torch.load(os.path.join(videos_path, 'ref_depths.pt'), map_location='cpu')  # [T, 1, H, W]
            input_depths = torch.load(os.path.join(videos_path, 'input_depths.pt'), map_location='cpu')  # [T, 1, H, W]
            warped_depths = torch.load(os.path.join(videos_path, 'warped_depths.pt'), map_location='cpu')  # [T, 1, H, W]
            
            # Still load RGB masks
            masks = self._read_video(os.path.join(videos_path, 'masks.mp4'))
            
            # Process depths - already in correct format [T, C, H, W] and [0, 1] range
            ref_video = ref_depths.float()  # [T, 1, H, W]
            input_video = input_depths.float()  # [T, 1, H, W]
            warped_video = warped_depths.float()  # [T, 1, H, W]
            
            # Take first 10 frames for reference
            ref_frames = min(10, ref_video.shape[0])
            ref_video = ref_video[:ref_frames]  # [ref_frames, 1, H, W]
            
            # Convert to [C, T, H, W] format to match RGB format
            ref_video = ref_video.permute(1, 0, 2, 3)  # [1, ref_frames, H, W]
            input_video = input_video.permute(1, 0, 2, 3)  # [1, T, H, W]
            warped_video = warped_video.permute(1, 0, 2, 3)  # [1, T, H, W]
            
        else:
            # Load RGB videos (original code)
            ref_video = self._read_video(os.path.join(videos_path, 'ref_video.mp4'))
            input_video = self._read_video(os.path.join(videos_path, 'input_video.mp4'))
            warped_video = self._read_video(os.path.join(videos_path, 'warped_video_twice.mp4'))
            masks = self._read_video(os.path.join(videos_path, 'masks.mp4'))
            
            # Convert to TrajectoryCrafter format WITHOUT batch dimension
            # ref_video: (T, H, W, C) -> (C, ref_frames, H, W) 
            ref_video = torch.from_numpy(ref_video).permute(3, 0, 1, 2) / 255.0
            ref_frames = min(10, ref_video.shape[1])  # Take first 10 frames for reference
            ref_video = ref_video[:, :ref_frames]
            
            # input_video: (T, H, W, C) -> (C, T, H, W)
            input_video = torch.from_numpy(input_video).permute(3, 0, 1, 2) / 255.0
            
            # warped_video: (T, H, W, C) -> (C, T, H, W)
            warped_video = torch.from_numpy(warped_video).permute(3, 0, 1, 2) / 255.0
        
        # Process masks (always RGB)
        if masks.shape[-1] == 3:  # RGB mask, take one channel
            masks = masks[:, :, :, 0:1]
        masks = torch.from_numpy(masks).permute(3, 0, 1, 2)
        
        # invert it
        masks = 255 - masks
        
        
        return {
            'caption': caption,
            'ref_video': ref_video.float(),
            'input_video': input_video.float(), 
            'warped_video': warped_video.float(),
            'masks': masks.float(),
            'sample_name': folder_name,
            'is_depth': self.use_depth
        }
    
    def _read_video(self, video_path):
        """Read video frames using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        return np.stack(frames, axis=0)  # (T, H, W, C)