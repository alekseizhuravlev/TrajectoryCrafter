import os
import torch
from torch.utils.data import Dataset
import glob


class LatentsDataset(Dataset):
    """Dataset to load latent files from nested directory structure"""
    
    def __init__(self, data_dir, use_depth_latents=False):
        """
        Args:
            data_dir: Directory containing latent files
            use_depth_latents: If True, load both video_latents.pt and depth_latents.pt
        """
        self.data_dir = data_dir
        self.use_depth_latents = use_depth_latents
        self.file_paths = []
        
        if use_depth_latents:
            # Find directories that contain both video and depth latents
            video_pattern = os.path.join(data_dir, "*/video_latents.pt")
            depth_pattern = os.path.join(data_dir, "*/depth_latents.pt")
            
            video_files = set(os.path.dirname(f) for f in glob.glob(video_pattern))
            depth_files = set(os.path.dirname(f) for f in glob.glob(depth_pattern))
            
            # Only include directories that have both files
            valid_dirs = video_files.intersection(depth_files)
            self.file_paths = sorted(list(valid_dirs))
            
            print(f"Found {len(self.file_paths)} directories with both video and depth latents")
            
            if len(self.file_paths) == 0:
                print(f"Warning: No directories with both video_latents.pt and depth_latents.pt found in {data_dir}")
                raise ValueError("No directories with both video and depth latent files found.")
        else:
            # Original behavior - just video latents
            pattern = os.path.join(data_dir, "*/video_latents.pt")
            self.file_paths = sorted(glob.glob(pattern))
            
            print(f"Found {len(self.file_paths)} video_latents.pt files")
            
            if len(self.file_paths) == 0:
                print(f"Warning: No video_latents.pt files found in {data_dir}")
                print(f"Looking for pattern: {pattern}")
                raise ValueError("No video latent files found.")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if self.use_depth_latents:
            # Load both video and depth latents from directory
            dir_path = self.file_paths[idx]
            video_path = os.path.join(dir_path, "video_latents.pt")
            depth_path = os.path.join(dir_path, "depth_latents.pt")
            
            try:
                video_data = torch.load(video_path, map_location='cpu', weights_only=True)
                depth_data = torch.load(depth_path, map_location='cpu', weights_only=True)
                
                return {
                    'video_latents': video_data,
                    'depth_latents': depth_data
                }
            except Exception as e:
                print(f"Error loading latents from {dir_path}: {e}")
                raise e
        else:
            # Original behavior - just video latents
            file_path = self.file_paths[idx]
            try:
                data = torch.load(file_path, map_location='cpu', weights_only=True)
                return data
            except Exception as e:
                print(f"Error loading video latents from {file_path}: {e}")
                raise e