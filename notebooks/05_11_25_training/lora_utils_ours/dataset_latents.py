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
            use_depth_latents: If True, load depth_latents.pt instead of video_latents.pt
        """
        self.data_dir = data_dir
        self.use_depth_latents = use_depth_latents
        self.file_paths = []
        
        # Choose the appropriate latent file type
        latent_filename = "depth_latents.pt" if use_depth_latents else "video_latents.pt"
        
        # Find all latent files in subdirectories
        pattern = os.path.join(data_dir, f"*/{latent_filename}")
        self.file_paths = sorted(glob.glob(pattern))
        
        latent_type = "depth" if use_depth_latents else "video"
        print(f"Found {len(self.file_paths)} {latent_filename} files ({latent_type} latents)")
        
        if len(self.file_paths) == 0:
            print(f"Warning: No {latent_filename} files found in {data_dir}")
            print(f"Looking for pattern: {pattern}")
            raise ValueError(f"No {latent_type} latent files found.")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=True)
            return data
        except Exception as e:
            latent_type = "depth" if self.use_depth_latents else "video"
            print(f"Error loading {latent_type} latents from {file_path}: {e}")
            # Return a dummy tensor or re-raise the exception
            raise e