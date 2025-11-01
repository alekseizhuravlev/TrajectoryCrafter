import os
import torch
from torch.utils.data import Dataset
import glob


class LatentsDataset(Dataset):
    """Simple dataset to load latent files from disk"""
    
    def __init__(self, data_dir):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "latents_*.pt")))
        print(f"Found {len(self.file_paths)} files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx], map_location='cpu')