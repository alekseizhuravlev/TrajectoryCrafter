import os
import torch
from torch.utils.data import Dataset
import glob


class LatentsDataset(Dataset):
    """Dataset to load latent files from nested directory structure"""
    
    def __init__(self, data_dir, use_depth_latents=False, num_ref_frames=30):
        """
        Args:
            data_dir: Directory containing latent files
            use_depth_latents: If True, load both video_latents.pt and depth_latents.pt
        """
        self.data_dir = data_dir
        self.use_depth_latents = use_depth_latents
        self.file_paths = []
        self.num_ref_frames = num_ref_frames
        print('Using num_ref_frames:', self.num_ref_frames)
        
        if use_depth_latents:
            # Find directories that contain both depth and ref_videos latents
            ref_video_pattern = os.path.join(data_dir, "*/ref_videos_latents.pt")
            depth_pattern = os.path.join(data_dir, "*/depth_latents.pt")
            
            ref_video_files = set(os.path.dirname(f) for f in glob.glob(ref_video_pattern))
            depth_files = set(os.path.dirname(f) for f in glob.glob(depth_pattern))
            
            # Only include directories that have both files
            valid_dirs = ref_video_files.intersection(depth_files)
            self.file_paths = sorted(list(valid_dirs))
            
            print(f"Found {len(self.file_paths)} directories with both depth and ref_videos latents")
            
            if len(self.file_paths) == 0:
                print(f"Warning: No directories with both depth_latents.pt and ref_videos_latents.pt found in {data_dir}")
                raise ValueError("No directories with both depth and ref_videos latent files found.")
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
        # if self.use_depth_latents:
        #     # Load both video and depth latents from directory
        #     dir_path = self.file_paths[idx]
        #     video_path = os.path.join(dir_path, "video_latents.pt")
        #     depth_path = os.path.join(dir_path, "depth_latents.pt")
            
        #     try:
        #         video_data = torch.load(video_path, map_location='cpu', weights_only=True)
        #         depth_data = torch.load(depth_path, map_location='cpu', weights_only=True)
                
        #         depth_data['ref_latents'] = video_data['gt_video_latents']
                
        #         return depth_data
            
        #     except Exception as e:
        #         print(f"Error loading latents from {dir_path}: {e}")
        #         raise e
        
        if self.use_depth_latents:
            # Load both ref_videos and depth latents from directory
            dir_path = self.file_paths[idx]
            ref_video_path = os.path.join(dir_path, "ref_videos_latents.pt")
            depth_path = os.path.join(dir_path, "depth_latents.pt")
            
            try:
                ref_video_data = torch.load(ref_video_path, map_location='cpu', weights_only=True)
                depth_data = torch.load(depth_path, map_location='cpu', weights_only=True)
                
                # 30 frames works
                
                if self.num_ref_frames == 49:
                    video_path = os.path.join(dir_path, "video_latents.pt")
                    video_data = torch.load(video_path, map_location='cpu', weights_only=True)

                    depth_data['ref_latents'] = video_data['gt_video_latents']
                
                else:
                    depth_data['ref_latents'] = ref_video_data[f'ref_latents_{self.num_ref_frames}_frames'][0]
                
                # ref_latents_10_frames torch.Size([1, 3, 16, 48, 84])
                # ref_latents_20_frames torch.Size([1, 5, 16, 48, 84])
                # ref_latents_30_frames torch.Size([1, 8, 16, 48, 84])
                # ref_latents_35_frames torch.Size([1, 9, 16, 48, 84])
                # ref_latents_40_frames torch.Size([1, 10, 16, 48, 84])
                
                return depth_data
            
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



# class LatentsDatasetSingle(Dataset):
#     """Dataset to load latent files from nested directory structure"""
    
#     def __init__(self, data_dir, use_depth_latents=False):
#         """
#         Args:
#             data_dir: Directory containing latent files
#             use_depth_latents: If True, load depth_latents.pt instead of video_latents.pt
#         """
#         self.data_dir = data_dir
#         self.use_depth_latents = use_depth_latents
#         self.file_paths = []
        
#         # Choose the appropriate latent file type
#         latent_filename = "depth_latents.pt" if use_depth_latents else "video_latents.pt"
        
#         # Find all latent files in subdirectories
#         pattern = os.path.join(data_dir, f"*/{latent_filename}")
#         self.file_paths = sorted(glob.glob(pattern))
        
#         latent_type = "depth" if use_depth_latents else "video"
#         print(f"Found {len(self.file_paths)} {latent_filename} files ({latent_type} latents)")
        
#         if len(self.file_paths) == 0:
#             print(f"Warning: No {latent_filename} files found in {data_dir}")
#             print(f"Looking for pattern: {pattern}")
#             raise ValueError(f"No {latent_type} latent files found.")
    
#     def __len__(self):
#         return len(self.file_paths)
    
#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         try:
#             data = torch.load(file_path, map_location='cpu', weights_only=True)
#             return data
#         except Exception as e:
#             latent_type = "depth" if self.use_depth_latents else "video"
#             print(f"Error loading {latent_type} latents from {file_path}: {e}")
#             # Return a dummy tensor or re-raise the exception
#             raise e