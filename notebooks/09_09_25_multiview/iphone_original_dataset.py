import os
import json
import numpy as np
import imageio.v3 as iio
import torch
from torch.utils.data import Dataset
from pathlib import Path

class iPhoneDataset(Dataset):
    """
    Dataset class for iPhone multi-view data
    
    Structure:
    iphone/{sequence}/camera/{camera_id}_{frame_id:05d}.json
    iphone/{sequence}/depth/{scale}x/{camera_id}_{frame_id:05d}.npy  (only camera 0)
    iphone/{sequence}/rgb/{scale}x/{camera_id}_{frame_id:05d}.png
    """
    
    def __init__(self, 
                 root_dir,
                 sequence_name,
                 scale='1x',
                 camera_ids=[0, 1, 2],
                 frame_range=None,
                 load_depth=True,
                 load_rgb=True,
                 min_sequence_length=49):
        """
        Args:
            root_dir: Path to 'iphone' directory
            sequence_name: Name of sequence (e.g., 'paper-windmill')
            scale: Image scale ('1x' or '2x')
            camera_ids: List of camera IDs to load [0, 1, 2]
            frame_range: Tuple (start, end) or None for all frames
            load_depth: Whether to load depth data (only available for camera 0)
            load_rgb: Whether to load RGB data
            min_sequence_length: Minimum length of contiguous sequences to keep
        """
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.scale = scale
        self.camera_ids = camera_ids
        self.load_depth = load_depth
        self.load_rgb = load_rgb
        self.min_sequence_length = min_sequence_length
        
        # Build paths
        self.sequence_dir = self.root_dir / sequence_name
        self.camera_dir = self.sequence_dir / 'camera'
        self.depth_dir = self.sequence_dir / 'depth' / scale
        self.rgb_dir = self.sequence_dir / 'rgb' / scale
        
        # Discover available frames
        self.frame_ids, self.contiguous_sequences = self._discover_frames(frame_range)
        
        if len(self.frame_ids) == 0:
            raise ValueError(f"No contiguous sequences of length >= {min_sequence_length} found!")
        
        print(f"iPhoneDataset initialized:")
        print(f"  Sequence: {sequence_name}")
        print(f"  Scale: {scale}")
        print(f"  Cameras: {camera_ids}")
        print(f"  Depth only available for camera 0")
        print(f"  Total valid frames: {len(self.frame_ids)} ({min(self.frame_ids):05d}-{max(self.frame_ids):05d})")
        print(f"  Contiguous sequences: {len(self.contiguous_sequences)}")
        
    def _discover_frames(self, frame_range):
        """Discover contiguous sequences where all cameras have data"""
        
        # First, collect all available frame IDs for each camera
        camera_frames = {}
        
        for cam_id in self.camera_ids:
            camera_files = list(self.camera_dir.glob(f"{cam_id}_*.json"))
            cam_frame_ids = set()
            
            for file_path in camera_files:
                frame_id = int(file_path.stem.split('_')[1])
                
                # Check if all required files exist for this camera and frame
                files_exist = True
                
                # Check camera file (already know it exists from glob)
                
                # Check depth file only for camera 0
                if self.load_depth and cam_id == 0:
                    depth_file = self.depth_dir / f"{cam_id}_{frame_id:05d}.npy"
                    if not depth_file.exists():
                        files_exist = False
                
                # Check RGB file (required for all cameras)
                if self.load_rgb:
                    rgb_file = self.rgb_dir / f"{cam_id}_{frame_id:05d}.png"
                    if not rgb_file.exists():
                        files_exist = False
                
                if files_exist:
                    cam_frame_ids.add(frame_id)
            
            camera_frames[cam_id] = sorted(list(cam_frame_ids))
            print(f"Camera {cam_id}: {len(cam_frame_ids)} frames available ({min(cam_frame_ids) if cam_frame_ids else 'N/A'}-{max(cam_frame_ids) if cam_frame_ids else 'N/A'})")
        
        # Find intersection - frames where ALL cameras have data
        common_frames = set(camera_frames[self.camera_ids[0]])
        for cam_id in self.camera_ids[1:]:
            common_frames = common_frames.intersection(set(camera_frames[cam_id]))
        
        common_frames = sorted(list(common_frames))
        print(f"Common frames across all cameras: {len(common_frames)}")
        
        if len(common_frames) == 0:
            return [], []
        
        # Apply frame range filter if specified
        if frame_range is not None:
            start, end = frame_range
            common_frames = [f for f in common_frames if start <= f <= end]
            print(f"After frame range filter [{start}-{end}]: {len(common_frames)} frames")
        
        # Find contiguous sequences of at least min_sequence_length
        contiguous_sequences = []
        current_sequence = []
        
        for i, frame_id in enumerate(common_frames):
            if i == 0 or frame_id == common_frames[i-1] + 1:
                # Continue current sequence
                current_sequence.append(frame_id)
            else:
                # Gap found, save current sequence if long enough
                if len(current_sequence) >= self.min_sequence_length:
                    contiguous_sequences.append(current_sequence)
                current_sequence = [frame_id]
        
        # Don't forget the last sequence
        if len(current_sequence) >= self.min_sequence_length:
            contiguous_sequences.append(current_sequence)
        
        print(f"Found {len(contiguous_sequences)} contiguous sequences of length >= {self.min_sequence_length}")
        for i, seq in enumerate(contiguous_sequences):
            print(f"  Sequence {i}: frames {seq[0]}-{seq[-1]} (length {len(seq)})")
        
        # Return all frames from valid sequences
        all_valid_frames = []
        for seq in contiguous_sequences:
            all_valid_frames.extend(seq)
        
        return sorted(all_valid_frames), contiguous_sequences
    
    def __len__(self):
        return len(self.frame_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
            - 'frame_id': int
            - 'cameras': dict with camera_id -> camera_data
            where camera_data contains:
                - 'camera_params': parsed camera parameters
                - 'K': intrinsic matrix [3, 3]
                - 'c2w': camera-to-world matrix [4, 4] 
                - 'depth': depth map [H, W] or None (only for camera 0)
                - 'rgb': RGB image [H, W, 3] (if load_rgb=True)
        """
        frame_id = self.frame_ids[idx]
        
        data = {
            'frame_id': frame_id,
            'cameras': {}
        }
        
        for cam_id in self.camera_ids:
            camera_data = self._load_camera_data(cam_id, frame_id)
            data['cameras'][cam_id] = camera_data
            
        return data
    
    def _load_camera_data(self, camera_id, frame_id):
        """Load all data for a specific camera and frame"""
        camera_data = {}
        
        # Load camera parameters
        camera_file = self.camera_dir / f"{camera_id}_{frame_id:05d}.json"
        with open(camera_file, 'r') as f:
            camera_params = json.load(f)
        
        camera_data['camera_params'] = camera_params
        
        # Parse camera parameters
        K, c2w = self._parse_camera_params(camera_params)
        camera_data['K'] = K
        camera_data['c2w'] = c2w
        
        # Load depth if requested and available (only for camera 0)
        if self.load_depth:
            if camera_id == 0:
                depth_file = self.depth_dir / f"{camera_id}_{frame_id:05d}.npy"
                if depth_file.exists():
                    depth = np.load(depth_file)
                    if len(depth.shape) == 3:
                        depth = depth.squeeze(-1)  # Remove singleton dimension
                    camera_data['depth'] = depth
                else:
                    camera_data['depth'] = None
                    print(f"Warning: Depth file not found for camera {camera_id}, frame {frame_id:05d}")
            else:
                # Depth not available for other cameras
                camera_data['depth'] = None
            
        # Load RGB if requested
        if self.load_rgb:
            rgb_file = self.rgb_dir / f"{camera_id}_{frame_id:05d}.png"
            if rgb_file.exists():
                rgb = iio.imread(rgb_file)
                if rgb.shape[-1] == 4:  # Remove alpha channel
                    rgb = rgb[:, :, :3]
                camera_data['rgb'] = rgb
            else:
                raise FileNotFoundError(f"RGB file not found: {rgb_file}")
            
        return camera_data
    
    def _parse_camera_params(self, camera_params):
        """Parse iPhone camera format to K and c2w matrices"""
        # Extract parameters
        focal_length = camera_params['focal_length']
        principal_point = camera_params['principal_point']
        position = np.array(camera_params['position'])
        orientation = np.array(camera_params['orientation'])
        
        # Create intrinsic matrix K
        K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Create camera-to-world matrix
        R = orientation  # 3x3 rotation matrix
        t = position     # 3D position
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.T  # Transpose for c2w
        c2w[:3, 3] = -R.T @ t  # Apply inverse transformation
        
        # c2w = np.eye(4, dtype=np.float32)
        # c2w[:3, :3] = R.T  # Direct assignment, no transpose
        # c2w[:3, 3] = t   # Direct assignment, no negative
        
        c2w = np.linalg.inv(c2w)  # Invert to get c2w
        
        
        return K, c2w
    
    def get_camera_info(self, camera_id, frame_id):
        """Get camera info for specific camera and frame"""
        idx = self.frame_ids.index(frame_id)
        data = self[idx]
        return data['cameras'][camera_id]
    
    def get_frame_by_id(self, frame_id):
        """Get data for specific frame ID"""
        idx = self.frame_ids.index(frame_id)
        return self[idx]
    
    def get_contiguous_subsequences(self, sequence_length):
        """Get all contiguous subsequences of specified length"""
        subsequences = []
        
        for seq in self.contiguous_sequences:
            if len(seq) >= sequence_length:
                for start_idx in range(len(seq) - sequence_length + 1):
                    subseq = seq[start_idx:start_idx + sequence_length]
                    subsequences.append(subseq)
        
        print(f"Generated {len(subsequences)} subsequences of length {sequence_length}")
        return subsequences
    
    def get_sequence_data(self, frame_ids):
        """Get data for a specific sequence of frame IDs"""
        sequence_data = []
        for frame_id in frame_ids:
            idx = self.frame_ids.index(frame_id)
            sequence_data.append(self[idx])
        return sequence_data
    
    def get_random_sequence(self, sequence_length):
        """Get a random contiguous sequence of specified length"""
        valid_sequences = []
        for seq in self.contiguous_sequences:
            if len(seq) >= sequence_length:
                valid_sequences.append(seq)
        
        if not valid_sequences:
            raise ValueError(f"No sequences of length >= {sequence_length} available")
        
        # Pick random sequence and random starting point
        chosen_seq = np.random.choice(valid_sequences)
        max_start = len(chosen_seq) - sequence_length
        start_idx = np.random.randint(0, max_start + 1)
        
        selected_frames = chosen_seq[start_idx:start_idx + sequence_length]
        return self.get_sequence_data(selected_frames)
    
    def has_depth(self, camera_id):
        """Check if depth data is available for given camera"""
        return camera_id == 0 and self.load_depth
