"""
Video segmentation using DINOv2 features and clustering
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class DINOVideoSegmenter:
    """Video segmentation using DINOv2 features"""
    
    def __init__(self, model_name='dinov2_vitb14', device='cuda'):
        """
        Initialize DINOv2 model for segmentation
        
        Args:
            model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            device: Device to run inference on
        """
        self.device = device
        self.model_name = model_name
        
        # Load DINOv2 model
        print(f"Loading {model_name}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Get model properties
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        
        print(f"Model loaded: patch_size={self.patch_size}, embed_dim={self.embed_dim}")
    
    def preprocess_frame(self, frame, target_size=518):
        """
        Preprocess frame for DINOv2
        
        Args:
            frame: Input frame (H, W, 3) in BGR format
            target_size: Target size for resizing
            
        Returns:
            tensor: Preprocessed tensor (1, 3, H, W)
            scale_factor: Scale factor for resizing back
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio
        h, w = frame_rgb.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Pad to make it divisible by patch_size
        pad_h = (self.patch_size - new_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - new_w % self.patch_size) % self.patch_size
        
        frame_padded = np.pad(frame_resized, 
                            ((0, pad_h), (0, pad_w), (0, 0)), 
                            mode='constant', constant_values=0)
        
        # Normalize to [0, 1] and convert to tensor
        frame_tensor = torch.from_numpy(frame_padded).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        return frame_tensor.to(self.device), scale, (new_h, new_w), (h, w)
    
    def extract_features(self, frame_tensor):
        """
        Extract DINOv2 features from frame
        
        Args:
            frame_tensor: Preprocessed frame tensor (1, 3, H, W)
            
        Returns:
            features: Feature tensor (num_patches, embed_dim)
            grid_shape: (height_patches, width_patches)
        """
        with torch.no_grad():
            # Get patch embeddings (excluding CLS token)
            features = self.model.forward_features(frame_tensor)
            patch_features = features['x_norm_patchtokens']  # (1, num_patches, embed_dim)
            
            # Remove batch dimension
            patch_features = patch_features.squeeze(0)  # (num_patches, embed_dim)
            
            # Calculate grid shape
            _, _, h, w = frame_tensor.shape
            h_patches = h // self.patch_size
            w_patches = w // self.patch_size
            
            return patch_features, (h_patches, w_patches)
    
    def cluster_features(self, features, n_clusters=8, use_pca=True, pca_dim=50):
        """
        Cluster features to create segmentation masks
        
        Args:
            features: Feature tensor (num_patches, embed_dim)
            n_clusters: Number of clusters
            use_pca: Whether to use PCA for dimensionality reduction
            pca_dim: PCA dimensions
            
        Returns:
            labels: Cluster labels (num_patches,)
            cluster_centers: Cluster centers
        """
        features_np = features.cpu().numpy()
        
        # Optional PCA for dimensionality reduction
        if use_pca and features_np.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim)
            features_np = pca.fit_transform(features_np)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_np)
        
        return labels, kmeans.cluster_centers_
    
    def create_segmentation_mask(self, labels, grid_shape, original_size):
        """
        Create segmentation mask from cluster labels
        
        Args:
            labels: Cluster labels (num_patches,)
            grid_shape: (height_patches, width_patches)
            original_size: (height, width) of original frame
            
        Returns:
            mask: Segmentation mask (height, width)
        """
        h_patches, w_patches = grid_shape
        
        # Reshape labels to grid
        label_grid = labels.reshape(h_patches, w_patches)
        
        # Resize to patch resolution
        patch_h = h_patches * self.patch_size
        patch_w = w_patches * self.patch_size
        
        # Upsample using nearest neighbor
        mask_patches = np.repeat(np.repeat(label_grid, self.patch_size, axis=0), 
                                self.patch_size, axis=1)
        
        # Resize to original size
        mask = cv2.resize(mask_patches.astype(np.uint8), 
                         (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def segment_frame(self, frame, n_clusters=8, use_pca=True):
        """
        Segment a single frame
        
        Args:
            frame: Input frame (H, W, 3) in BGR format
            n_clusters: Number of segments
            use_pca: Whether to use PCA
            
        Returns:
            mask: Segmentation mask
            colored_mask: Colored visualization
        """
        # Preprocess
        frame_tensor, scale, resized_shape, original_shape = self.preprocess_frame(frame)
        
        # Extract features
        features, grid_shape = self.extract_features(frame_tensor)
        
        # Cluster features
        labels, centers = self.cluster_features(features, n_clusters, use_pca)
        
        # Create mask
        mask = self.create_segmentation_mask(labels, grid_shape, original_shape)
        
        # Create colored mask for visualization
        colored_mask = self.create_colored_mask(mask, n_clusters)
        
        return mask, colored_mask
    
    def create_colored_mask(self, mask, n_clusters):
        """Create a colored visualization of the segmentation mask"""
        # Create colormap
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))[:, :3]  # RGB values
        colors = (colors * 255).astype(np.uint8)
        
        # Apply colors
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i in range(n_clusters):
            colored_mask[mask == i] = colors[i]
        
        return colored_mask
    
    def segment_video(self, video_path, output_dir, n_clusters=8, max_frames=None, 
                     save_masks=True, save_overlay=True, use_pca=True):
        """
        Segment entire video
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            n_clusters: Number of segments
            max_frames: Maximum frames to process (None for all)
            save_masks: Whether to save segmentation masks
            save_overlay: Whether to save overlay visualizations
            use_pca: Whether to use PCA
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        if save_masks:
            (output_dir / "masks").mkdir(exist_ok=True)
        if save_overlay:
            (output_dir / "overlays").mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing {total_frames} frames at {fps} FPS...")
        
        # Process frames
        frame_idx = 0
        pbar = tqdm(total=total_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break
            
            # Segment frame
            mask, colored_mask = self.segment_frame(frame, n_clusters, use_pca)
            
            # Save results
            if save_masks:
                mask_path = output_dir / "masks" / f"mask_{frame_idx:06d}.png"
                cv2.imwrite(str(mask_path), mask * (255 // n_clusters))
                
                colored_path = output_dir / "masks" / f"colored_{frame_idx:06d}.png"
                cv2.imwrite(str(colored_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
            
            if save_overlay:
                # Create overlay
                overlay = cv2.addWeighted(frame, 0.7, 
                                        cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR), 0.3, 0)
                overlay_path = output_dir / "overlays" / f"overlay_{frame_idx:06d}.png"
                cv2.imwrite(str(overlay_path), overlay)
            
            frame_idx += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        print(f"Segmentation complete! Results saved to {output_dir}")
        
        # Save summary
        summary = {
            'video_path': str(video_path),
            'total_frames': frame_idx,
            'n_clusters': n_clusters,
            'model_name': self.model_name,
            'fps': fps
        }
        
        import json
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Segment video using DINOv2")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output_dir", default="./segmentation_output", 
                       help="Output directory")
    parser.add_argument("--model", default="dinov2_vitb14",
                       choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                       help="DINOv2 model variant")
    parser.add_argument("--clusters", type=int, default=8,
                       help="Number of clusters/segments")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--no_pca", action="store_true",
                       help="Disable PCA dimensionality reduction")
    
    args = parser.parse_args()
    
    # Create segmenter
    segmenter = DINOVideoSegmenter(args.model, args.device)
    
    # Segment video
    segmenter.segment_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        n_clusters=args.clusters,
        max_frames=args.max_frames,
        use_pca=not args.no_pca
    )


if __name__ == "__main__":
    main()