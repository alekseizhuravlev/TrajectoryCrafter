import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import random
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse  # Add this import
import logging


def reshape_patches_to_spatial_temporal(features, num_frames=13, patch_height=24, patch_width=42):
    """
    Convert flattened patch features back to spatial-temporal grid
    
    Args:
        features: [batch, 13104, feature_dim] or [13104, feature_dim]
        num_frames: 13
        patch_height: 24 (48 // patch_size)
        patch_width: 42 (84 // patch_size)
    
    Returns:
        reshaped_features: [batch, 13, 24, 42, feature_dim] or [13, 24, 42, feature_dim]
    """
    if features.dim() == 2:
        # [13104, feature_dim] → [13, 24, 42, feature_dim]
        return features.view(num_frames, patch_height, patch_width, -1)
    else:
        # [batch, 13104, feature_dim] → [batch, 13, 24, 42, feature_dim]
        batch_size = features.shape[0]
        feature_dim = features.shape[-1]
        return features.view(batch_size, num_frames, patch_height, patch_width, feature_dim)


def resize_depth_to_feature_space(depth, target_frames=13, patch_height=24, patch_width=42):
    """
    Resize depth to match feature dimensions following the original patchification process.
    
    Args:
        depth: [num_frames, 1, height, width] - e.g., [49, 1, 540, 960]
        target_frames: 13 (from feature shape)
        patch_height: 24 (from feature shape) 
        patch_width: 42 (from feature shape)
    
    Returns:
        resized_depth: [target_frames, patch_height, patch_width]
    """
    # Ensure we have the right format: [batch, channels, frames, height, width]
    if depth.dim() == 4:  # [49, 1, 540, 960]
        # Rearrange to [1, 1, 49, 540, 960] for trilinear interpolation
        depth = depth.permute(1, 0, 2, 3).unsqueeze(0)
    elif depth.dim() == 3:  # [49, 540, 960]
        # Add batch and channel dimensions: [1, 1, 49, 540, 960]
        depth = depth.unsqueeze(0).unsqueeze(0)
    
    # logger.info(f"Depth shape before interpolation: {depth.shape}")
    
    # Temporal + spatial downsampling to match patch dimensions
    resized_depth = F.interpolate(
        depth,
        size=(target_frames, patch_height, patch_width),
        mode='trilinear',
        align_corners=False
    )
    
    # Remove batch and channel dimensions: [1, 1, 13, 24, 42] → [13, 24, 42]
    return resized_depth.squeeze(0).squeeze(0)

def normalize_features_per_channel(features):
    """
    features: [1, 13104, 3072] 
    Normalize each of the 3072 feature dimensions independently
    """
    # Method 1: Using broadcasting
    mean_per_channel = features.mean(dim=1, keepdim=True)  # [1, 1, 3072]
    std_per_channel = features.std(dim=1, keepdim=True)    # [1, 1, 3072]
    
    normalized_features = (features - mean_per_channel) / (std_per_channel + 1e-8)

    return normalized_features


class DepthProbingDataset(Dataset):
    def __init__(self, root_dir, timestep, feature_name):
        """
        Args:
            root_dir: Root directory containing all monkaa_* folders
            timestep: Feature timestep (e.g., 39)
            feature_name: Feature file name (e.g., 'cross_attn_0')
        """
        self.root_dir = Path(root_dir)
        self.timestep = timestep
        self.feature_name = feature_name
        
        # Find all dataset entries (monkaa_*)
        self.entries = sorted([d for d in self.root_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('monkaa_')])
        
        # Validate that required files exist
        self.valid_entries = []
        for entry in self.entries:
            depth_path = entry / 'depths' / 'depths.pt'
            feature_path = entry / 'features' / f'{timestep}' / f'{feature_name}.pt'
            if depth_path.exists() and feature_path.exists():
                self.valid_entries.append(entry)
        
        logger.info(f"Found {len(self.valid_entries)} valid entries")
    
    def __len__(self):
        return len(self.valid_entries)
    
    def __getitem__(self, idx):
        entry = self.valid_entries[idx]
        
        # Load depth
        depth_path = entry / 'depths' / 'depths.pt'
        depth = torch.load(depth_path, weights_only=True)
        
        # bug: keep last 49 frames
        depth = depth[10:]
        
        # log normalize depth
        depth = torch.log1p(depth)
        
        # Load features
        feature_path = entry / 'features' / f'{self.timestep}' / f'{self.feature_name}.pt'
        features = torch.load(feature_path, weights_only=True).float()
        
        # Normalize features per channel
        features = normalize_features_per_channel(features)
        
        
        # Resize depth and features to same spatial-temporal dimensions
        features = reshape_patches_to_spatial_temporal(features)[0]
        depth_resized = resize_depth_to_feature_space(depth)
        
        return {
            'features': features,
            'depth': depth_resized,
            'entry_name': entry.name
        }


class ConvProbe(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, feats):
        # feats: [B, T, H, W, C]
        B, T, H, W, C = feats.shape
        
        # Reshape to process all frames at once
        feats = feats.view(B * T, H, W, C)    # [B*T, H, W, C]
        feats = feats.permute(0, 3, 1, 2)     # [B*T, C, H, W]
        
        # feats = self.norm(feats)
        depth = self.proj(feats)              # [B*T, 1, H, W]
        depth = depth.squeeze(1)              # [B*T, H, W]
        
        # Reshape back to temporal format
        depth = depth.view(B, T, H, W)        # [B, T, H, W]
        return depth


# Evaluation function
def evaluate_probe(probe, data, data_name=""):
    probe.eval()
    losses = []
    predictions = []
    ground_truths = []
    relative_errors = []  # Add this
    
    with torch.no_grad():
        for data_i in tqdm(data, desc=f"Evaluating {data_name}"):
            feats = data_i['features'].unsqueeze(0).cuda()
            depth_gt = data_i['depth'].unsqueeze(0).cuda()
            
            pred = probe(feats)
            loss = criterion(pred, depth_gt)
            
            # Calculate relative error
            epsilon = 1e-8
            rel_error = torch.abs((pred - depth_gt) / (depth_gt + epsilon)) * 100
            mean_rel_error = rel_error.mean().item()
            
            losses.append(loss.item())
            relative_errors.append(mean_rel_error)
            predictions.append(pred.cpu())
            ground_truths.append(depth_gt.cpu())
    
    avg_loss = sum(losses) / len(losses)
    avg_rel_error = sum(relative_errors) / len(relative_errors)
    return avg_loss, avg_rel_error, predictions, ground_truths



if __name__ == "__main__":
    
    # Add argument parsing
    parser = argparse.ArgumentParser(description='MLP Probing with different timesteps and features')
    parser.add_argument('--timestep', type=str, required=True, help='Timestep (e.g., timestep_839)')
    parser.add_argument('--feature_name', type=str, required=True, help='Feature name (e.g., transformer_block_24)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--exp_name', type=str, 
                       help='Experiment name for output directory (default: mlp_probes_fixed_lr_50)')
    parser.add_argument('--data_dir', type=str,
                       help='Data directory name (default: linear_probing_fixed)')
    
    args = parser.parse_args()
    
    # Use the parsed arguments
    timestep = args.timestep
    feature_name = args.feature_name
    num_epochs = args.num_epochs
    exp_name = args.exp_name
    data_dir = args.data_dir
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)
    
    
    logger.info(f"Running with timestep: {timestep}, feature: {feature_name}, epochs: {num_epochs}")
    logger.info(f"Experiment: {exp_name}, Data dir: {data_dir}")
    
    
    save_dir = f'/home/azhuravl/scratch/mlp_probes/{exp_name}/{timestep}/{feature_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(save_dir, 'tb_logs'))
    
    # load dataset with configurable data directory
    dataset = DepthProbingDataset(
        f'/home/azhuravl/scratch/{data_dir}',
        f'{timestep}', 
        f'{feature_name}'
    )
    
    # Inspect a sample
    data_10 = dataset[10]
    logger.info(f"Depth range: {data_10['depth'].min():.4f} to {data_10['depth'].max():.4f}")
    logger.info(f"Features range: {data_10['features'].min():.4f} to {data_10['features'].max():.4f}")
    
    
    
    total_samples = len(dataset)
    train_size = int(0.75 * total_samples)
    test_size = total_samples - train_size

    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Train size: {train_size} ({train_size/total_samples*100:.1f}%)")
    logger.info(f"Test size: {test_size} ({test_size/total_samples*100:.1f}%)")


    # Preload training data (first 75% of entries)
    logger.info("Preloading training data...")
    train_data = []
    for i in tqdm(range(train_size)):
        data_i = dataset[i]
        train_data.append({
            'features': data_i['features'].clone(),  # Clone to avoid reference issues
            'depth': data_i['depth'].clone(),
            'entry_name': data_i['entry_name']
        })

    # Preload test data (last 25% of entries)
    logger.info("Preloading test data...")
    test_data = []
    for i in tqdm(range(train_size, total_samples)):
        data_i = dataset[i]
        test_data.append({
            'features': data_i['features'].clone(),
            'depth': data_i['depth'].clone(),
            'entry_name': data_i['entry_name']
        })

    logger.info(f"Loaded {len(train_data)} training samples, {len(test_data)} test samples")
    
    
    # setup training
    probe = ConvProbe(3072).cuda()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    losses = []
    
    # training loop
    for epoch in range(num_epochs):
        # Create shuffled indices
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        iterator = tqdm(indices, desc=f"Epoch {epoch+1}")
        for batch_idx, idx in enumerate(iterator):
            data_i = train_data[idx]
            feats = data_i['features'].unsqueeze(0).cuda()
            depth_gt = data_i['depth'].unsqueeze(0).cuda()
            
            pred = probe(feats)
            loss = criterion(pred, depth_gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Log to TensorBoard
            global_step = epoch * len(train_data) + batch_idx
            writer.add_scalar('Loss/Train_Step', loss.item(), global_step)

            iterator.set_description(f"Epoch {epoch+1} Loss: {sum(losses[-10:]) / len(losses[-10:]):.4f}")
            
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
    
        


    # save the checkpoint
    torch.save(probe.state_dict(), os.path.join(save_dir, 'conv_probe.pth'))
    
    # plot training loss
    window = 10
    losses_smooth = np.convolve(losses, np.ones(window)/window, mode='valid')

    plt.plot(losses_smooth)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss (smoothed)')
    plt.title(f'Training Loss (moving avg, window={window})')
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()
    
    # Evaluate on both train and test sets
    logger.info("Evaluating on training set...")
    train_loss, train_rel_error, train_preds, train_gts = evaluate_probe(probe, train_data, "Train")

    logger.info("Evaluating on test set...")
    test_loss, test_rel_error, test_preds, test_gts = evaluate_probe(probe, test_data, "Test")

    logger.info(f"\nTrain Loss: {train_loss:.4f}")
    logger.info(f"Train Relative Error: {train_rel_error:.2f}%")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Relative Error: {test_rel_error:.2f}%")
    logger.info(f"Generalization Gap (MSE): {test_loss - train_loss:.4f}")
    logger.info(f"Generalization Gap (Rel Error): {test_rel_error - train_rel_error:.2f}%")
    
    # Log final metrics to TensorBoard
    writer.add_scalar('Loss/Final_Train', train_loss, 0)
    writer.add_scalar('Loss/Final_Test', test_loss, 0)
    writer.add_scalar('Loss/Generalization_Gap', test_loss - train_loss, 0)
    
    # Add relative error metrics
    writer.add_scalar('Relative_Error/Final_Train', train_rel_error, 0)
    writer.add_scalar('Relative_Error/Final_Test', test_rel_error, 0)
    writer.add_scalar('Relative_Error/Generalization_Gap', test_rel_error - train_rel_error, 0)
    
    # Log hyperparameters and final metrics
    writer.add_hparams(
        {
            'timestep': timestep,
            'feature_name': feature_name,
            'learning_rate': 1e-3,
            'epochs': num_epochs,
            'train_samples': len(train_data),
            'test_samples': len(test_data)
        },
        {
            'final_train_loss': train_loss,
            'final_test_loss': test_loss,
            'generalization_gap_mse': test_loss - train_loss,
            'final_train_rel_error': train_rel_error,
            'final_test_rel_error': test_rel_error,
            'generalization_gap_rel_error': test_rel_error - train_rel_error
        }
    )
    
    
    # Close the writer
    writer.close()
    
    
    # Visualize some predictions
    ids_vis = range(0, len(test_data), 5)

    for idx in ids_vis:
        # plot gt, pred, difference for frame 6 only. use same color scale
        gt_depth = test_gts[idx][0, 6].squeeze().numpy()
        pred_depth = test_preds[idx][0, 6].squeeze().numpy()
        
        # Calculate relative error in percentage
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-8
        rel_error = ((pred_depth - gt_depth) / (gt_depth + epsilon)) * 100
        
        vmin = min(gt_depth.min(), pred_depth.min())
        vmax = max(gt_depth.max(), pred_depth.max())
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        im0 = axs[0].imshow(gt_depth, cmap='plasma', vmin=vmin, vmax=vmax)
        axs[0].set_title('Ground Truth Depth Frame 6')
        plt.colorbar(im0, ax=axs[0], fraction=0.03, pad=0.04)
        # remove axes
        axs[0].axis('off')
        
        # Prediction
        im1 = axs[1].imshow(pred_depth, cmap='plasma', vmin=vmin, vmax=vmax)
        axs[1].set_title('Predicted Depth Frame 6')
        plt.colorbar(im1, ax=axs[1], fraction=0.03, pad=0.04)
        axs[1].axis('off')
        
        # Print some statistics
        mean_abs_rel_error = np.abs(rel_error).mean()
        logger.info(f"Sample {idx}: Mean Absolute Relative Error: {mean_abs_rel_error:.2f}%")
        
        
        # Relative error in percentage
        im2 = axs[2].imshow(rel_error, cmap='RdBu_r', vmin=-50, vmax=50)  # Symmetric range
        axs[2].set_title(f'Relative Error: {mean_abs_rel_error:.2f}%')
        plt.colorbar(im2, ax=axs[2], label='Error %', fraction=0.03, pad=0.04)
        axs[2].axis('off')
        
        plt.savefig(os.path.join(save_dir, f'prediction_vis_{idx}.png'))
        plt.close()