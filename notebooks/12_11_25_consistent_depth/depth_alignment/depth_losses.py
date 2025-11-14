import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/home/azhuravl/work/Video-Depth-Anything')

from loss.loss import VideoDepthLoss



class SimpleDepthLoss(nn.Module):
    def __init__(self, l1_weight=1.0, rmse_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.rmse_weight = rmse_weight

    def forward(self, prediction, target, mask):
        '''
            prediction: predicted depth tensor
            target: target depth tensor  
            mask: valid pixel mask
        '''
        loss_dict = {}
        
        if mask.sum() == 0:
            # No valid pixels, return zero losses
            device = prediction.device
            zero_loss = torch.tensor(0.0, device=device, dtype=prediction.dtype)
            loss_dict['l1_loss'] = zero_loss
            loss_dict['rmse_loss'] = zero_loss
            loss_dict['total_loss'] = zero_loss
            return loss_dict
        
        # Extract valid pixels
        pred_valid = prediction[mask]
        target_valid = target[mask]
        
        # Compute individual losses
        loss_dict['l1_loss'] = F.l1_loss(pred_valid, target_valid) * self.l1_weight
        loss_dict['rmse_loss'] = torch.sqrt(F.mse_loss(pred_valid, target_valid)) * self.rmse_weight
        
        # Total loss
        loss_dict['total_loss'] = loss_dict['l1_loss'] + loss_dict['rmse_loss']
        
        return loss_dict



# Updated Combined Loss Class
class CombinedDepthLossWithTAE(nn.Module):
    def __init__(self, vda_weight=1.0, simple_weight=1.0, tae_weight=0.1, 
                 l1_weight=1.0, rmse_weight=1.0):
        super().__init__()
        self.vda_loss = VideoDepthLoss()
        self.simple_loss = SimpleDepthLoss(l1_weight=l1_weight, rmse_weight=rmse_weight)
        self.tae_loss = DifferentiableTAELoss(weight=tae_weight)
        self.vda_weight = vda_weight
        self.simple_weight = simple_weight
        self.tae_weight = tae_weight

    def forward(self, prediction, target, mask, intrinsics=None, extrinsics=None):
        '''
            prediction: predicted depth tensor [B, T, H, W]
            target: target depth tensor [B, T, H, W]
            mask: valid pixel mask [B, T, H, W]
            intrinsics: [3, 3] camera intrinsics (for TAE)
            extrinsics: [T, 4, 4] camera poses (for TAE)
        '''
        # Get VDA and Simple losses
        vda_loss_dict = self.vda_loss(prediction, target, mask)
        simple_loss_dict = self.simple_loss(prediction, target, mask)
        
        # Combine losses
        loss_dict = {}
        
        # Add individual VDA and Simple losses
        for key, value in vda_loss_dict.items():
            if key != 'total_loss':
                loss_dict[f'vda_{key}'] = value
        
        for key, value in simple_loss_dict.items():
            if key != 'total_loss':
                loss_dict[f'simple_{key}'] = value
        
        # Compute TAE loss if intrinsics and extrinsics are provided
        if intrinsics is not None and extrinsics is not None:
            tae_loss_value = self.tae_loss(prediction, intrinsics, extrinsics)
            loss_dict['tae_loss'] = tae_loss_value
        else:
            loss_dict['tae_loss'] = torch.tensor(0.0, device=prediction.device)
        
        # Weighted combination
        vda_total = vda_loss_dict['total_loss'] * self.vda_weight
        simple_total = simple_loss_dict['total_loss'] * self.simple_weight
        tae_total = loss_dict['tae_loss'] * self.tae_weight
        
        loss_dict['vda_total'] = vda_total
        loss_dict['simple_total'] = simple_total
        loss_dict['tae_total'] = tae_total
        loss_dict['total_loss'] = vda_total + simple_total + tae_total
        
        return loss_dict


class DifferentiableTAELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, depth_sequence, intrinsics, extrinsics, mask_threshold=1e-3, max_depth=100.0):
        """
        Differentiable Temporal Alignment Error loss.
        
        Args:
            depth_sequence: [B, T, H, W] or [T, H, W] depth tensor
            intrinsics: [3, 3] intrinsic matrix
            extrinsics: [T, 4, 4] camera poses tensor
            mask_threshold: minimum valid depth value
            max_depth: maximum valid depth value
        
        Returns:
            TAE loss (scalar tensor)
        """
        if len(depth_sequence.shape) == 3:
            depth_sequence = depth_sequence.unsqueeze(0)  # Add batch dim
        
        B, T, H, W = depth_sequence.shape
        device = depth_sequence.device
        
        # Ensure float32 for numerical stability
        depth_sequence = depth_sequence.float()
        intrinsics = intrinsics.float().to(device)
        extrinsics = extrinsics.float().to(device)
        
        # Extract intrinsic parameters
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        total_error = 0.0
        valid_pairs = 0
        
        for i in range(T - 1):
            depth1 = depth_sequence[:, i]  # [B, H, W]
            depth2 = depth_sequence[:, i + 1]  # [B, H, W]
            
            # Camera poses
            T_1 = extrinsics[i]      # [4, 4]
            T_2 = extrinsics[i + 1]  # [4, 4]
            
            # Relative transformation
            T_2_1 = torch.linalg.inv(T_2) @ T_1
            R = T_2_1[:3, :3]  # [3, 3]
            t = T_2_1[:3, 3]   # [3]
            
            # Valid depth masks
            mask1 = (depth1 > mask_threshold) & (depth1 < max_depth)  # [B, H, W]
            mask2 = (depth2 > mask_threshold) & (depth2 < max_depth)  # [B, H, W]
            
            # Convert pixels to 3D points in frame 1
            X1 = (x_coords - cx) * depth1 / fx  # [B, H, W]
            Y1 = (y_coords - cy) * depth1 / fy  # [B, H, W]
            Z1 = depth1  # [B, H, W]
            
            # Stack to get 3D points [B, 3, H, W]
            points_3d_1 = torch.stack([X1, Y1, Z1], dim=1)
            
            # Transform points to frame 2 coordinate system
            # Reshape for matrix multiplication: [B, 3, H*W]
            points_flat = points_3d_1.reshape(B, 3, -1)
            
            # Apply rotation and translation
            points_2_flat = R @ points_flat + t.unsqueeze(-1)  # [3, H*W]
            points_2 = points_2_flat.reshape(B, 3, H, W)  # [B, 3, H, W]
            
            # Project back to frame 2 image coordinates
            X2, Y2, Z2 = points_2[:, 0], points_2[:, 1], points_2[:, 2]
            
            # Avoid division by zero
            Z2_safe = torch.clamp(Z2, min=1e-6)
            u2 = fx * X2 / Z2_safe + cx
            v2 = fy * Y2 / Z2_safe + cy
            
            # Check if projected coordinates are within image bounds
            valid_proj = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H) & (Z2 > mask_threshold)
            
            # Combined mask
            combined_mask = mask1 & valid_proj
            
            if combined_mask.sum() > 0:
                # Sample depth2 at projected coordinates using bilinear interpolation
                # Normalize coordinates to [-1, 1] for grid_sample
                grid_x = (u2 / (W - 1)) * 2.0 - 1.0  # [B, H, W]
                grid_y = (v2 / (H - 1)) * 2.0 - 1.0  # [B, H, W]
                grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]
                
                # Sample depth2 using bilinear interpolation
                depth2_sampled = F.grid_sample(
                    depth2.unsqueeze(1),  # [B, 1, H, W]
                    grid,  # [B, H, W, 2]
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                ).squeeze(1)  # [B, H, W]
                
                # Compute depth error only where mask is valid
                projected_depth = Z2  # Depth of projected point
                depth_error = torch.abs(projected_depth - depth2_sampled)
                
                # Relative error (more robust)
                relative_error = depth_error / (depth2_sampled + 1e-6)
                
                # Average over valid pixels
                error = (relative_error * combined_mask.float()).sum() / (combined_mask.float().sum() + 1e-6)
                total_error += error
                valid_pairs += 1
        
        if valid_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return (total_error / valid_pairs) * self.weight

