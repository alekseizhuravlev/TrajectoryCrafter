import torch
import torch.nn as nn
from depth_losses import CombinedDepthLossWithTAE, DifferentiableTAELoss
import logging
import sys

class DepthAlignmentTrainer:
    def __init__(self, video_depth_model, lr=2e-3, device='cuda', logger=None):
        self.video_depth_model = video_depth_model
        self.device = device
        self.lr = lr
        self.logger = logger
        
        # instantiate logger if not provided
        if self.logger is None:

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[
                    # logging.FileHandler(log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            self.logger = logging.getLogger(__name__)

        # Disable gradients for the main model
        for param in self.video_depth_model.parameters():
            param.requires_grad = False
        
        # Instantiate loss functions with hard-coded parameters
        self.loss_fn = CombinedDepthLossWithTAE(
            vda_weight=1.0, 
            simple_weight=1.0, 
            tae_weight=0.0,  # Start with small weight
            l1_weight=1.0, 
            rmse_weight=1.0
        )
        
        # Create TAE loss for evaluation
        self.loss_tae = DifferentiableTAELoss(weight=1.0)
    
    def train(self, rgb, sparse_depth, sparse_mask, 
              intrinsics_torch=None, extrinsics_torch=None, epochs=50):
        """
        Train visual prompt to align depth predictions.
        
        Args:
            rgb: [B, T, C, H, W] RGB frames
            sparse_depth: [B, T, H, W] sparse depth target
            sparse_mask: [B, T, H, W] validity mask
            intrinsics_torch: [3, 3] camera intrinsics
            extrinsics_torch: [T, 4, 4] camera poses
            epochs: number of training epochs
            logger: logger instance
        
        Returns:
            corrected_depth: [B, T, H, W] aligned depth prediction
            visual_prompt: optimized visual prompt
            final_scale: scale parameter
            final_shift: shift parameter
        """
        # Initialize learnable prompt
        visual_prompt = torch.nn.Parameter(torch.zeros_like(rgb, dtype=torch.bfloat16, device=self.device))
        optimizer = torch.optim.AdamW([{'params': visual_prompt, 'lr': self.lr}])
        
        # Training loop
        for epoch in range(epochs):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                new_rgb = rgb + visual_prompt
                pre_depth_ = self.video_depth_model.forward(new_rgb)
                
                # Compute scale and shift
                scale, shift = compute_scale_and_shift(pre_depth_.flatten(1,2), sparse_depth.flatten(1,2), sparse_mask.flatten(1,2))
                pre_depth = scale.view(-1, 1, 1, 1) * pre_depth_ + shift.view(-1, 1, 1, 1)
                
                # Compute loss
                loss_dict = self.loss_fn(
                    pre_depth, 
                    sparse_depth, 
                    sparse_mask,
                    intrinsics=intrinsics_torch,
                    extrinsics=extrinsics_torch
                )
                loss = loss_dict['total_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                rde = (torch.abs(pre_depth[sparse_mask] - sparse_depth[sparse_mask]) / (sparse_depth[sparse_mask] + 1e-8)).mean()

                # TAE evaluation
                tae = evaluate_tae(
                    pre_depth.squeeze(0),
                    intrinsics_torch,
                    extrinsics_torch
                )
                tae_diff = self.loss_tae(
                    pre_depth.squeeze(0),
                    intrinsics_torch,
                    extrinsics_torch
                ).item()

            # Logging
            if self.logger:
                loss_str_parts = []
                for loss_name, loss_value in loss_dict.items():
                    if loss_name != 'total_loss':
                        loss_str_parts.append(f"{loss_name}: {loss_value.item():.4f}")
                
                loss_str_parts.extend([
                    "\n"
                    f"rde: {rde.item():.4f}",
                    f"tae: {tae:.4f}",
                    f"tae_diff: {tae_diff:.4f}",
                    f"scale: {scale.item():.4f}",
                    f"shift: {shift.item():.4f}"
                ])
                
                loss_str = ", ".join(loss_str_parts)
                self.logger.info(f'{epoch:3d}/{epochs} {loss_str}')

        # Final prediction
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                new_rgb = rgb + visual_prompt
                corrected_depth_ = self.video_depth_model.forward(new_rgb)
                
                # Final scale and shift
                final_scale, final_shift = compute_scale_and_shift(corrected_depth_.flatten(1,2), sparse_depth.flatten(1,2), sparse_mask.flatten(1,2))
                corrected_depth = final_scale.view(-1, 1, 1, 1) * corrected_depth_ + final_shift.view(-1, 1, 1, 1)

        return corrected_depth, visual_prompt, final_scale, final_shift
    
    def validate(self, corrected_depth, depth, sparse_depth, gt_mask, sparse_mask,
                intrinsics_torch, extrinsics_torch):
        """
        Compute validation metrics - preserve your exact metrics and logging.
        """
        with torch.no_grad():
            # Preserve your exact ground truth metrics computation
            gt_mask_cpu = gt_mask.cpu()
            final_rmse_gt = torch.sqrt(((corrected_depth[gt_mask_cpu] - depth[gt_mask_cpu]) ** 2).mean())
            final_mae_gt = torch.abs(corrected_depth[gt_mask_cpu] - depth[gt_mask_cpu]).mean()
            final_rde_gt = (torch.abs(corrected_depth[gt_mask_cpu] - depth[gt_mask_cpu]) / depth[gt_mask_cpu]).mean()
            
            # Preserve your exact sparse depth metrics computation
            sparse_mask_cpu = sparse_mask.cpu()
            final_rmse_sparse = torch.sqrt(((corrected_depth[sparse_mask_cpu] - sparse_depth[sparse_mask_cpu]) ** 2).mean())
            final_mae_sparse = torch.abs(corrected_depth[sparse_mask_cpu] - sparse_depth[sparse_mask_cpu]).mean()
            final_rde_sparse = (torch.abs(corrected_depth[sparse_mask_cpu] - sparse_depth[sparse_mask_cpu]) / sparse_depth[sparse_mask_cpu]).mean()
            
            # Preserve your exact TAE computation
            final_tae = evaluate_tae(
                corrected_depth.squeeze(0),
                intrinsics_torch,
                extrinsics_torch
            )
            gt_tae = evaluate_tae(
                depth.float().squeeze(0),
                intrinsics_torch,
                extrinsics_torch
            )

        # Preserve your exact logging format
        if self.logger:
            self.logger.info("Final Metrics:")
            self.logger.info(f"Ground Truth  - RMSE: {final_rmse_gt.item():.4f}, MAE: {final_mae_gt.item():.4f}, RDE: {final_rde_gt.item():.4f}")
            self.logger.info(f"Sparse Depth  - RMSE: {final_rmse_sparse.item():.4f}, MAE: {final_mae_sparse.item():.4f}, RDE: {final_rde_sparse.item():.4f}")
            self.logger.info(f"Temporal Alignment Error (TAE): {final_tae:.4f} (GT TAE: {gt_tae:.4f})")

        return {
            'gt_rmse': final_rmse_gt.item(),
            'gt_mae': final_mae_gt.item(), 
            'gt_rde': final_rde_gt.item(),
            'sparse_rmse': final_rmse_sparse.item(),
            'sparse_mae': final_mae_sparse.item(),
            'sparse_rde': final_rde_sparse.item(),
            'final_tae': final_tae,
            'gt_tae': gt_tae
        }


# Updated main training code - replace your training loop with this:
if __name__ == '__main__':
    # ... all your existing setup code stays exactly the same ...
    
    # Replace the training loop section (around line 814) with:
    trainer = DepthAlignmentTrainer(
        video_depth_anything, 
        loss_fn, 
        lr=2e-3, 
        device=DEVICE
    )
    
    # Train - preserve all your variable names and structure
    corrected_depth, visual_prompt, final_scale, final_shift = trainer.train(
        rgb, sparse_depth, sparse_mask,
        intrinsics_torch=intrinsics_torch,
        extrinsics_torch=extrinsics_torch,
        epochs=args_ttt.epochs,
        logger=logger
    )
    
    # Validate - preserve your exact logging
    validation_metrics = trainer.validate(
        corrected_depth, depth, sparse_depth, gt_mask, sparse_mask,
        intrinsics_torch, extrinsics_torch, logger=logger
    )
    
    # Preserve your exact final scale/shift logging
    logger.info(f"Final Scale: {final_scale.item():.4f}, Final Shift: {final_shift.item():.4f}")
    
    # ... all your existing saving and visualization code stays exactly the same ...