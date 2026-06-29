"""
UNet Model for BubbleFlow Prediction.

This module implements a UNet architecture for predicting physical fields from 
conditioning inputs - a direct regression approach (no diffusion).

Supports multiple tasks through task_cfg configuration:
- temperature_from_sdf: Predict temperature from SDF (Task 1)
- velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)

This is a FRAME-TO-FRAME model (non-autoregressive):
    Input:  conditioning
    Target: output fields

Classes:
    UNet2d: UNet architecture adapted from PDEBench
    UNetLightning: PyTorch Lightning wrapper

Reference:
    UNet implementation adapted from:
    https://github.com/HPCForge/BubbleML/blob/main/sciml/models/pdebench/unet.py
"""

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional


class UNet2d(nn.Module):
    """
    U-Net architecture for 2D field prediction.
    
    This is the same architecture used in unet_ar.py, adapted from PDEBench.
    Uses GELU activation and ConvTranspose2d for upsampling.
    
    Args:
        in_channels: Number of input channels (conditioning)
        out_channels: Number of output channels (targets)
        init_features: Initial number of features (default: 32)
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32):
        super(UNet2d, self).__init__()
        features = init_features
        
        # Encoder
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet2d._block(features * 8, features * 16, name="bottleneck")

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        # Output convolution
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        """
        Basic UNet block with two convolutions, batch norm, and GELU activation.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "gelu1", nn.GELU()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "gelu2", nn.GELU()),
                ]
            )
        )


class UNetLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for UNet prediction model.
    
    This is a FRAME-TO-FRAME model (non-autoregressive):
    - Input: conditioning
    - Target: output fields
    
    Supports multiple tasks through task_cfg configuration:
    - temperature_from_sdf: Predict temperature from SDF (Task 1)
    - velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)
    
    Args:
        model_cfg: Model configuration containing UNet parameters
        optim_cfg: Optimizer configuration
        scheduler_cfg: Learning rate scheduler configuration
        task_cfg: Task configuration defining which channels to use
        normalization_stats: Pre-computed normalization statistics from training data
    """
    
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig, scheduler_cfg: DictConfig,
                 task_cfg: Optional[DictConfig] = None, normalization_stats: Optional[dict] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])
        
        # Store normalization statistics for accurate denormalization during logging
        self.normalization_stats = normalization_stats
        
        # Store task configuration
        self.task_cfg = task_cfg
        
        # Parse task config for channel indices
        if task_cfg is not None:
            self.conditioning_channels = list(task_cfg.get('conditioning_channels', [0]))
            self.target_channels = list(task_cfg.get('target_channels', [0]))
            self.task_name = task_cfg.get('name', 'unknown')
            
            # Check if this is Task 3 (noisy variant)
            self.is_noisy_task = 'noisy' in self.task_name.lower()
            noise_cfg = task_cfg.get('noise_cfg', None)
            self.has_noise = noise_cfg is not None and noise_cfg.get('enabled', True)
            
            print(f"🎯 Task: {self.task_name}")
            print(f"   Conditioning channels: {self.conditioning_channels} ({task_cfg.get('conditioning_names', [])})")
            print(f"   Target channels: {self.target_channels} ({task_cfg.get('target_names', [])})")
            
            # Log noise configuration for Task 3
            if self.has_noise:
                print(f"   🔊 Noise injection: ENABLED (simulating optical flow uncertainty)")
                print(f"      SDF noise: std={noise_cfg.get('sdf_noise_std', 0.1)}")
                print(f"      Velocity noise: base={noise_cfg.get('vel_base_noise_std', 0.05)}, scale={noise_cfg.get('vel_scale_factor', 0.15)}")
        else:
            # Default to Task 1 behavior (backward compatibility)
            self.conditioning_channels = [0]  # SDF only
            self.target_channels = [0]  # Temperature only
            self.task_name = 'temperature_from_sdf'
            self.is_noisy_task = False
            self.has_noise = False
            print("⚠️  No task_cfg provided, defaulting to temperature_from_sdf task")
        
        # Compute derived channel counts
        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)
        
        # Compute in_channels and out_channels from task_cfg
        in_channels = self.num_conditioning_channels
        out_channels = self.num_target_channels
        
        print(f"\n🔧 UNet Configuration:")
        print(f"   UNet in_channels: {in_channels} (conditioning)")
        print(f"   UNet out_channels: {out_channels} (targets)")
        
        # Initialize UNet model
        self.unet = UNet2d(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=model_cfg.get('init_features', 32)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # =============================================================================
        # NORMALIZATION CONFIGURATION
        # =============================================================================
        # Get downsample factor from normalization_stats (passed through from data config)
        self.downsample_factor = 1
        if normalization_stats is not None:
            self.downsample_factor = normalization_stats.get('downsample_factor', 1)
        
        # Temperature normalization parameters
        # Use computed stats if available, otherwise fall back to config values
        if normalization_stats is not None and 'temperature' in normalization_stats:
            temp_stats = normalization_stats['temperature']
            self.temp_min = temp_stats['min']
            self.temp_max = temp_stats['max']
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = normalization_stats.get('unified_velocity_scale', 1.0)
            self.sdf_scale = normalization_stats.get('sdf', {}).get('scale', 1.0)
            print(f"\n📊 Using computed normalization stats for logging:")
            print(f"   Temperature: [{self.temp_min:.2f}, {self.temp_max:.2f}]°C")
            print(f"   Velocity scale: {self.unified_velocity_scale:.4f}")
            print(f"   SDF scale: {self.sdf_scale:.4f}")
            print(f"   Downsample factor: {self.downsample_factor}")
        else:
            # Fall back to config values (for backward compatibility with old checkpoints)
            self.temp_min = model_cfg.get('temp_min', 55.0)
            self.temp_max = model_cfg.get('temp_max', 120.0)
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
            self.sdf_scale = 1.0
            print(f"\n⚠️  Using config normalization params (no stats provided)")
        
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        
    def forward(self, conditioning):
        """
        Forward pass: conditioning -> prediction
        
        Args:
            conditioning: Input tensor [B, C_cond, H, W]
            
        Returns:
            Predicted output [B, C_out, H, W]
        """
        return self.unet(conditioning)
    
    def denormalize_temperature(self, temperature_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize temperature from [-1, 1] to Celsius."""
        return (temperature_norm + 1.0) / 2.0 * self.temp_range + self.temp_min
    
    def denormalize_velocity(self, velocity_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize velocity from scaled value back to original."""
        return velocity_norm * self.unified_velocity_scale
    
    def _extract_channels(self, tensor: torch.Tensor, channel_indices: list) -> torch.Tensor:
        """Extract specific channels from a tensor based on channel indices."""
        return tensor[:, channel_indices, :, :]
    
    def training_step(self, batch, batch_idx):
        """Training step for frame-to-frame prediction."""
        # Extract data from batch
        # BulkFlow dataset returns: (input_data, output_data)
        input_data, output_data = batch
        
        # input_data: [B, C_in, H, W] (e.g., sdf, velx_interface, vely_interface)
        # output_data: [B, C_out, H, W] (e.g., temperature, velx, vely)
        
        # Extract conditioning and target channels based on task_cfg
        conditioning = self._extract_channels(input_data, self.conditioning_channels)  # [B, num_cond, H, W]
        target = self._extract_channels(output_data, self.target_channels)  # [B, num_target, H, W]
        
        # Predict output from conditioning
        prediction = self.forward(conditioning)
        
        # Compute MSE loss
        loss = self.loss_fn(prediction, target)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step."""
        # Determine validation type
        is_clean_val = (dataloader_idx == 0) if self.has_noise else True
        val_prefix = "val_clean" if is_clean_val else "val_noisy"
        
        # Extract data from batch
        input_data, output_data = batch
        
        # Extract conditioning and target channels based on task_cfg
        conditioning = self._extract_channels(input_data, self.conditioning_channels)
        target = self._extract_channels(output_data, self.target_channels)
        
        # Predict output from conditioning
        prediction = self.forward(conditioning)
        
        # Compute MSE loss
        loss = self.loss_fn(prediction, target)
        
        # Log losses
        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        
        # Log statistics for first batch
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, prediction.shape[0])
                
                # Log normalized statistics
                pred_mean = prediction[:num_samples].mean()
                pred_std = prediction[:num_samples].std()
                target_mean = target[:num_samples].mean()
                target_std = target[:num_samples].std()
                
                self.log(f'{val_prefix}_pred_mean_norm', pred_mean, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_pred_std_norm', pred_std, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_mean_norm', target_mean, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_std_norm', target_std, on_step=False, on_epoch=True, add_dataloader_idx=False)
                
                # Log temperature statistics if applicable
                if self.task_cfg is not None and 'temperature' in self.task_cfg.get('target_names', []):
                    temp_idx = list(self.task_cfg.get('target_names', [])).index('temperature')
                    pred_temp = prediction[:num_samples, temp_idx:temp_idx+1, :, :]
                    target_temp = target[:num_samples, temp_idx:temp_idx+1, :, :]
                    
                    pred_celsius = self.denormalize_temperature(pred_temp)
                    target_celsius = self.denormalize_temperature(target_temp)
                    
                    self.log(f'{val_prefix}_pred_temp_min_C', pred_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_pred_temp_max_C', pred_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_min_C', target_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_max_C', target_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                elif self.task_cfg is None:
                    # Backward compatibility: assume single temperature channel
                    pred_celsius = self.denormalize_temperature(prediction[:num_samples])
                    target_celsius = self.denormalize_temperature(target[:num_samples])
                    
                    self.log(f'{val_prefix}_pred_temp_min_C', pred_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_pred_temp_max_C', pred_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_min_C', target_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_max_C', target_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        return loss
    
    def configure_optimizers(self):
        # Configure optimizer
        if self.optim_cfg.name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-3),
                weight_decay=self.optim_cfg.get('weight_decay', 0.0)
            )
        elif self.optim_cfg.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-3),
                weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
            )
        elif self.optim_cfg.name.lower() == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-4),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
                )
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-3),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_cfg.name}")
        
        # Configure scheduler
        if self.scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.scheduler_cfg.name.lower() == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_cfg.get('T_0', 10),
                T_mult=self.scheduler_cfg.get('T_mult', 2),
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
