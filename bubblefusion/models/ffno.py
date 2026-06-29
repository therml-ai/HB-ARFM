"""
Factorized Fourier Neural Operator (FFNO) for BubbleFlow Prediction.

This module implements a Factorized FNO architecture for predicting physical fields from 
conditioning inputs - a direct regression approach (no diffusion).

Supports multiple tasks through task_cfg configuration:
- temperature_from_sdf: Predict temperature from SDF (Task 1)
- velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)

This is a FRAME-TO-FRAME model (non-autoregressive):
    Input:  conditioning
    Target: output fields

Reference:
    Factorized FNO implementation adapted from:
    https://github.com/HPCForge/BubbleML/blob/main/sciml/models/factorized_fno/factorized_fno.py
    Original: https://github.com/alasdairtran/fourierflow/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional
from einops import rearrange


# ==============================================================================
# Helper Modules for FFNO
# ==============================================================================

class WNLinear(nn.Module):
    """
    Weight-Normalized Linear layer with optional weight normalization.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        wnorm: Whether to use weight normalization
    """
    def __init__(self, in_features: int, out_features: int, wnorm: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if wnorm:
            self.linear = nn.utils.weight_norm(self.linear)
    
    def forward(self, x):
        return self.linear(x)


class FeedForward(nn.Module):
    """
    Feedforward network with optional weight normalization and layer normalization.
    
    Args:
        dim: Input/output dimension
        factor: Expansion factor for hidden dimension
        ff_weight_norm: Whether to use weight normalization
        n_layers: Number of linear layers
        layer_norm: Whether to use layer normalization
        dropout: Dropout probability
    """
    def __init__(self, dim: int, factor: int = 2, ff_weight_norm: bool = False,
                 n_layers: int = 2, layer_norm: bool = False, dropout: float = 0.0):
        super().__init__()
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        
        layers = []
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            layers.append(WNLinear(in_dim, out_dim, wnorm=ff_weight_norm))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        
        if layer_norm:
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        out = self.layers(x)
        if self.layer_norm:
            out = self.norm(out)
        return out


class SpectralConv2d(nn.Module):
    """
    Spectral convolution layer using Fourier transform.
    
    Applies learnable transformations in Fourier space, keeping only n_modes modes.
    
    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels
        n_modes: Number of Fourier modes to keep
        forecast_ff: Pre-built forecast feedforward network (for weight sharing)
        backcast_ff: Pre-built backcast feedforward network (for weight sharing)
        fourier_weight: Pre-built Fourier weights (for weight sharing)
        factor: Feedforward expansion factor
        ff_weight_norm: Whether to use weight normalization in feedforward
        n_ff_layers: Number of layers in feedforward network
        layer_norm: Whether to use layer normalization
        use_fork: Whether to use fork architecture (forecast + backcast)
        dropout: Dropout probability
        mode: Fourier mode ('full' for learned transform, 'low-pass' for simple filter)
    """
    def __init__(self, in_dim: int, out_dim: int, n_modes: int,
                 forecast_ff=None, backcast_ff=None, fourier_weight=None,
                 factor: int = 2, ff_weight_norm: bool = False,
                 n_ff_layers: int = 2, layer_norm: bool = False,
                 use_fork: bool = False, dropout: float = 0.0, mode: str = 'full'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        # Initialize Fourier weights if not provided
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, H, W, in_dim] (channels last!)
            
        Returns:
            Tuple of (backcast, forecast) tensors
        """
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        """Apply Fourier transform and spectral convolution."""
        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, H, W]

        B, I, M, N = x.shape

        # Dimension Y (last dimension)
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)

        if self.mode == 'full':
            out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')

        # Dimension X (second-to-last dimension)
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)

        if self.mode == 'full':
            out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        # Combine dimensions
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        return x


class FNOFactorized2DBlock(nn.Module):
    """
    Factorized Fourier Neural Operator 2D Block.
    
    Multi-layer spectral convolution block with optional forking.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes: Number of Fourier modes to keep
        width: Internal width (hidden dimension)
        dropout: Dropout probability
        in_dropout: Input dropout probability
        n_layers: Number of spectral layers
        share_weight: Whether to share Fourier weights across layers
        share_fork: Whether to share fork networks across layers
        factor: Feedforward expansion factor
        ff_weight_norm: Whether to use weight normalization
        n_ff_layers: Number of feedforward layers
        gain: Initialization gain for Fourier weights
        layer_norm: Whether to use layer normalization
        use_fork: Whether to use fork architecture
        mode: Fourier mode ('full' or 'low-pass')
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int, width: int,
                 dropout: float = 0.0, in_dropout: float = 0.0, n_layers: int = 4,
                 share_weight: bool = False, share_fork: bool = False, factor: int = 2,
                 ff_weight_norm: bool = False, n_ff_layers: int = 2, gain: float = 1.0,
                 layer_norm: bool = False, use_fork: bool = False, mode: str = 'full'):
        super().__init__()
        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.use_fork = use_fork

        # Input projection (channels last format)
        self.in_proj = WNLinear(in_channels, width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)

        # Shared feedforward networks (optional)
        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            if use_fork:
                self.forecast_ff = FeedForward(
                    width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
            self.backcast_ff = FeedForward(
                width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        # Shared Fourier weights (optional)
        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(width, width, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param, gain=gain)
                self.fourier_weight.append(param)

        # Build spectral layers
        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(
                in_dim=width,
                out_dim=width,
                n_modes=modes,
                forecast_ff=self.forecast_ff,
                backcast_ff=self.backcast_ff,
                fourier_weight=self.fourier_weight,
                factor=factor,
                ff_weight_norm=ff_weight_norm,
                n_ff_layers=n_ff_layers,
                layer_norm=layer_norm,
                use_fork=use_fork,
                dropout=dropout,
                mode=mode))

        # Output projection
        self.out = nn.Sequential(
            WNLinear(width, 128, wnorm=ff_weight_norm),
            WNLinear(128, out_channels, wnorm=ff_weight_norm))

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C_in, H, W] (channels first, standard PyTorch format)
            
        Returns:
            Output tensor [B, C_out, H, W]
        """
        # Convert to channels-last format
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C_in]

        forecast = 0
        x = self.in_proj(x)
        x = self.drop(x)
        
        forecast_list = []
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, f = layer(x)

            if self.use_fork:
                f_out = self.out(f)
                forecast = forecast + f_out
                forecast_list.append(f_out)

            x = x + b

        if not self.use_fork:
            forecast = self.out(b)

        # Convert back to channels-first format
        forecast = forecast.permute(0, 3, 1, 2)  # [B, C_out, H, W]

        return forecast


# ==============================================================================
# Main FFNO Model
# ==============================================================================

class FFNO2d(nn.Module):
    """
    Factorized Fourier Neural Operator for 2D field prediction.
    
    This wraps FNOFactorized2DBlock to provide a simple interface
    matching the UNet architecture.
    
    Args:
        in_channels: Number of input channels (conditioning)
        out_channels: Number of output channels (targets)
        modes: Number of Fourier modes to keep (default: 12)
        width: Internal channel width (default: 64)
        n_layers: Number of spectral layers (default: 4)
        dropout: Dropout probability (default: 0.0)
        use_fork: Whether to use fork architecture (default: False)
        mode: Fourier mode ('full' or 'low-pass', default: 'full')
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 modes: int = 12, width: int = 64, n_layers: int = 4,
                 dropout: float = 0.0, use_fork: bool = False, mode: str = 'full'):
        super().__init__()
        
        self.fno_block = FNOFactorized2DBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            n_layers=n_layers,
            dropout=dropout,
            use_fork=use_fork,
            mode=mode,
            share_weight=True,  # Share Fourier weights for efficiency
            share_fork=True,    # Share feedforward networks
            ff_weight_norm=True,  # Weight normalization for stability
            n_ff_layers=2,
            factor=2,
            layer_norm=True
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C_in, H, W]
            
        Returns:
            Output tensor [B, C_out, H, W]
        """
        return self.fno_block(x)


# ==============================================================================
# PyTorch Lightning Module
# ==============================================================================

class FFNOLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for Factorized FNO prediction model.
    
    This is a FRAME-TO-FRAME model (non-autoregressive):
    - Input: conditioning
    - Target: output fields
    
    Supports multiple tasks through task_cfg configuration:
    - temperature_from_sdf: Predict temperature from SDF (Task 1)
    - velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)
    
    Args:
        model_cfg: Model configuration containing FFNO parameters
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
        
        print(f"\n🔧 FFNO Configuration:")
        print(f"   FFNO in_channels: {in_channels} (conditioning)")
        print(f"   FFNO out_channels: {out_channels} (targets)")
        print(f"   Fourier modes: {model_cfg.get('modes', 12)}")
        print(f"   Width: {model_cfg.get('width', 64)}")
        print(f"   Layers: {model_cfg.get('n_layers', 4)}")
        
        # Initialize FFNO model
        self.ffno = FFNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=model_cfg.get('modes', 12),
            width=model_cfg.get('width', 64),
            n_layers=model_cfg.get('n_layers', 4),
            dropout=model_cfg.get('dropout', 0.0),
            use_fork=model_cfg.get('use_fork', False),
            mode=model_cfg.get('fourier_mode', 'full')
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
        return self.ffno(conditioning)
    
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
