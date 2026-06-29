"""
Conditional Diffusion Model (DDPM) for BubbleFlow dataset.

This module implements a conditional DDPM architecture for predicting bulk liquid velocity 
and temperature from bubble position and interface velocity.

Classes:
    SinusoidalPositionalEmbedding: Sinusoidal timestep embeddings
    TimestepEmbedding: MLP for processing timestep embeddings
    ResidualBlock: ResNet-style block with timestep conditioning
    ConditionalUNet: UNet architecture with timestep embeddings for DDPM
    BubbleDDPM: Main DDPM model with forward/reverse processes
    BubbleDDPMLightning: PyTorch Lightning wrapper for DDPM

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple
import numpy as np


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimestepEmbedding(nn.Module):
    """MLP for processing timestep embeddings."""
    
    def __init__(self, time_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        
        self.time_embed = SinusoidalPositionalEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, time):
        time_emb = self.time_embed(time)
        time_emb = self.time_mlp(time_emb)
        return time_emb


class ResidualBlock(nn.Module):
    """ResNet-style block with timestep conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, 
                 dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for the UNet."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(dim=1)
        
        # Compute attention
        # q, k, v shape: (b, num_heads, head_dim, h*w)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return x + out


class ConditionalUNet(nn.Module):
    """
    UNet architecture with timestep embeddings for DDPM.
    
    Args:
        in_channels: Number of input channels (6: noisy_target + conditioning)
        out_channels: Number of output channels (3: noise prediction)
        base_channels: Base number of channels for the first layer
        time_embed_dim: Dimension of timestep embeddings
        num_res_blocks: Number of residual blocks per level
        use_attention: Whether to use attention blocks
        dropout: Dropout probability
    """
    
    def __init__(self, in_channels: int = 6, out_channels: int = 3, 
                 base_channels: int = 64, time_embed_dim: int = 256,
                 num_res_blocks: int = 2, use_attention: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_embed_dim = time_embed_dim
        
        # Timestep embedding
        self.time_embedding = TimestepEmbedding(time_embed_dim, time_embed_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, in_ch, time_embed_dim, dropout))
                in_ch = in_ch
            
            # Add attention at middle resolution
            if use_attention and i == 1:
                blocks.append(AttentionBlock(in_ch))
            
            blocks.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
            self.down_blocks.append(blocks)
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(channels[-1], channels[-1], time_embed_dim, dropout),
            AttentionBlock(channels[-1]) if use_attention else nn.Identity(),
            ResidualBlock(channels[-1], channels[-1], time_embed_dim, dropout),
        ])
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        channels.reverse()
        
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            blocks = nn.ModuleList()
            blocks.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            
            # First block handles concatenated channels (from skip connection)
            first_block_in_ch = out_ch * 2  # After concatenation with skip
            blocks.append(ResidualBlock(first_block_in_ch, out_ch, time_embed_dim, dropout))
            
            # Subsequent blocks use normal channel count
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(out_ch, out_ch, time_embed_dim, dropout))
            
            # Add attention at middle resolution
            if use_attention and i == len(channels) - 3:
                blocks.append(AttentionBlock(out_ch))
            
            self.up_blocks.append(blocks)
        
        # Final output
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, x, time):
        # Get time embeddings
        time_emb = self.time_embedding(time)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Store skip connections
        skip_connections = [x]
        
        # Encoder
        for blocks in self.down_blocks:
            for block in blocks[:-1]:  # All except downsampling
                if isinstance(block, ResidualBlock):
                    x = block(x, time_emb)
                else:
                    x = block(x)
            skip_connections.append(x)
            x = blocks[-1](x)  # Downsampling
        
        # Middle
        for block in self.middle_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, time_emb)
            else:
                x = block(x)
        
        # Decoder
        for blocks in self.up_blocks:
            x = blocks[0](x)  # Upsampling
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            
            for block in blocks[1:]:
                if isinstance(block, ResidualBlock):
                    x = block(x, time_emb)
                else:
                    x = block(x)
        
        # Final output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x


class BubbleDDPM(nn.Module):
    """
    DDPM model for bubble flow prediction.
    
    Args:
        unet: The UNet denoiser model
        num_timesteps: Number of diffusion timesteps
        beta_start: Start value for noise schedule
        beta_end: End value for noise schedule
    """
    
    def __init__(self, unet: ConditionalUNet, num_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        
        self.unet = unet
        self.num_timesteps = num_timesteps
        
        # Create noise schedule
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_timesteps))
        
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', 
                           F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Precompute useful quantities
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise to x_start.
        
        Args:
            x_start: Clean target data [B, 3, H, W]
            t: Timesteps [B]
            noise: Random noise (optional)
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(self, x_noisy: torch.Tensor, condition: torch.Tensor, 
                t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise given noisy input and conditioning.
        
        Args:
            x_noisy: Noisy target data [B, 3, H, W]
            condition: Conditioning input [B, 3, H, W]
            t: Timesteps [B]
            
        Returns:
            Predicted noise [B, 3, H, W]
        """
        # Concatenate noisy target with conditioning
        x_input = torch.cat([x_noisy, condition], dim=1)  # [B, 6, H, W]
        
        # Predict noise
        return self.unet(x_input, t)
    
    @torch.no_grad()
    def p_sample_loop(self, condition: torch.Tensor, shape: Tuple[int, ...],
                      device: torch.device) -> torch.Tensor:
        """
        Generate samples using reverse diffusion process.
        
        Args:
            condition: Conditioning input [B, 3, H, W]
            shape: Shape of the output (B, C, H, W)
            device: Device to generate on
            
        Returns:
            Generated samples [B, 3, H, W]
        """
        b = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(x, condition, t)
            
            # Compute denoising step
            alpha_t = self.alphas_cumprod[i]
            alpha_t_prev = self.alphas_cumprod_prev[i]
            beta_t = self.betas[i]
            
            # Compute x_{t-1}
            pred_x_start = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            if i > 0:
                # Add noise for non-final step
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t_prev) * pred_x_start + torch.sqrt(1 - alpha_t_prev) * noise
            else:
                x = pred_x_start
        
        return x


class BubbleDDPMLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for Bubble DDPM model.
    
    This is a FRAME-TO-FRAME model (non-autoregressive):
    - Input: [noisy_state, conditioning]
    - Target: current_output
    
    Supports multiple tasks through task_cfg configuration:
    - temperature_from_sdf: Predict temperature from SDF (Task 1)
    - velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)
    
    Args:
        model_cfg: Model configuration containing DDPM parameters
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
        # in_channels = num_target_channels (noisy state) + num_conditioning_channels
        # out_channels = num_target_channels (noise prediction)
        computed_in_channels = self.num_target_channels + self.num_conditioning_channels
        computed_out_channels = self.num_target_channels
        
        in_channels = computed_in_channels
        out_channels = computed_out_channels
        
        print(f"\n🔄 Frame-to-Frame DDPM Configuration:")
        print(f"   UNet in_channels: {in_channels} = {self.num_target_channels} (x_t) + {self.num_conditioning_channels} (cond)")
        print(f"   UNet out_channels: {out_channels}")

        # Initialize DDPM model
        unet = ConditionalUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=model_cfg.get('base_channels', 64),
            time_embed_dim=model_cfg.get('time_embed_dim', 256),
            num_res_blocks=model_cfg.get('num_res_blocks', 2),
            use_attention=model_cfg.get('use_attention', True),
            dropout=model_cfg.get('dropout', 0.0)
        )
        
        self.ddpm = BubbleDDPM(
            unet=unet,
            num_timesteps=model_cfg.get('num_timesteps', 1000),
            beta_start=model_cfg.get('beta_start', 1e-4),
            beta_end=model_cfg.get('beta_end', 2e-2)
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
        
        # DDPM inference parameters
        self.num_inference_steps = model_cfg.get('num_inference_steps', 1000)
        print(f"\n🔧 Default Inference Settings:")
        print(f"   Inference steps: {self.num_inference_steps}")
        
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        
    def forward(self, x_noisy, condition, t):
        return self.ddpm(x_noisy, condition, t)
    
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
        
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Sample random timesteps
        t = torch.randint(0, self.ddpm.num_timesteps, (batch_size,), device=device)
        
        # Add noise to target
        noise = torch.randn_like(target)
        x_noisy = self.ddpm.q_sample(target, t, noise)
        
        # Predict noise
        predicted_noise = self.ddpm(x_noisy, conditioning, t)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, noise)
        
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
        
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Sample random timesteps
        t = torch.randint(0, self.ddpm.num_timesteps, (batch_size,), device=device)
        
        # Add noise to target
        noise = torch.randn_like(target)
        x_noisy = self.ddpm.q_sample(target, t, noise)
        
        # Predict noise
        predicted_noise = self.ddpm(x_noisy, conditioning, t)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, noise)
        
        # Log losses
        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        
        # Generate samples and log statistics
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, batch_size)
                
                # Generate samples using reverse diffusion
                samples = self.ddpm.p_sample_loop(
                    conditioning[:num_samples],
                    (num_samples, self.num_target_channels, target.shape[2], target.shape[3]),
                    device
                )
                
                # Log statistics
                sample_mean = samples.mean()
                sample_std = samples.std()
                target_mean = target[:num_samples].mean()
                target_std = target[:num_samples].std()
                
                self.log(f'{val_prefix}_sample_mean_norm', sample_mean, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_sample_std_norm', sample_std, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_mean_norm', target_mean, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_std_norm', target_std, on_step=False, on_epoch=True, add_dataloader_idx=False)
                
                # Log temperature statistics if applicable
                if self.task_cfg is not None and 'temperature' in self.task_cfg.get('target_names', []):
                    temp_idx = list(self.task_cfg.get('target_names', [])).index('temperature')
                    samples_temp = samples[:, temp_idx:temp_idx+1, :, :]
                    target_temp = target[:num_samples, temp_idx:temp_idx+1, :, :]
                    
                    samples_celsius = self.denormalize_temperature(samples_temp)
                    target_celsius = self.denormalize_temperature(target_temp)
                    
                    self.log(f'{val_prefix}_pred_temp_min_C', samples_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_pred_temp_max_C', samples_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_min_C', target_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_max_C', target_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                elif self.task_cfg is None:
                    # Backward compatibility: assume single temperature channel
                    samples_celsius = self.denormalize_temperature(samples)
                    target_celsius = self.denormalize_temperature(target[:num_samples])
                    
                    self.log(f'{val_prefix}_pred_temp_min_C', samples_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_pred_temp_max_C', samples_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_min_C', target_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_max_C', target_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        return loss
    
    def configure_optimizers(self):
        # Configure optimizer
        if self.optim_cfg.name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-4),
                weight_decay=self.optim_cfg.get('weight_decay', 0.0)
            )
        elif self.optim_cfg.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-4),
                weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
            )
        elif self.optim_cfg.name.lower() == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-5),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
                )
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-4),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2)
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_cfg.name}")
        
        # Configure scheduler
        if self.scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.optim_cfg.get('lr', 1e-4) * 0.01
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
                eta_min=self.optim_cfg.get('lr', 1e-4) * 0.01
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
