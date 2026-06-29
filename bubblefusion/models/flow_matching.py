"""
Optimal Transport Conditional Flow Matching for BubbleFlow Prediction.

This module implements an Optimal Transport Conditional Flow Matching (OT-CFM) 
architecture for predicting physical fields from conditioning inputs.

Supports multiple tasks through task_cfg configuration:
- temperature_from_sdf: Predict temperature from SDF (Task 1)
- velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)

This is a FRAME-TO-FRAME model (non-autoregressive):
    Input:  [current_state_noisy, conditioning]  
    Target: current_output

Flow Matching is a generative model that learns continuous normalizing flows
from noise to data. OT-CFM uses optimal transport paths for efficient training.

Improvements from DiffusionPDE/EDM that are applicable to flow matching:
- Bottleneck-only attention (attention_type='bottleneck') - global context, efficient
- Adaptive scale time conditioning (adaptive_scale=True) - FiLM-style modulation
- Skip connection scaling (skip_scale=True) - better gradient flow

References:
    - "Flow Matching for Generative Modeling" (Lipman et al., 2023)
    - "Improving and generalizing flow-based generative models with minibatch 
       optimal transport" (Tong et al., 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple, Literal
import numpy as np

from bubblefusion.models.loss_utils import build_loss_fn


class SinusoidalTimeEmbedding(nn.Module):
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


class TimeEmbedding(nn.Module):
    """MLP for processing timestep embeddings."""
    
    def __init__(self, time_embed_dim: int = 320):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, 4 * time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * time_embed_dim, 4 * time_embed_dim),
        )
        
    def forward(self, time):
        time_emb = self.time_embed(time)
        time_emb = self.time_mlp(time_emb)
        return time_emb


def get_groups(channels: int) -> int:
    """Get appropriate group size for GroupNorm that divides evenly."""
    if channels >= 32 and channels % 32 == 0:
        return 32
    elif channels >= 16 and channels % 16 == 0:
        return 16
    elif channels >= 8 and channels % 8 == 0:
        return 8
    elif channels >= 4 and channels % 4 == 0:
        return 4
    else:
        return 1


class ResidualBlock(nn.Module):
    """
    ResNet-style block with timestep conditioning.
    
    Supports two time conditioning modes:
    - Additive: h = h + time_emb (simple, original flow matching)
    - Adaptive scale (FiLM): h = scale * norm(h) + shift (may help with multi-scale PDEs)
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_embed_dim: Dimension of time embedding
        dropout: Dropout probability
        adaptive_scale: If True, use FiLM-style scale+shift conditioning
        skip_scale: If True, scale residual by 1/sqrt(2) for better gradient flow
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int = 1280, 
                 dropout: float = 0.0, adaptive_scale: bool = False, skip_scale: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adaptive_scale = adaptive_scale
        self.skip_scale_factor = 1.0 / np.sqrt(2) if skip_scale else 1.0
        
        groups = get_groups(in_channels)
        out_groups = get_groups(out_channels)
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Adaptive scale uses 2x output channels (scale + shift)
        time_out_dim = out_channels * 2 if adaptive_scale else out_channels
        self.time_emb_proj = nn.Linear(time_embed_dim, time_out_dim)
        
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
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
        
        # Time conditioning
        time_emb = F.silu(time_emb)
        time_proj = self.time_emb_proj(time_emb)
        
        if self.adaptive_scale:
            # FiLM-style: scale and shift
            scale, shift = time_proj.chunk(2, dim=1)
            h = self.norm2(h)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = F.silu(h)
        else:
            # Simple additive conditioning
            h = h + time_proj[:, :, None, None]
            h = self.norm2(h)
            h = F.silu(h)
        
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection with optional scaling
        out = h + self.shortcut(x)
        return out * self.skip_scale_factor


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features with FP32 stability."""
    
    def __init__(self, channels: int, num_heads: int = 8, skip_scale: bool = False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.skip_scale_factor = 1.0 / np.sqrt(2) if skip_scale else 1.0
        
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        groups = get_groups(channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(dim=1)
        
        # Compute attention in FP32 for numerical stability
        q_fp32 = q.float()
        k_fp32 = k.float()
        attn = torch.einsum('bhdn,bhdm->bhnm', q_fp32, k_fp32) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1).to(q.dtype)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return (x + out) * self.skip_scale_factor


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class FlowMatchingUNet(nn.Module):
    """
    UNet architecture for flow matching velocity field prediction.
    
    Includes improvements that are applicable to flow matching:
    - Bottleneck-only attention for efficiency and global context
    - Adaptive scale time conditioning (FiLM-style)
    - Skip connection scaling for better gradient flow
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels for the first layer
        time_embed_dim: Dimension of timestep embeddings
        num_res_blocks: Number of residual blocks per level
        attention_type: 'none' or 'bottleneck' (attention at lowest resolution only)
        dropout: Dropout probability
        adaptive_scale: Use FiLM-style time conditioning
        skip_scale: Scale skip connections by 1/sqrt(2)
    """
    
    def __init__(self, in_channels: int = 2, out_channels: int = 1, 
                 base_channels: int = 64, time_embed_dim: int = 320,
                 num_res_blocks: int = 2,
                 attention_type: Literal['none', 'bottleneck'] = 'none',
                 dropout: float = 0.0,
                 adaptive_scale: bool = False,
                 skip_scale: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.attention_type = attention_type
        
        # Timestep embedding (time_embed_dim -> 4 * time_embed_dim)
        self.time_embedding = TimeEmbedding(time_embed_dim)
        time_emb_dim = 4 * time_embed_dim
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Define channel progression for simple 3-level UNet
        self.ch1 = base_channels      # e.g., 64
        self.ch2 = base_channels * 2  # e.g., 128
        self.ch3 = base_channels * 4  # e.g., 256
        
        # Common kwargs for residual blocks
        res_kwargs = dict(dropout=dropout, adaptive_scale=adaptive_scale, skip_scale=skip_scale)
        attn_kwargs = dict(skip_scale=skip_scale)
        
        # Encoder Level 1 (full resolution)
        self.enc1_res1 = ResidualBlock(self.ch1, self.ch1, time_emb_dim, **res_kwargs)
        self.enc1_res2 = ResidualBlock(self.ch1, self.ch1, time_emb_dim, **res_kwargs)
        self.enc1_down = Downsample(self.ch1)
        
        # Encoder Level 2 (half resolution)
        self.enc2_res1 = ResidualBlock(self.ch1, self.ch2, time_emb_dim, **res_kwargs)
        self.enc2_res2 = ResidualBlock(self.ch2, self.ch2, time_emb_dim, **res_kwargs)
        self.enc2_down = Downsample(self.ch2)
        
        # Encoder Level 3 (quarter resolution)
        self.enc3_res1 = ResidualBlock(self.ch2, self.ch3, time_emb_dim, **res_kwargs)
        self.enc3_res2 = ResidualBlock(self.ch3, self.ch3, time_emb_dim, **res_kwargs)
        
        # Middle (bottleneck) - attention here if enabled
        self.mid_res1 = ResidualBlock(self.ch3, self.ch3, time_emb_dim, **res_kwargs)
        self.mid_attn = AttentionBlock(self.ch3, **attn_kwargs) if attention_type == 'bottleneck' else nn.Identity()
        self.mid_res2 = ResidualBlock(self.ch3, self.ch3, time_emb_dim, **res_kwargs)
        
        # Decoder Level 3
        self.dec3_res1 = ResidualBlock(self.ch3 + self.ch3, self.ch3, time_emb_dim, **res_kwargs)
        self.dec3_res2 = ResidualBlock(self.ch3, self.ch3, time_emb_dim, **res_kwargs)
        self.dec3_up = Upsample(self.ch3)
        
        # Decoder Level 2
        self.dec2_res1 = ResidualBlock(self.ch3 + self.ch2, self.ch2, time_emb_dim, **res_kwargs)
        self.dec2_res2 = ResidualBlock(self.ch2, self.ch2, time_emb_dim, **res_kwargs)
        self.dec2_up = Upsample(self.ch2)
        
        # Decoder Level 1
        self.dec1_res1 = ResidualBlock(self.ch2 + self.ch1, self.ch1, time_emb_dim, **res_kwargs)
        self.dec1_res2 = ResidualBlock(self.ch1, self.ch1, time_emb_dim, **res_kwargs)
        
        # Final output
        groups = get_groups(base_channels)
        self.norm_out = nn.GroupNorm(groups, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, x, time):
        # Get time embeddings
        time_emb = self.time_embedding(time)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Encoder Level 1
        enc1 = self.enc1_res1(x, time_emb)
        enc1 = self.enc1_res2(enc1, time_emb)
        x = self.enc1_down(enc1)
        
        # Encoder Level 2
        x = self.enc2_res1(x, time_emb)
        enc2 = self.enc2_res2(x, time_emb)
        x = self.enc2_down(enc2)
        
        # Encoder Level 3
        x = self.enc3_res1(x, time_emb)
        enc3 = self.enc3_res2(x, time_emb)
        
        # Middle (bottleneck with attention)
        x = self.mid_res1(enc3, time_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, time_emb)
        
        # Decoder Level 3
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3_res1(x, time_emb)
        x = self.dec3_res2(x, time_emb)
        x = self.dec3_up(x)
        
        # Decoder Level 2
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2_res1(x, time_emb)
        x = self.dec2_res2(x, time_emb)
        x = self.dec2_up(x)
        
        # Decoder Level 1
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1_res1(x, time_emb)
        x = self.dec1_res2(x, time_emb)
        
        # Final output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x


class FlowMatchingSampler:
    """
    Flow Matching Sampler for inference using ODE integration.
    
    This class is kept for backward compatibility with flow_matching_ar.py.
    """
    
    def __init__(self, num_integration_steps: int = 50, solver: str = 'euler'):
        """
        Args:
            num_integration_steps: Number of steps for ODE integration
            solver: ODE solver - 'euler', 'heun', 'rk4', or 'midpoint'
        """
        self.num_integration_steps = num_integration_steps
        self.dt = 1.0 / num_integration_steps
        self.solver = solver
    
    def sample(self, model: nn.Module, condition: torch.Tensor, 
               shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Generate samples using flow matching ODE integration.
        
        Args:
            model: The flow matching model (must have forward(x, condition, t))
            condition: Conditioning inputs [B, C_cond, H, W]
            shape: Shape of the output (B, C_out, H, W)
            device: Device to generate on
            
        Returns:
            Generated samples [B, C_out, H, W]
        """
        x = torch.randn(shape, device=device)
        dt = self.dt
        
        for step in range(self.num_integration_steps):
            t = step * dt
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.float32)
            
            if self.solver == 'euler':
                with torch.no_grad():
                    velocity = model(x, condition, t_tensor)
                x = x + velocity * dt
                
            elif self.solver == 'heun':
                with torch.no_grad():
                    v1 = model(x, condition, t_tensor)
                    x_pred = x + v1 * dt
                    t_next = torch.full((shape[0],), min(t + dt, 1.0), device=device, dtype=torch.float32)
                    v2 = model(x_pred, condition, t_next)
                x = x + 0.5 * (v1 + v2) * dt
                
            elif self.solver == 'midpoint':
                with torch.no_grad():
                    v1 = model(x, condition, t_tensor)
                    x_mid = x + v1 * (dt / 2)
                    t_mid = torch.full((shape[0],), t + dt/2, device=device, dtype=torch.float32)
                    v_mid = model(x_mid, condition, t_mid)
                x = x + v_mid * dt
                
            elif self.solver == 'rk4':
                with torch.no_grad():
                    t_half = torch.full((shape[0],), t + dt/2, device=device, dtype=torch.float32)
                    t_next = torch.full((shape[0],), min(t + dt, 1.0), device=device, dtype=torch.float32)
                    k1 = model(x, condition, t_tensor)
                    k2 = model(x + k1 * dt/2, condition, t_half)
                    k3 = model(x + k2 * dt/2, condition, t_half)
                    k4 = model(x + k3 * dt, condition, t_next)
                x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)
            else:
                raise ValueError(f"Unknown solver: {self.solver}")
        
        return x


class ConditionalFlowMatching(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) model.
    
    Flow matching learns a continuous flow from noise to data:
        dx/dt = v_θ(x(t), t, condition)
    
    where v_θ is the velocity field predicted by the UNet.
    
    OT-CFM uses optimal transport paths for training:
        x(t) = t * x_1 + (1-t) * x_0
        velocity_target = x_1 - x_0
    
    Args:
        unet: The UNet velocity field predictor
    """
    
    def __init__(self, unet: FlowMatchingUNet):
        super().__init__()
        self.unet = unet
        
    def compute_conditional_flow(self, x_0: torch.Tensor, x_1: torch.Tensor, 
                                  t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow path using optimal transport (linear interpolation).
        
        Args:
            x_0: Source samples (noise) [B, C, H, W]
            x_1: Target samples (clean data) [B, C, H, W]
            t: Time values in [0, 1] [B]
            
        Returns:
            x_t: Interpolated samples [B, C, H, W]
            velocity_target: Target velocity field [B, C, H, W]
        """
        t = t.view(-1, 1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        velocity_target = x_1 - x_0
        return x_t, velocity_target
    
    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, 
                t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field given current state and conditioning.
        
        Args:
            x_t: Current state (interpolated) [B, C_out, H, W]
            condition: Conditioning [B, C_cond, H, W]
            t: Time values in [0, 1] [B]
            
        Returns:
            Predicted velocity field [B, C_out, H, W]
        """
        x_input = torch.cat([x_t, condition], dim=1)
        return self.unet(x_input, t)
    
    @torch.no_grad()
    def sample(self, condition: torch.Tensor, shape: Tuple[int, ...],
               device: torch.device, num_integration_steps: int = 50,
               solver: str = 'euler') -> torch.Tensor:
        """
        Generate samples using ODE integration.
        
        Args:
            condition: Conditioning inputs [B, C_cond, H, W]
            shape: Shape of the output (B, C_out, H, W)
            device: Device to generate on
            num_integration_steps: Number of integration steps
            solver: ODE solver - 'euler', 'heun', 'rk4', or 'midpoint'
            
        Returns:
            Generated samples [B, C_out, H, W]
        """
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_integration_steps
        
        for step in range(num_integration_steps):
            t = step * dt
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.float32)
            
            if solver == 'euler':
                velocity = self.forward(x, condition, t_tensor)
                x = x + velocity * dt
                
            elif solver == 'heun':
                v1 = self.forward(x, condition, t_tensor)
                x_pred = x + v1 * dt
                t_next = torch.full((shape[0],), min(t + dt, 1.0), device=device, dtype=torch.float32)
                v2 = self.forward(x_pred, condition, t_next)
                x = x + 0.5 * (v1 + v2) * dt
                
            elif solver == 'midpoint':
                v1 = self.forward(x, condition, t_tensor)
                x_mid = x + v1 * (dt / 2)
                t_mid = torch.full((shape[0],), t + dt/2, device=device, dtype=torch.float32)
                v_mid = self.forward(x_mid, condition, t_mid)
                x = x + v_mid * dt
                
            elif solver == 'rk4':
                t_half = torch.full((shape[0],), t + dt/2, device=device, dtype=torch.float32)
                t_next = torch.full((shape[0],), min(t + dt, 1.0), device=device, dtype=torch.float32)
                k1 = self.forward(x, condition, t_tensor)
                k2 = self.forward(x + k1 * dt/2, condition, t_half)
                k3 = self.forward(x + k2 * dt/2, condition, t_half)
                k4 = self.forward(x + k3 * dt, condition, t_next)
                x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)
            else:
                raise ValueError(f"Unknown solver: {solver}")
        
        return x


class ConditionalFlowMatchingLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for Conditional Flow Matching model.
    
    This is a FRAME-TO-FRAME model (non-autoregressive):
    - Input: [current_state_noisy, conditioning]
    - Target: current_output
    
    Supports multiple tasks through task_cfg configuration.
    
    Improvements that are applicable to flow matching (toggleable via model_cfg):
    - attention_type: 'none' or 'bottleneck' (default: 'none')
    - adaptive_scale: True/False (FiLM-style time conditioning)
    - skip_scale: True/False (skip connection scaling for gradient flow)
    
    Args:
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        scheduler_cfg: Learning rate scheduler configuration
        task_cfg: Task configuration defining which channels to use
        normalization_stats: Pre-computed normalization statistics
    """
    
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig, scheduler_cfg: DictConfig,
                 task_cfg: Optional[DictConfig] = None, normalization_stats: Optional[dict] = None,
                 norm_mode: str = 'all'):
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])
        
        # Store normalization statistics
        self.normalization_stats = normalization_stats
        self.norm_mode = norm_mode
        self.task_cfg = task_cfg
        
        # Parse task config for channel indices
        if task_cfg is not None:
            self.conditioning_channels = list(task_cfg.get('conditioning_channels', [0]))
            self.target_channels = list(task_cfg.get('target_channels', [0]))
            self.task_name = task_cfg.get('name', 'unknown')
            self.is_noisy_task = 'noisy' in self.task_name.lower()
            noise_cfg = task_cfg.get('noise_cfg', None)
            self.has_noise = noise_cfg is not None and noise_cfg.get('enabled', True)
            
            print(f"🎯 Task: {self.task_name}")
            print(f"   Conditioning channels: {self.conditioning_channels} ({task_cfg.get('conditioning_names', [])})")
            print(f"   Target channels: {self.target_channels} ({task_cfg.get('target_names', [])})")
            
            if self.has_noise:
                print(f"   🔊 Noise injection: ENABLED")
        else:
            self.conditioning_channels = [0]
            self.target_channels = [0]
            self.task_name = 'temperature_from_sdf'
            self.is_noisy_task = False
            self.has_noise = False
            print("⚠️  No task_cfg provided, defaulting to temperature_from_sdf task")
        
        # Compute derived channel counts
        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)
        
        in_channels = self.num_target_channels + self.num_conditioning_channels
        out_channels = self.num_target_channels
        
        # =============================================================================
        # ARCHITECTURE IMPROVEMENTS (applicable to flow matching)
        # =============================================================================
        # Handle backward compatibility: convert old use_attention bool to attention_type
        use_attention_old = model_cfg.get('use_attention', False)
        attention_type = model_cfg.get('attention_type', 'bottleneck' if use_attention_old else 'none')
        
        adaptive_scale = model_cfg.get('adaptive_scale', False)
        skip_scale = model_cfg.get('skip_scale', False)
        
        print(f"\n🔄 Flow Matching Configuration:")
        print(f"   UNet in_channels: {in_channels} = {self.num_target_channels} (x_t) + {self.num_conditioning_channels} (cond)")
        print(f"   UNet out_channels: {out_channels}")
        print(f"\n🔧 Architecture Options:")
        print(f"   attention_type: {attention_type}")
        print(f"   adaptive_scale: {adaptive_scale}")
        print(f"   skip_scale: {skip_scale}")
        
        # Initialize UNet with improvements
        unet = FlowMatchingUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=model_cfg.get('base_channels', 64),
            time_embed_dim=model_cfg.get('time_embed_dim', 320),
            num_res_blocks=model_cfg.get('num_res_blocks', 2),
            attention_type=attention_type,
            dropout=model_cfg.get('dropout', 0.0),
            adaptive_scale=adaptive_scale,
            skip_scale=skip_scale,
        )
        
        self.flow_matching = ConditionalFlowMatching(unet=unet)
        
        self.loss_fn = build_loss_fn(model_cfg)
        
        # =============================================================================
        # NORMALIZATION CONFIGURATION
        # =============================================================================
        self.downsample_factor = 1
        if normalization_stats is not None:
            self.downsample_factor = normalization_stats.get('downsample_factor', 1)
        
        if normalization_stats is not None and 'temperature' in normalization_stats:
            temp_stats = normalization_stats['temperature']
            self.temp_min = temp_stats['min']
            self.temp_max = temp_stats['max']
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = normalization_stats.get('unified_velocity_scale', 1.0)
            self.sdf_scale = normalization_stats.get('sdf', {}).get('scale', 1.0)
            print(f"\n📊 Using computed normalization stats:")
            print(f"   Temperature: [{self.temp_min:.2f}, {self.temp_max:.2f}]°C")
            print(f"   Velocity scale: {self.unified_velocity_scale:.4f}")
        else:
            self.temp_min = model_cfg.get('temp_min', 55.0)
            self.temp_max = model_cfg.get('temp_max', 120.0)
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
            self.sdf_scale = 1.0
            print(f"\n⚠️  Using config normalization params")
        
        # Flow matching parameters
        self.num_integration_steps = model_cfg.get('num_integration_steps', 50)
        
        # Inference configuration
        inference_cfg = model_cfg.get('inference', {})
        self.default_solver = inference_cfg.get('solver', 'heun')
        self.default_guidance_scale = inference_cfg.get('guidance_scale', 1.0)
        print(f"\n🔧 Inference Settings:")
        print(f"   Solver: {self.default_solver}")
        print(f"   Integration steps: {self.num_integration_steps}")
        
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        
    def forward(self, x_t, condition, t):
        return self.flow_matching(x_t, condition, t)
    
    def denormalize_temperature(self, temperature_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize temperature from [-1, 1] to Celsius."""
        if self.norm_mode == 'none':
            return temperature_norm
        return (temperature_norm + 1.0) / 2.0 * self.temp_range + self.temp_min
    
    def denormalize_velocity(self, velocity_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize velocity from scaled value back to original."""
        if self.norm_mode in ('none', 'temperature_only'):
            return velocity_norm
        return velocity_norm * self.unified_velocity_scale
    
    def _extract_channels(self, tensor: torch.Tensor, channel_indices: list) -> torch.Tensor:
        """Extract specific channels from a tensor based on channel indices."""
        return tensor[:, channel_indices, :, :]
    
    def training_step(self, batch, batch_idx):
        """Training step with uniform time sampling and loss weighting (theoretically correct)."""
        input_data, output_data = batch
        
        conditioning = self._extract_channels(input_data, self.conditioning_channels)
        target = self._extract_channels(output_data, self.target_channels)
        
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Uniform time sampling (correct for flow matching)
        t = torch.rand(batch_size, device=device)
        
        # Sample noise (source distribution)
        x_0 = torch.randn_like(target)
        
        # Compute conditional flow path and target velocity
        x_t, velocity_target = self.flow_matching.compute_conditional_flow(x_0, target, t)
        
        # Predict velocity field
        velocity_pred = self.flow_matching(x_t, conditioning, t)
        
        # Uniform MSE loss (correct for flow matching)
        loss = self.loss_fn(velocity_pred, velocity_target)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step."""
        is_clean_val = (dataloader_idx == 0) if self.has_noise else True
        val_prefix = "val_clean" if is_clean_val else "val_noisy"
        
        input_data, output_data = batch
        
        conditioning = self._extract_channels(input_data, self.conditioning_channels)
        target = self._extract_channels(output_data, self.target_channels)
        
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        t = torch.rand(batch_size, device=device)
        x_0 = torch.randn_like(target)
        x_t, velocity_target = self.flow_matching.compute_conditional_flow(x_0, target, t)
        velocity_pred = self.flow_matching(x_t, conditioning, t)
        
        loss = self.loss_fn(velocity_pred, velocity_target)
        
        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        
        # Generate samples and log statistics
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, batch_size)
                
                samples = self.flow_matching.sample(
                    conditioning[:num_samples],
                    (num_samples, self.num_target_channels, target.shape[2], target.shape[3]),
                    device,
                    num_integration_steps=self.num_integration_steps,
                    solver=self.default_solver
                )
                
                sample_mean = samples.mean()
                sample_std = samples.std()
                target_mean = target[:num_samples].mean()
                target_std = target[:num_samples].std()
                
                self.log(f'{val_prefix}_sample_mean_norm', sample_mean, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_sample_std_norm', sample_std, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_mean_norm', target_mean, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_std_norm', target_std, on_step=False, on_epoch=True, add_dataloader_idx=False)
                
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
        
        return loss
    
    def configure_optimizers(self):
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
        
        if self.scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
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
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
            }
        else:
            return optimizer
