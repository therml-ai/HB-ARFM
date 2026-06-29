"""
Variance Exploding Score-Based Model with VE-SDE for BubbleFlow Temperature Prediction.

This module implements a Score-Based Generative Model using Variance Exploding (VE) SDE
for predicting temperature and velocity fields from bubble SDF conditioning.

Task: Given conditioning inputs -> Predict target fields using score-based diffusion
- Task 1: SDF -> Temperature
- Task 2: SDF + Interface Velocity -> Bulk Velocity + Temperature
- Task 3: Noisy SDF + Noisy Interface Velocity -> Bulk Velocity + Temperature
         (Noise augmentation simulates optical flow prediction errors)

Implementation follows the original score_sde repository by Yang Song:
    https://github.com/yang-song/score_sde

Key insight: Use noise prediction (ε-parameterization) for numerical stability.
The network predicts the noise z, and score is computed as: s = -z/σ

VE-SDE forward process:
    x(σ) = x + σ * z,  where z ~ N(0, I)

Training uses Denoising Score Matching (DSM) with noise prediction:
    L = E[||ε_θ(x + σz, σ) - z||²]

References:
    - "Score-Based Generative Modeling through Stochastic Differential Equations" 
      (Song et al., ICLR 2021)
    - Original implementation: https://github.com/yang-song/score_sde
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple
import numpy as np


class SinusoidalSigmaEmbedding(nn.Module):
    """Sinusoidal positional embeddings for noise levels (sigma)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, sigma):
        """
        Args:
            sigma: Noise levels [B] or [B, 1]
            
        Returns:
            Embeddings [B, dim]
        """
        device = sigma.device
        if sigma.dim() > 1:
            sigma = sigma.squeeze(-1)
        
        # Use log-scale for sigma embeddings (standard practice for VE-SDE)
        # Clamp to avoid log(0)
        log_sigma = torch.log(sigma.clamp(min=1e-10))
        
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = log_sigma[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SigmaEmbedding(nn.Module):
    """MLP for processing sigma (noise level) embeddings."""
    
    def __init__(self, sigma_embed_dim: int = 256):
        super().__init__()
        self.sigma_embed_dim = sigma_embed_dim
        
        self.sigma_embed = SinusoidalSigmaEmbedding(sigma_embed_dim)
        self.sigma_mlp = nn.Sequential(
            nn.Linear(sigma_embed_dim, 4 * sigma_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * sigma_embed_dim, 4 * sigma_embed_dim),
        )
        
    def forward(self, sigma):
        sigma_emb = self.sigma_embed(sigma)
        sigma_emb = self.sigma_mlp(sigma_emb)
        return sigma_emb


class ResidualBlock(nn.Module):
    """ResNet-style block with sigma (noise level) conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, sigma_embed_dim: int = 1024, 
                 dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Use appropriate group size for GroupNorm
        groups = min(32, in_channels) if in_channels >= 8 else 1
        out_groups = min(32, out_channels) if out_channels >= 8 else 1
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.sigma_emb_proj = nn.Linear(sigma_embed_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, sigma_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add sigma embedding
        sigma_emb_out = F.silu(sigma_emb)
        sigma_emb_out = self.sigma_emb_proj(sigma_emb_out)
        h = h + sigma_emb_out[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        groups = min(32, channels) if channels >= 8 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(dim=1)
        
        # Compute attention with scaled dot-product
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return x + out


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


class NCSNpp(nn.Module):
    """
    Noise Conditional Score Network++ (NCSN++) for VE-SDE.
    
    This is the UNet architecture from the score_sde paper.
    Uses noise prediction (ε-parameterization) for numerical stability.
    
    The network predicts the noise ε, and score is computed as: s = -ε/σ
    
    NOTE: Attention is ONLY applied at the lowest resolution levels (after 2+ downsamples)
    to avoid OOM on high-resolution inputs like 512x512.
    
    Args:
        in_channels: Number of input channels (noisy_state + condition)
        out_channels: Number of output channels (noise prediction for target fields)
        base_channels: Base number of channels for the first layer
        sigma_embed_dim: Dimension of sigma embeddings
        num_res_blocks: Number of residual blocks per level
        use_attention: Whether to use attention blocks (only at low-res levels)
        dropout: Dropout probability
        attention_resolution: Only apply attention when resolution <= this (default 64)
    """
    
    def __init__(self, in_channels: int = 2, out_channels: int = 1, 
                 base_channels: int = 64, sigma_embed_dim: int = 256,
                 num_res_blocks: int = 2, use_attention: bool = True,
                 dropout: float = 0.0, attention_resolution: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.attention_resolution = attention_resolution
        
        # Sigma embedding (sigma_embed_dim -> 4 * sigma_embed_dim)
        self.sigma_embedding = SigmaEmbedding(sigma_embed_dim)
        sigma_emb_dim = 4 * sigma_embed_dim
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Define channel progression for simple 3-level UNet
        self.ch1 = base_channels      # 64
        self.ch2 = base_channels * 2  # 128
        self.ch3 = base_channels * 4  # 256
        
        # Encoder
        # Level 1: base_channels (full resolution - NO attention to save memory!)
        self.enc1_res1 = ResidualBlock(self.ch1, self.ch1, sigma_emb_dim, dropout)
        self.enc1_res2 = ResidualBlock(self.ch1, self.ch1, sigma_emb_dim, dropout)
        # No attention at level 1 (512x512 is too large)
        self.enc1_down = Downsample(self.ch1)
        
        # Level 2: base_channels * 2 (half resolution - NO attention)
        self.enc2_res1 = ResidualBlock(self.ch1, self.ch2, sigma_emb_dim, dropout)
        self.enc2_res2 = ResidualBlock(self.ch2, self.ch2, sigma_emb_dim, dropout)
        # No attention at level 2 (256x256 still too large)
        self.enc2_down = Downsample(self.ch2)
        
        # Level 3: base_channels * 4 (1/4 resolution = 128x128 - attention OK)
        self.enc3_res1 = ResidualBlock(self.ch2, self.ch3, sigma_emb_dim, dropout)
        self.enc3_res2 = ResidualBlock(self.ch3, self.ch3, sigma_emb_dim, dropout)
        self.enc3_attn = AttentionBlock(self.ch3) if use_attention else nn.Identity()
        
        # Middle (1/4 resolution - attention OK)
        self.mid_res1 = ResidualBlock(self.ch3, self.ch3, sigma_emb_dim, dropout)
        self.mid_attn = AttentionBlock(self.ch3) if use_attention else nn.Identity()
        self.mid_res2 = ResidualBlock(self.ch3, self.ch3, sigma_emb_dim, dropout)
        
        # Decoder
        # Level 3 decoder (1/4 resolution - attention OK)
        self.dec3_res1 = ResidualBlock(self.ch3 + self.ch3, self.ch3, sigma_emb_dim, dropout)
        self.dec3_res2 = ResidualBlock(self.ch3, self.ch3, sigma_emb_dim, dropout)
        self.dec3_attn = AttentionBlock(self.ch3) if use_attention else nn.Identity()
        self.dec3_up = Upsample(self.ch3)
        
        # Level 2 decoder (half resolution - NO attention)
        self.dec2_res1 = ResidualBlock(self.ch3 + self.ch2, self.ch2, sigma_emb_dim, dropout)
        self.dec2_res2 = ResidualBlock(self.ch2, self.ch2, sigma_emb_dim, dropout)
        # No attention at level 2
        self.dec2_up = Upsample(self.ch2)
        
        # Level 1 decoder (full resolution - NO attention)
        self.dec1_res1 = ResidualBlock(self.ch2 + self.ch1, self.ch1, sigma_emb_dim, dropout)
        self.dec1_res2 = ResidualBlock(self.ch1, self.ch1, sigma_emb_dim, dropout)
        # No attention at level 1
        
        # Final output
        groups = min(32, base_channels) if base_channels >= 8 else 1
        self.norm_out = nn.GroupNorm(groups, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        # Initialize final conv to zero (important for stable training!)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        
    def forward(self, x, sigma):
        """
        Predict noise given noisy input and sigma.
        
        Args:
            x: Concatenated noisy state and condition [B, in_channels, H, W]
            sigma: Noise levels [B]
            
        Returns:
            Predicted noise ε [B, out_channels, H, W]
        """
        # Get sigma embeddings
        sigma_emb = self.sigma_embedding(sigma)
        
        # Initial convolution
        x = self.conv_in(x)  # [B, ch1, H, W]
        
        # Encoder Level 1 (full resolution - no attention)
        enc1 = self.enc1_res1(x, sigma_emb)
        enc1 = self.enc1_res2(enc1, sigma_emb)
        x = self.enc1_down(enc1)  # [B, ch1, H/2, W/2]
        
        # Encoder Level 2 (half resolution - no attention)
        x = self.enc2_res1(x, sigma_emb)
        enc2 = self.enc2_res2(x, sigma_emb)
        x = self.enc2_down(enc2)  # [B, ch2, H/4, W/4]
        
        # Encoder Level 3 (1/4 resolution - attention OK)
        x = self.enc3_res1(x, sigma_emb)
        enc3 = self.enc3_res2(x, sigma_emb)
        enc3 = self.enc3_attn(enc3)  # [B, ch3, H/4, W/4]
        
        # Middle (1/4 resolution - attention OK)
        x = self.mid_res1(enc3, sigma_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, sigma_emb)  # [B, ch3, H/4, W/4]
        
        # Decoder Level 3 (1/4 resolution - attention OK)
        x = torch.cat([x, enc3], dim=1)  # [B, ch3*2, H/4, W/4]
        x = self.dec3_res1(x, sigma_emb)
        x = self.dec3_res2(x, sigma_emb)
        x = self.dec3_attn(x)
        x = self.dec3_up(x)  # [B, ch3, H/2, W/2]
        
        # Decoder Level 2 (half resolution - no attention)
        x = torch.cat([x, enc2], dim=1)  # [B, ch3+ch2, H/2, W/2]
        x = self.dec2_res1(x, sigma_emb)
        x = self.dec2_res2(x, sigma_emb)
        x = self.dec2_up(x)  # [B, ch2, H, W]
        
        # Decoder Level 1 (full resolution - no attention)
        x = torch.cat([x, enc1], dim=1)  # [B, ch2+ch1, H, W]
        x = self.dec1_res1(x, sigma_emb)
        x = self.dec1_res2(x, sigma_emb)
        
        # Final output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)  # [B, out_channels, H, W]
        
        return x


class WallTempBias(nn.Module):
    """Simple learned bias based on wall temperature."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.bias_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, wall_temp):
        if wall_temp.dim() == 1:
            wall_temp = wall_temp.unsqueeze(-1)
        wall_temp_norm = (wall_temp - 87.5) / 32.5
        bias = self.bias_net(wall_temp_norm)
        bias = bias.view(-1, 1, 1, 1)
        return bias


class FiLMLayer(nn.Module):
    """FiLM: Feature-wise Linear Modulation layer."""
    
    def __init__(self, num_channels: int = 1, hidden_dim: int = 64, gamma_range: float = 0.1):
        super().__init__()
        self.num_channels = num_channels
        self.gamma_range = gamma_range
        
        self.film_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias[:num_channels])
        nn.init.zeros_(self.film_net[-1].bias[num_channels:])
        
    def forward(self, x, wall_temp):
        if wall_temp.dim() == 1:
            wall_temp = wall_temp.unsqueeze(-1)
        wall_temp_norm = (wall_temp - 87.5) / 32.5
        film_params = self.film_net(wall_temp_norm)
        gamma_raw, beta = torch.chunk(film_params, 2, dim=1)
        gamma = (1 - self.gamma_range) + 2 * self.gamma_range * torch.sigmoid(gamma_raw)
        gamma = gamma.view(-1, self.num_channels, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1)
        return gamma * x + beta


class VESDESampler:
    """
    Sampler for VE-SDE Score-Based Models.
    
    Based on the original score_sde implementation.
    Uses noise prediction and converts to score for sampling.
    """
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0, 
                 num_steps: int = 500, method: str = 'pc',
                 snr: float = 0.16, corrector_steps: int = 1):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.method = method
        self.snr = snr
        self.corrector_steps = corrector_steps
        
        # Create geometric noise schedule (same as original score_sde)
        self.sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), num_steps + 1))
        ).float()
    
    @torch.no_grad()
    def sample(self, model, condition: torch.Tensor,
               shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate samples using the specified method."""
        sigmas = self.sigmas.to(device)
        
        # Start from noise at sigma_max
        x = torch.randn(shape, device=device) * self.sigma_max
        
        if self.method == 'pc':
            return self._sample_pc(model, x, condition, sigmas)
        elif self.method == 'ode':
            return self._sample_ode(model, x, condition, sigmas)
        else:
            return self._sample_euler(model, x, condition, sigmas)
    
    def _sample_pc(self, model, x, condition, sigmas):
        """
        Predictor-Corrector sampling (Algorithm 2 from score_sde paper).
        
        Predictor: Reverse diffusion (ancestral sampling)
        Corrector: Langevin MCMC
        """
        for i in range(self.num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # === Predictor step (reverse diffusion) ===
            sigma_tensor = torch.full((x.shape[0],), sigma.item(), device=x.device)
            
            # Get noise prediction from model
            x_input = torch.cat([x, condition], dim=1)
            noise_pred = model.noise_net(x_input, sigma_tensor)
            
            # Convert to score: s = -noise_pred / sigma
            score = -noise_pred / sigma
            
            # Reverse VE-SDE step (ancestral sampling)
            # x_next = x + (sigma^2 - sigma_next^2) * score + sqrt(sigma^2 - sigma_next^2) * z
            diff = sigma**2 - sigma_next**2
            x = x + diff * score
            if sigma_next > self.sigma_min:
                x = x + torch.sqrt(diff) * torch.randn_like(x)
            
            # === Corrector step (Langevin dynamics) ===
            if sigma_next > self.sigma_min:
                for _ in range(self.corrector_steps):
                    sigma_tensor = torch.full((x.shape[0],), sigma_next.item(), device=x.device)
                    x_input = torch.cat([x, condition], dim=1)
                    noise_pred = model.noise_net(x_input, sigma_tensor)
                    score = -noise_pred / sigma_next
                    
                    # Langevin step with adaptive step size
                    grad_norm = torch.sqrt(torch.mean(score**2))
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2
                    step_size = min(step_size.item(), (sigma_next ** 2) * 2)  # Clip step size
                    
                    x = x + step_size * score + torch.sqrt(2 * step_size) * torch.randn_like(x)
        
        return x
    
    def _sample_ode(self, model, x, condition, sigmas):
        """Probability flow ODE sampling (deterministic).
        
        Uses the same update as PC predictor but without noise injection.
        This is equivalent to the deterministic probability flow ODE.
        """
        for i in range(self.num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            sigma_tensor = torch.full((x.shape[0],), sigma.item(), device=x.device)
            x_input = torch.cat([x, condition], dim=1)
            noise_pred = model.noise_net(x_input, sigma_tensor)
            score = -noise_pred / sigma
            
            # Same formula as PC predictor (reverse VE-SDE)
            diff = sigma**2 - sigma_next**2
            x = x + diff * score
        
        return x
    
    def _sample_euler(self, model, x, condition, sigmas):
        """Euler-Maruyama sampling (stochastic, simpler than PC).
        
        Uses the same update formula as PC predictor with noise injection,
        but without the corrector step.
        """
        for i in range(self.num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            sigma_tensor = torch.full((x.shape[0],), sigma.item(), device=x.device)
            x_input = torch.cat([x, condition], dim=1)
            noise_pred = model.noise_net(x_input, sigma_tensor)
            score = -noise_pred / sigma
            
            # Same formula as PC predictor
            diff = sigma**2 - sigma_next**2
            x = x + diff * score
            
            # Add noise (Euler-Maruyama stochastic term)
            if sigma_next > self.sigma_min:
                x = x + torch.sqrt(diff) * torch.randn_like(x)
        
        return x


class ScoreBasedVESDE(nn.Module):
    """
    Variance Exploding SDE Score-Based Model.
    
    Uses noise prediction (ε-parameterization) following the original score_sde:
    - Network predicts noise ε
    - Score is computed as: s = -ε / σ
    
    Forward process: x_noisy = x_clean + σ * z
    
    Training loss (Denoising Score Matching):
        L = E_σ[ ||ε_θ(x + σz, σ) - z||² ]
    
    Args:
        noise_net: The noise prediction network (NCSN++)
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
    """
    
    def __init__(self, noise_net: NCSNpp, sigma_min: float = 0.01, 
                 sigma_max: float = 1.0):
        super().__init__()
        self.noise_net = noise_net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def get_score(self, x_noisy: torch.Tensor, condition: torch.Tensor,
                  sigma: torch.Tensor) -> torch.Tensor:
        """
        Get score from noise prediction.
        
        Score = -ε / σ (following score_sde convention)
        """
        x_input = torch.cat([x_noisy, condition], dim=1)
        noise_pred = self.noise_net(x_input, sigma)
        sigma_view = sigma.view(-1, 1, 1, 1)
        score = -noise_pred / sigma_view
        return score
    
    def forward(self, x_noisy: torch.Tensor, condition: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy state and conditioning."""
        x_input = torch.cat([x_noisy, condition], dim=1)
        return self.noise_net(x_input, sigma)
    
    def compute_loss(self, x_clean: torch.Tensor, condition: torch.Tensor,
                     sigma: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute denoising score matching loss with noise prediction.
        
        Loss = E_σ[ ||ε_θ(x + σz, σ) - z||² ]
        
        This is equivalent to the original DSM loss but using noise prediction
        for numerical stability.
        """
        batch_size = x_clean.shape[0]
        device = x_clean.device
        
        # Sample noise levels uniformly in log space
        if sigma is None:
            log_sigma = torch.rand(batch_size, device=device) * \
                       (np.log(self.sigma_max) - np.log(self.sigma_min)) + \
                       np.log(self.sigma_min)
            sigma = torch.exp(log_sigma)
        
        # Sample noise
        z = torch.randn_like(x_clean)
        
        # Add noise to clean data: x_noisy = x_clean + σ * z
        sigma_view = sigma.view(-1, 1, 1, 1)
        x_noisy = x_clean + sigma_view * z
        
        # Predict noise (network output)
        noise_pred = self.forward(x_noisy, condition, sigma)
        
        # Simple MSE loss on noise prediction (no weighting needed!)
        # This is the key fix - we predict noise directly
        loss = F.mse_loss(noise_pred, z)
        
        return loss, sigma
    
    @torch.no_grad()
    def sample(self, condition: torch.Tensor, shape: Tuple[int, ...],
               device: torch.device, num_steps: int = 500,
               method: str = 'pc', snr: float = 0.16) -> torch.Tensor:
        """Generate samples using the specified sampling method."""
        sampler = VESDESampler(
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            num_steps=num_steps,
            method=method,
            snr=snr
        )
        return sampler.sample(self, condition, shape, device)


class ScoreBasedVESDELightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for Score-Based VE-SDE Model.
    
    This is a FRAME-TO-FRAME model (non-autoregressive):
    - Input: [noisy_state, conditioning]
    - Target: current_output
    
    Supports multiple tasks through task_cfg configuration:
    - temperature_from_sdf: Predict temperature from SDF (Task 1)
    - velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)
    - noisy_velocity_from_interface: Predict velocity + temperature from NOISY inputs (Task 3)
      (Noise is applied at dataset level via noise_augmentation config)
    
    Args:
        model_cfg: Model configuration containing VE-SDE parameters
        optim_cfg: Optimizer configuration
        scheduler_cfg: Learning rate scheduler configuration
        task_cfg: Task configuration defining which channels to use
        normalization_stats: Pre-computed normalization statistics from training data
    """
    
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig, 
                 scheduler_cfg: DictConfig, task_cfg: Optional[DictConfig] = None,
                 normalization_stats: Optional[dict] = None):
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
        
        # Wall temperature conditioning strategy
        self.conditioning_strategy = model_cfg.get('conditioning_strategy', 'none')
        
        # Compute in_channels and out_channels from task_cfg
        # in_channels = num_target_channels (noisy state) + num_conditioning_channels
        # out_channels = num_target_channels (noise prediction)
        computed_in_channels = self.num_target_channels + self.num_conditioning_channels
        computed_out_channels = self.num_target_channels
        
        in_channels = model_cfg.get('in_channels', computed_in_channels)
        out_channels = model_cfg.get('out_channels', computed_out_channels)
        
        print(f"\n🔄 Frame-to-Frame VE-SDE Configuration:")
        print(f"   NCSN++ in_channels: {in_channels} = {self.num_target_channels} (x_noisy) + {self.num_conditioning_channels} (cond)")
        print(f"   NCSN++ out_channels: {out_channels}")
        
        # VE-SDE parameters - USE SMALLER sigma_max FOR NORMALIZED DATA!
        # Data is in [-1, 1], so sigma_max=1.0 is appropriate (not 50!)
        self.sigma_min = model_cfg.get('sigma_min', 0.01)
        self.sigma_max = model_cfg.get('sigma_max', 1.0)  # Changed from 50 to 1.0!
        self.num_sampling_steps = model_cfg.get('num_sampling_steps', 500)
        self.sampling_method = model_cfg.get('sampling_method', 'pc')
        self.snr = model_cfg.get('snr', 0.16)
        
        print(f"\n🔊 VE-SDE: σ_min={self.sigma_min}, σ_max={self.sigma_max}")
        print(f"   Sampling: {self.sampling_method}, steps={self.num_sampling_steps}, SNR={self.snr}")
        
        # Initialize noise prediction network (NCSN++)
        noise_net = NCSNpp(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=model_cfg.get('base_channels', 64),
            sigma_embed_dim=model_cfg.get('sigma_embed_dim', 256),
            num_res_blocks=model_cfg.get('num_res_blocks', 2),
            use_attention=model_cfg.get('use_attention', True),
            dropout=model_cfg.get('dropout', 0.0)
        )
        
        self.ve_sde = ScoreBasedVESDE(
            noise_net=noise_net,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max
        )
        
        # Initialize conditioning module
        self.wall_temp_conditioner = None
        if self.conditioning_strategy == 'bias':
            self.wall_temp_conditioner = WallTempBias(
                hidden_dim=model_cfg.get('wall_temp_bias_hidden', 64)
            )
            print("🌡️  Conditioning: SIMPLE BIAS")
        elif self.conditioning_strategy == 'film':
            gamma_range = model_cfg.get('wall_temp_film_gamma_range', 0.1)
            self.wall_temp_conditioner = FiLMLayer(
                num_channels=1,
                hidden_dim=model_cfg.get('wall_temp_film_hidden', 64),
                gamma_range=gamma_range
            )
            print(f"🌡️  Conditioning: FiLM")
        else:
            print("🌡️  Conditioning: NONE")
        
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
        
        # VE-SDE inference parameters
        self.num_inference_steps = model_cfg.get('num_inference_steps', self.num_sampling_steps)
        print(f"\n🔧 Default Inference Settings:")
        print(f"   Inference steps: {self.num_inference_steps}")

        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        
    def forward(self, x_noisy, condition, sigma):
        return self.ve_sde(x_noisy, condition, sigma)
    
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
        # BulkFlow dataset returns: (input_data, output_data) or (input_data, output_data, wall_temp)
        if self.conditioning_strategy != 'none':
            input_data, output_data, wall_temp = batch
        else:
            input_data, output_data = batch
            wall_temp = None
        
        # input_data: [B, C_in, H, W] (e.g., sdf, velx_interface, vely_interface)
        # output_data: [B, C_out, H, W] (e.g., temperature, velx, vely)
        
        # Extract conditioning and target channels based on task_cfg
        conditioning = self._extract_channels(input_data, self.conditioning_channels)  # [B, num_cond, H, W]
        target = self._extract_channels(output_data, self.target_channels)  # [B, num_target, H, W]
        
        # Apply wall temperature conditioning (only for single-channel targets)
        if self.conditioning_strategy == 'bias' and wall_temp is not None and self.num_target_channels == 1:
            wall_temp_cond = self.wall_temp_conditioner(wall_temp)
            target_conditioned = target + wall_temp_cond
        elif self.conditioning_strategy == 'film' and wall_temp is not None and self.num_target_channels == 1:
            target_conditioned = self.wall_temp_conditioner(target, wall_temp)
        else:
            target_conditioned = target
        
        # Compute loss
        loss, sigma = self.ve_sde.compute_loss(target_conditioned, conditioning)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_sigma_mean', sigma.mean(), on_step=False, on_epoch=True)
        self.log('train_sigma_min', sigma.min(), on_step=False, on_epoch=True)
        self.log('train_sigma_max', sigma.max(), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step."""
        # Determine validation type
        is_clean_val = (dataloader_idx == 0) if self.has_noise else True
        val_prefix = "val_clean" if is_clean_val else "val_noisy"
        
        # Extract data from batch
        if self.conditioning_strategy != 'none':
            input_data, output_data, wall_temp = batch
        else:
            input_data, output_data = batch
            wall_temp = None
        
        # Extract conditioning and target channels based on task_cfg
        conditioning = self._extract_channels(input_data, self.conditioning_channels)
        target = self._extract_channels(output_data, self.target_channels)
        
        # Apply conditioning (only for single-channel targets)
        if self.conditioning_strategy == 'bias' and wall_temp is not None and self.num_target_channels == 1:
            wall_temp_cond = self.wall_temp_conditioner(wall_temp)
            target_conditioned = target + wall_temp_cond
        elif self.conditioning_strategy == 'film' and wall_temp is not None and self.num_target_channels == 1:
            target_conditioned = self.wall_temp_conditioner(target, wall_temp)
        else:
            target_conditioned = target
        
        # Compute validation loss
        loss, sigma = self.ve_sde.compute_loss(target_conditioned, conditioning)
        
        # Log losses
        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        
        # Generate samples and log statistics (only batch 0 and clean validation)
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, conditioning.shape[0])
                val_steps = min(100, self.num_sampling_steps)
                
                # Generate samples using reverse diffusion
                samples = self.ve_sde.sample(
                    conditioning[:num_samples],
                    (num_samples, self.num_target_channels, target.shape[2], target.shape[3]),
                    conditioning.device,
                    num_steps=val_steps,
                    method='ode',  # Use ODE for deterministic validation
                    snr=self.snr
                )
                
                # Remove conditioning
                if self.conditioning_strategy == 'bias' and wall_temp is not None and self.num_target_channels == 1:
                    wall_temp_cond = self.wall_temp_conditioner(wall_temp[:num_samples])
                    samples = samples - wall_temp_cond
                elif self.conditioning_strategy == 'film' and wall_temp is not None and self.num_target_channels == 1:
                    wall_temp_sample = wall_temp[:num_samples]
                    if wall_temp_sample.dim() == 1:
                        wall_temp_sample = wall_temp_sample.unsqueeze(-1)
                    wall_temp_norm = (wall_temp_sample - 87.5) / 32.5
                    film_params = self.wall_temp_conditioner.film_net(wall_temp_norm)
                    gamma_raw, beta = torch.chunk(film_params, 2, dim=1)
                    gamma = (1 - self.wall_temp_conditioner.gamma_range) + \
                            2 * self.wall_temp_conditioner.gamma_range * torch.sigmoid(gamma_raw)
                    gamma = gamma.view(-1, 1, 1, 1)
                    beta = beta.view(-1, 1, 1, 1)
                    samples = (samples - beta) / (gamma + 1e-8)
                
                # Log normalized statistics
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
        # Use lower learning rate for stability (1e-4 is typical for score models)
        lr = self.optim_cfg.get('lr', 1e-4)
        
        if self.optim_cfg.name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=self.optim_cfg.get('weight_decay', 0.0),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.optim_cfg.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=self.optim_cfg.get('weight_decay', 1e-4),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=1e-4
            )
        
        # Use warmup + cosine decay for stable training
        if self.scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=lr * 0.01
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
                eta_min=lr * 0.01
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
