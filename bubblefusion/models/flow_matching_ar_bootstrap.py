"""
Autoregressive Conditional Flow Matching with Bootstrap Initialization.

This module implements an Autoregressive Conditional Flow Matching model
that can operate in two modes:

1. Bootstrap Mode (previous state missing):
   - Uses a history encoder to infer the initial bulk state from
     a sequence of conditioning inputs (SDF, interface velocity)
   - No zeros fed - the model explicitly learns to estimate initial state

2. Autoregressive Mode (previous state exists):
   - Standard AR prediction using previous timestep output
   - Same as flow_matching_ar.py

Key Design Principles:
- Never feed zeros and hope the model figures it out
- Explicitly tell the model whether previous state exists via availability mask
- Train both modes jointly in the same rollout for end-to-end learning

Training Strategy:
- Sample trajectory segments of length L
- First frame uses bootstrap mode (infer initial state from history)
- Subsequent frames use AR mode with teacher forcing or scheduled sampling
- Both losses trained jointly

Architecture:
- HistoryEncoder: Temporal CNN that takes [B, T_hist, C_cond, H, W] → [B, C_out, H, W]
- FlowMatchingUNet: Main velocity field predictor (same as flow_matching_ar.py)
- Availability mask channel: Indicates whether previous state is available

References:
    - "Flow Matching for Generative Modeling" (Lipman et al., 2023)
    - "Scheduled Sampling for Sequence Prediction with RNNs" (Bengio et al., 2015)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple, List
import numpy as np

from bubblefusion.models.flow_matching import (
    FlowMatchingUNet,
    TimeEmbedding,
    ResidualBlock,
)
from bubblefusion.models.flow_matching_ar import (
    SpectralLoss,
    GradientLoss,
)


class TemporalConvBlock(nn.Module):
    """
    Temporal convolution block for processing conditioning history.
    
    Uses 3D convolutions to capture spatial-temporal patterns in the
    conditioning history (SDF, interface velocity over time).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        padding: str = 'same',
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            padding=padding if padding == 'same' else tuple(k // 2 for k in kernel_size)
        )
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] - batch, channels, time, height, width
        Returns:
            [B, C_out, T, H, W]
        """
        return self.act(self.norm(self.conv(x)))


class TemporalMixer(nn.Module):
    """
    Fast temporal mixer for bootstrap initialization.
    
    Instead of expensive Conv3D operations, this flattens the temporal
    dimension into channels and uses efficient 2D convolutions.
    
    Architecture:
    1. Flatten temporal history into channels: [B, T, C, H, W] → [B, T*C, H, W]
    2. Mix with 1x1 convs (channel mixing)
    3. Optionally add spatial context with 3x3 conv
    4. Project to output channels
    
    This is ~3-5x faster than Conv3D-based HistoryEncoder while maintaining
    reasonable quality for bootstrap initialization.
    
    Args:
        in_channels: Number of conditioning channels per timestep (e.g., 3 for SDF + velx + vely)
        out_channels: Number of output channels (e.g., 3 for velx + vely + temp)
        history_length: Number of timesteps in history (T)
        hidden_channels: Hidden dimension for mixing layers
        use_spatial_conv: Whether to add 3x3 conv for spatial context (slightly slower but better)
        use_temporal_weights: Whether to learn per-timestep importance weights
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        history_length: int = 10,
        hidden_channels: int = 32,
        use_spatial_conv: bool = True,
        use_temporal_weights: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.history_length = history_length
        self.use_temporal_weights = use_temporal_weights
        
        # Total input channels after flattening time
        # +1 for current conditioning if provided
        total_in_channels = in_channels * (history_length + 1)  # +1 for current_conditioning
        
        # Learnable temporal importance weights (recent frames should matter more)
        if use_temporal_weights:
            # Initialize with exponential decay favoring recent frames
            init_weights = torch.exp(torch.linspace(-2, 0, history_length + 1))
            init_weights = init_weights / init_weights.sum()  # Normalize
            self.temporal_weights = nn.Parameter(init_weights.view(1, history_length + 1, 1, 1, 1))
        
        # Channel mixing: flatten temporal → hidden
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(total_in_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        
        # Optional spatial context (3x3 conv)
        if use_spatial_conv:
            self.spatial_mixer = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                nn.SiLU(),
            )
        else:
            self.spatial_mixer = nn.Identity()
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
        # Learnable output scale and bias (for training stability)
        self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    
    def forward(
        self,
        conditioning_history: torch.Tensor,
        current_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode conditioning history to produce initial state estimate.
        
        Args:
            conditioning_history: [B, T, C_cond, H, W] - history of conditioning inputs
            current_conditioning: [B, C_cond, H, W] - current timestep conditioning (optional)
                                 If provided, appended to history for better context
                                 
        Returns:
            initial_state: [B, C_out, H, W] - estimated initial bulk state
        """
        B, T, C, H, W = conditioning_history.shape
        
        # Append current conditioning if provided
        if current_conditioning is not None:
            current_expanded = current_conditioning.unsqueeze(1)  # [B, 1, C, H, W]
            conditioning_history = torch.cat([conditioning_history, current_expanded], dim=1)
            T = T + 1
        
        # Apply temporal importance weights if enabled
        if self.use_temporal_weights:
            # Adjust weights size if needed (in case T differs from expected)
            if T != self.temporal_weights.shape[1]:
                # Interpolate weights to match actual T
                weights = F.interpolate(
                    self.temporal_weights.squeeze(0).squeeze(-1).squeeze(-1),  # [1, T_expected]
                    size=T,
                    mode='linear',
                    align_corners=False
                ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, T, 1, 1, 1]
            else:
                weights = self.temporal_weights
            
            # Apply weights: [B, T, C, H, W] * [1, T, 1, 1, 1]
            conditioning_history = conditioning_history * weights * T  # Scale to preserve magnitude
        
        # Flatten temporal dimension into channels: [B, T, C, H, W] → [B, T*C, H, W]
        x = conditioning_history.view(B, T * C, H, W)
        
        # Channel mixing
        x = self.channel_mixer(x)  # [B, hidden, H, W]
        
        # Spatial mixing (optional 3x3 conv)
        x = self.spatial_mixer(x)  # [B, hidden, H, W]
        
        # Output projection with learnable scale
        initial_state = self.output_proj(x) * self.output_scale + self.output_bias
        
        return initial_state


class HistoryEncoder(nn.Module):
    """
    Encodes conditioning history to produce an initial bulk state estimate.
    
    Takes a sequence of conditioning inputs (SDF, interface velocity) and
    produces an estimate of the bulk state (velocity + temperature) that
    would be consistent with this conditioning history.
    
    Architecture:
    1. Temporal convolutions to aggregate information across time
    2. Spatial convolutions to produce final state estimate
    3. Progressive temporal reduction: T → T//2 → ... → 1
    
    Args:
        in_channels: Number of conditioning channels (e.g., 3 for SDF + velx + vely)
        out_channels: Number of output channels (e.g., 3 for velx + vely + temp)
        hidden_channels: Hidden channel dimension
        num_temporal_blocks: Number of temporal conv blocks
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: int = 64,
        num_temporal_blocks: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial projection
        self.input_proj = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        
        # Temporal conv blocks with progressive temporal reduction
        self.temporal_blocks = nn.ModuleList()
        for i in range(num_temporal_blocks):
            self.temporal_blocks.append(
                TemporalConvBlock(
                    hidden_channels, hidden_channels,
                    kernel_size=(3, 3, 3)
                )
            )
        
        # Temporal pooling layers (reduce time dimension progressively)
        self.temporal_pools = nn.ModuleList()
        for i in range(num_temporal_blocks - 1):
            # Use adaptive pooling to handle variable history lengths
            self.temporal_pools.append(
                nn.AdaptiveAvgPool3d((None, None, None))  # Placeholder, actual pooling done in forward
            )
        
        # Final temporal aggregation (collapse time dimension)
        self.temporal_aggregate = nn.AdaptiveAvgPool3d((1, None, None))
        
        # Spatial refinement after temporal aggregation
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, out_channels, 1)
        
        # Learnable scale and bias for output (helps with training stability)
        self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
    def forward(
        self, 
        conditioning_history: torch.Tensor,
        current_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode conditioning history to produce initial state estimate.
        
        Args:
            conditioning_history: [B, T, C_cond, H, W] - history of conditioning inputs
            current_conditioning: [B, C_cond, H, W] - current timestep conditioning (optional)
                                 If provided, appended to history for better context
                                 
        Returns:
            initial_state: [B, C_out, H, W] - estimated initial bulk state
        """
        B, T, C, H, W = conditioning_history.shape
        
        # Optionally append current conditioning to history
        if current_conditioning is not None:
            # current_conditioning: [B, C, H, W] -> [B, 1, C, H, W]
            current_expanded = current_conditioning.unsqueeze(1)
            conditioning_history = torch.cat([conditioning_history, current_expanded], dim=1)
            T = T + 1
        
        # Rearrange to [B, C, T, H, W] for Conv3D
        x = conditioning_history.permute(0, 2, 1, 3, 4)
        
        # Initial projection
        x = self.input_proj(x)  # [B, hidden, T, H, W]
        
        # Temporal conv blocks with progressive reduction
        for i, block in enumerate(self.temporal_blocks):
            x = block(x)
            
            # Reduce temporal dimension (except last block)
            if i < len(self.temporal_blocks) - 1 and x.shape[2] > 1:
                # Pool temporally by factor of 2
                new_t = max(1, x.shape[2] // 2)
                x = F.adaptive_avg_pool3d(x, (new_t, x.shape[3], x.shape[4]))
        
        # Final temporal aggregation
        x = self.temporal_aggregate(x)  # [B, hidden, 1, H, W]
        x = x.squeeze(2)  # [B, hidden, H, W]
        
        # Spatial refinement
        x = self.spatial_refine(x)
        
        # Output projection with learnable scale
        initial_state = self.output_proj(x) * self.output_scale + self.output_bias
        
        return initial_state


# =============================================================================
# Attention-Based History Encoder (ENMA-inspired factored space-time attention)
# =============================================================================

def _get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings for a (grid_h, grid_w) grid."""
    assert embed_dim % 2 == 0
    gh = np.arange(grid_h, dtype=np.float32)
    gw = np.arange(grid_w, dtype=np.float32)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, gh)  # [grid_h, D/2]
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, gw)  # [grid_w, D/2]
    emb_h = np.repeat(emb_h, grid_w, axis=0)               # [grid_h*grid_w, D/2]
    emb_w = np.tile(emb_w, (grid_h, 1))                    # [grid_h*grid_w, D/2]
    return np.concatenate([emb_h, emb_w], axis=1)           # [grid_h*grid_w, D]


class SpaceTimeAttnBlock(nn.Module):
    """
    Pre-norm transformer block with multi-head self-attention and SwiGLU FFN.

    Uses F.scaled_dot_product_attention for automatic Flash/Memory-Efficient
    dispatch. The *is_causal* flag enables causal masking for temporal layers.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.is_causal = is_causal
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.w_gate = nn.Linear(embed_dim, hidden, bias=False)
        self.w_up = nn.Linear(embed_dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, embed_dim, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B_local, S, D]  (S = N_patches for spatial, T for temporal)
        """
        # --- Self-Attention ---
        h = self.norm1(x)
        B, S, D = h.shape
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, S, head_dim]
        q, k, v = qkv.unbind(0)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=self.is_causal,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        x = x + self.out_proj(attn_out)

        # --- SwiGLU FFN ---
        h = self.norm2(x)
        x = x + self.ffn_drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))
        return x


class AttentionHistoryEncoder(nn.Module):
    """
    Attention-based history encoder using factored space-time transformer.

    Patchifies each conditioning frame, adds spatial and temporal positional
    embeddings, then alternates spatial self-attention (within-frame) and
    causal temporal self-attention (across frames). The last frame's tokens
    are decoded back to pixel space to produce the initial state estimate.

    This is a drop-in replacement for TemporalMixer / HistoryEncoder, with
    the same forward() signature and *out_channels* attribute.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        patch_size: int = 8,
        img_size: int = 128,
        max_history_length: int = 50,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        output_head_type: str = 'linear',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.max_history_length = max_history_length

        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # Patch embedding (Conv2d acting as non-overlapping patchify)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # Spatial positional embeddings (fixed sinusoidal)
        spatial_pe = get_2d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_w)
        self.register_buffer(
            'spatial_pos_embed',
            torch.from_numpy(spatial_pe).float().unsqueeze(0),  # [1, N, D]
        )

        # Temporal positional embeddings (learned, broadcast over patches)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_history_length, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Alternating spatial / temporal transformer blocks
        self.spatial_blocks = nn.ModuleList([
            SpaceTimeAttnBlock(embed_dim, num_heads, mlp_ratio, dropout, is_causal=False)
            for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            SpaceTimeAttnBlock(embed_dim, num_heads, mlp_ratio, dropout, is_causal=True)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Output head
        if output_head_type == 'conv_decoder':
            mid = embed_dim // 2
            self.output_head = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, mid, kernel_size=patch_size // 2, stride=patch_size // 2),
                nn.GroupNorm(8, mid),
                nn.SiLU(),
                nn.ConvTranspose2d(mid, out_channels, kernel_size=2, stride=2),
            )
            self._output_head_type = 'conv_decoder'
        else:
            self.output_head = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
            self._output_head_type = 'linear'

        # Learnable scale/bias for training stability (same interface as TemporalMixer)
        self.output_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.shape[0], -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.shape[0], -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, patch_size**2 * C_out] -> [B, C_out, H, W]"""
        p = self.patch_size
        c = self.out_channels
        h, w = self.grid_h, self.grid_w
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(
        self,
        conditioning_history: torch.Tensor,
        current_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            conditioning_history: [B, T, C_cond, H, W]
            current_conditioning: [B, C_cond, H, W] (optional, appended as last frame)

        Returns:
            initial_state: [B, C_out, H, W]
        """
        B, T, C, H, W = conditioning_history.shape

        if current_conditioning is not None:
            conditioning_history = torch.cat(
                [conditioning_history, current_conditioning.unsqueeze(1)], dim=1
            )
            T = T + 1

        # Patchify all frames: [B*T, C, H, W] -> [B*T, N, D]
        x = conditioning_history.reshape(B * T, C, H, W)
        x = self.patch_embed(x)                          # [B*T, D, gh, gw]
        x = x.flatten(2).transpose(1, 2)                 # [B*T, N, D]

        # Add spatial positional embeddings (broadcast over batch & time)
        x = x + self.spatial_pos_embed                    # [B*T, N, D]

        # Reshape to [B, T, N, D] and add temporal positional embeddings
        x = x.reshape(B, T, self.num_patches, self.embed_dim)
        x = x + self.temporal_pos_embed[:, :T, :, :]     # [B, T, N, D]

        # Alternating spatial and temporal attention
        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: attend within each frame  [B*T, N, D]
            x = x.reshape(B * T, self.num_patches, self.embed_dim)
            x = s_block(x)

            # Temporal: attend across frames per patch  [B*N, T, D]
            x = x.reshape(B, T, self.num_patches, self.embed_dim)
            x = x.permute(0, 2, 1, 3).reshape(B * self.num_patches, T, self.embed_dim)
            x = t_block(x)
            x = x.reshape(B, self.num_patches, T, self.embed_dim).permute(0, 2, 1, 3)

        # Take last frame's tokens
        x = x[:, -1, :, :]  # [B, N, D]
        x = self.final_norm(x)

        # Decode to pixel space
        if self._output_head_type == 'conv_decoder':
            x = x.transpose(1, 2).reshape(B, self.embed_dim, self.grid_h, self.grid_w)
            initial_state = self.output_head(x)
        else:
            x = self.output_head(x)          # [B, N, p*p*C_out]
            initial_state = self._unpatchify(x)

        # Crop/interpolate if image size doesn't evenly divide by patch_size
        if initial_state.shape[-2:] != (H, W):
            initial_state = F.interpolate(initial_state, size=(H, W), mode='bilinear', align_corners=False)

        initial_state = initial_state * self.output_scale + self.output_bias
        return initial_state


class ConditionalFlowMatchingARBootstrap(nn.Module):
    """
    Autoregressive Flow Matching model with Bootstrap Initialization.
    
    This model can operate in two modes:
    1. Bootstrap mode: Infer initial state from conditioning history
    2. AR mode: Predict next state using previous state
    
    The model uses an availability mask to distinguish between modes,
    allowing end-to-end training of both capabilities.
    
    Args:
        unet: The UNet velocity field predictor
        history_encoder: The history encoder for bootstrap mode
        use_availability_mask: Whether to use availability mask channel
    """
    
    def __init__(
        self, 
        unet: FlowMatchingUNet,
        history_encoder: HistoryEncoder,
        use_availability_mask: bool = True,
    ):
        super().__init__()
        self.unet = unet
        self.history_encoder = history_encoder
        self.use_availability_mask = use_availability_mask
        
    def compute_conditional_flow(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow path using optimal transport (linear interpolation).
        """
        t = t.view(-1, 1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        velocity_target = x_1 - x_0
        return x_t, velocity_target
    
    def forward(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        t: torch.Tensor,
        availability_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity field given current state, conditioning, and previous output.
        
        Args:
            x_t: Current noisy state [B, C_out, H, W]
            condition: Current conditioning (SDF, interface vel) [B, C_cond, H, W]
            prev_output: Previous timestep output [B, C_out, H, W]
                        (from history encoder in bootstrap mode, or actual prev in AR mode)
            t: Time values in [0, 1] [B]
            availability_mask: Optional [B, 1, H, W] indicating if prev_output is real (1) or bootstrapped (0)
            
        Returns:
            Predicted velocity field [B, C_out, H, W]
        """
        # Build input: [x_t, condition, prev_output]
        if self.use_availability_mask and availability_mask is not None:
            # Add availability mask as extra channel
            x_input = torch.cat([x_t, condition, prev_output, availability_mask], dim=1)
        else:
            x_input = torch.cat([x_t, condition, prev_output], dim=1)
            
        return self.unet(x_input, t)
    
    def bootstrap_initial_state(
        self,
        conditioning_history: torch.Tensor,
        current_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Bootstrap initial state from conditioning history.
        
        Args:
            conditioning_history: [B, T, C_cond, H, W]
            current_conditioning: [B, C_cond, H, W] (optional)
            
        Returns:
            initial_state: [B, C_out, H, W]
        """
        return self.history_encoder(conditioning_history, current_conditioning)
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        num_integration_steps: int = 50,
        solver: str = 'euler',
        availability_mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples using ODE integration.
        
        Args:
            condition: Current conditioning [B, C_cond, H, W]
            prev_output: Previous timestep output [B, C_out, H, W]
            shape: Output shape (B, C_out, H, W)
            device: Device for generation
            num_integration_steps: Number of integration steps
            solver: ODE solver - 'euler', 'heun', 'midpoint', or 'rk4'
            availability_mask: [B, 1, H, W] indicating real (1) vs bootstrapped (0) prev_output
            guidance_scale: CFG scale (1.0 = no guidance)
            
        Returns:
            Generated samples [B, C_out, H, W]
        """
        # Start from noise
        x = torch.randn(shape, device=device)
        
        dt = 1.0 / num_integration_steps
        
        for step in range(num_integration_steps):
            t = step * dt
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.float32)
            
            if solver == 'euler':
                velocity = self(x, condition, prev_output, t_tensor, availability_mask)
                x = x + velocity * dt
                
            elif solver == 'heun':
                v1 = self(x, condition, prev_output, t_tensor, availability_mask)
                x_pred = x + v1 * dt
                
                t_next = torch.full((shape[0],), min(t + dt, 1.0), device=device, dtype=torch.float32)
                v2 = self(x_pred, condition, prev_output, t_next, availability_mask)
                
                x = x + 0.5 * (v1 + v2) * dt
                
            elif solver == 'midpoint':
                v1 = self(x, condition, prev_output, t_tensor, availability_mask)
                x_mid = x + v1 * (dt / 2)
                
                t_mid = torch.full((shape[0],), t + dt/2, device=device, dtype=torch.float32)
                v_mid = self(x_mid, condition, prev_output, t_mid, availability_mask)
                x = x + v_mid * dt
                
            elif solver == 'rk4':
                t_half = torch.full((shape[0],), t + dt/2, device=device, dtype=torch.float32)
                t_next = torch.full((shape[0],), min(t + dt, 1.0), device=device, dtype=torch.float32)
                
                k1 = self(x, condition, prev_output, t_tensor, availability_mask)
                k2 = self(x + k1 * dt/2, condition, prev_output, t_half, availability_mask)
                k3 = self(x + k2 * dt/2, condition, prev_output, t_half, availability_mask)
                k4 = self(x + k3 * dt, condition, prev_output, t_next, availability_mask)
                
                x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)
            else:
                raise ValueError(f"Unknown solver: {solver}")
        
        return x
    
    @torch.no_grad()
    def sample_trajectory(
        self,
        conditions: torch.Tensor,
        conditioning_history: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        num_integration_steps: int = 50,
        solver: str = 'heun',
    ) -> torch.Tensor:
        """
        Generate a full trajectory with automatic bootstrap initialization.
        
        This is the main inference method. If initial_state is None, uses
        conditioning_history to bootstrap the initial state.
        
        Args:
            conditions: Conditioning for all timesteps [T, B, C_cond, H, W]
            conditioning_history: History for bootstrap [B, T_hist, C_cond, H, W]
                                 Required if initial_state is None
            initial_state: Optional initial state [B, C_out, H, W]
                          If None, bootstrapped from history
            num_integration_steps: Steps per frame
            solver: ODE solver
            
        Returns:
            Generated trajectory [T, B, C_out, H, W]
        """
        T = conditions.shape[0]
        device = conditions.device
        B = conditions.shape[1]
        H, W = conditions.shape[3], conditions.shape[4]
        C_out = self.history_encoder.out_channels
        
        # Determine initial state
        if initial_state is None:
            if conditioning_history is None:
                raise ValueError(
                    "Either initial_state or conditioning_history must be provided. "
                    "For bootstrap mode, provide conditioning_history."
                )
            # Bootstrap initial state
            prev_output = self.bootstrap_initial_state(
                conditioning_history, 
                current_conditioning=conditions[0]
            )
            is_bootstrapped = True
        else:
            prev_output = initial_state
            is_bootstrapped = False
        
        trajectory = []
        
        for t in range(T):
            condition_t = conditions[t]  # [B, C_cond, H, W]
            
            # Create availability mask
            if self.use_availability_mask:
                if t == 0 and is_bootstrapped:
                    # First frame with bootstrapped prev_output
                    availability_mask = torch.zeros(B, 1, H, W, device=device)
                else:
                    # Subsequent frames with real previous output
                    availability_mask = torch.ones(B, 1, H, W, device=device)
            else:
                availability_mask = None
            
            # Generate output at timestep t
            output_t = self.sample(
                condition_t,
                prev_output,
                (B, C_out, H, W),
                device,
                num_integration_steps,
                solver=solver,
                availability_mask=availability_mask,
            )
            
            trajectory.append(output_t)
            prev_output = output_t
        
        return torch.stack(trajectory, dim=0)


class ConditionalFlowMatchingARBootstrapLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for AR Flow Matching with Bootstrap.
    
    Training Strategy:
    - Sample trajectory segments of length rollout_length
    - First frame uses bootstrap mode (infer initial state from history)
    - Subsequent frames use AR mode with teacher forcing or scheduled sampling
    - Both bootstrap and AR losses trained jointly
    
    The model learns to:
    1. Infer reasonable initial states from conditioning history (bootstrap)
    2. Predict accurate state transitions given previous state (AR)
    
    Args:
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        scheduler_cfg: Learning rate scheduler configuration
        task_cfg: Task configuration defining channels
        normalization_stats: Normalization statistics
    """
    
    def __init__(
        self,
        model_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        task_cfg: Optional[DictConfig] = None,
        normalization_stats: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])
        
        self.normalization_stats = normalization_stats
        self.task_cfg = task_cfg
        
        # Parse task config for channel indices
        if task_cfg is not None:
            self.conditioning_channels = list(task_cfg.get('conditioning_channels', [0]))
            self.target_channels = list(task_cfg.get('target_channels', [0]))
            self.task_name = task_cfg.get('name', 'unknown')
            
            print(f"🎯 Task: {self.task_name}")
            print(f"   Conditioning channels: {self.conditioning_channels}")
            print(f"   Target channels: {self.target_channels}")
        else:
            self.conditioning_channels = [0]
            self.target_channels = [0]
            self.task_name = 'temperature_from_sdf'
        
        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)
        
        # Bootstrap configuration
        self.history_length = model_cfg.get('history_length', 10)
        self.history_stride = model_cfg.get('history_stride', 1)
        self.rollout_length = model_cfg.get('rollout_length', 5)
        self.use_availability_mask = model_cfg.get('use_availability_mask', True)
        
        # Loss weights
        self.bootstrap_loss_weight = model_cfg.get('bootstrap_loss_weight', 1.0)
        self.ar_loss_weight = model_cfg.get('ar_loss_weight', 1.0)
        
        # AR temporal decay: weight early frames more (exponential decay)
        # This helps "kill spin-up" by prioritizing early frame accuracy
        ar_decay_cfg = model_cfg.get('ar_temporal_decay', {})
        self.ar_temporal_decay_enabled = ar_decay_cfg.get('enabled', False)
        self.ar_temporal_decay_gamma = ar_decay_cfg.get('gamma', 0.85)
        
        print(f"\n🚀 Bootstrap AR Flow Matching Configuration:")
        print(f"   History length: {self.history_length}")
        print(f"   History stride: {self.history_stride} (spans {self.history_length * self.history_stride} timesteps)")
        print(f"   Rollout length: {self.rollout_length}")
        print(f"   Use availability mask: {self.use_availability_mask}")
        print(f"   Bootstrap loss weight: {self.bootstrap_loss_weight}")
        print(f"   AR loss weight: {self.ar_loss_weight}")
        if self.ar_temporal_decay_enabled:
            print(f"   📉 AR temporal decay: ENABLED (gamma={self.ar_temporal_decay_gamma})")
            print(f"      Frame weights: 1={1.0:.2f}, 2={self.ar_temporal_decay_gamma:.2f}, 3={self.ar_temporal_decay_gamma**2:.2f}, 4={self.ar_temporal_decay_gamma**3:.2f}")
        else:
            print(f"   📉 AR temporal decay: DISABLED (uniform weights)")
        
        # UNet configuration
        # in_channels = num_target_channels (x_t) + num_conditioning_channels + num_target_channels (prev) + 1 (mask)
        extra_mask_channels = 1 if self.use_availability_mask else 0
        in_channels = (self.num_target_channels + self.num_conditioning_channels + 
                      self.num_target_channels + extra_mask_channels)
        out_channels = self.num_target_channels
        
        print(f"   UNet in_channels: {in_channels}")
        print(f"   UNet out_channels: {out_channels}")
        
        # Initialize UNet
        # Handle both old (use_attention: bool) and new (attention_type: str) configs
        use_attention = model_cfg.get('use_attention', False)
        attention_type = model_cfg.get('attention_type', 'bottleneck' if use_attention else 'none')
        
        unet = FlowMatchingUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=model_cfg.get('base_channels', 64),
            time_embed_dim=model_cfg.get('time_embed_dim', 320),
            num_res_blocks=model_cfg.get('num_res_blocks', 2),
            attention_type=attention_type,
            dropout=model_cfg.get('dropout', 0.0),
        )
        
        # Initialize history encoder (choose between Conv3D, TemporalMixer, or Attention)
        history_encoder_type = model_cfg.get('history_encoder_type', 'conv3d')
        
        if history_encoder_type == 'temporal_mixer':
            # Fast temporal mixer: ~3-5x faster than Conv3D
            history_encoder = TemporalMixer(
                in_channels=self.num_conditioning_channels,
                out_channels=self.num_target_channels,
                history_length=self.history_length,
                hidden_channels=model_cfg.get('history_encoder_hidden', 32),
                use_spatial_conv=model_cfg.get('temporal_mixer_spatial_conv', True),
                use_temporal_weights=model_cfg.get('temporal_mixer_temporal_weights', True),
            )
            print(f"   🚀 History Encoder: TemporalMixer (fast)")
            print(f"      Hidden channels: {model_cfg.get('history_encoder_hidden', 32)}")
            print(f"      Spatial conv: {model_cfg.get('temporal_mixer_spatial_conv', True)}")
            print(f"      Temporal weights: {model_cfg.get('temporal_mixer_temporal_weights', True)}")
        elif history_encoder_type == 'attention':
            img_size = model_cfg.get('img_size', 128)
            history_encoder = AttentionHistoryEncoder(
                in_channels=self.num_conditioning_channels,
                out_channels=self.num_target_channels,
                embed_dim=model_cfg.get('attention_encoder_embed_dim', 256),
                num_heads=model_cfg.get('attention_encoder_num_heads', 8),
                depth=model_cfg.get('attention_encoder_depth', 4),
                patch_size=model_cfg.get('attention_encoder_patch_size', 8),
                img_size=img_size,
                max_history_length=model_cfg.get('attention_encoder_max_history_length', 50),
                mlp_ratio=model_cfg.get('attention_encoder_mlp_ratio', 4.0),
                dropout=model_cfg.get('attention_encoder_dropout', 0.0),
                output_head_type=model_cfg.get('attention_encoder_output_head', 'linear'),
            )
            print(f"   🧠 History Encoder: Attention (factored space-time transformer)")
            print(f"      Embed dim: {model_cfg.get('attention_encoder_embed_dim', 256)}")
            print(f"      Heads: {model_cfg.get('attention_encoder_num_heads', 8)}")
            print(f"      Depth: {model_cfg.get('attention_encoder_depth', 4)}")
            print(f"      Patch size: {model_cfg.get('attention_encoder_patch_size', 8)}")
            print(f"      Image size: {img_size}")
            print(f"      Max history length: {model_cfg.get('attention_encoder_max_history_length', 50)}")
            print(f"      Output head: {model_cfg.get('attention_encoder_output_head', 'linear')}")
        else:
            # Default: Conv3D-based history encoder (more expressive but slower)
            history_encoder = HistoryEncoder(
                in_channels=self.num_conditioning_channels,
                out_channels=self.num_target_channels,
                hidden_channels=model_cfg.get('history_encoder_hidden', 64),
                num_temporal_blocks=model_cfg.get('history_encoder_blocks', 3),
            )
            print(f"   🧠 History Encoder: Conv3D (expressive)")
            print(f"      Hidden channels: {model_cfg.get('history_encoder_hidden', 64)}")
            print(f"      Temporal blocks: {model_cfg.get('history_encoder_blocks', 3)}")
        
        # Initialize combined model
        self.flow_matching = ConditionalFlowMatchingARBootstrap(
            unet=unet,
            history_encoder=history_encoder,
            use_availability_mask=self.use_availability_mask,
        )
        
        # Loss functions
        self.loss_fn = nn.MSELoss()
        
        # Auxiliary losses
        loss_cfg = model_cfg.get('auxiliary_losses', {})
        
        self.use_spectral_loss = loss_cfg.get('spectral_enabled', False)
        if self.use_spectral_loss:
            self.spectral_loss = SpectralLoss(
                weight=loss_cfg.get('spectral_weight', 0.1),
            )
            print(f"📊 Spectral Loss: ENABLED")
        else:
            self.spectral_loss = None
        
        self.use_gradient_loss = loss_cfg.get('gradient_enabled', False)
        if self.use_gradient_loss:
            self.gradient_loss = GradientLoss(
                weight=loss_cfg.get('gradient_weight', 0.1),
            )
            print(f"📐 Gradient Loss: ENABLED")
        else:
            self.gradient_loss = None
        
        # Bootstrap state supervision loss
        # Direct supervision on the bootstrapped initial state
        self.bootstrap_state_loss_weight = model_cfg.get('bootstrap_state_loss_weight', 0.5)
        print(f"🎯 Bootstrap state supervision weight: {self.bootstrap_state_loss_weight}")
        
        # Inference configuration
        inference_cfg = model_cfg.get('inference', {})
        self.default_solver = inference_cfg.get('solver', 'heun')
        self.num_integration_steps = model_cfg.get('num_integration_steps', 50)
        
        # Normalization parameters
        if normalization_stats is not None and 'temperature' in normalization_stats:
            temp_stats = normalization_stats['temperature']
            self.temp_min = temp_stats['min']
            self.temp_max = temp_stats['max']
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = normalization_stats.get('unified_velocity_scale', 1.0)
        else:
            self.temp_min = 55.0
            self.temp_max = 120.0
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
        
        # Scheduled sampling (for AR mode)
        ss_cfg = model_cfg.get('scheduled_sampling', {})
        self.scheduled_sampling_enabled = ss_cfg.get('enabled', False)
        if self.scheduled_sampling_enabled:
            self.ss_schedule_type = ss_cfg.get('schedule_type', 'linear')
            self.ss_warmup_epochs = ss_cfg.get('warmup_epochs', 5)
            self.ss_transition_epochs = ss_cfg.get('transition_epochs', 40)
            self.ss_min_teacher_ratio = ss_cfg.get('min_teacher_ratio', 0.0)
            self.ss_exponential_decay = ss_cfg.get('exponential_decay_rate', 0.95)
            self.ss_sigmoid_k = ss_cfg.get('sigmoid_k', 5.0)
            self.ss_sampling_steps = ss_cfg.get('sampling_steps', 20)
            print(f"\n📊 Scheduled Sampling: ENABLED")
            print(f"   Schedule type: {self.ss_schedule_type}")
            print(f"   Warmup epochs: {self.ss_warmup_epochs} (pure teacher forcing)")
            print(f"   Transition epochs: {self.ss_transition_epochs}")
            print(f"   Final teacher ratio: {self.ss_min_teacher_ratio:.1%}")
            print(f"   Sampling steps for predictions: {self.ss_sampling_steps}")
        else:
            print(f"\n📊 Scheduled Sampling: DISABLED")
        
        # Push Forward Trick (alternative to scheduled sampling)
        pf_cfg = model_cfg.get('push_forward', {})
        self.push_forward_enabled = pf_cfg.get('enabled', False)
        if self.push_forward_enabled:
            self.pf_warmup_epochs = pf_cfg.get('warmup_epochs', 3)
            self.pf_max_push_steps = pf_cfg.get('max_push_steps', 3)
            self.pf_step_increase_epochs = pf_cfg.get('step_increase_epochs', 10)
            self.pf_sampling_steps = pf_cfg.get('sampling_steps', 15)
            self.pf_loss_on_all_pushed = pf_cfg.get('loss_on_all_pushed', True)
            self.pf_detach_pushed = pf_cfg.get('detach_pushed', False)
            print(f"\n🚀 Push Forward Trick: ENABLED")
            print(f"   Warmup epochs: {self.pf_warmup_epochs} (pure teacher forcing)")
            print(f"   Max push steps: {self.pf_max_push_steps}")
            print(f"   Step increase every: {self.pf_step_increase_epochs} epochs")
            print(f"   Sampling steps: {self.pf_sampling_steps}")
            print(f"   Loss on all pushed frames: {self.pf_loss_on_all_pushed}")
            print(f"   Detach pushed predictions: {self.pf_detach_pushed}")
            
            # Warn if both scheduled sampling and push forward are enabled
            if self.scheduled_sampling_enabled:
                print(f"   ⚠️  WARNING: Both scheduled sampling and push forward are enabled!")
                print(f"   ⚠️  This is not recommended - push forward will take precedence.")
                # Disable scheduled sampling when push forward is enabled
                self.scheduled_sampling_enabled = False
        else:
            print(f"🚀 Push Forward Trick: DISABLED")
        
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        
    def forward(self, x_t, condition, prev_output, t, availability_mask=None):
        return self.flow_matching(x_t, condition, prev_output, t, availability_mask)
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        availability_mask: Optional[torch.Tensor] = None,
        num_integration_steps: Optional[int] = None,
        solver: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate samples."""
        if num_integration_steps is None:
            num_integration_steps = self.num_integration_steps
        if solver is None:
            solver = self.default_solver
            
        return self.flow_matching.sample(
            condition, prev_output, shape, device,
            num_integration_steps, solver, availability_mask
        )
    
    @torch.no_grad()
    def bootstrap_initial_state(
        self,
        conditioning_history: torch.Tensor,
        current_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bootstrap initial state from conditioning history."""
        return self.flow_matching.bootstrap_initial_state(
            conditioning_history, current_conditioning
        )
    
    @torch.no_grad()
    def sample_trajectory(
        self,
        conditions: torch.Tensor,
        conditioning_history: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        num_integration_steps: Optional[int] = None,
        solver: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate full trajectory with automatic bootstrap."""
        if num_integration_steps is None:
            num_integration_steps = self.num_integration_steps
        if solver is None:
            solver = self.default_solver
            
        return self.flow_matching.sample_trajectory(
            conditions, conditioning_history, initial_state,
            num_integration_steps, solver
        )
    
    def denormalize_temperature(self, temperature_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize temperature from [-1, 1] to Celsius."""
        return (temperature_norm + 1.0) / 2.0 * self.temp_range + self.temp_min
    
    def _extract_channels(self, tensor: torch.Tensor, channel_indices: list) -> torch.Tensor:
        """Extract specific channels from a tensor [B, C, H, W] or [B, T, C, H, W]."""
        if tensor.dim() == 4:
            return tensor[:, channel_indices, :, :]
        elif tensor.dim() == 5:
            return tensor[:, :, channel_indices, :, :]
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    def get_teacher_forcing_ratio(self) -> float:
        """
        Compute teacher forcing ratio for scheduled sampling.
        
        Supports multiple schedule types:
        - linear: Linear decay from 1.0 to min_ratio
        - exponential: Exponential decay with configurable rate
        - inverse_sigmoid: Smooth S-curve transition
        
        Returns:
            Teacher forcing ratio between 0 and 1 (1 = pure teacher forcing)
        """
        if not self.scheduled_sampling_enabled:
            return 1.0  # Pure teacher forcing
        
        current_epoch = self.current_epoch
        
        # During warmup: pure teacher forcing
        if current_epoch < self.ss_warmup_epochs:
            return 1.0
        
        # Epochs into the transition phase
        transition_epoch = current_epoch - self.ss_warmup_epochs
        
        # After transition: minimum teacher ratio
        if transition_epoch >= self.ss_transition_epochs:
            return self.ss_min_teacher_ratio
        
        # Progress through transition (0 to 1)
        progress = transition_epoch / self.ss_transition_epochs
        
        if self.ss_schedule_type == 'linear':
            # Linear decay: ratio = 1 - progress * (1 - min_ratio)
            teacher_ratio = 1.0 - progress * (1.0 - self.ss_min_teacher_ratio)
        
        elif self.ss_schedule_type == 'exponential':
            # Exponential decay: ratio = max(min_ratio, decay^epoch)
            teacher_ratio = max(
                self.ss_min_teacher_ratio,
                self.ss_exponential_decay ** transition_epoch
            )
        
        elif self.ss_schedule_type == 'inverse_sigmoid':
            # Inverse sigmoid: smooth S-curve transition
            # Starts slow, accelerates in middle, slows at end
            k = self.ss_sigmoid_k
            # Map progress to sigmoid input centered at 0.5
            x = k * (progress - 0.5)
            sigmoid_val = 1.0 / (1.0 + math.exp(-x))
            # Map from [sigmoid(-k/2), sigmoid(k/2)] to [1, min_ratio]
            teacher_ratio = 1.0 - sigmoid_val * (1.0 - self.ss_min_teacher_ratio)
        
        else:
            # Default to linear
            teacher_ratio = 1.0 - progress * (1.0 - self.ss_min_teacher_ratio)
        
        return max(self.ss_min_teacher_ratio, min(1.0, teacher_ratio))
    
    def get_push_forward_steps(self) -> int:
        """
        Compute number of push forward steps for current epoch.
        
        The push forward trick gradually increases the number of steps
        where the model uses its own predictions instead of ground truth.
        
        Returns:
            Number of push forward steps (0 = pure teacher forcing)
        """
        if not self.push_forward_enabled:
            return 0
        
        current_epoch = self.current_epoch
        
        # During warmup: pure teacher forcing (0 push steps)
        if current_epoch < self.pf_warmup_epochs:
            return 0
        
        # Calculate number of push steps based on epochs since warmup
        epochs_since_warmup = current_epoch - self.pf_warmup_epochs
        
        # Increase push steps every pf_step_increase_epochs
        push_steps = 1 + (epochs_since_warmup // self.pf_step_increase_epochs)
        
        # Clamp to max push steps (and ensure it doesn't exceed rollout_length - 1)
        push_steps = min(push_steps, self.pf_max_push_steps, self.rollout_length - 1)
        
        return push_steps
    
    def training_step(self, batch, batch_idx):
        """
        Training step with joint bootstrap and AR training.
        
        Batch contains a trajectory segment:
        - conditioning_history: [B, T_hist, C_cond, H, W] - for bootstrap
        - conditioning_sequence: [B, L, C_cond, H, W] - rollout conditioning
        - target_sequence: [B, L, C_out, H, W] - rollout targets
        """
        # Unpack batch
        conditioning_history, conditioning_sequence, target_sequence = batch[:3]
        
        # Extract relevant channels
        cond_hist = self._extract_channels(conditioning_history, self.conditioning_channels)
        cond_seq = self._extract_channels(conditioning_sequence, self.conditioning_channels)
        target_seq = self._extract_channels(target_sequence, self.target_channels)
        
        B, L, C_cond, H, W = cond_seq.shape
        C_out = target_seq.shape[2]
        device = cond_seq.device
        
        total_loss = 0.0
        bootstrap_loss_total = 0.0
        ar_loss_total = 0.0
        
        # ==========================================
        # BOOTSTRAP TRAINING (first frame)
        # ==========================================
        # Bootstrap initial state from conditioning history
        current_cond_0 = cond_seq[:, 0]  # [B, C_cond, H, W]
        bootstrapped_state = self.flow_matching.bootstrap_initial_state(
            cond_hist, current_cond_0
        )
        
        # Direct supervision on bootstrapped state
        target_0 = target_seq[:, 0]  # [B, C_out, H, W]
        bootstrap_state_loss = self.loss_fn(bootstrapped_state, target_0)
        bootstrap_loss_total = bootstrap_loss_total + bootstrap_state_loss * self.bootstrap_state_loss_weight
        
        # Flow matching loss for first frame (bootstrap mode)
        t_0 = torch.rand(B, device=device)
        x_0_noise = torch.randn_like(target_0)
        x_t_0, velocity_target_0 = self.flow_matching.compute_conditional_flow(x_0_noise, target_0, t_0)
        
        # Create availability mask (0 = bootstrapped)
        if self.use_availability_mask:
            availability_mask_0 = torch.zeros(B, 1, H, W, device=device)
        else:
            availability_mask_0 = None
        
        velocity_pred_0 = self.flow_matching(
            x_t_0, current_cond_0, bootstrapped_state, t_0, availability_mask_0
        )
        bootstrap_fm_loss = self.loss_fn(velocity_pred_0, velocity_target_0)
        bootstrap_loss_total = bootstrap_loss_total + bootstrap_fm_loss * self.bootstrap_loss_weight
        
        # ==========================================
        # AR TRAINING (subsequent frames)
        # ==========================================
        teacher_ratio = self.get_teacher_forcing_ratio()
        push_steps = self.get_push_forward_steps()
        
        # ROLLOUT-AWARE BOOTSTRAP: Start AR training from bootstrapped_state
        # instead of GT. This way:
        # 1. Frame 1 sees the actual bootstrap quality (not perfect GT)
        # 2. Gradients flow back to HistoryEncoder, improving bootstrap
        # 3. Training better matches inference where frame 1 sees imperfect prev
        # Note: We DON'T detach so gradients flow to history encoder
        prev_output = bootstrapped_state  # Rollout-aware: let AR see bootstrap quality
        
        # Track weight sum for temporal decay normalization
        ar_weight_sum = 0.0
        
        # ==========================================
        # PUSH FORWARD TRICK (if enabled)
        # ==========================================
        # Push forward: Generate predictions for push_steps frames using model's
        # own predictions, then compute loss on those frames.
        if self.push_forward_enabled and push_steps > 0:
            # List to store pushed predictions and their targets
            pushed_predictions = []
            pushed_targets = []
            pushed_conditions = []
            
            # Generate predictions for push_steps frames using model's own output
            current_prev = prev_output
            for push_l in range(1, min(push_steps + 1, L)):
                current_cond = cond_seq[:, push_l]  # [B, C_cond, H, W]
                target_l = target_seq[:, push_l]    # [B, C_out, H, W]
                
                if self.use_availability_mask:
                    availability_mask = torch.ones(B, 1, H, W, device=device)
                else:
                    availability_mask = None
                
                # Generate prediction using model's own previous output
                # Note: self.sample() has @torch.no_grad() decorator, so gradients
                # don't flow through the sampling process itself. The push forward
                # trick still helps because:
                # 1. The model sees its own error distribution during training
                # 2. Loss is computed on velocity predictions conditioned on pushed frames
                # 3. This teaches the model to be robust to imperfect previous states
                pred_l = self.sample(
                    current_cond, current_prev,
                    (B, C_out, H, W), device,
                    availability_mask=availability_mask,
                    num_integration_steps=self.pf_sampling_steps,
                    solver='euler',
                )
                
                pushed_predictions.append(pred_l)
                pushed_targets.append(target_l)
                pushed_conditions.append(current_cond)
                
                # Update current_prev to use model's prediction
                current_prev = pred_l.detach() if self.pf_detach_pushed else pred_l
            
            # Compute loss on pushed frames
            if self.pf_loss_on_all_pushed:
                # Loss on all pushed frames
                for push_idx, (pred, target, cond) in enumerate(zip(pushed_predictions, pushed_targets, pushed_conditions)):
                    frame_l = push_idx + 1  # l=1, 2, ...
                    
                    # Flow matching loss on the pushed prediction
                    t_l = torch.rand(B, device=device)
                    x_0_noise = torch.randn_like(target)
                    x_t_l, velocity_target_l = self.flow_matching.compute_conditional_flow(
                        x_0_noise, target, t_l
                    )
                    
                    if self.use_availability_mask:
                        availability_mask = torch.ones(B, 1, H, W, device=device)
                    else:
                        availability_mask = None
                    
                    # Use the pushed prediction as prev_output for velocity prediction
                    prev_for_loss = pushed_predictions[push_idx - 1].detach() if push_idx > 0 else bootstrapped_state
                    velocity_pred_l = self.flow_matching(
                        x_t_l, cond, prev_for_loss, t_l, availability_mask
                    )
                    
                    push_loss_l = self.loss_fn(velocity_pred_l, velocity_target_l)
                    
                    # Apply temporal decay weighting if enabled
                    if self.ar_temporal_decay_enabled:
                        frame_weight = self.ar_temporal_decay_gamma ** (frame_l - 1)
                        ar_loss_total = ar_loss_total + push_loss_l * frame_weight
                        ar_weight_sum = ar_weight_sum + frame_weight
                    else:
                        ar_loss_total = ar_loss_total + push_loss_l
            else:
                # Loss only on the last pushed frame
                pred = pushed_predictions[-1]
                target = pushed_targets[-1]
                cond = pushed_conditions[-1]
                frame_l = len(pushed_predictions)
                
                t_l = torch.rand(B, device=device)
                x_0_noise = torch.randn_like(target)
                x_t_l, velocity_target_l = self.flow_matching.compute_conditional_flow(
                    x_0_noise, target, t_l
                )
                
                if self.use_availability_mask:
                    availability_mask = torch.ones(B, 1, H, W, device=device)
                else:
                    availability_mask = None
                
                prev_for_loss = pushed_predictions[-2].detach() if len(pushed_predictions) > 1 else bootstrapped_state
                velocity_pred_l = self.flow_matching(
                    x_t_l, cond, prev_for_loss, t_l, availability_mask
                )
                
                push_loss_l = self.loss_fn(velocity_pred_l, velocity_target_l)
                
                if self.ar_temporal_decay_enabled:
                    frame_weight = self.ar_temporal_decay_gamma ** (frame_l - 1)
                    ar_loss_total = ar_loss_total + push_loss_l * frame_weight
                    ar_weight_sum = ar_weight_sum + frame_weight
                else:
                    ar_loss_total = ar_loss_total + push_loss_l
            
            # Continue with teacher forcing for remaining frames (after push_steps)
            # Update prev_output to the last GT frame before the remaining frames
            if push_steps < L - 1:
                prev_output = target_seq[:, push_steps].clone()
            
            # Start from frame after push_steps
            start_frame = push_steps + 1
        else:
            # No push forward - start from frame 1
            start_frame = 1
        
        # ==========================================
        # STANDARD AR TRAINING (remaining frames)
        # ==========================================
        for l in range(start_frame, L):
            current_cond = cond_seq[:, l]  # [B, C_cond, H, W]
            target_l = target_seq[:, l]    # [B, C_out, H, W]
            
            # ============================================================
            # STEP 1: Compute loss using current prev_output (frame l-1)
            # ============================================================
            # prev_output here is frame l-1 (either GT or prediction from previous iteration)
            
            # Sample time and compute flow
            t_l = torch.rand(B, device=device)
            x_0_noise = torch.randn_like(target_l)
            x_t_l, velocity_target_l = self.flow_matching.compute_conditional_flow(
                x_0_noise, target_l, t_l
            )
            
            # Create availability mask (1 = real prev_output)
            if self.use_availability_mask:
                availability_mask = torch.ones(B, 1, H, W, device=device)
            else:
                availability_mask = None
            
            # Predict velocity using prev_output (frame l-1) to predict frame l
            velocity_pred_l = self.flow_matching(
                x_t_l, current_cond, prev_output, t_l, availability_mask
            )
            
            ar_loss_l = self.loss_fn(velocity_pred_l, velocity_target_l)
            
            # Apply temporal decay weighting if enabled
            # Earlier frames (l=1,2) get higher weight than later frames (l=3,4)
            if self.ar_temporal_decay_enabled:
                frame_weight = self.ar_temporal_decay_gamma ** (l - 1)  # l=1->1.0, l=2->gamma, l=3->gamma^2
                ar_loss_total = ar_loss_total + ar_loss_l * frame_weight
                ar_weight_sum = ar_weight_sum + frame_weight
            else:
                ar_loss_total = ar_loss_total + ar_loss_l
            
            # ============================================================
            # STEP 2: Update prev_output for NEXT iteration
            # ============================================================
            # Decide what frame l+1 will see as its "previous frame":
            # - Teacher forcing: use GT frame l
            # - Scheduled sampling: use model's prediction of frame l
            
            if self.scheduled_sampling_enabled and teacher_ratio < 1.0:
                use_teacher = torch.rand(1).item() < teacher_ratio
                if not use_teacher:
                    # Generate prediction for frame l to use as prev_output for frame l+1
                    with torch.no_grad():
                        prev_output = self.sample(
                            current_cond, prev_output,  # Use current prev_output (frame l-1) to predict frame l
                            (B, C_out, H, W), device,
                            availability_mask=availability_mask,
                            num_integration_steps=self.ss_sampling_steps,
                            solver='euler',
                        )
                else:
                    # Teacher forcing: use GT frame l
                    prev_output = target_l.clone()
            else:
                # Pure teacher forcing: always use GT frame l
                prev_output = target_l.clone()
        
        # Average AR loss over rollout (normalize by weight sum if using decay)
        if L > 1:
            if self.ar_temporal_decay_enabled:
                # ar_weight_sum already accounts for all frames (pushed + remaining)
                if ar_weight_sum > 0:
                    ar_loss_total = ar_loss_total / ar_weight_sum
            else:
                # Count total AR frames: pushed frames + remaining frames
                if self.push_forward_enabled and push_steps > 0:
                    if self.pf_loss_on_all_pushed:
                        num_ar_frames = min(push_steps, L - 1) + (L - 1 - start_frame + 1) if start_frame <= L - 1 else min(push_steps, L - 1)
                    else:
                        num_ar_frames = 1 + (L - 1 - start_frame + 1) if start_frame <= L - 1 else 1
                else:
                    num_ar_frames = L - 1
                if num_ar_frames > 0:
                    ar_loss_total = ar_loss_total / num_ar_frames
        
        # Combined loss
        total_loss = bootstrap_loss_total + ar_loss_total * self.ar_loss_weight
        
        # Auxiliary losses (on last frame for efficiency)
        if self.use_spectral_loss and self.spectral_loss is not None:
            spec_loss = self.spectral_loss(velocity_pred_l, velocity_target_l)
            total_loss = total_loss + spec_loss
            self.log('train_spectral_loss', spec_loss, on_step=False, on_epoch=True)
        
        if self.use_gradient_loss and self.gradient_loss is not None:
            grad_loss = self.gradient_loss(velocity_pred_l, velocity_target_l)
            total_loss = total_loss + grad_loss
            self.log('train_gradient_loss', grad_loss, on_step=False, on_epoch=True)
        
        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_bootstrap_loss', bootstrap_loss_total, on_step=False, on_epoch=True)
        self.log('train_bootstrap_state_loss', bootstrap_state_loss, on_step=False, on_epoch=True)
        self.log('train_ar_loss', ar_loss_total, on_step=False, on_epoch=True)
        
        if self.scheduled_sampling_enabled:
            self.log('teacher_ratio', teacher_ratio, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.push_forward_enabled:
            self.log('push_forward_steps', float(push_steps), on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step.
        
        Supports dual validation for Task 3 (noisy inputs):
        - dataloader_idx=0: clean inputs (physics fidelity)
        - dataloader_idx=1: noisy inputs (deployment performance)
        """
        conditioning_history, conditioning_sequence, target_sequence = batch[:3]
        
        cond_hist = self._extract_channels(conditioning_history, self.conditioning_channels)
        cond_seq = self._extract_channels(conditioning_sequence, self.conditioning_channels)
        target_seq = self._extract_channels(target_sequence, self.target_channels)
        
        B, L, C_cond, H, W = cond_seq.shape
        C_out = target_seq.shape[2]
        device = cond_seq.device
        
        # Determine suffix for dual validation (clean vs noisy)
        # dataloader_idx=0 → clean, dataloader_idx=1 → noisy
        is_clean_val = (dataloader_idx == 0)
        suffix = "_clean" if is_clean_val else "_noisy"
        
        # Bootstrap evaluation
        current_cond_0 = cond_seq[:, 0]
        bootstrapped_state = self.flow_matching.bootstrap_initial_state(
            cond_hist, current_cond_0
        )
        target_0 = target_seq[:, 0]
        bootstrap_state_loss = self.loss_fn(bootstrapped_state, target_0)
        
        # Flow matching loss for bootstrap
        t_0 = torch.rand(B, device=device)
        x_0_noise = torch.randn_like(target_0)
        x_t_0, velocity_target_0 = self.flow_matching.compute_conditional_flow(
            x_0_noise, target_0, t_0
        )
        
        if self.use_availability_mask:
            availability_mask_0 = torch.zeros(B, 1, H, W, device=device)
        else:
            availability_mask_0 = None
        
        velocity_pred_0 = self.flow_matching(
            x_t_0, current_cond_0, bootstrapped_state, t_0, availability_mask_0
        )
        bootstrap_fm_loss = self.loss_fn(velocity_pred_0, velocity_target_0)
        
        # AR evaluation (rollout-aware: start from bootstrap, not GT)
        ar_loss_total = 0.0
        ar_weight_sum = 0.0
        prev_output = bootstrapped_state  # Match training: start AR from bootstrap
        
        for l in range(1, L):
            current_cond = cond_seq[:, l]
            target_l = target_seq[:, l]
            
            t_l = torch.rand(B, device=device)
            x_0_noise = torch.randn_like(target_l)
            x_t_l, velocity_target_l = self.flow_matching.compute_conditional_flow(
                x_0_noise, target_l, t_l
            )
            
            if self.use_availability_mask:
                availability_mask = torch.ones(B, 1, H, W, device=device)
            else:
                availability_mask = None
            
            velocity_pred_l = self.flow_matching(
                x_t_l, current_cond, prev_output, t_l, availability_mask
            )
            
            ar_loss_l = self.loss_fn(velocity_pred_l, velocity_target_l)
            
            # Apply temporal decay weighting if enabled (match training)
            if self.ar_temporal_decay_enabled:
                frame_weight = self.ar_temporal_decay_gamma ** (l - 1)
                ar_loss_total = ar_loss_total + ar_loss_l * frame_weight
                ar_weight_sum = ar_weight_sum + frame_weight
            else:
                ar_loss_total = ar_loss_total + ar_loss_l
            
            prev_output = target_l
        
        if L > 1:
            if self.ar_temporal_decay_enabled:
                ar_loss_total = ar_loss_total / ar_weight_sum
            else:
                ar_loss_total = ar_loss_total / (L - 1)
        
        val_loss = bootstrap_fm_loss + bootstrap_state_loss * self.bootstrap_state_loss_weight + ar_loss_total
        
        # Logging with suffix for dual validation
        # Use clean validation loss as the primary val_loss for checkpointing
        if is_clean_val:
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log(f'val_loss{suffix}', val_loss, on_step=False, on_epoch=True, prog_bar=not is_clean_val)
        self.log(f'val_bootstrap_state_loss{suffix}', bootstrap_state_loss, on_step=False, on_epoch=True)
        self.log(f'val_bootstrap_fm_loss{suffix}', bootstrap_fm_loss, on_step=False, on_epoch=True)
        self.log(f'val_ar_loss{suffix}', ar_loss_total, on_step=False, on_epoch=True)
        
        # Generate samples for first batch (only for clean validation)
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(2, B)
                
                # Sample with bootstrap
                sample_0 = self.sample(
                    condition=current_cond_0[:num_samples],
                    prev_output=bootstrapped_state[:num_samples],
                    shape=(num_samples, C_out, H, W),
                    device=device,
                    availability_mask=availability_mask_0[:num_samples] if availability_mask_0 is not None else None,
                )
                
                self.log('val_sample_mean', sample_0.mean(), on_step=False, on_epoch=True)
                self.log('val_sample_std', sample_0.std(), on_step=False, on_epoch=True)
                self.log('val_bootstrap_mean', bootstrapped_state[:num_samples].mean(), on_step=False, on_epoch=True)
                self.log('val_bootstrap_std', bootstrapped_state[:num_samples].std(), on_step=False, on_epoch=True)
        
        return val_loss
    
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
