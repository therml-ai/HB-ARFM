"""
JiT (Just image Transformer) Flow Matching for BubbleFlow Prediction.

Adapts the JiT Vision Transformer architecture from:
  "Back to Basics: Let Denoising Generative Models Denoise" (Li & He, 2025)
  https://github.com/LTH14/JiT

Key features:
- Vision Transformer backbone with RoPE, SwiGLU FFN, AdaLN modulation, QK-norm
- Data prediction parameterization with velocity loss
- Log-normal time sampling for improved training dynamics
- Spatial conditioning via channel concatenation (adapted for physics problems)

Differences from original JiT:
- No class label conditioning (spatial conditioning via input channels instead)
- No in-context tokens
- Decoupled in_channels (target + conditioning) and out_channels (target only)
- Device-agnostic (no hardcoded .cuda() calls)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple
import numpy as np

from bubblefusion.models.loss_utils import build_loss_fn


# =============================================================================
# Utility functions (adapted from JiT util/model_util.py, no einops dependency)
# =============================================================================

def rotate_half(x):
    """Rotate pairs of elements for rotary position embeddings."""
    d = x.shape[-1]
    x_reshaped = x.reshape(*x.shape[:-1], d // 2, 2)
    x1, x2 = x_reshaped.unbind(dim=-1)
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.reshape(x.shape)


class VisionRotaryEmbeddingFast(nn.Module):
    """2D Rotary Position Embedding for vision transformers (device-agnostic)."""

    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len).float() / ft_seq_len * pt_seq_len

        freqs = torch.einsum('i,j->ij', t, freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)

        freqs_2d = torch.cat([
            freqs[:, None, :].expand(-1, ft_seq_len, -1),
            freqs[None, :, :].expand(ft_seq_len, -1, -1),
        ], dim=-1)

        freqs_flat = freqs_2d.reshape(-1, freqs_2d.shape[-1])
        self.register_buffer('freqs_cos', freqs_flat.cos())
        self.register_buffer('freqs_sin', freqs_flat.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed(embed_dim, pos):
    """Generate 1D sinusoidal positional embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# =============================================================================
# JiT Architecture Components
# =============================================================================

def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """Image to Patch Embedding with bottleneck projection."""

    def __init__(self, img_size=64, patch_size=8, in_chans=6,
                 pca_dim=64, embed_dim=384, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj1 = nn.Conv2d(in_chans, pca_dim,
                               kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim,
                               kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class Attention(nn.Module):
    """Multi-head attention with QK-norm and rotary position embeddings."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q.float() @ k.float().transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(
            attn_weight, self.attn_drop.p if self.training else 0.0, train=self.training
        )
        x = attn_weight @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class JiTBlock(nn.Module):
    """JiT transformer block with AdaLN modulation."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Final layer of JiT with AdaLN modulation."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# =============================================================================
# JiT Backbone (adapted for spatial conditioning, no class labels)
# =============================================================================

class JiTBackbone(nn.Module):
    """
    JiT Vision Transformer adapted for physics-based flow matching.

    Replaces class label conditioning with spatial conditioning via input
    channels.  Uses timestep-only AdaLN modulation.
    """

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=6,
        out_channels=3,
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_size = img_size

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.x_embedder = BottleneckPatchEmbed(
            img_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = img_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len
        )

        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
            )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """x: (N, T, patch_size**2 * C) -> (N, C, H, W)"""
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t):
        """
        Args:
            x: (N, C_in, H, W)  concatenated [x_t, conditioning]
            t: (N,) timestep values in [0, 1]
        Returns:
            (N, C_out, H, W) data prediction
        """
        c = self.t_embedder(t)

        x = self.x_embedder(x)
        x += self.pos_embed

        for block in self.blocks:
            x = block(x, c, self.feat_rope)

        x = self.final_layer(x, c)
        return self.unpatchify(x, self.patch_size)


# =============================================================================
# Flow Matching with Data Prediction + Velocity Loss
# =============================================================================

class ConditionalFlowMatchingJiT(nn.Module):
    """
    Flow Matching with JiT backbone using data prediction + velocity loss.

    The model predicts clean data x_1, and the loss is computed on the
    derived velocity:
        v_pred   = (x_pred - z_t) / (1-t)
        v_target = (x_1   - z_t) / (1-t)  =  x_1 - x_0

    This gives an implicit 1/(1-t)^2 weighting that upweights accuracy
    at late timesteps (fine details).
    """

    def __init__(self, backbone: JiTBackbone,
                 noise_scale: float = 1.0, t_eps: float = 1e-5):
        super().__init__()
        self.backbone = backbone
        self.noise_scale = noise_scale
        self.t_eps = t_eps

    def compute_conditional_flow(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """OT-CFM interpolation.  t=0 -> noise, t=1 -> data."""
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        velocity_target = x_1 - x_0
        return x_t, velocity_target

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Returns data prediction x_1_hat."""
        x_input = torch.cat([x_t, condition], dim=1)
        return self.backbone(x_input, t)

    def compute_velocity_loss(self, x_pred, x_t, x_1, t, loss_fn=None):
        """JiT-style velocity loss from data prediction.
        
        Derives v_pred and v_target from data prediction, then applies loss_fn.
        The 1/(1-t) factor provides implicit upweighting of late timesteps.
        If loss_fn is None, falls back to MSE.
        """
        t_expanded = t.view(-1, 1, 1, 1)
        denom = (1 - t_expanded).clamp_min(self.t_eps)

        v_pred = (x_pred - x_t) / denom
        v_target = (x_1 - x_t) / denom

        if loss_fn is not None:
            return loss_fn(v_pred, v_target)
        loss = (v_pred - v_target).pow(2)
        return loss.mean(dim=(1, 2, 3)).mean()

    # ------------------------------------------------------------------
    # Sampling (ODE integration deriving velocity from data prediction)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, condition: torch.Tensor, shape: Tuple[int, ...],
               device: torch.device, num_integration_steps: int = 50,
               solver: str = 'euler') -> torch.Tensor:
        z = torch.randn(shape, device=device) * self.noise_scale
        timesteps = torch.linspace(0.0, 1.0, num_integration_steps + 1, device=device)

        for i in range(num_integration_steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_tensor = torch.full((shape[0],), t.item(), device=device)

            if solver == 'euler':
                z = self._euler_step(z, t, t_next, t_tensor, condition)
            elif solver == 'heun':
                z = self._heun_step(z, t, t_next, t_tensor, condition, shape)
            elif solver == 'midpoint':
                z = self._midpoint_step(z, t, t_next, t_tensor, condition, shape)
            elif solver == 'rk4':
                z = self._rk4_step(z, t, t_next, t_tensor, condition, shape)
            else:
                raise ValueError(f"Unknown solver: {solver}")

        # Last step always Euler (following JiT)
        t = timesteps[-2]
        t_next = timesteps[-1]
        t_tensor = torch.full((shape[0],), t.item(), device=device)
        z = self._euler_step(z, t, t_next, t_tensor, condition)
        return z

    def _get_velocity(self, z, t_scalar, t_tensor, condition):
        x_pred = self.forward(z, condition, t_tensor)
        denom = max(1.0 - t_scalar.item(), self.t_eps)
        return (x_pred - z) / denom

    def _euler_step(self, z, t, t_next, t_tensor, condition):
        v = self._get_velocity(z, t, t_tensor, condition)
        return z + (t_next - t) * v

    def _heun_step(self, z, t, t_next, t_tensor, condition, shape):
        v1 = self._get_velocity(z, t, t_tensor, condition)
        z_euler = z + (t_next - t) * v1

        t_next_tensor = torch.full((shape[0],), t_next.item(), device=z.device)
        v2 = self._get_velocity(z_euler, t_next, t_next_tensor, condition)
        return z + (t_next - t) * 0.5 * (v1 + v2)

    def _midpoint_step(self, z, t, t_next, t_tensor, condition, shape):
        dt = t_next - t
        v1 = self._get_velocity(z, t, t_tensor, condition)
        z_mid = z + v1 * (dt / 2)

        t_mid = t + dt / 2
        t_mid_tensor = torch.full((shape[0],), t_mid.item(), device=z.device)
        v_mid = self._get_velocity(z_mid, t_mid, t_mid_tensor, condition)
        return z + dt * v_mid

    def _rk4_step(self, z, t, t_next, t_tensor, condition, shape):
        dt = t_next - t
        t_half = t + dt / 2
        t_half_tensor = torch.full((shape[0],), t_half.item(), device=z.device)
        t_next_tensor = torch.full((shape[0],), t_next.item(), device=z.device)

        k1 = self._get_velocity(z, t, t_tensor, condition)
        k2 = self._get_velocity(z + k1 * dt / 2, t_half, t_half_tensor, condition)
        k3 = self._get_velocity(z + k2 * dt / 2, t_half, t_half_tensor, condition)
        k4 = self._get_velocity(z + k3 * dt, t_next, t_next_tensor, condition)
        return z + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)


# =============================================================================
# PyTorch Lightning Wrapper
# =============================================================================

class ConditionalFlowMatchingJiTLightning(L.LightningModule):
    """
    Lightning wrapper for JiT Flow Matching.

    Uses data prediction with velocity loss and log-normal time sampling.
    Same interface as ConditionalFlowMatchingLightning for drop-in compatibility.
    """

    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig,
                 scheduler_cfg: DictConfig,
                 task_cfg: Optional[DictConfig] = None,
                 normalization_stats: Optional[dict] = None,
                 norm_mode: str = 'all'):
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])
        
        self.norm_mode = norm_mode
        self.normalization_stats = normalization_stats
        self.task_cfg = task_cfg

        # --- task channels ---------------------------------------------------
        if task_cfg is not None:
            self.conditioning_channels = list(task_cfg.get('conditioning_channels', [0]))
            self.target_channels = list(task_cfg.get('target_channels', [0]))
            self.task_name = task_cfg.get('name', 'unknown')
            self.is_noisy_task = 'noisy' in self.task_name.lower()
            noise_cfg = task_cfg.get('noise_cfg', None)
            self.has_noise = noise_cfg is not None and noise_cfg.get('enabled', True)

            print(f"Task: {self.task_name}")
            print(f"   Conditioning channels: {self.conditioning_channels} "
                  f"({task_cfg.get('conditioning_names', [])})")
            print(f"   Target channels: {self.target_channels} "
                  f"({task_cfg.get('target_names', [])})")
            if self.has_noise:
                print(f"   Noise injection: ENABLED")
        else:
            self.conditioning_channels = [0]
            self.target_channels = [0]
            self.task_name = 'temperature_from_sdf'
            self.is_noisy_task = False
            self.has_noise = False

        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)

        in_channels = self.num_target_channels + self.num_conditioning_channels
        out_channels = self.num_target_channels

        # --- JiT architecture -------------------------------------------------
        img_size = model_cfg.get('img_size', 64)
        patch_size = model_cfg.get('patch_size', 8)
        hidden_size = model_cfg.get('hidden_size', 384)
        depth = model_cfg.get('depth', 8)
        num_heads = model_cfg.get('num_heads', 6)
        mlp_ratio = model_cfg.get('mlp_ratio', 4.0)
        bottleneck_dim = model_cfg.get('bottleneck_dim', 64)
        dropout = model_cfg.get('dropout', 0.0)

        print(f"\nJiT Flow Matching Configuration:")
        print(f"   Image size: {img_size}x{img_size}")
        print(f"   Patch size: {patch_size} -> {(img_size // patch_size) ** 2} tokens")
        print(f"   Hidden size: {hidden_size}, Depth: {depth}, Heads: {num_heads}")
        print(f"   Bottleneck dim: {bottleneck_dim}")
        print(f"   in_channels: {in_channels} = {self.num_target_channels} (x_t) "
              f"+ {self.num_conditioning_channels} (cond)")
        print(f"   out_channels: {out_channels}")

        backbone = JiTBackbone(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=dropout,
            proj_drop=dropout,
            bottleneck_dim=bottleneck_dim,
        )

        noise_scale = model_cfg.get('noise_scale', 1.0)
        t_eps = model_cfg.get('t_eps', 1e-5)

        self.flow_matching = ConditionalFlowMatchingJiT(
            backbone=backbone,
            noise_scale=noise_scale,
            t_eps=t_eps,
        )

        self.loss_fn = build_loss_fn(model_cfg)

        # --- JiT training parameters -----------------------------------------
        self.P_mean = model_cfg.get('P_mean', -0.8)
        self.P_std = model_cfg.get('P_std', 0.8)
        self.noise_scale = noise_scale
        self.t_eps = t_eps

        print(f"   P_mean: {self.P_mean}, P_std: {self.P_std}")
        print(f"   Noise scale: {self.noise_scale}, t_eps: {self.t_eps}")

        # --- normalization ----------------------------------------------------
        self.downsample_factor = 1
        if normalization_stats is not None:
            self.downsample_factor = normalization_stats.get('downsample_factor', 1)

        if normalization_stats is not None and 'temperature' in normalization_stats:
            temp_stats = normalization_stats['temperature']
            self.temp_min = temp_stats['min']
            self.temp_max = temp_stats['max']
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = normalization_stats.get(
                'unified_velocity_scale', 1.0)
            self.sdf_scale = normalization_stats.get('sdf', {}).get('scale', 1.0)
            print(f"   Normalization: T=[{self.temp_min:.2f}, {self.temp_max:.2f}]C, "
                  f"V_scale={self.unified_velocity_scale:.4f}")
        else:
            self.temp_min = model_cfg.get('temp_min', 55.0)
            self.temp_max = model_cfg.get('temp_max', 120.0)
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
            self.sdf_scale = 1.0

        # --- inference --------------------------------------------------------
        self.num_integration_steps = model_cfg.get('num_integration_steps', 50)
        inference_cfg = model_cfg.get('inference', {})
        self.default_solver = inference_cfg.get('solver', 'heun')
        self.default_guidance_scale = inference_cfg.get('guidance_scale', 1.0)
        print(f"   Solver: {self.default_solver}, Steps: {self.num_integration_steps}")

        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg

    # ------------------------------------------------------------------
    # Forward / helpers
    # ------------------------------------------------------------------
    def forward(self, x_t, condition, t):
        """Returns data prediction x_1_hat."""
        return self.flow_matching(x_t, condition, t)

    def denormalize_temperature(self, temperature_norm: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == 'none':
            return temperature_norm
        return (temperature_norm + 1.0) / 2.0 * self.temp_range + self.temp_min

    def denormalize_velocity(self, velocity_norm: torch.Tensor) -> torch.Tensor:
        if self.norm_mode in ('none', 'temperature_only'):
            return velocity_norm
        return velocity_norm * self.unified_velocity_scale

    def _extract_channels(self, tensor: torch.Tensor,
                          channel_indices: list) -> torch.Tensor:
        return tensor[:, channel_indices, :, :]

    def _sample_time(self, n: int, device: torch.device) -> torch.Tensor:
        """Log-normal time sampling (JiT)."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        input_data, output_data = batch

        conditioning = self._extract_channels(input_data, self.conditioning_channels)
        target = self._extract_channels(output_data, self.target_channels)

        batch_size = conditioning.shape[0]
        device = conditioning.device

        t = self._sample_time(batch_size, device)
        x_0 = torch.randn_like(target) * self.noise_scale

        x_t, _ = self.flow_matching.compute_conditional_flow(x_0, target, t)
        x_pred = self.flow_matching(x_t, conditioning, t)
        loss = self.flow_matching.compute_velocity_loss(x_pred, x_t, target, t,
                                                        loss_fn=self.loss_fn)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        is_clean_val = (dataloader_idx == 0) if self.has_noise else True
        val_prefix = "val_clean" if is_clean_val else "val_noisy"

        input_data, output_data = batch
        conditioning = self._extract_channels(input_data, self.conditioning_channels)
        target = self._extract_channels(output_data, self.target_channels)

        batch_size = conditioning.shape[0]
        device = conditioning.device

        t = self._sample_time(batch_size, device)
        x_0 = torch.randn_like(target) * self.noise_scale
        x_t, _ = self.flow_matching.compute_conditional_flow(x_0, target, t)
        x_pred = self.flow_matching(x_t, conditioning, t)
        loss = self.flow_matching.compute_velocity_loss(x_pred, x_t, target, t,
                                                        loss_fn=self.loss_fn)

        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, add_dataloader_idx=False)
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=False, add_dataloader_idx=False)

        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, batch_size)
                samples = self.flow_matching.sample(
                    conditioning[:num_samples],
                    (num_samples, self.num_target_channels,
                     target.shape[2], target.shape[3]),
                    device,
                    num_integration_steps=self.num_integration_steps,
                    solver=self.default_solver,
                )

                self.log(f'{val_prefix}_sample_mean_norm', samples.mean(),
                         on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_sample_std_norm', samples.std(),
                         on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_mean_norm',
                         target[:num_samples].mean(),
                         on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_std_norm',
                         target[:num_samples].std(),
                         on_step=False, on_epoch=True, add_dataloader_idx=False)

                if (self.task_cfg is not None
                        and 'temperature' in self.task_cfg.get('target_names', [])):
                    temp_idx = list(
                        self.task_cfg.get('target_names', [])
                    ).index('temperature')
                    samples_temp = samples[:, temp_idx:temp_idx + 1, :, :]
                    target_temp = target[:num_samples, temp_idx:temp_idx + 1, :, :]

                    samples_celsius = self.denormalize_temperature(samples_temp)
                    target_celsius = self.denormalize_temperature(target_temp)

                    self.log(f'{val_prefix}_pred_temp_min_C',
                             samples_celsius.min(), on_step=False,
                             on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_pred_temp_max_C',
                             samples_celsius.max(), on_step=False,
                             on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_min_C',
                             target_celsius.min(), on_step=False,
                             on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_max_C',
                             target_celsius.max(), on_step=False,
                             on_epoch=True, prog_bar=True, add_dataloader_idx=False)

        return loss

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        if self.optim_cfg.name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-3),
                weight_decay=self.optim_cfg.get('weight_decay', 0.0),
            )
        elif self.optim_cfg.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-3),
                weight_decay=self.optim_cfg.get('weight_decay', 1e-2),
            )
        elif self.optim_cfg.name.lower() == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-4),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2),
                )
            except ImportError:
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-3),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2),
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_cfg.name}")

        if self.scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
            }
        elif self.scheduler_cfg.name.lower() == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_cfg.get('T_0', 10),
                T_mult=self.scheduler_cfg.get('T_mult', 2),
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
            }
        else:
            return optimizer
