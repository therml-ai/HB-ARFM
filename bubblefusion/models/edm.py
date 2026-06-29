"""
EDM: Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)

Baseline implementation adapted from:
https://github.com/jhhuangchloe/DiffusionPDE

This module implements the EDM-style diffusion model architecture for
conditional prediction of physical fields.

Supports multiple tasks through task_cfg configuration:
- temperature_from_sdf: Predict temperature from SDF (Task 1)
- velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)

This is a FRAME-TO-FRAME model (non-autoregressive):
    Input:  conditioning
    Target: output fields

Reference:
    - DiffusionPDE: https://arxiv.org/abs/2406.17763
    - EDM: https://arxiv.org/abs/2206.00364
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple


# ==============================================================================
# Building Blocks (adapted from EDM/DiffusionPDE networks.py)
# ==============================================================================

def weight_init(shape, mode, fan_in, fan_out):
    """Unified routine for initializing weights and biases."""
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(nn.Module):
    """Fully-connected layer with custom initialization."""
    
    def __init__(self, in_features, out_features, bias=True, 
                 init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(nn.Module):
    """Convolutional layer with optional up/downsampling."""
    
    def __init__(self, in_channels, out_channels, kernel, bias=True, 
                 up=False, down=False, resample_filter=[1, 1], fused_resample=False,
                 init_mode='kaiming_normal', init_weight=1, init_bias=0):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), 
                                   groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = F.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = F.conv2d(x, w, padding=w_pad + f_pad)
            x = F.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), 
                                       groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = F.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), 
                            groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = F.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(nn.Module):
    """Group normalization with automatic group calculation."""
    
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = F.group_norm(x, num_groups=self.num_groups, 
                        weight=self.weight.to(x.dtype), 
                        bias=self.bias.to(x.dtype), eps=self.eps)
        return x


class AttentionOp(torch.autograd.Function):
    """Attention weight computation with FP32 for numerical stability."""
    
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), 
                        (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), 
                                          output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


class UNetBlock(nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention.
    From DDPM++, NCSN++, and ADM architectures.
    """
    
    def __init__(self, in_channels, out_channels, emb_channels, up=False, down=False, 
                 attention=False, num_heads=None, channels_per_head=64, dropout=0, 
                 skip_scale=1, eps=1e-5, resample_filter=[1, 1], resample_proj=False, 
                 adaptive_scale=True, init=dict(), init_zero=dict(init_weight=0), init_attn=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, 
                           up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, 
                            out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, 
                              up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, 
                             **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))

        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, 
                                                       x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timesteps (DDPM++ style)."""
    
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(nn.Module):
    """Random Fourier feature embeddings for timesteps (NCSN++ style)."""
    
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ==============================================================================
# SongUNet: Main U-Net Architecture from DDPM++ / NCSN++
# ==============================================================================

class SongUNet(nn.Module):
    """
    U-Net architecture from "Score-Based Generative Modeling through SDEs".
    Supports both DDPM++ and NCSN++ configurations.
    
    Args:
        img_resolution: Image resolution at input/output
        in_channels: Number of input channels
        out_channels: Number of output channels
        label_dim: Number of class labels (0 = unconditional)
        augment_dim: Augmentation label dimensionality
        model_channels: Base multiplier for channels
        channel_mult: Per-resolution channel multipliers
        channel_mult_emb: Embedding dimension multiplier
        num_blocks: Number of residual blocks per resolution
        attn_resolutions: List of resolutions with self-attention
        dropout: Dropout probability
        label_dropout: Label dropout for classifier-free guidance
        embedding_type: 'positional' (DDPM++) or 'fourier' (NCSN++)
        channel_mult_noise: Timestep embedding size multiplier
        encoder_type: Encoder architecture type
        decoder_type: Decoder architecture type
        resample_filter: Resampling filter
    """
    
    def __init__(self, img_resolution, in_channels, out_channels, label_dim=0, augment_dim=0,
                 model_channels=128, channel_mult=[1, 2, 2, 2], channel_mult_emb=4, num_blocks=4,
                 attn_resolutions=[16], dropout=0.10, label_dropout=0,
                 embedding_type='positional', channel_mult_noise=1, 
                 encoder_type='standard', decoder_type='standard', resample_filter=[1, 1]):
        
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping network
        if embedding_type == 'positional':
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        else:
            self.map_noise = FourierEmbedding(num_channels=noise_channels)
        
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder
        self.enc = nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, 
                                                               down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, 
                                                                   down=True, resample_filter=resample_filter, 
                                                                   fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, 
                                                                 attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, 
                                                         attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, 
                                                                 attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                                             kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, 
                                                           kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels=None, augment_labels=None):
        # Mapping
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))

        # Encoder
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(F.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


# ==============================================================================
# EDM Preconditioning Wrapper
# ==============================================================================

class EDMPrecond(nn.Module):
    """
    Improved preconditioning from "Elucidating the Design Space of Diffusion-Based 
    Generative Models" (EDM).
    
    This wrapper handles the noise-level-dependent preconditioning and provides
    a clean interface for conditional prediction.
    
    Args:
        img_resolution: Image resolution
        img_channels: Number of image channels
        cond_channels: Number of conditioning channels
        label_dim: Number of class labels
        use_fp16: Use FP16 precision
        sigma_min: Minimum supported noise level
        sigma_max: Maximum supported noise level
        sigma_data: Expected standard deviation of training data
        model_type: Underlying model architecture
        **model_kwargs: Additional model arguments
    """
    
    def __init__(self, img_resolution, img_channels, cond_channels=0, label_dim=0, use_fp16=False,
                 sigma_min=0, sigma_max=float('inf'), sigma_data=0.5, model_type='SongUNet',
                 **model_kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.cond_channels = cond_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        # Total input channels = noisy target + conditioning
        in_channels = img_channels + cond_channels
        
        self.model = SongUNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs
        )

    def forward(self, x, sigma, condition=None, class_labels=None, force_fp32=False, **model_kwargs):
        """
        Forward pass with EDM preconditioning.
        
        Args:
            x: Noisy input tensor [B, img_channels, H, W]
            sigma: Noise levels [B]
            condition: Conditioning tensor [B, cond_channels, H, W]
            class_labels: Class labels for conditional generation
            force_fp32: Force FP32 computation
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # EDM preconditioning
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Concatenate conditioning with scaled noisy input
        if condition is not None:
            model_input = torch.cat([c_in * x, condition], dim=1)
        else:
            model_input = c_in * x

        F_x = self.model(model_input.to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ==============================================================================
# Conditional EDM Model
# ==============================================================================

class ConditionalEDM(nn.Module):
    """
    Conditional EDM model for frame-to-frame prediction.
    
    Uses EDM-style diffusion with conditioning on physical fields.
    
    Args:
        edm_precond: EDMPrecond model
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        sigma_data: Expected data standard deviation
        rho: Noise schedule exponent
    """
    
    def __init__(self, edm_precond: EDMPrecond, sigma_min=0.002, sigma_max=80, 
                 sigma_data=0.5, rho=7):
        super().__init__()
        self.edm_precond = edm_precond
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
    
    def sample_noise_level(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise levels using log-normal distribution (EDM style)."""
        rnd_normal = torch.randn([batch_size], device=device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()  # P_mean=-1.2, P_std=1.2
        return sigma
    
    def forward(self, x_noisy: torch.Tensor, sigma: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        """
        Denoising network forward pass.
        
        Args:
            x_noisy: Noisy target tensor [B, C_out, H, W]
            sigma: Noise levels [B]
            condition: Conditioning tensor [B, C_cond, H, W]
            
        Returns:
            Denoised prediction [B, C_out, H, W]
        """
        return self.edm_precond(x_noisy, sigma, condition=condition)
    
    @torch.no_grad()
    def sample(self, condition: torch.Tensor, shape: Tuple[int, ...], 
               device: torch.device, num_steps: int = 50, 
               solver: str = 'euler') -> torch.Tensor:
        """
        Generate samples using EDM sampling (deterministic or stochastic).
        
        Args:
            condition: Conditioning tensor [B, C_cond, H, W]
            shape: Output shape (B, C_out, H, W)
            device: Device for computation
            num_steps: Number of sampling steps
            solver: 'euler' or 'heun'
            
        Returns:
            Generated samples [B, C_out, H, W]
        """
        # Noise schedule (time steps from sigma_max to sigma_min)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (num_steps - 1) * 
                  (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
        
        # Initialize with pure noise at sigma_max
        x_next = torch.randn(shape, device=device) * t_steps[0]
        
        # Main sampling loop
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            
            # Increase noise temporarily for stochastic sampling
            t_hat = t_cur
            
            # Euler step
            sigma = t_hat.expand(shape[0])
            denoised = self.forward(x_cur, sigma, condition)
            d_cur = (x_cur - denoised) / t_hat
            x_next = x_cur + (t_next - t_hat) * d_cur
            
            # Apply 2nd order correction (Heun's method)
            if solver == 'heun' and t_next > 0:
                sigma_next = t_next.expand(shape[0])
                denoised_next = self.forward(x_next, sigma_next, condition)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_cur + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_next


# ==============================================================================
# PyTorch Lightning Module
# ==============================================================================

class EDMLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for EDM model.
    
    This is a FRAME-TO-FRAME model (non-autoregressive):
    - Input: conditioning
    - Target: output fields
    
    Uses EDM-style diffusion training and sampling.
    
    Supports multiple tasks through task_cfg configuration:
    - temperature_from_sdf: Predict temperature from SDF (Task 1)
    - velocity_from_interface: Predict velocity + temperature from SDF + interface velocity (Task 2)
    
    Args:
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        scheduler_cfg: Learning rate scheduler configuration
        task_cfg: Task configuration defining which channels to use
        normalization_stats: Pre-computed normalization statistics from training data
    """
    
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig, scheduler_cfg: DictConfig,
                 task_cfg: Optional[DictConfig] = None, normalization_stats: Optional[dict] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])
        
        # Store normalization statistics
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
            
            if self.has_noise:
                print(f"   🔊 Noise injection: ENABLED")
        else:
            # Default to Task 1 behavior
            self.conditioning_channels = [0]
            self.target_channels = [0]
            self.task_name = 'temperature_from_sdf'
            self.is_noisy_task = False
            self.has_noise = False
            print("⚠️  No task_cfg provided, defaulting to temperature_from_sdf task")
        
        # Compute derived channel counts
        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)
        
        # =============================================================================
        # NORMALIZATION CONFIGURATION - Must be done before model initialization
        # =============================================================================
        # Get downsample factor from model_cfg (from data config) or normalization_stats (backward compatibility)
        # downsample_factor comes from data_cfg, not normalization_stats
        self.downsample_factor = model_cfg.get('downsample_factor', 1)
        if self.downsample_factor == 1 and normalization_stats is not None:
            # Fallback for backward compatibility with old checkpoints
            self.downsample_factor = normalization_stats.get('downsample_factor', 1)
        
        # Compute image resolution based on downsample factor
        # Base resolution is 512 for BubbleML dataset
        base_resolution = model_cfg.get('base_resolution', 512)
        img_resolution = base_resolution // self.downsample_factor
        
        # Compute attention resolutions dynamically based on image resolution
        # We want attention at ~16x16 resolution (after downsampling in the UNet)
        # With 4 channel_mult levels [1,2,2,2], we downsample 3 times (factor of 8)
        # So for 128x128 input, bottleneck is 16x16
        # For 512x512 input, bottleneck is 64x64
        default_attn_res = max(img_resolution // 8, 8)  # At least 8x8
        attn_resolutions = model_cfg.get('attn_resolutions', [default_attn_res])
        
        print(f"\n🔬 EDM Configuration:")
        print(f"   Base resolution: {base_resolution}x{base_resolution}")
        print(f"   Downsample factor: {self.downsample_factor}")
        print(f"   Image resolution: {img_resolution}x{img_resolution}")
        print(f"   Attention resolutions: {attn_resolutions}")
        print(f"   Conditioning channels: {self.num_conditioning_channels}")
        print(f"   Target channels: {self.num_target_channels}")
        print(f"   Model channels: {model_cfg.get('model_channels', 128)}")
        print(f"   Channel mult: {model_cfg.get('channel_mult', [1, 2, 2, 2])}")
        
        # Initialize EDM preconditioned model
        edm_precond = EDMPrecond(
            img_resolution=img_resolution,
            img_channels=self.num_target_channels,
            cond_channels=self.num_conditioning_channels,
            label_dim=0,
            use_fp16=model_cfg.get('use_fp16', False),
            sigma_min=model_cfg.get('sigma_min', 0.002),
            sigma_max=model_cfg.get('sigma_max', 80),
            sigma_data=model_cfg.get('sigma_data', 0.5),
            model_type='SongUNet',
            # SongUNet parameters
            model_channels=model_cfg.get('model_channels', 128),
            channel_mult=model_cfg.get('channel_mult', [1, 2, 2, 2]),
            channel_mult_emb=model_cfg.get('channel_mult_emb', 4),
            num_blocks=model_cfg.get('num_blocks', 4),
            attn_resolutions=attn_resolutions,  # Use dynamically computed attention resolutions
            dropout=model_cfg.get('dropout', 0.10),
            embedding_type=model_cfg.get('embedding_type', 'positional'),
            channel_mult_noise=model_cfg.get('channel_mult_noise', 1),
            encoder_type=model_cfg.get('encoder_type', 'standard'),
            decoder_type=model_cfg.get('decoder_type', 'standard'),
            resample_filter=model_cfg.get('resample_filter', [1, 1]),
        )
        
        # Wrap in ConditionalEDM
        self.edm = ConditionalEDM(
            edm_precond=edm_precond,
            sigma_min=model_cfg.get('sigma_min', 0.002),
            sigma_max=model_cfg.get('sigma_max', 80),
            sigma_data=model_cfg.get('sigma_data', 0.5),
            rho=model_cfg.get('rho', 7),
        )
        
        # Loss function (weighted MSE in original EDM)
        self.loss_fn = nn.MSELoss()
        
        # =============================================================================
        # TEMPERATURE/VELOCITY NORMALIZATION PARAMETERS (for logging/denormalization)
        # =============================================================================
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
            print(f"   SDF scale: {self.sdf_scale:.4f}")
            print(f"   Downsample factor: {self.downsample_factor}")
        else:
            self.temp_min = model_cfg.get('temp_min', 55.0)
            self.temp_max = model_cfg.get('temp_max', 120.0)
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
            self.sdf_scale = 1.0
            print(f"\n⚠️  Using config normalization params (no stats provided)")
        
        # Sampling configuration
        self.num_sampling_steps = model_cfg.get('num_sampling_steps', 50)
        self.default_solver = model_cfg.get('solver', 'heun')
        
        print(f"\n🔧 Sampling Settings:")
        print(f"   Steps: {self.num_sampling_steps}")
        print(f"   Solver: {self.default_solver}")
        
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
    
    def forward(self, x_noisy, sigma, condition):
        """Forward pass through denoising network."""
        return self.edm(x_noisy, sigma, condition)
    
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
        """Training step using EDM loss."""
        input_data, output_data = batch
        
        # Extract conditioning and target channels
        conditioning = self._extract_channels(input_data, self.conditioning_channels)
        target = self._extract_channels(output_data, self.target_channels)
        
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Sample noise levels (log-normal distribution, EDM style)
        rnd_normal = torch.randn([batch_size], device=device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()  # P_mean=-1.2, P_std=1.2
        
        # Add noise to target
        noise = torch.randn_like(target)
        x_noisy = target + noise * sigma.view(-1, 1, 1, 1)
        
        # Predict denoised output
        denoised = self.forward(x_noisy, sigma, conditioning)
        
        # EDM loss weighting
        weight = (sigma ** 2 + self.edm.sigma_data ** 2) / \
                 (sigma * self.edm.sigma_data) ** 2
        weight = weight.view(-1, 1, 1, 1)
        
        # Compute weighted MSE loss
        loss = (weight * (denoised - target) ** 2).mean()
        
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
        
        # Sample noise levels
        rnd_normal = torch.randn([batch_size], device=device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()
        
        # Add noise to target
        noise = torch.randn_like(target)
        x_noisy = target + noise * sigma.view(-1, 1, 1, 1)
        
        # Predict denoised output
        denoised = self.forward(x_noisy, sigma, conditioning)
        
        # EDM loss weighting
        weight = (sigma ** 2 + self.edm.sigma_data ** 2) / \
                 (sigma * self.edm.sigma_data) ** 2
        weight = weight.view(-1, 1, 1, 1)
        
        # Compute weighted MSE loss
        loss = (weight * (denoised - target) ** 2).mean()
        
        # Log losses
        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        
        # Generate samples and log statistics
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, batch_size)
                
                # Generate samples
                samples = self.edm.sample(
                    conditioning[:num_samples],
                    (num_samples, self.num_target_channels, target.shape[2], target.shape[3]),
                    device,
                    num_steps=min(self.num_sampling_steps, 25),  # Faster validation
                    solver=self.default_solver
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
