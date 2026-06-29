"""
Autoregressive UNet for BubbleFlow Prediction.

This module implements an Autoregressive UNet architecture that conditions on
the previous timestep output to enforce temporal consistency.

Key Idea:
- Direct regression (no diffusion/flow matching)
- Model conditions on: [conditioning_t, output_{t-1}]
- During training: uses teacher forcing (ground truth previous state)
- During inference: uses model's own predictions (autoregressive rollout)

Training Modes:
1. Teacher Forcing (scheduled_sampling.enabled=False):
   - Previous state always comes from ground truth
   - Efficient but may cause exposure bias

2. Scheduled Sampling (scheduled_sampling.enabled=True):
   - Gradually transitions from teacher forcing to model predictions
   - Early: 100% ground truth (stability)
   - Mid: mixture of ground truth and predictions
   - Late: 100% predictions (like inference)
   - Reduces exposure bias

Autoregressive Formulation:
    Input:  [conditioning_t, output_{t-1}]  # [B, 6, H, W] for Task 2
    Target: output_t                         # [B, 3, H, W]

Where:
    - conditioning_t = [SDF_t, velx_interface_t, vely_interface_t]
    - output_{t-1} = [velx_{t-1}, vely_{t-1}, temp_{t-1}] (teacher forcing or predicted)
    - output_t = [velx_t, vely_t, temp_t]

This encourages temporal consistency by construction, as the model must
learn to produce outputs that are consistent with its previous predictions.

Classes:
    UNet2dAR: UNet architecture for autoregressive prediction
    UNetARLightning: PyTorch Lightning wrapper

Reference:
    UNet implementation adapted from PDEBench and extended with AR conditioning
"""

import math
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple
import numpy as np


class SpectralLoss(nn.Module):
    """
    Spectral loss to preserve high-frequency details.
    
    Computes loss in frequency domain to prevent blur/smoothing.
    Weights higher frequencies more heavily to encourage sharp predictions.
    """
    
    def __init__(
        self, 
        weight: float = 0.1,
        high_freq_weight: float = 2.0,
        freq_threshold: float = 0.3,
    ):
        super().__init__()
        self.weight = weight
        self.high_freq_weight = high_freq_weight
        self.freq_threshold = freq_threshold
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # Compute magnitude spectra
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Compute phase difference
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_diff = torch.abs(pred_phase - target_phase)
        phase_diff = torch.minimum(phase_diff, 2*np.pi - phase_diff)
        
        # Create frequency weight mask
        B, C, H, W = pred.shape
        freq_y = torch.fft.fftfreq(H, device=pred.device).view(-1, 1).abs()
        freq_x = torch.fft.fftfreq(W, device=pred.device).view(1, -1).abs()
        freq_magnitude = torch.sqrt(freq_y**2 + freq_x**2)
        freq_magnitude = freq_magnitude / freq_magnitude.max()
        
        weight_mask = torch.where(
            freq_magnitude > self.freq_threshold,
            torch.full_like(freq_magnitude, self.high_freq_weight),
            torch.ones_like(freq_magnitude)
        )
        weight_mask = weight_mask.unsqueeze(0).unsqueeze(0)
        
        mag_loss = ((pred_mag - target_mag).abs() * weight_mask).mean()
        phase_loss = (phase_diff * weight_mask).mean() * 0.1
        
        return self.weight * (mag_loss + phase_loss)


class GradientLoss(nn.Module):
    """
    Gradient loss to preserve edges and sharp transitions.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        
        total_loss = 0.0
        for c in range(C):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target[:, c:c+1, :, :]
            
            pred_gx = F.conv2d(pred_c, self.sobel_x, padding=1)
            pred_gy = F.conv2d(pred_c, self.sobel_y, padding=1)
            target_gx = F.conv2d(target_c, self.sobel_x, padding=1)
            target_gy = F.conv2d(target_c, self.sobel_y, padding=1)
            
            loss_x = F.l1_loss(pred_gx, target_gx)
            loss_y = F.l1_loss(pred_gy, target_gy)
            
            total_loss = total_loss + loss_x + loss_y
        
        return self.weight * total_loss / C


class WallTempBias(nn.Module):
    """
    Simple learned bias based on wall temperature.
    """
    
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
    """
    FiLM: Feature-wise Linear Modulation layer.
    """
    
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


class UNet2dAR(nn.Module):
    """
    U-Net architecture for autoregressive prediction.
    
    Takes concatenated [conditioning, prev_output] as input and produces current output.
    
    Args:
        in_channels: Number of input channels (conditioning + prev_output)
        out_channels: Number of output channels
        init_features: Initial number of features (default: 32)
    """
    
    def __init__(self, in_channels: int = 6, out_channels: int = 3, init_features: int = 32):
        super(UNet2dAR, self).__init__()
        features = init_features
        
        # Encoder
        self.encoder1 = UNet2dAR._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2dAR._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2dAR._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2dAR._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet2dAR._block(features * 8, features * 16, name="bottleneck")

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2dAR._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2dAR._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2dAR._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet2dAR._block(features * 2, features, name="dec1")

        # Output convolution
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Concatenated input [conditioning, prev_output] of shape [B, in_channels, H, W]
            
        Returns:
            Predicted output [B, out_channels, H, W]
        """
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


class UNetARLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for Autoregressive UNet model.
    
    Supports autoregressive training with teacher forcing and scheduled sampling:
    - Input: [conditioning_t, output_{t-1}]
    - Target: output_t
    
    Direct regression without diffusion/flow matching.
    
    Args:
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        scheduler_cfg: Learning rate scheduler configuration
        task_cfg: Task configuration defining channels
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
        
        # Store normalization statistics for accurate denormalization during logging
        self.normalization_stats = normalization_stats
        
        # Store task configuration
        self.task_cfg = task_cfg
        
        # Parse task config for channel indices
        if task_cfg is not None:
            self.conditioning_channels = list(task_cfg.get('conditioning_channels', [0]))
            self.target_channels = list(task_cfg.get('target_channels', [0]))
            self.task_name = task_cfg.get('name', 'unknown')
            
            # Check for noise configuration
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
        
        # Compute channel counts
        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)
        
        # UNet configuration for autoregressive model
        # in_channels = num_conditioning_channels + num_target_channels (prev_output)
        # For Task 2: 3 + 3 = 6
        in_channels = self.num_conditioning_channels + self.num_target_channels
        out_channels = self.num_target_channels
        
        print(f"\n🔄 Autoregressive UNet Configuration:")
        print(f"   UNet in_channels: {in_channels} = {self.num_conditioning_channels} (cond) + {self.num_target_channels} (prev)")
        print(f"   UNet out_channels: {out_channels}")
        
        # Wall temperature conditioning (optional)
        self.conditioning_strategy = model_cfg.get('conditioning_strategy', 'none')
        
        # Initialize UNet
        self.unet = UNet2dAR(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=model_cfg.get('init_features', 32),
        )
        
        # Initialize wall temp conditioner if needed
        self.wall_temp_conditioner = None
        if self.conditioning_strategy == 'bias':
            self.wall_temp_conditioner = WallTempBias(
                hidden_dim=model_cfg.get('wall_temp_bias_hidden', 64)
            )
            print("🌡️  Conditioning: SIMPLE BIAS")
        elif self.conditioning_strategy == 'film':
            gamma_range = model_cfg.get('wall_temp_film_gamma_range', 0.1)
            self.wall_temp_conditioner = FiLMLayer(
                num_channels=out_channels,
                hidden_dim=model_cfg.get('wall_temp_film_hidden', 64),
                gamma_range=gamma_range
            )
            print(f"🌡️  Conditioning: FiLM")
        else:
            print("🌡️  Conditioning: NONE")
        
        # =============================================================================
        # LOSS FUNCTION CONFIGURATION
        # =============================================================================
        self.loss_fn = nn.MSELoss()
        
        # Auxiliary losses to prevent blur
        loss_cfg = model_cfg.get('auxiliary_losses', {})
        
        # Spectral loss
        self.use_spectral_loss = loss_cfg.get('spectral_enabled', False)
        if self.use_spectral_loss:
            self.spectral_loss = SpectralLoss(
                weight=loss_cfg.get('spectral_weight', 0.1),
                high_freq_weight=loss_cfg.get('spectral_high_freq_weight', 2.0),
                freq_threshold=loss_cfg.get('spectral_freq_threshold', 0.3),
            )
            print(f"📊 Spectral Loss: ENABLED (weight={loss_cfg.get('spectral_weight', 0.1)})")
        else:
            self.spectral_loss = None
            print(f"📊 Spectral Loss: DISABLED")
        
        # Gradient loss
        self.use_gradient_loss = loss_cfg.get('gradient_enabled', False)
        if self.use_gradient_loss:
            self.gradient_loss = GradientLoss(
                weight=loss_cfg.get('gradient_weight', 0.1),
            )
            print(f"📐 Gradient Loss: ENABLED (weight={loss_cfg.get('gradient_weight', 0.1)})")
        else:
            self.gradient_loss = None
            print(f"📐 Gradient Loss: DISABLED")
        
        # =============================================================================
        # RESIDUAL/DELTA PREDICTION MODE
        # =============================================================================
        self.residual_prediction = model_cfg.get('residual_prediction', False)
        if self.residual_prediction:
            print(f"\n🔄 Residual Prediction Mode: ENABLED")
            print(f"   Model predicts Δoutput = output_t - output_{{t-1}}")
            print(f"   Reconstruction: output_t = output_{{t-1}} + Δoutput")
        else:
            print(f"\n🔄 Residual Prediction Mode: DISABLED (absolute values)")
        
        # Temperature normalization parameters
        # Use computed stats if available, otherwise fall back to config values
        if normalization_stats is not None and 'temperature' in normalization_stats:
            temp_stats = normalization_stats['temperature']
            self.temp_min = temp_stats['min']
            self.temp_max = temp_stats['max']
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = normalization_stats.get('unified_velocity_scale', 1.0)
            print(f"📊 Using computed normalization stats for logging:")
            print(f"   Temperature: [{self.temp_min:.2f}, {self.temp_max:.2f}]°C")
        else:
            self.temp_min = model_cfg.get('temp_min', 55.0)
            self.temp_max = model_cfg.get('temp_max', 120.0)
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
            print(f"⚠️  Using config normalization params (no stats provided)")

        # =============================================================================
        # SCHEDULED SAMPLING CONFIGURATION
        # =============================================================================
        ss_cfg = model_cfg.get('scheduled_sampling', {})
        self.scheduled_sampling_enabled = ss_cfg.get('enabled', False)
        
        if self.scheduled_sampling_enabled:
            self.ss_schedule_type = ss_cfg.get('schedule_type', 'linear')
            self.ss_warmup_epochs = ss_cfg.get('warmup_epochs', 5)
            self.ss_transition_epochs = ss_cfg.get('transition_epochs', 40)
            self.ss_min_teacher_ratio = ss_cfg.get('min_teacher_ratio', 0.0)
            self.ss_exponential_decay = ss_cfg.get('exponential_decay_rate', 0.95)
            self.ss_sigmoid_k = ss_cfg.get('sigmoid_k', 5.0)
            
            print(f"\n📊 Scheduled Sampling: ENABLED")
            print(f"   Schedule type: {self.ss_schedule_type}")
            print(f"   Warmup epochs: {self.ss_warmup_epochs} (pure teacher forcing)")
            print(f"   Transition epochs: {self.ss_transition_epochs}")
            print(f"   Final teacher ratio: {self.ss_min_teacher_ratio:.1%}")
        else:
            print(f"\n📊 Scheduled Sampling: DISABLED (pure teacher forcing)")
        
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        
    def forward(self, condition: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict current output from conditioning and previous output.
        
        Args:
            condition: Conditioning (SDF, interface vel) [B, C_cond, H, W]
            prev_output: Previous timestep output [B, C_out, H, W]
            
        Returns:
            Predicted output [B, C_out, H, W]
        """
        # Concatenate conditioning and previous output
        x = torch.cat([condition, prev_output], dim=1)
        return self.unet(x)
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate samples (direct prediction - no iterative sampling needed).
        
        Handles residual reconstruction automatically if residual_prediction is enabled.
        
        Args:
            condition: Current conditioning [B, C_cond, H, W]
            prev_output: Previous timestep output [B, C_out, H, W]
            
        Returns:
            Generated samples [B, C_out, H, W] (absolute values)
        """
        # Forward pass
        raw_output = self.forward(condition, prev_output)
        
        # Residual reconstruction if enabled
        if self.residual_prediction:
            output = prev_output + raw_output
        else:
            output = raw_output
        
        return output
    
    @torch.no_grad()
    def create_initial_state(
        self,
        shape: tuple,
        device: torch.device,
        mode: str = 'zeros',
        conditioning: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Create an initial state for autoregressive inference when ground truth is not available.
        
        Args:
            shape: Output shape (B, C_out, H, W)
            device: Device for the tensor
            mode: Initialization mode:
                - 'zeros': All zeros (neutral state for normalized data)
                - 'small_noise': Small random noise around zero
                - 'from_conditioning': Derive from conditioning (use interface vel as bulk vel)
            conditioning: Optional conditioning tensor [B, C_cond, H, W] for 'from_conditioning' mode
            
        Returns:
            Initial state tensor [B, C_out, H, W]
        """
        B, C_out, H, W = shape
        
        if mode == 'zeros':
            return torch.zeros(shape, device=device)
        elif mode == 'small_noise':
            return torch.randn(shape, device=device) * 0.01
        elif mode == 'from_conditioning':
            if conditioning is None:
                print("⚠️ 'from_conditioning' mode requires conditioning tensor, falling back to zeros")
                return torch.zeros(shape, device=device)
            
            initial = torch.zeros(shape, device=device)
            # Use interface velocity as initial bulk velocity estimate
            if conditioning.shape[1] >= 3 and C_out >= 2:
                initial[:, 0, :, :] = conditioning[:, 1, :, :]  # velx_interface -> velx
                initial[:, 1, :, :] = conditioning[:, 2, :, :]  # vely_interface -> vely
            return initial
        else:
            raise ValueError(f"Unknown initial state mode: {mode}")
    
    @torch.no_grad()
    def sample_trajectory(
        self,
        conditions: torch.Tensor,
        initial_state: torch.Tensor = None,
        initial_state_mode: str = 'zeros',
    ) -> torch.Tensor:
        """
        Generate a full trajectory autoregressively.
        
        Args:
            conditions: Conditioning for all timesteps [T, B, C_cond, H, W]
            initial_state: Initial state (output at t=0) [B, C_out, H, W].
                          If None, creates initial state based on initial_state_mode.
            initial_state_mode: Mode for creating initial state when initial_state is None.
            
        Returns:
            Generated trajectory [T, B, C_out, H, W]
        """
        T = conditions.shape[0]
        B = conditions.shape[1]
        C_cond = conditions.shape[2]
        H, W = conditions.shape[3], conditions.shape[4]
        
        # Create initial state if not provided
        if initial_state is None:
            C_out = self.num_target_channels
            shape = (B, C_out, H, W)
            prev_output = self.create_initial_state(
                shape=shape,
                device=conditions.device,
                mode=initial_state_mode,
                conditioning=conditions[0] if initial_state_mode == 'from_conditioning' else None
            )
        else:
            prev_output = initial_state
        
        trajectory = []
        
        for t in range(T):
            condition_t = conditions[t]  # [B, C_cond, H, W]
            output_t = self.sample(condition_t, prev_output)
            trajectory.append(output_t)
            prev_output = output_t
        
        return torch.stack(trajectory, dim=0)
    
    def denormalize_temperature(self, temperature_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize temperature from [-1, 1] to Celsius."""
        return (temperature_norm + 1.0) / 2.0 * self.temp_range + self.temp_min
    
    def _extract_channels(self, tensor: torch.Tensor, channel_indices: list) -> torch.Tensor:
        """Extract specific channels from a tensor [B, C, H, W]."""
        return tensor[:, channel_indices, :, :]
    
    def get_teacher_forcing_ratio(self) -> float:
        """
        Compute the teacher forcing ratio based on current epoch and schedule.
        """
        if not self.scheduled_sampling_enabled:
            return 1.0
        
        current_epoch = self.current_epoch
        
        if current_epoch < self.ss_warmup_epochs:
            return 1.0
        
        transition_epoch = current_epoch - self.ss_warmup_epochs
        
        if transition_epoch >= self.ss_transition_epochs:
            return self.ss_min_teacher_ratio
        
        progress = transition_epoch / self.ss_transition_epochs
        
        if self.ss_schedule_type == 'linear':
            teacher_ratio = 1.0 - progress * (1.0 - self.ss_min_teacher_ratio)
        elif self.ss_schedule_type == 'exponential':
            teacher_ratio = max(
                self.ss_min_teacher_ratio,
                self.ss_exponential_decay ** transition_epoch
            )
        elif self.ss_schedule_type == 'inverse_sigmoid':
            k = self.ss_sigmoid_k
            x = k * (progress - 0.5)
            sigmoid_val = 1.0 / (1.0 + math.exp(-x))
            teacher_ratio = 1.0 - sigmoid_val * (1.0 - self.ss_min_teacher_ratio)
        else:
            teacher_ratio = 1.0 - progress * (1.0 - self.ss_min_teacher_ratio)
        
        return max(self.ss_min_teacher_ratio, min(1.0, teacher_ratio))
    
    @torch.no_grad()
    def _generate_predicted_prev_output(
        self,
        conditioning_t_minus_1: torch.Tensor,
        output_t_minus_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate model's prediction for output at t-1 (for scheduled sampling).
        """
        return self.sample(conditioning_t_minus_1, output_t_minus_2)
    
    def training_step(self, batch, batch_idx):
        """
        Training step with teacher forcing or scheduled sampling.
        """
        use_scheduled_sampling = (
            self.scheduled_sampling_enabled and 
            len(batch) >= 5
        )
        
        # Extract data from batch
        if use_scheduled_sampling:
            if self.conditioning_strategy != 'none':
                (inp_data_t, prev_output_raw, out_data_t,
                 conditioning_t_minus_1_raw, output_t_minus_2_raw, wall_temp) = batch
            else:
                (inp_data_t, prev_output_raw, out_data_t,
                 conditioning_t_minus_1_raw, output_t_minus_2_raw) = batch
                wall_temp = None
        else:
            if self.conditioning_strategy != 'none':
                inp_data_t, prev_output_raw, out_data_t, wall_temp = batch
            else:
                inp_data_t, prev_output_raw, out_data_t = batch
                wall_temp = None
        
        # Extract conditioning and target channels
        conditioning = self._extract_channels(inp_data_t, self.conditioning_channels)
        target = self._extract_channels(out_data_t, self.target_channels)
        prev_output_gt = self._extract_channels(prev_output_raw, self.target_channels)
        
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Determine prev_output to use
        if use_scheduled_sampling:
            teacher_ratio = self.get_teacher_forcing_ratio()
            self.log('teacher_ratio', teacher_ratio, on_step=False, on_epoch=True, prog_bar=True)
            
            if teacher_ratio < 1.0:
                conditioning_t_minus_1 = self._extract_channels(
                    conditioning_t_minus_1_raw, self.conditioning_channels
                )
                output_t_minus_2 = self._extract_channels(
                    output_t_minus_2_raw, self.target_channels
                )
                
                use_teacher = torch.rand(batch_size, device=device) < teacher_ratio
                num_predictions = (~use_teacher).sum().item()
                
                if num_predictions > 0:
                    pred_indices = torch.where(~use_teacher)[0]
                    
                    predicted_prev_output = self._generate_predicted_prev_output(
                        conditioning_t_minus_1[pred_indices],
                        output_t_minus_2[pred_indices],
                    )
                    
                    prev_output = prev_output_gt.clone()
                    prev_output[pred_indices] = predicted_prev_output
                    
                    self.log('pct_predictions', 100.0 * num_predictions / batch_size, 
                            on_step=False, on_epoch=True, prog_bar=False)
                else:
                    prev_output = prev_output_gt
            else:
                prev_output = prev_output_gt
        else:
            prev_output = prev_output_gt
        
        # =======================================================================
        # RESIDUAL PREDICTION MODE
        # =======================================================================
        if self.residual_prediction:
            delta_target = target - prev_output
            pred_target = delta_target
            
            if batch_idx % 100 == 0:
                delta_mean = delta_target.abs().mean()
                delta_max = delta_target.abs().max()
                self.log('delta_mean', delta_mean, on_step=False, on_epoch=True)
                self.log('delta_max', delta_max, on_step=False, on_epoch=True)
        else:
            pred_target = target
        
        # Forward pass
        predicted = self.forward(conditioning, prev_output)
        
        # Compute primary MSE loss
        mse_loss = self.loss_fn(predicted, pred_target)
        
        # Compute auxiliary losses
        total_loss = mse_loss
        
        if self.use_spectral_loss and self.spectral_loss is not None:
            spec_loss = self.spectral_loss(predicted, pred_target)
            total_loss = total_loss + spec_loss
            self.log('train_spectral_loss', spec_loss, on_step=False, on_epoch=True)
        
        if self.use_gradient_loss and self.gradient_loss is not None:
            grad_loss = self.gradient_loss(predicted, pred_target)
            total_loss = total_loss + grad_loss
            self.log('train_gradient_loss', grad_loss, on_step=False, on_epoch=True)
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse_loss', mse_loss, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step."""
        is_clean_val = (dataloader_idx == 0) if self.has_noise else True
        val_prefix = "val_clean" if is_clean_val else "val_noisy"
        
        # Extract data
        if self.conditioning_strategy != 'none':
            inp_data_t, prev_output_raw, out_data_t, wall_temp = batch
        else:
            inp_data_t, prev_output_raw, out_data_t = batch
            wall_temp = None
        
        # Extract channels
        conditioning = self._extract_channels(inp_data_t, self.conditioning_channels)
        target = self._extract_channels(out_data_t, self.target_channels)
        prev_output = self._extract_channels(prev_output_raw, self.target_channels)
        
        # Residual prediction mode
        if self.residual_prediction:
            delta_target = target - prev_output
            pred_target = delta_target
        else:
            pred_target = target
        
        # Forward pass
        predicted = self.forward(conditioning, prev_output)
        
        # Compute loss
        loss = self.loss_fn(predicted, pred_target)
        
        # Log losses
        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        
        # Generate samples and log statistics
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, conditioning.shape[0])
                
                # Generate samples (handles residual reconstruction)
                samples = self.sample(
                    conditioning[:num_samples],
                    prev_output[:num_samples]
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
