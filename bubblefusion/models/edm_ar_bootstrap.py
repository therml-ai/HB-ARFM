"""
Autoregressive EDM with Bootstrap Initialization.

This module implements an Autoregressive EDM (Elucidating the Design Space of
Diffusion-Based Generative Models) model that can operate in two modes:

1. Bootstrap Mode (previous state missing):
   - Uses a history encoder to infer the initial bulk state from
     a sequence of conditioning inputs (SDF, interface velocity)
   - No zeros fed - the model explicitly learns to estimate initial state

2. Autoregressive Mode (previous state exists):
   - Standard AR prediction using previous timestep output
   - EDM-style diffusion for each frame

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
- HistoryEncoder / TemporalMixer: Temporal encoder from flow_matching_ar_bootstrap
- EDMPrecond + SongUNet: EDM denoising backbone from edm.py
- Availability mask channel: Indicates whether previous state is available

References:
    - EDM: "Elucidating the Design Space of Diffusion-Based Generative Models" (Karras et al., 2022)
    - "Scheduled Sampling for Sequence Prediction with RNNs" (Bengio et al., 2015)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple
import numpy as np

from bubblefusion.models.edm import EDMPrecond
from bubblefusion.models.flow_matching_ar_bootstrap import (
    HistoryEncoder,
    TemporalMixer,
    AttentionHistoryEncoder,
)


class ConditionalEDMARBootstrap(nn.Module):
    """
    Autoregressive EDM model with Bootstrap Initialization.

    Wraps EDMPrecond with a HistoryEncoder. The conditioning tensor passed
    to EDMPrecond is extended to include prev_output and an optional
    availability mask channel.

    Args:
        edm_precond: The EDM preconditioned denoising network
        history_encoder: The history encoder for bootstrap mode
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        sigma_data: Expected data standard deviation
        rho: Noise schedule exponent
        use_availability_mask: Whether to use availability mask channel
    """

    def __init__(
        self,
        edm_precond: EDMPrecond,
        history_encoder: nn.Module,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        sigma_data: float = 0.5,
        rho: float = 7,
        use_availability_mask: bool = True,
    ):
        super().__init__()
        self.edm_precond = edm_precond
        self.history_encoder = history_encoder
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.use_availability_mask = use_availability_mask

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        availability_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoising forward pass with AR conditioning.

        Args:
            x_noisy: Noisy target [B, C_out, H, W]
            sigma: Noise levels [B]
            condition: Current conditioning (SDF, interface vel) [B, C_cond, H, W]
            prev_output: Previous timestep output [B, C_out, H, W]
            availability_mask: [B, 1, H, W] - 1=real prev, 0=bootstrapped

        Returns:
            Denoised prediction [B, C_out, H, W]
        """
        if self.use_availability_mask and availability_mask is not None:
            full_condition = torch.cat([condition, prev_output, availability_mask], dim=1)
        else:
            full_condition = torch.cat([condition, prev_output], dim=1)

        return self.edm_precond(x_noisy, sigma, condition=full_condition)

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

    def sample_noise_level(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise levels using log-normal distribution (EDM style)."""
        rnd_normal = torch.randn([batch_size], device=device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()
        return sigma

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: int = 50,
        solver: str = 'euler',
        availability_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples using EDM ODE sampling.

        Args:
            condition: Current conditioning [B, C_cond, H, W]
            prev_output: Previous timestep output [B, C_out, H, W]
            shape: Output shape (B, C_out, H, W)
            device: Device
            num_steps: Number of sampling steps
            solver: 'euler' or 'heun'
            availability_mask: [B, 1, H, W]

        Returns:
            Generated samples [B, C_out, H, W]
        """
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices / (num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_next = torch.randn(shape, device=device) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            t_hat = t_cur

            sigma = t_hat.expand(shape[0])
            denoised = self.forward(x_cur, sigma, condition, prev_output, availability_mask)
            d_cur = (x_cur - denoised) / t_hat
            x_next = x_cur + (t_next - t_hat) * d_cur

            if solver == 'heun' and t_next > 0:
                sigma_next = t_next.expand(shape[0])
                denoised_next = self.forward(x_next, sigma_next, condition, prev_output, availability_mask)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_cur + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    @torch.no_grad()
    def sample_trajectory(
        self,
        conditions: torch.Tensor,
        conditioning_history: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        solver: str = 'heun',
    ) -> torch.Tensor:
        """
        Generate a full trajectory with automatic bootstrap initialization.

        Args:
            conditions: Conditioning for all timesteps [T, B, C_cond, H, W]
            conditioning_history: History for bootstrap [B, T_hist, C_cond, H, W]
            initial_state: Optional initial state [B, C_out, H, W]
            num_steps: Steps per frame
            solver: ODE solver

        Returns:
            Generated trajectory [T, B, C_out, H, W]
        """
        T = conditions.shape[0]
        device = conditions.device
        B = conditions.shape[1]
        H, W = conditions.shape[3], conditions.shape[4]
        C_out = self.history_encoder.out_channels

        if initial_state is None:
            if conditioning_history is None:
                raise ValueError(
                    "Either initial_state or conditioning_history must be provided."
                )
            prev_output = self.bootstrap_initial_state(
                conditioning_history,
                current_conditioning=conditions[0],
            )
            is_bootstrapped = True
        else:
            prev_output = initial_state
            is_bootstrapped = False

        trajectory = []

        for t in range(T):
            condition_t = conditions[t]

            if self.use_availability_mask:
                if t == 0 and is_bootstrapped:
                    availability_mask = torch.zeros(B, 1, H, W, device=device)
                else:
                    availability_mask = torch.ones(B, 1, H, W, device=device)
            else:
                availability_mask = None

            output_t = self.sample(
                condition_t,
                prev_output,
                (B, C_out, H, W),
                device,
                num_steps,
                solver=solver,
                availability_mask=availability_mask,
            )

            trajectory.append(output_t)
            prev_output = output_t

        return torch.stack(trajectory, dim=0)


class EDMARBootstrapLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for AR EDM with Bootstrap.

    Training Strategy:
    - Sample trajectory segments of length rollout_length
    - First frame uses bootstrap mode (infer initial state from history)
    - Subsequent frames use AR mode with teacher forcing or scheduled sampling
    - Both bootstrap and AR losses trained jointly (EDM sigma-weighted MSE)

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

        # Parse task config
        if task_cfg is not None:
            self.conditioning_channels = list(task_cfg.get('conditioning_channels', [0]))
            self.target_channels = list(task_cfg.get('target_channels', [0]))
            self.task_name = task_cfg.get('name', 'unknown')
            print(f"Task: {self.task_name}")
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

        print(f"\nBootstrap AR EDM Configuration:")
        print(f"   History length: {self.history_length}")
        print(f"   History stride: {self.history_stride} (spans {self.history_length * self.history_stride} timesteps)")
        print(f"   Rollout length: {self.rollout_length}")
        print(f"   Use availability mask: {self.use_availability_mask}")

        # EDM parameters
        sigma_min = model_cfg.get('sigma_min', 0.002)
        sigma_max = model_cfg.get('sigma_max', 80)
        sigma_data = model_cfg.get('sigma_data', 0.5)
        rho = model_cfg.get('rho', 7)

        # Resolution
        self.downsample_factor = model_cfg.get('downsample_factor', 1)
        if self.downsample_factor == 1 and normalization_stats is not None:
            self.downsample_factor = normalization_stats.get('downsample_factor', 1)

        base_resolution = model_cfg.get('base_resolution', 512)
        img_resolution = base_resolution // self.downsample_factor

        default_attn_res = max(img_resolution // 8, 8)
        attn_resolutions = model_cfg.get('attn_resolutions', [default_attn_res])

        # Conditioning channels for EDMPrecond:
        # cond_channels = C_cond + C_out (prev) + 1 (mask, if enabled)
        extra_mask_channels = 1 if self.use_availability_mask else 0
        cond_channels = (
            self.num_conditioning_channels
            + self.num_target_channels
            + extra_mask_channels
        )

        print(f"   Image resolution: {img_resolution}x{img_resolution}")
        print(f"   Attention resolutions: {attn_resolutions}")
        print(f"   EDMPrecond img_channels: {self.num_target_channels}")
        print(f"   EDMPrecond cond_channels: {cond_channels}")
        print(f"   Model channels: {model_cfg.get('model_channels', 32)}")

        # Build EDMPrecond
        edm_precond = EDMPrecond(
            img_resolution=img_resolution,
            img_channels=self.num_target_channels,
            cond_channels=cond_channels,
            label_dim=0,
            use_fp16=model_cfg.get('use_fp16', False),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            model_type='SongUNet',
            model_channels=model_cfg.get('model_channels', 32),
            channel_mult=model_cfg.get('channel_mult', [1, 2, 4]),
            channel_mult_emb=model_cfg.get('channel_mult_emb', 4),
            num_blocks=model_cfg.get('num_blocks', 2),
            attn_resolutions=attn_resolutions,
            dropout=model_cfg.get('dropout', 0.10),
            embedding_type=model_cfg.get('embedding_type', 'positional'),
            channel_mult_noise=model_cfg.get('channel_mult_noise', 1),
            encoder_type=model_cfg.get('encoder_type', 'standard'),
            decoder_type=model_cfg.get('decoder_type', 'standard'),
            resample_filter=model_cfg.get('resample_filter', [1, 1]),
        )

        # Build history encoder
        history_encoder_type = model_cfg.get('history_encoder_type', 'conv3d')
        if history_encoder_type == 'temporal_mixer':
            history_encoder = TemporalMixer(
                in_channels=self.num_conditioning_channels,
                out_channels=self.num_target_channels,
                history_length=self.history_length,
                hidden_channels=model_cfg.get('history_encoder_hidden', 32),
                use_spatial_conv=model_cfg.get('temporal_mixer_spatial_conv', True),
                use_temporal_weights=model_cfg.get('temporal_mixer_temporal_weights', True),
            )
            print(f"   History Encoder: TemporalMixer (fast)")
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
            print(f"   History Encoder: Attention (factored space-time transformer)")
        else:
            history_encoder = HistoryEncoder(
                in_channels=self.num_conditioning_channels,
                out_channels=self.num_target_channels,
                hidden_channels=model_cfg.get('history_encoder_hidden', 64),
                num_temporal_blocks=model_cfg.get('history_encoder_blocks', 3),
            )
            print(f"   History Encoder: Conv3D (expressive)")

        # Build combined model
        self.edm_ar = ConditionalEDMARBootstrap(
            edm_precond=edm_precond,
            history_encoder=history_encoder,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
            use_availability_mask=self.use_availability_mask,
        )

        self.loss_fn = nn.MSELoss()

        # Inference configuration
        inference_cfg = model_cfg.get('inference', {})
        self.default_solver = inference_cfg.get('solver', 'heun')
        self.num_sampling_steps = model_cfg.get('num_sampling_steps', 50)

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

        # Scheduled sampling
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
            print(f"\n   Scheduled Sampling: ENABLED")
            print(f"   Schedule type: {self.ss_schedule_type}")
            print(f"   Warmup epochs: {self.ss_warmup_epochs}")
            print(f"   Transition epochs: {self.ss_transition_epochs}")
            print(f"   Final teacher ratio: {self.ss_min_teacher_ratio:.1%}")
        else:
            print(f"\n   Scheduled Sampling: DISABLED")

        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x_noisy, sigma, condition, prev_output, availability_mask=None):
        return self.edm_ar(x_noisy, sigma, condition, prev_output, availability_mask)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        prev_output: torch.Tensor,
        shape: Tuple[int, ...],
        device: torch.device,
        availability_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        # Accept and ignore extra kwargs for API compatibility
        num_integration_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using EDM ODE."""
        if num_steps is None:
            num_steps = num_integration_steps if num_integration_steps is not None else self.num_sampling_steps
        if solver is None:
            solver = self.default_solver
        return self.edm_ar.sample(
            condition, prev_output, shape, device,
            num_steps, solver, availability_mask,
        )

    @torch.no_grad()
    def bootstrap_initial_state(
        self,
        conditioning_history: torch.Tensor,
        current_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bootstrap initial state from conditioning history."""
        return self.edm_ar.bootstrap_initial_state(
            conditioning_history, current_conditioning
        )

    @torch.no_grad()
    def sample_trajectory(
        self,
        conditions: torch.Tensor,
        conditioning_history: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate full trajectory with automatic bootstrap."""
        if num_steps is None:
            num_steps = self.num_sampling_steps
        if solver is None:
            solver = self.default_solver
        return self.edm_ar.sample_trajectory(
            conditions, conditioning_history, initial_state,
            num_steps, solver,
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
        """Compute teacher forcing ratio for scheduled sampling."""
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
                self.ss_exponential_decay ** transition_epoch,
            )
        elif self.ss_schedule_type == 'inverse_sigmoid':
            k = self.ss_sigmoid_k
            x = k * (progress - 0.5)
            sigmoid_val = 1.0 / (1.0 + math.exp(-x))
            teacher_ratio = 1.0 - sigmoid_val * (1.0 - self.ss_min_teacher_ratio)
        else:
            teacher_ratio = 1.0 - progress * (1.0 - self.ss_min_teacher_ratio)

        return max(self.ss_min_teacher_ratio, min(1.0, teacher_ratio))

    def training_step(self, batch, batch_idx):
        """
        Training step with joint bootstrap and AR training using EDM loss.

        Batch contains a trajectory segment:
        - conditioning_history: [B, T_hist, C_cond, H, W]
        - conditioning_sequence: [B, L, C_cond, H, W]
        - target_sequence: [B, L, C_out, H, W]
        """
        conditioning_history, conditioning_sequence, target_sequence = batch[:3]

        cond_hist = self._extract_channels(conditioning_history, self.conditioning_channels)
        cond_seq = self._extract_channels(conditioning_sequence, self.conditioning_channels)
        target_seq = self._extract_channels(target_sequence, self.target_channels)

        B, L, C_cond, H, W = cond_seq.shape
        C_out = target_seq.shape[2]
        device = cond_seq.device
        sigma_data = self.edm_ar.sigma_data

        total_loss = 0.0
        bootstrap_loss_total = 0.0
        ar_loss_total = 0.0

        # ==== BOOTSTRAP TRAINING (first frame) ====
        current_cond_0 = cond_seq[:, 0]
        bootstrapped_state = self.edm_ar.bootstrap_initial_state(cond_hist, current_cond_0)

        # Direct supervision on bootstrapped state
        target_0 = target_seq[:, 0]
        bootstrap_state_loss = self.loss_fn(bootstrapped_state, target_0)
        bootstrap_loss_total = bootstrap_loss_total + bootstrap_state_loss * 0.5

        # EDM diffusion loss for first frame (bootstrap mode)
        rnd_normal_0 = torch.randn([B], device=device)
        sigma_0 = (rnd_normal_0 * 1.2 - 1.2).exp()
        noise_0 = torch.randn_like(target_0)
        x_noisy_0 = target_0 + noise_0 * sigma_0.view(-1, 1, 1, 1)

        if self.use_availability_mask:
            availability_mask_0 = torch.zeros(B, 1, H, W, device=device)
        else:
            availability_mask_0 = None

        denoised_0 = self.edm_ar(x_noisy_0, sigma_0, current_cond_0, bootstrapped_state, availability_mask_0)

        weight_0 = (sigma_0 ** 2 + sigma_data ** 2) / (sigma_0 * sigma_data) ** 2
        weight_0 = weight_0.view(-1, 1, 1, 1)
        bootstrap_fm_loss = (weight_0 * (denoised_0 - target_0) ** 2).mean()
        bootstrap_loss_total = bootstrap_loss_total + bootstrap_fm_loss

        # ==== AR TRAINING (subsequent frames) ====
        teacher_ratio = self.get_teacher_forcing_ratio()
        prev_output = bootstrapped_state

        for l in range(1, L):
            current_cond = cond_seq[:, l]
            target_l = target_seq[:, l]

            # Sample sigma and add noise
            rnd_normal_l = torch.randn([B], device=device)
            sigma_l = (rnd_normal_l * 1.2 - 1.2).exp()
            noise_l = torch.randn_like(target_l)
            x_noisy_l = target_l + noise_l * sigma_l.view(-1, 1, 1, 1)

            if self.use_availability_mask:
                availability_mask = torch.ones(B, 1, H, W, device=device)
            else:
                availability_mask = None

            denoised_l = self.edm_ar(x_noisy_l, sigma_l, current_cond, prev_output, availability_mask)

            weight_l = (sigma_l ** 2 + sigma_data ** 2) / (sigma_l * sigma_data) ** 2
            weight_l = weight_l.view(-1, 1, 1, 1)
            ar_loss_l = (weight_l * (denoised_l - target_l) ** 2).mean()
            ar_loss_total = ar_loss_total + ar_loss_l

            # Update prev_output for next iteration
            if self.scheduled_sampling_enabled and teacher_ratio < 1.0:
                use_teacher = torch.rand(1).item() < teacher_ratio
                if not use_teacher:
                    with torch.no_grad():
                        prev_output = self.sample(
                            current_cond, prev_output,
                            (B, C_out, H, W), device,
                            availability_mask=availability_mask,
                            num_steps=self.ss_sampling_steps,
                            solver='euler',
                        )
                else:
                    prev_output = target_l.clone()
            else:
                prev_output = target_l.clone()

        # Average AR loss
        if L > 1:
            ar_loss_total = ar_loss_total / (L - 1)

        total_loss = bootstrap_loss_total + ar_loss_total

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_bootstrap_loss', bootstrap_loss_total, on_step=False, on_epoch=True)
        self.log('train_bootstrap_state_loss', bootstrap_state_loss, on_step=False, on_epoch=True)
        self.log('train_ar_loss', ar_loss_total, on_step=False, on_epoch=True)

        if self.scheduled_sampling_enabled:
            self.log('teacher_ratio', teacher_ratio, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step."""
        conditioning_history, conditioning_sequence, target_sequence = batch[:3]

        cond_hist = self._extract_channels(conditioning_history, self.conditioning_channels)
        cond_seq = self._extract_channels(conditioning_sequence, self.conditioning_channels)
        target_seq = self._extract_channels(target_sequence, self.target_channels)

        B, L, C_cond, H, W = cond_seq.shape
        C_out = target_seq.shape[2]
        device = cond_seq.device
        sigma_data = self.edm_ar.sigma_data

        is_clean_val = (dataloader_idx == 0)
        suffix = "_clean" if is_clean_val else "_noisy"

        # Bootstrap evaluation
        current_cond_0 = cond_seq[:, 0]
        bootstrapped_state = self.edm_ar.bootstrap_initial_state(cond_hist, current_cond_0)
        target_0 = target_seq[:, 0]
        bootstrap_state_loss = self.loss_fn(bootstrapped_state, target_0)

        # EDM diffusion loss for bootstrap frame
        rnd_normal_0 = torch.randn([B], device=device)
        sigma_0 = (rnd_normal_0 * 1.2 - 1.2).exp()
        noise_0 = torch.randn_like(target_0)
        x_noisy_0 = target_0 + noise_0 * sigma_0.view(-1, 1, 1, 1)

        if self.use_availability_mask:
            availability_mask_0 = torch.zeros(B, 1, H, W, device=device)
        else:
            availability_mask_0 = None

        denoised_0 = self.edm_ar(x_noisy_0, sigma_0, current_cond_0, bootstrapped_state, availability_mask_0)
        weight_0 = (sigma_0 ** 2 + sigma_data ** 2) / (sigma_0 * sigma_data) ** 2
        weight_0 = weight_0.view(-1, 1, 1, 1)
        bootstrap_fm_loss = (weight_0 * (denoised_0 - target_0) ** 2).mean()

        # AR evaluation (rollout-aware: start from bootstrap)
        ar_loss_total = 0.0
        prev_output = bootstrapped_state

        for l in range(1, L):
            current_cond = cond_seq[:, l]
            target_l = target_seq[:, l]

            rnd_normal_l = torch.randn([B], device=device)
            sigma_l = (rnd_normal_l * 1.2 - 1.2).exp()
            noise_l = torch.randn_like(target_l)
            x_noisy_l = target_l + noise_l * sigma_l.view(-1, 1, 1, 1)

            if self.use_availability_mask:
                availability_mask = torch.ones(B, 1, H, W, device=device)
            else:
                availability_mask = None

            denoised_l = self.edm_ar(x_noisy_l, sigma_l, current_cond, prev_output, availability_mask)
            weight_l = (sigma_l ** 2 + sigma_data ** 2) / (sigma_l * sigma_data) ** 2
            weight_l = weight_l.view(-1, 1, 1, 1)
            ar_loss_l = (weight_l * (denoised_l - target_l) ** 2).mean()
            ar_loss_total = ar_loss_total + ar_loss_l
            prev_output = target_l

        if L > 1:
            ar_loss_total = ar_loss_total / (L - 1)

        val_loss = bootstrap_fm_loss + bootstrap_state_loss * 0.5 + ar_loss_total

        if is_clean_val:
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log(f'val_loss{suffix}', val_loss, on_step=False, on_epoch=True, prog_bar=not is_clean_val)
        self.log(f'val_bootstrap_state_loss{suffix}', bootstrap_state_loss, on_step=False, on_epoch=True)
        self.log(f'val_bootstrap_fm_loss{suffix}', bootstrap_fm_loss, on_step=False, on_epoch=True)
        self.log(f'val_ar_loss{suffix}', ar_loss_total, on_step=False, on_epoch=True)

        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(2, B)
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
                print("Lion optimizer not available, falling back to AdamW")
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
