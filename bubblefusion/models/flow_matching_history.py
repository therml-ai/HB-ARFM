"""
History-Window Conditional Flow Matching for BubbleFlow Prediction.

This module implements a history-conditioned flow matching baseline that
uses a sliding window of W conditioning frames to predict the target at the
last timestep. Pure channel concatenation -- no temporal encoder, no AR loop.

    Input:  [x_t_noisy, SDF_{t-W+1}, iVel_{t-W+1}, ..., SDF_t, iVel_t]
    Target: [velx_t, vely_t, temp_t]

Training equals inference: every sample sees the same sliding window of
observable conditioning (SDF + interface velocity). There is no autoregressive
chaining and therefore no exposure bias.

This serves as a strong history-aware baseline to isolate the contribution of
architectural components (bootstrap encoder, availability mask, push-forward)
in the AR Bootstrap model.

References:
    - "Flow Matching for Generative Modeling" (Lipman et al., 2023)
"""

import torch
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple

from bubblefusion.models.flow_matching import (
    FlowMatchingUNet,
    ConditionalFlowMatching,
)
from bubblefusion.models.loss_utils import build_loss_fn


class ConditionalFlowMatchingHistoryLightning(L.LightningModule):
    """
    History-window flow matching model (non-autoregressive).

    The dataset provides a flattened window of W conditioning frames as the
    input tensor: [W * C_cond, H, W].  The model concatenates this with the
    noisy target x_t and predicts the velocity field, exactly like the
    frame-to-frame ConditionalFlowMatchingLightning but with a wider input.

    Args:
        model_cfg: Model configuration (must include ``history_window``)
        optim_cfg: Optimizer configuration
        scheduler_cfg: LR scheduler configuration
        task_cfg: Task configuration (channel indices)
        normalization_stats: Pre-computed normalization statistics
        norm_mode: Normalization mode ('all', 'none', 'temperature_only')
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        optim_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        task_cfg: Optional[DictConfig] = None,
        normalization_stats: Optional[dict] = None,
        norm_mode: str = 'all',
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])

        self.normalization_stats = normalization_stats
        self.norm_mode = norm_mode
        self.task_cfg = task_cfg

        # --- Task channels ---------------------------------------------------
        if task_cfg is not None:
            self.conditioning_channels = list(task_cfg.get('conditioning_channels', [0]))
            self.target_channels = list(task_cfg.get('target_channels', [0]))
            self.task_name = task_cfg.get('name', 'unknown')
            self.is_noisy_task = 'noisy' in self.task_name.lower()
            noise_cfg = task_cfg.get('noise_cfg', None)
            self.has_noise = noise_cfg is not None and noise_cfg.get('enabled', True)
        else:
            self.conditioning_channels = [0]
            self.target_channels = [0]
            self.task_name = 'temperature_from_sdf'
            self.is_noisy_task = False
            self.has_noise = False

        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)

        # --- History window ---------------------------------------------------
        # history_window: number of conditioning frames W in the window.
        # history_stride: stride S between those frames in raw timesteps.
        #   The window spans (W-1)*S + 1 raw frames, but the channel count
        #   (and therefore the compute) only depends on W.  So increasing S
        #   lets the model see a longer time horizon for free.
        self.history_window = model_cfg.get('history_window', 10)
        self.history_stride = int(model_cfg.get('history_stride', 1))
        if self.history_stride < 1:
            raise ValueError(
                f"history_stride must be >= 1, got {self.history_stride}"
            )
        self.history_span = (self.history_window - 1) * self.history_stride + 1

        # BulkFlowHistory always emits a fixed set of raw conditioning channels
        # per frame ([sdf, velx_interface, vely_interface]).  Task-specific
        # selection is done by the model so that the dataset can stay
        # task-agnostic, exactly like the frame-to-frame flow_matching model.
        self._raw_cond_channels_per_frame = int(
            model_cfg.get('raw_cond_channels_per_frame', 3)
        )
        max_cond_idx = max(self.conditioning_channels) if self.conditioning_channels else -1
        if max_cond_idx >= self._raw_cond_channels_per_frame:
            raise ValueError(
                f"task_cfg.conditioning_channels contains index {max_cond_idx}, "
                f"but the dataset only emits {self._raw_cond_channels_per_frame} "
                f"conditioning channels per frame."
            )

        # Precompute the flattened-window indices we need for each task.  The
        # dataset returns frames concatenated along channels:
        #   [c0_t0, c1_t0, ..., c_{R-1}_t0, c0_t1, ..., c_{R-1}_t_{W-1}]
        # where R = raw_cond_channels_per_frame.  We pick the task-specific
        # conditioning_channels from every frame.
        self._history_channel_indices = [
            k * self._raw_cond_channels_per_frame + c
            for k in range(self.history_window)
            for c in self.conditioning_channels
        ]

        # UNet channels: x_t (C_out) + W * C_cond (flattened history window)
        in_channels = self.num_target_channels + self.history_window * self.num_conditioning_channels
        out_channels = self.num_target_channels

        print(f"\n🔄 History-Window Flow Matching Configuration:")
        print(f"   History window (W): {self.history_window}")
        print(f"   History stride (S): {self.history_stride}")
        print(f"   Temporal span:      {self.history_span} raw frames "
              f"(= (W-1)*S + 1)")
        print(f"   Raw conditioning channels per frame (dataset): "
              f"{self._raw_cond_channels_per_frame}")
        print(f"   Selected conditioning channels per frame (task): "
              f"{self.num_conditioning_channels} -> {self.conditioning_channels}")
        print(f"   UNet in_channels: {in_channels} = {self.num_target_channels} (x_t) + {self.history_window}*{self.num_conditioning_channels} (history)")
        print(f"   UNet out_channels: {out_channels}")

        # --- Architecture options ---------------------------------------------
        use_attention_old = model_cfg.get('use_attention', False)
        attention_type = model_cfg.get('attention_type', 'bottleneck' if use_attention_old else 'none')
        adaptive_scale = model_cfg.get('adaptive_scale', False)
        skip_scale = model_cfg.get('skip_scale', False)

        print(f"   attention_type: {attention_type}")
        print(f"   adaptive_scale: {adaptive_scale}")
        print(f"   skip_scale: {skip_scale}")

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

        # --- Normalization ----------------------------------------------------
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
        else:
            self.temp_min = model_cfg.get('temp_min', 55.0)
            self.temp_max = model_cfg.get('temp_max', 120.0)
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
            self.sdf_scale = 1.0

        # --- Inference --------------------------------------------------------
        self.num_integration_steps = model_cfg.get('num_integration_steps', 50)
        inference_cfg = model_cfg.get('inference', {})
        self.default_solver = inference_cfg.get('solver', 'heun')

        print(f"   Solver: {self.default_solver}")
        print(f"   Integration steps: {self.num_integration_steps}")

        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg

    # ------------------------------------------------------------------
    # Forward / helpers
    # ------------------------------------------------------------------

    def forward(self, x_t, condition, t):
        return self.flow_matching(x_t, condition, t)

    def denormalize_temperature(self, temperature_norm: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == 'none':
            return temperature_norm
        return (temperature_norm + 1.0) / 2.0 * self.temp_range + self.temp_min

    def denormalize_velocity(self, velocity_norm: torch.Tensor) -> torch.Tensor:
        if self.norm_mode in ('none', 'temperature_only'):
            return velocity_norm
        return velocity_norm * self.unified_velocity_scale

    def _extract_channels(self, tensor: torch.Tensor, channel_indices: list) -> torch.Tensor:
        return tensor[:, channel_indices, :, :]

    def extract_history_conditioning(self, history_flat: torch.Tensor) -> torch.Tensor:
        """Select task conditioning_channels from every frame of a flattened window.

        Args:
            history_flat: [B, W * raw_cond_channels_per_frame, H, W] tensor as
                emitted by ``BulkFlowHistory``.

        Returns:
            [B, W * num_conditioning_channels, H, W] tensor with only the
            channels listed in ``task_cfg.conditioning_channels`` kept for each
            of the W frames.
        """
        return history_flat[:, self._history_channel_indices, :, :]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        # Dataset returns (conditioning_window_flat, output_data)
        # conditioning_window_flat: [B, W * raw_cond_channels_per_frame, H, W]
        # output_data:              [B, C_out_raw, H, W]
        input_data, output_data = batch

        conditioning = self.extract_history_conditioning(input_data)
        target = self._extract_channels(output_data, self.target_channels)

        batch_size = conditioning.shape[0]
        device = conditioning.device

        t = torch.rand(batch_size, device=device)
        x_0 = torch.randn_like(target)
        x_t, velocity_target = self.flow_matching.compute_conditional_flow(x_0, target, t)
        velocity_pred = self.flow_matching(x_t, conditioning, t)

        loss = self.loss_fn(velocity_pred, velocity_target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        is_clean_val = (dataloader_idx == 0) if self.has_noise else True
        val_prefix = "val_clean" if is_clean_val else "val_noisy"

        input_data, output_data = batch

        conditioning = self.extract_history_conditioning(input_data)
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

        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, batch_size)
                samples = self.flow_matching.sample(
                    conditioning[:num_samples],
                    (num_samples, self.num_target_channels, target.shape[2], target.shape[3]),
                    device,
                    num_integration_steps=self.num_integration_steps,
                    solver=self.default_solver,
                )

                self.log(f'{val_prefix}_sample_mean_norm', samples.mean(), on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_sample_std_norm', samples.std(), on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_mean_norm', target[:num_samples].mean(), on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_std_norm', target[:num_samples].std(), on_step=False, on_epoch=True, add_dataloader_idx=False)

                if self.task_cfg is not None and 'temperature' in self.task_cfg.get('target_names', []):
                    temp_idx = list(self.task_cfg.get('target_names', [])).index('temperature')
                    samples_celsius = self.denormalize_temperature(samples[:, temp_idx:temp_idx+1])
                    target_celsius = self.denormalize_temperature(target[:num_samples, temp_idx:temp_idx+1])

                    self.log(f'{val_prefix}_pred_temp_min_C', samples_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_pred_temp_max_C', samples_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_min_C', target_celsius.min(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_max_C', target_celsius.max(), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

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
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        elif self.scheduler_cfg.name.lower() == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_cfg.get('T_0', 10),
                T_mult=self.scheduler_cfg.get('T_mult', 2),
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01,
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        else:
            return optimizer

