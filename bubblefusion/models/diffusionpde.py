"""
DiffusionPDE: Generative PDE-Solving Under Partial Observation (NeurIPS 2024)

Faithful baseline implementation of:
    https://github.com/jhhuangchloe/DiffusionPDE
    https://arxiv.org/abs/2406.17763

Key differences from our conditional EDM (edm.py):
    1. Training is UNCONDITIONAL on the JOINT distribution of all fields
       (observed + target stacked as channels).
    2. At inference, two gradient-based guidance signals steer the sampling:
       - Observation guidance: pushes denoised estimate to match known fields
       - PDE guidance: enforces divergence-free constraint on velocity

The guidance loop follows the original two-phase schedule:
    - First 80% of steps: observation guidance only
    - Last  20% of steps: reduced observation + PDE (divergence-free) guidance

This is a FRAME-TO-FRAME model (non-autoregressive):
    Input:  conditioning (used as observation for guidance at inference)
    Target: output fields

References:
    - DiffusionPDE: https://arxiv.org/abs/2406.17763
    - EDM: https://arxiv.org/abs/2206.00364
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from typing import Optional, Tuple

from bubblefusion.models.edm import EDMPrecond


# ==============================================================================
# PDE Guidance Utilities
# ==============================================================================

def compute_divergence(velx: torch.Tensor, vely: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
    """
    Compute velocity divergence via central finite differences.

    div(V) = du/dx + dv/dy

    Uses the same kernel as DiffusionPDE's generate_ns_bounded.py:
        [-1, 0, 1] / 2  (central differences)

    Args:
        velx: [B, 1, H, W] x-velocity component
        vely: [B, 1, H, W] y-velocity component
        device: torch device

    Returns:
        divergence: [B, 1, H, W]
    """
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=velx.dtype, device=device).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=vely.dtype, device=device).view(1, 1, 3, 1) / 2

    dudx = F.conv2d(velx, deriv_x, padding=(0, 1))
    dvdy = F.conv2d(vely, deriv_y, padding=(1, 0))
    div = dudx + dvdy

    # Zero out boundary rows/columns (following original DiffusionPDE)
    div[:, :, 0, :] = 0
    div[:, :, -1, :] = 0
    div[:, :, :, 0] = 0
    div[:, :, :, -1] = 0
    return div


# ==============================================================================
# DiffusionPDE Model (unconditional EDM with guided sampling)
# ==============================================================================

class DiffusionPDEModel(nn.Module):
    """
    Unconditional EDM diffusion model with PDE-guided sampling.

    Training: learns the joint distribution p(observed, target) via standard
    EDM denoising on all channels simultaneously with no conditioning.

    Inference: Heun ODE sampler augmented with gradient guidance from
    observation consistency and divergence-free PDE constraint.

    Args:
        edm_precond: EDMPrecond model (cond_channels=0, unconditional)
        sigma_min / sigma_max / sigma_data / rho: EDM noise schedule params
        num_observed: number of leading channels that are "observed"
        velx_joint_idx: index of velx in the joint channel layout
        vely_joint_idx: index of vely in the joint channel layout
        sdf_joint_idx: index of SDF in the joint channel layout
        zeta_obs: observation guidance weight
        zeta_pde: PDE guidance weight
        pde_start_fraction: fraction of steps before PDE guidance kicks in
        pde_obs_decay: observation weight multiplier during PDE phase
        bulk_sdf_threshold: SDF > threshold is considered bulk liquid
    """

    def __init__(self, edm_precond: EDMPrecond,
                 sigma_min=0.002, sigma_max=80, sigma_data=0.5, rho=7,
                 num_observed: int = 3,
                 velx_joint_idx: int = 3, vely_joint_idx: int = 4,
                 sdf_joint_idx: int = 0,
                 zeta_obs: float = 1.0, zeta_pde: float = 0.5,
                 pde_start_fraction: float = 0.8, pde_obs_decay: float = 0.1,
                 bulk_sdf_threshold: float = 0.05):
        super().__init__()
        self.edm_precond = edm_precond
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

        self.num_observed = num_observed
        self.velx_joint_idx = velx_joint_idx
        self.vely_joint_idx = vely_joint_idx
        self.sdf_joint_idx = sdf_joint_idx

        self.zeta_obs = zeta_obs
        self.zeta_pde = zeta_pde
        self.pde_start_fraction = pde_start_fraction
        self.pde_obs_decay = pde_obs_decay
        self.bulk_sdf_threshold = bulk_sdf_threshold

    def forward(self, x_noisy: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
        """Unconditional denoising pass (no condition argument)."""
        return self.edm_precond(x_noisy, sigma)

    @torch.no_grad()
    def sample_unguided(self, shape: Tuple[int, ...],
                        device: torch.device, num_steps: int = 50,
                        solver: str = 'heun') -> torch.Tensor:
        """Standard (unguided) EDM sampling for validation speed."""
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (self.sigma_max ** (1 / self.rho) +
                   step_indices / (num_steps - 1) *
                   (self.sigma_min ** (1 / self.rho) -
                    self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_next = torch.randn(shape, device=device) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            sigma = t_cur.expand(shape[0])
            denoised = self.forward(x_cur, sigma)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            if solver == 'heun' and t_next > 0:
                sigma_next = t_next.expand(shape[0])
                denoised_next = self.forward(x_next, sigma_next)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sample_with_guidance(self, observed_gt: torch.Tensor,
                             shape: Tuple[int, ...],
                             device: torch.device,
                             num_steps: int = 50,
                             solver: str = 'heun') -> torch.Tensor:
        """
        DiffusionPDE guided sampling.

        Args:
            observed_gt: ground-truth observed channels [B, num_observed, H, W]
            shape: full joint shape (B, C_joint, H, W)
            device: torch device
            num_steps: number of denoising steps
            solver: 'euler' or 'heun'

        Returns:
            Predicted target channels [B, C_target, H, W]
        """
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (self.sigma_max ** (1 / self.rho) +
                   step_indices / (num_steps - 1) *
                   (self.sigma_min ** (1 / self.rho) -
                    self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_next = torch.randn(shape, device=device, dtype=torch.float64) * t_steps[0]

        pde_start_step = int(self.pde_start_fraction * num_steps)
        has_velocity = (
            self.velx_joint_idx >= 0
            and self.vely_joint_idx >= 0
            and self.velx_joint_idx < shape[1]
            and self.vely_joint_idx < shape[1]
        )

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)

            sigma = t_cur.expand(shape[0])

            # --- Heun denoising step (with gradient tracking) ---
            denoised = self.forward(x_cur.float(), sigma.float()).to(torch.float64)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            if solver == 'heun' and i < num_steps - 1:
                sigma_next = t_next.expand(shape[0])
                denoised_2 = self.forward(x_next.float(), sigma_next.float()).to(torch.float64)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

            # --- Observation guidance ---
            obs_pred = denoised[:, :self.num_observed]
            obs_gt = observed_gt.to(denoised.dtype)
            L_obs = torch.norm(obs_pred - obs_gt, 2)
            grad_obs = torch.autograd.grad(L_obs, x_cur, retain_graph=True)[0]

            # --- PDE guidance (divergence-free on velocity) ---
            grad_pde = torch.zeros_like(x_cur)
            if has_velocity and i > pde_start_step:
                velx = denoised[:, self.velx_joint_idx:self.velx_joint_idx + 1]
                vely = denoised[:, self.vely_joint_idx:self.vely_joint_idx + 1]
                div = compute_divergence(velx, vely, device)

                # Mask to bulk liquid (SDF > threshold)
                sdf = denoised[:, self.sdf_joint_idx:self.sdf_joint_idx + 1]
                bulk_mask = (sdf > self.bulk_sdf_threshold).to(div.dtype)
                div = div * bulk_mask

                H, W = shape[2], shape[3]
                L_pde = torch.norm(div, 2) / (H * W)
                grad_pde = torch.autograd.grad(L_pde, x_cur)[0]

            # --- Apply guidance (two-phase schedule) ---
            if i <= pde_start_step:
                x_next = x_next - self.zeta_obs * grad_obs
            else:
                x_next = (x_next
                          - self.pde_obs_decay * self.zeta_obs * grad_obs
                          - self.zeta_pde * grad_pde)

        # Return only the target (non-observed) channels, detached from the autograd graph
        return x_next[:, self.num_observed:].detach().float()


# ==============================================================================
# PyTorch Lightning Module
# ==============================================================================

class DiffusionPDELightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for the DiffusionPDE baseline.

    Training: unconditional EDM on the joint distribution of all fields.
    Inference: PDE-guided + observation-guided sampling.

    Args:
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        scheduler_cfg: Learning rate scheduler configuration
        task_cfg: Task configuration defining channel layout
        normalization_stats: Pre-computed normalization statistics
    """

    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig,
                 scheduler_cfg: DictConfig,
                 task_cfg: Optional[DictConfig] = None,
                 normalization_stats: Optional[dict] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])

        self.normalization_stats = normalization_stats
        self.task_cfg = task_cfg

        # ----- Parse task config -----
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
        else:
            self.conditioning_channels = [0]
            self.target_channels = [0]
            self.task_name = 'temperature_from_sdf'
            self.is_noisy_task = False
            self.has_noise = False

        self.num_conditioning_channels = len(self.conditioning_channels)
        self.num_target_channels = len(self.target_channels)

        # Joint channels = observed (conditioning) + target
        self.num_joint_channels = self.num_conditioning_channels + self.num_target_channels

        # Velocity channel indices inside the joint tensor:
        # joint layout = [observed..., target...]
        # For Task 2: observed=[sdf, vix, viy], target=[vx, vy, T]
        #   velx_joint_idx = 3 (first target channel = vx)
        #   vely_joint_idx = 4
        # For Task 1: observed=[sdf], target=[T] -> no velocity guidance
        target_names = list(task_cfg.get('target_names', [])) if task_cfg else []
        self.velx_joint_idx = -1
        self.vely_joint_idx = -1
        for i, name in enumerate(target_names):
            joint_i = self.num_conditioning_channels + i
            if name == 'velx':
                self.velx_joint_idx = joint_i
            elif name == 'vely':
                self.vely_joint_idx = joint_i

        # ----- Resolution / downsample -----
        self.downsample_factor = model_cfg.get('downsample_factor', 1)
        if self.downsample_factor == 1 and normalization_stats is not None:
            self.downsample_factor = normalization_stats.get('downsample_factor', 1)

        base_resolution = model_cfg.get('base_resolution', 512)
        img_resolution = base_resolution // self.downsample_factor

        default_attn_res = max(img_resolution // 8, 8)
        attn_resolutions = model_cfg.get('attn_resolutions', [default_attn_res])

        print(f"\nDiffusionPDE Configuration:")
        print(f"   Image resolution: {img_resolution}x{img_resolution}")
        print(f"   Joint channels: {self.num_joint_channels} "
              f"({self.num_conditioning_channels} observed + "
              f"{self.num_target_channels} target)")
        print(f"   Model channels: {model_cfg.get('model_channels', 128)}")
        print(f"   Channel mult: {model_cfg.get('channel_mult', [1, 2, 2, 2])}")
        print(f"   Attention resolutions: {attn_resolutions}")

        # ----- Build unconditional EDMPrecond -----
        edm_precond = EDMPrecond(
            img_resolution=img_resolution,
            img_channels=self.num_joint_channels,
            cond_channels=0,
            label_dim=0,
            use_fp16=model_cfg.get('use_fp16', False),
            sigma_min=model_cfg.get('sigma_min', 0.002),
            sigma_max=model_cfg.get('sigma_max', 80),
            sigma_data=model_cfg.get('sigma_data', 0.5),
            model_type='SongUNet',
            model_channels=model_cfg.get('model_channels', 128),
            channel_mult=model_cfg.get('channel_mult', [1, 2, 2, 2]),
            channel_mult_emb=model_cfg.get('channel_mult_emb', 4),
            num_blocks=model_cfg.get('num_blocks', 4),
            attn_resolutions=attn_resolutions,
            dropout=model_cfg.get('dropout', 0.10),
            embedding_type=model_cfg.get('embedding_type', 'positional'),
            channel_mult_noise=model_cfg.get('channel_mult_noise', 1),
            encoder_type=model_cfg.get('encoder_type', 'standard'),
            decoder_type=model_cfg.get('decoder_type', 'standard'),
            resample_filter=model_cfg.get('resample_filter', [1, 1]),
        )

        # Guidance hyperparameters
        zeta_obs = model_cfg.get('zeta_obs', 1.0)
        zeta_pde = model_cfg.get('zeta_pde', 0.5)
        pde_start_fraction = model_cfg.get('pde_start_fraction', 0.8)
        pde_obs_decay = model_cfg.get('pde_obs_decay', 0.1)
        bulk_sdf_threshold = model_cfg.get('bulk_sdf_threshold', 0.05)

        print(f"\n   Guidance:")
        print(f"     zeta_obs={zeta_obs}, zeta_pde={zeta_pde}")
        print(f"     PDE starts at step fraction {pde_start_fraction}")
        print(f"     Obs decay in PDE phase: {pde_obs_decay}")
        print(f"     Bulk SDF threshold: {bulk_sdf_threshold}")
        if self.velx_joint_idx < 0:
            print(f"     (No velocity channels found -- PDE guidance disabled)")

        self.diffusion_pde = DiffusionPDEModel(
            edm_precond=edm_precond,
            sigma_min=model_cfg.get('sigma_min', 0.002),
            sigma_max=model_cfg.get('sigma_max', 80),
            sigma_data=model_cfg.get('sigma_data', 0.5),
            rho=model_cfg.get('rho', 7),
            num_observed=self.num_conditioning_channels,
            velx_joint_idx=self.velx_joint_idx,
            vely_joint_idx=self.vely_joint_idx,
            sdf_joint_idx=0,
            zeta_obs=zeta_obs,
            zeta_pde=zeta_pde,
            pde_start_fraction=pde_start_fraction,
            pde_obs_decay=pde_obs_decay,
            bulk_sdf_threshold=bulk_sdf_threshold,
        )

        # ----- Normalization stats -----
        if normalization_stats is not None and 'temperature' in normalization_stats:
            temp_stats = normalization_stats['temperature']
            self.temp_min = temp_stats['min']
            self.temp_max = temp_stats['max']
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = normalization_stats.get(
                'unified_velocity_scale', 1.0)
            self.sdf_scale = normalization_stats.get('sdf', {}).get('scale', 1.0)
            print(f"\n   Normalization stats:")
            print(f"     Temperature: [{self.temp_min:.2f}, {self.temp_max:.2f}]C")
            print(f"     Velocity scale: {self.unified_velocity_scale:.4f}")
            print(f"     SDF scale: {self.sdf_scale:.4f}")
        else:
            self.temp_min = model_cfg.get('temp_min', 55.0)
            self.temp_max = model_cfg.get('temp_max', 120.0)
            self.temp_range = self.temp_max - self.temp_min
            self.unified_velocity_scale = 1.0
            self.sdf_scale = 1.0

        self.num_sampling_steps = model_cfg.get('num_sampling_steps', 50)
        self.default_solver = model_cfg.get('solver', 'heun')

        print(f"\n   Sampling: {self.num_sampling_steps} steps, solver={self.default_solver}")

        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_channels(self, tensor: torch.Tensor,
                          channel_indices: list) -> torch.Tensor:
        return tensor[:, channel_indices, :, :]

    def denormalize_temperature(self, t_norm: torch.Tensor) -> torch.Tensor:
        return (t_norm + 1.0) / 2.0 * self.temp_range + self.temp_min

    def denormalize_velocity(self, v_norm: torch.Tensor) -> torch.Tensor:
        return v_norm * self.unified_velocity_scale

    def _build_joint(self, input_data: torch.Tensor,
                     output_data: torch.Tensor) -> torch.Tensor:
        """Concatenate observed (from input) and target (from output) channels."""
        observed = self._extract_channels(input_data, self.conditioning_channels)
        target = self._extract_channels(output_data, self.target_channels)
        return torch.cat([observed, target], dim=1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        input_data, output_data = batch
        joint = self._build_joint(input_data, output_data)

        batch_size = joint.shape[0]
        device = joint.device

        rnd_normal = torch.randn([batch_size], device=device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()

        noise = torch.randn_like(joint)
        x_noisy = joint + noise * sigma.view(-1, 1, 1, 1)

        denoised = self.diffusion_pde(x_noisy, sigma)

        weight = (sigma ** 2 + self.diffusion_pde.sigma_data ** 2) / \
                 (sigma * self.diffusion_pde.sigma_data) ** 2
        weight = weight.view(-1, 1, 1, 1)

        loss = (weight * (denoised - joint) ** 2).mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        is_clean_val = (dataloader_idx == 0) if self.has_noise else True
        val_prefix = "val_clean" if is_clean_val else "val_noisy"

        input_data, output_data = batch
        joint = self._build_joint(input_data, output_data)

        batch_size = joint.shape[0]
        device = joint.device

        rnd_normal = torch.randn([batch_size], device=device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()

        noise = torch.randn_like(joint)
        x_noisy = joint + noise * sigma.view(-1, 1, 1, 1)

        denoised = self.diffusion_pde(x_noisy, sigma)

        weight = (sigma ** 2 + self.diffusion_pde.sigma_data ** 2) / \
                 (sigma * self.diffusion_pde.sigma_data) ** 2
        weight = weight.view(-1, 1, 1, 1)

        loss = (weight * (denoised - joint) ** 2).mean()

        self.log(f'{val_prefix}_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, add_dataloader_idx=False)
        if is_clean_val:
            self.log('val_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=False, add_dataloader_idx=False)

        # Quick unguided sample for statistics (faster than guided)
        if batch_idx == 0 and is_clean_val:
            with torch.no_grad():
                num_samples = min(4, batch_size)
                joint_shape = (num_samples, self.num_joint_channels,
                               joint.shape[2], joint.shape[3])
                samples = self.diffusion_pde.sample_unguided(
                    joint_shape, device,
                    num_steps=min(self.num_sampling_steps, 25),
                    solver=self.default_solver,
                )
                target = self._extract_channels(output_data[:num_samples],
                                                self.target_channels)
                # Target channels in the joint sample
                samples_target = samples[:, self.num_conditioning_channels:]

                self.log(f'{val_prefix}_sample_mean_norm',
                         samples_target.mean(), on_step=False, on_epoch=True,
                         add_dataloader_idx=False)
                self.log(f'{val_prefix}_sample_std_norm',
                         samples_target.std(), on_step=False, on_epoch=True,
                         add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_mean_norm',
                         target.mean(), on_step=False, on_epoch=True,
                         add_dataloader_idx=False)
                self.log(f'{val_prefix}_target_std_norm',
                         target.std(), on_step=False, on_epoch=True,
                         add_dataloader_idx=False)

                if self.task_cfg is not None and 'temperature' in self.task_cfg.get('target_names', []):
                    target_names = list(self.task_cfg.get('target_names', []))
                    temp_idx = target_names.index('temperature')
                    s_temp = self.denormalize_temperature(
                        samples_target[:, temp_idx:temp_idx + 1])
                    t_temp = self.denormalize_temperature(
                        target[:, temp_idx:temp_idx + 1])
                    self.log(f'{val_prefix}_pred_temp_min_C', s_temp.min(),
                             on_step=False, on_epoch=True, prog_bar=True,
                             add_dataloader_idx=False)
                    self.log(f'{val_prefix}_pred_temp_max_C', s_temp.max(),
                             on_step=False, on_epoch=True, prog_bar=True,
                             add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_min_C', t_temp.min(),
                             on_step=False, on_epoch=True, prog_bar=True,
                             add_dataloader_idx=False)
                    self.log(f'{val_prefix}_target_temp_max_C', t_temp.max(),
                             on_step=False, on_epoch=True, prog_bar=True,
                             add_dataloader_idx=False)

        return loss

    # ------------------------------------------------------------------
    # Optimizer / Scheduler (identical to EDMLightning)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        if self.optim_cfg.name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-3),
                weight_decay=self.optim_cfg.get('weight_decay', 0.0))
        elif self.optim_cfg.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optim_cfg.get('lr', 1e-3),
                weight_decay=self.optim_cfg.get('weight_decay', 1e-2))
        elif self.optim_cfg.name.lower() == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-4),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2))
            except ImportError:
                print("Lion optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.optim_cfg.get('lr', 1e-3),
                    weight_decay=self.optim_cfg.get('weight_decay', 1e-2))
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_cfg.name}")

        if self.scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01)
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        elif self.scheduler_cfg.name.lower() == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_cfg.get('T_0', 10),
                T_mult=self.scheduler_cfg.get('T_mult', 2),
                eta_min=self.optim_cfg.get('lr', 1e-3) * 0.01)
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        else:
            return optimizer
