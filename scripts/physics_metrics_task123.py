#!/usr/bin/env python3
"""
Physics Inference Metrics Script for Task 1, Task 2 & Task 3

Evaluates model's inference physics metrics compared with ground truth simulation data.
Outputs a comprehensive CSV table with the following metrics:

Task 1 (temperature_from_sdf) - Temperature only metrics:
1. Interface temperature at the liquid-vapor interface (where SDF ≈ 0)
2. Averaged wall heat flux across time and frames
3. Row temperature analysis for rows 0, 8, 16, 24, 32

Task 2/3 (velocity_from_interface) - Full metrics:
1. Velocity divergence (mass conservation): ∇·V = ∂u/∂x + ∂v/∂y
2. Interface temperature at the liquid-vapor interface (where SDF ≈ 0)
3. Averaged wall heat flux across time and frames
4. Row temperature analysis for rows 0, 8, 16, 24, 32

Supported Models:
- flow_matching: Standard Flow Matching model (velocity prediction)
- flow_matching_jit: Data-prediction Flow Matching model
- flow_matching_ar: Autoregressive Flow Matching model
- flow_matching_ar_bootstrap: AR Flow Matching with Bootstrap
- unet_ar: Autoregressive UNet (direct regression)

Usage:
    python physics_metrics_task2.py --checkpoint /path/to/checkpoint.ckpt --data-file /path/to/data.hdf5
    
    # For AR models (autoregressive rollout):
    python physics_metrics_task2.py --checkpoint /path/to/ar_model.ckpt --model-type flow_matching_ar
    python physics_metrics_task2.py --checkpoint /path/to/unet_ar.ckpt --model-type unet_ar
    
Output:
    CSV file with physics metrics comparison between ground truth and predictions

Note on Units (BubbleML Dataset):
---------------------------------
The BubbleML simulation data uses NON-DIMENSIONAL quantities:
  - Coordinates: Scaled by characteristic length lc = 0.73e-3 m (FC-72)
  - Velocities: Non-dimensional (v* = v / U_ref)
  - Temperature: Dimensional (°C) - NOT non-dimensionalized
  - SDF: Non-dimensional (scaled by lc)
  - Heat flux: Converted to W/m² using lc and thermal conductivity

The grid uses:
  - x ∈ [-8, 8], y ∈ [0, 16] (non-dimensional)
  - dx = dy = 1/32 (non-dimensional grid spacing)
  - Physical grid spacing: dx_phys = (1/32) * 0.73e-3 m ≈ 22.8 µm
"""

import sys
import os
import argparse
import json
import numpy as np
import h5py as h5
import torch
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bubblefusion.models.flow_matching import ConditionalFlowMatchingLightning
from bubblefusion.models.flow_matching_history import ConditionalFlowMatchingHistoryLightning
from bubblefusion.models.flow_matching_jit import ConditionalFlowMatchingJiTLightning
from bubblefusion.models.flow_matching_ar import ConditionalFlowMatchingARLightning
from bubblefusion.models.flow_matching_ar_bootstrap import ConditionalFlowMatchingARBootstrapLightning
from bubblefusion.models.unet import UNetLightning
from bubblefusion.models.unet_ar import UNetARLightning
from bubblefusion.models.ddpm import BubbleDDPMLightning
from bubblefusion.models.ve_sde import ScoreBasedVESDELightning
from bubblefusion.models.ffno import FFNOLightning
from bubblefusion.models.edm import EDMLightning
from bubblefusion.models.edm_ar_bootstrap import EDMARBootstrapLightning
from bubblefusion.models.diffusionpde import DiffusionPDELightning
from bubblefusion.data import BulkFlow, BulkFlowAutoregressive, BulkFlowHistory
from bubblefusion.data.bubbleml import compute_normalization_stats, BulkFlowARBootstrap

# ============================================================================
# Utility Functions
# ============================================================================


def compute_bootstrap_ablation_state(
    ablation_mode: str,
    cond_hist_extracted: torch.Tensor,
    C_out: int,
    device: torch.device,
    target_names: list = None,
    conditioning_names: list = None,
) -> torch.Tensor:
    """Compute an alternative initial state for bootstrap ablation experiments.

    Instead of using the learned history encoder, produce a simple initial
    ``prev_output`` in **target space** for the AR bootstrap model.

    Args:
        ablation_mode: One of ``'zeros'`` or ``'mean_conditioning_naive'``.
        cond_hist_extracted: Conditioning history ``[B, T, C_cond, H, W]``.
        C_out: Number of target channels.
        device: Torch device.
        target_names: List of target field names (e.g. ``['velx', 'vely', 'temperature']``).
        conditioning_names: List of conditioning field names
            (e.g. ``['sdf', 'velx_interface', 'vely_interface']``).

    Returns:
        Tensor ``[B, C_out, H, W]`` in target space.
    """
    B, T, C_cond, H, W = cond_hist_extracted.shape

    if ablation_mode == 'zeros':
        return torch.zeros(B, C_out, H, W, device=device)

    if ablation_mode == 'mean_conditioning_naive':
        prev_output = torch.zeros(B, C_out, H, W, device=device)
        mean_cond = cond_hist_extracted.mean(dim=1)  # [B, C_cond, H, W]

        if target_names is None:
            target_names = ['velx', 'vely', 'temperature']
        if conditioning_names is None:
            conditioning_names = ['sdf', 'velx_interface', 'vely_interface']

        if 'velx_interface' in conditioning_names and 'velx' in target_names:
            src = conditioning_names.index('velx_interface')
            dst = target_names.index('velx')
            prev_output[:, dst, :, :] = mean_cond[:, src, :, :]
        if 'vely_interface' in conditioning_names and 'vely' in target_names:
            src = conditioning_names.index('vely_interface')
            dst = target_names.index('vely')
            prev_output[:, dst, :, :] = mean_cond[:, src, :, :]
        return prev_output

    raise ValueError(
        f"Unknown bootstrap_ablation mode: '{ablation_mode}'. "
        "Use 'zeros' or 'mean_conditioning_naive'."
    )


def load_task_config(task_name: str = 'velocity_from_interface') -> DictConfig:
    """Load task configuration from YAML file."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'bubblefusion', 'config', 'task_cfg', f'{task_name}.yaml'
    )
    
    if os.path.exists(config_path):
        task_cfg = OmegaConf.load(config_path)
        print(f"✓ Loaded task config: {task_name}")
        print(f"   Conditioning channels: {task_cfg.conditioning_channels} ({task_cfg.conditioning_names})")
        print(f"   Target channels: {task_cfg.target_channels} ({task_cfg.target_names})")
        # Log noise configuration for Task 3
        if 'noise_cfg' in task_cfg and task_cfg.noise_cfg.get('enabled', False):
            noise_type = task_cfg.noise_cfg.get('noise_type', 'optical_flow')
            print(f"   🔊 Noise enabled: {noise_type}")
            if noise_type in ['gaussian', 'simple']:
                print(f"      SDF noise std: {task_cfg.noise_cfg.get('sdf_noise_std', 0.1)}")
                print(f"      Vel noise std: {task_cfg.noise_cfg.get('vel_noise_std', 0.05)}")
        return task_cfg
    else:
        print(f"⚠️  Task config not found: {config_path}")
        # Return default for velocity_from_interface
        return DictConfig({
            'name': 'velocity_from_interface',
            'conditioning_channels': [0, 1, 2],
            'conditioning_names': ['sdf', 'velx_interface', 'vely_interface'],
            'target_channels': [1, 2, 0],  # velx, vely, temperature from [temp, velx, vely]
            'target_names': ['velx', 'vely', 'temperature']
        })


def load_model_from_checkpoint(checkpoint_path: str, model_cfg: DictConfig, 
                               optim_cfg: DictConfig, scheduler_cfg: DictConfig,
                               task_cfg: DictConfig = None,
                               model_type: str = 'flow_matching',
                               normalization_stats: dict = None,
                               norm_mode: str = 'all'):
    """Load the trained model from checkpoint with task config.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        scheduler_cfg: Scheduler configuration
        task_cfg: Task configuration
        model_type: Model type ('flow_matching', 'flow_matching_ar', 'flow_matching_ar_bootstrap',
                               'unet_ar', 'bubble_ddpm')
        normalization_stats: Normalization statistics for accurate denormalization during inference
        norm_mode: Normalization mode ('none', 'all', 'temperature_only')
    """
    print(f"\n🤖 Loading model from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if model_type == 'bubble_ddpm':
        print(f"📦 Loading DDPM model...")
        model = BubbleDDPMLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False  # Allow missing/extra keys
        )
        print(f"✓ DDPM model loaded successfully!")
        print(f"   Timesteps: {model.ddpm.num_timesteps}")
        print(f"   Inference steps: {model.num_inference_steps}")
    elif model_type == 've_sde':
        print(f"📦 Loading VE-SDE Score-Based model...")
        model = ScoreBasedVESDELightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False  # Allow missing/extra keys
        )
        print(f"✓ VE-SDE model loaded successfully!")
        print(f"   σ_min: {model.sigma_min}, σ_max: {model.sigma_max}")
        print(f"   Sampling method: {model.sampling_method}")
        print(f"   Sampling steps: {model.num_sampling_steps}")
    elif model_type == 'flow_matching_ar':
        print(f"📦 Loading Autoregressive Flow Matching model...")
        model = ConditionalFlowMatchingARLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False  # Allow missing/extra keys
        )
        print(f"✓ Autoregressive Flow Matching model loaded successfully!")
        print(f"   Residual prediction: {model.residual_prediction}")
        print(f"   Default solver: {model.default_solver}")
        print(f"   Default guidance: {model.default_guidance_scale}")
    elif model_type == 'flow_matching_ar_bootstrap':
        print(f"📦 Loading Autoregressive Flow Matching with Bootstrap model...")
        model = ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False  # Allow missing/extra keys
        )
        print(f"✓ Autoregressive Flow Matching with Bootstrap model loaded successfully!")
        print(f"   Bootstrap: Uses history encoder to infer initial state")
        print(f"   History length: {model_cfg.get('history_length', 10)} frames")
        print(f"   Rollout length: {model_cfg.get('rollout_length', 5)} frames")
        print(f"   Default solver: {model.default_solver}")
        print(f"   Use availability mask: {model.use_availability_mask}")
    elif model_type == 'edm_ar_bootstrap':
        print(f"📦 Loading Autoregressive EDM with Bootstrap model...")
        model = EDMARBootstrapLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False,
        )
        print(f"✓ Autoregressive EDM with Bootstrap model loaded successfully!")
        print(f"   Bootstrap: Uses history encoder to infer initial state")
        print(f"   History length: {model_cfg.get('history_length', 10)} frames")
        print(f"   Rollout length: {model_cfg.get('rollout_length', 5)} frames")
        print(f"   Default solver: {model.default_solver}")
        print(f"   Use availability mask: {model.use_availability_mask}")
        print(f"   Sampling steps: {model.num_sampling_steps}")
    elif model_type == 'unet_ar':
        print(f"📦 Loading Autoregressive UNet model (direct regression)...")
        model = UNetARLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False
        )
        print(f"✓ Autoregressive UNet model loaded successfully!")
        print(f"   Residual prediction: {model.residual_prediction}")
        print(f"   Note: Direct regression (no diffusion/flow matching)")
    elif model_type == 'unet':
        print(f"📦 Loading UNet model (direct regression)...")
        model = UNetLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False
        )
        print(f"✓ UNet model loaded successfully!")
        print(f"   Note: Direct regression (no diffusion/flow matching)")
        print(f"   Single forward pass (fast inference)")
    elif model_type == 'ffno':
        print(f"📦 Loading FFNO model (Factorized Fourier Neural Operator)...")
        model = FFNOLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False
        )
        print(f"✓ FFNO model loaded successfully!")
        print(f"   Note: Spectral method (Fourier Neural Operator)")
        print(f"   Single forward pass (fast inference)")
    elif model_type == 'edm':
        print(f"📦 Loading EDM model (EDM-style diffusion)...")
        model = EDMLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False
        )
        print(f"✓ EDM model loaded successfully!")
        print(f"   Note: EDM-style diffusion (NeurIPS 2024 baseline)")
        print(f"   Model channels: {model_cfg.get('model_channels', 128)}")
        print(f"   Sampling steps: {model.num_sampling_steps}")
    elif model_type == 'diffusionpde':
        print(f"📦 Loading DiffusionPDE model (unconditional joint diffusion + guided sampling)...")
        model = DiffusionPDELightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False,
        )
        print(f"✓ DiffusionPDE model loaded successfully!")
        print(f"   Model channels: {model_cfg.get('model_channels', 128)}")
        print(f"   Sampling steps: {model.num_sampling_steps}")
        print(f"   Guidance: zeta_obs={model_cfg.get('zeta_obs', 1.0)}, "
              f"zeta_pde={model_cfg.get('zeta_pde', 0.5)}")
    elif model_type == 'flow_matching_jit':
        print(f"📦 Loading JiT Flow Matching model (Vision Transformer, data prediction)...")
        model = ConditionalFlowMatchingJiTLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            norm_mode=norm_mode,
            strict=False
        )
        print(f"✓ JiT Flow Matching model loaded successfully!")
        print(f"   Architecture: Vision Transformer (JiT)")
        print(f"   Solver: {model.default_solver}")
    elif model_type == 'flow_matching_history':
        print(f"📦 Loading History-Window Flow Matching model...")
        model = ConditionalFlowMatchingHistoryLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            norm_mode=norm_mode,
            strict=False,
        )
        print(f"✓ History-Window Flow Matching model loaded successfully!")
        print(f"   History window (W): {getattr(model, 'history_window', 'n/a')}")
        print(f"   History stride (S): {getattr(model, 'history_stride', 1)}")
        print(f"   Solver: {model.default_solver}")
    elif model_type == 'flow_matching':
        print(f"📦 Loading Flow Matching model...")
        model = ConditionalFlowMatchingLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            norm_mode=norm_mode
        )
        print(f"✓ Flow Matching model loaded successfully!")
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: flow_matching, flow_matching_history, flow_matching_jit, "
                        f"flow_matching_ar, flow_matching_ar_bootstrap, unet, unet_ar, "
                        f"bubble_ddpm, ve_sde, ffno, edm, diffusionpde")
    
    model.eval()
    print(f"   Task: {model.task_name}")
    print(f"   Conditioning channels: {model.conditioning_channels}")
    print(f"   Target channels: {model.target_channels}")
    return model


def load_dataset(
    data_file_path: str,
    output_fields=None,
    start_time: int = 100,
    return_wall_temp: bool = False,
    noise_cfg: Optional[Dict] = None,
    use_clean_inputs: bool = False,
    is_temporal: bool = False,
    is_autoregressive: bool = False,
    is_ar_bootstrap: bool = False,
    is_history_model: bool = False,
    history_window: int = 10,
    history_length: int = 50,
    temporal_stride: int = 1,
    history_stride: int = 1,
    rollout_length: int = 5,
    downsample_factor: int = 1,
    normalization_stats: Optional[Dict] = None,
    norm_mode: str = 'all',
    # Legacy parameters - ignored
    normalize_temperature: bool = True,
    normalizer=None,
    stats_file: str = None,
):
    """Load the BulkFlow, BulkFlowAutoregressive, or BulkFlowARBootstrap dataset.

    All fields are normalized according to NORMALIZATION_REQUIREMENTS.md:
    - SDF: Zero-preserving normalization
    - Velocity: Unified scale normalization
    - Temperature: Tanh normalization to [-1, 1]

    For Task 3, this can optionally apply the same noise model used at training
    (deployment-style evaluation) or disable it for clean-input physics checks.
    
    Args:
        data_file_path: Path to HDF5 data file
        output_fields: Output field names
        start_time: Starting timestep
        return_wall_temp: Whether to return wall temperature
        noise_cfg: Noise configuration dict (for Task 3)
        use_clean_inputs: If True, disable noise (for Task 3 clean inference)
        is_temporal: Unused (legacy parameter)
        is_autoregressive: If True, use BulkFlowAutoregressive for AR models
        is_ar_bootstrap: If True, use BulkFlowARBootstrap for AR Bootstrap models
        history_length: Number of historical frames (for bootstrap models)
        temporal_stride: Unused (legacy parameter)
        history_stride: Stride between history frames for AR bootstrap (1=consecutive)
        rollout_length: Number of frames in rollout segment (for AR Bootstrap models)
        downsample_factor: Factor to downsample spatial resolution (1 = no downsampling)
        normalization_stats: Pre-computed normalization statistics (if None, computed from data_file)
    """
    print(f"\n📂 Loading data from: {data_file_path}")

    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    if output_fields is None:
        output_fields = ['temperature', 'velx', 'vely']

    # Decide whether to apply noise (Task 3 only)
    effective_noise_cfg = None
    if noise_cfg is not None and not use_clean_inputs:
        effective_noise_cfg = dict(noise_cfg)
        print("   🔊 Using NOISY inputs for physics metrics (Task 3 deployment-style)")
    elif noise_cfg is not None and use_clean_inputs:
        print("   ✨ Using CLEAN inputs for physics metrics (Task 3 physics fidelity)")
    else:
        print("   ✨ No noise (Task 2 or clean evaluation)")
    
    # Log downsampling configuration
    if downsample_factor > 1:
        print(f"   📐 Downsampling: {downsample_factor}x (e.g., 512x512 → {512 // downsample_factor}x{512 // downsample_factor})")
    
    # Compute normalization stats if not provided
    if normalization_stats is None:
        print(f"   📊 Computing normalization stats from inference file...")
        normalization_stats = compute_normalization_stats(
            filenames=[data_file_path],
            start_time=start_time,
            verbose=True
        )
    else:
        print(f"   📊 Using provided normalization statistics")

    if is_ar_bootstrap:
        print(f"   🚀 Using AR BOOTSTRAP dataset:")
        print(f"      History length: {history_length} (for bootstrap)")
        print(f"      History stride: {history_stride} (spans {history_length * history_stride} timesteps)")
        print(f"      Rollout length: {rollout_length}")
        
        dataset = BulkFlowARBootstrap(
            filenames=[data_file_path],
            output_fields=output_fields,
            start_time=start_time,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=effective_noise_cfg,
            history_length=history_length,
            history_stride=history_stride,
            rollout_length=rollout_length,
            downsample_factor=downsample_factor,
            norm_mode=norm_mode
        )
        print(f"✓ AR Bootstrap dataset loaded: {len(dataset)} samples")
        print(f"   Each sample: condition_history [T_hist, C, H, W], condition_seq [L, C, H, W], target_seq [L, C, H, W]")
    elif is_autoregressive:
        print(f"   🔄 Using AUTOREGRESSIVE dataset:")
        print("      Returns: (conditioning_t, prev_output_{{t-1}}, target_t)")
        
        dataset = BulkFlowAutoregressive(
            filenames=[data_file_path],
            output_fields=output_fields,
            start_time=start_time,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=effective_noise_cfg,
            downsample_factor=downsample_factor,
            norm_mode=norm_mode
        )
        print(f"✓ Autoregressive dataset loaded: {len(dataset)} samples")
        print("   Each sample: (input_t, prev_output_{{t-1}}, target_t)")
    elif is_history_model:
        print(f"   Using HISTORY-WINDOW dataset (W={history_window}, S={history_stride})")
        dataset = BulkFlowHistory(
            filenames=[data_file_path],
            output_fields=output_fields,
            start_time=start_time,
            history_window=history_window,
            history_stride=history_stride,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=effective_noise_cfg,
            downsample_factor=downsample_factor,
            norm_mode=norm_mode
        )
        print(f"   History-window dataset loaded: {len(dataset)} samples")
    else:
        dataset = BulkFlow(
            filenames=[data_file_path],
            output_fields=output_fields,
            start_time=start_time,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=effective_noise_cfg,
            downsample_factor=downsample_factor,
            norm_mode=norm_mode
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples (from timestep {start_time} onwards)")

    print(f"   Output fields: {output_fields}")
    print(f"   ✓ All fields are normalized (temperature, velocity, SDF)")
    return dataset


def load_ground_truth_data(data_file_path: str, start_time: int = 100, 
                           downsample_factor: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load ground truth SDF, velocity, interface velocity, and temperature data from HDF5 file.
    
    Args:
        data_file_path: Path to HDF5 data file
        start_time: Starting timestep
        downsample_factor: Factor to downsample spatial resolution (1 = no downsampling)
        
    Returns:
        dfun: SDF field (T, H, W)
        temp: Temperature field (T, H, W)
        velx: Bulk x-velocity (T, H, W)
        vely: Bulk y-velocity (T, H, W)
        velx_interface: Interface x-velocity / mass flux (T, H, W)
        vely_interface: Interface y-velocity / mass flux (T, H, W)
        heater_temp: Heater temperature in Celsius
    """
    print(f"\n📂 Loading ground truth data from: {data_file_path}")
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    with h5.File(data_file_path, 'r') as f:
        dfun = f['dfun'][start_time:]  # (T, H, W)
        temp = f['temperature'][start_time:]  # (T, H, W)
        velx = f['velx'][start_time:]  # (T, H, W)
        vely = f['vely'][start_time:]  # (T, H, W)
        
        # Load interface velocities (mass flux fields)
        # These represent velocity at the bubble interface where phase change occurs
        if 'velx_interface' in f and 'vely_interface' in f:
            velx_interface = f['velx_interface'][start_time:]  # (T, H, W)
            vely_interface = f['vely_interface'][start_time:]  # (T, H, W)
            print(f"  ✓ Loaded interface velocities from HDF5")
        else:
            # Compute interface velocities from bulk velocities using SDF masking
            # Interface is where SDF is close to zero (within ~1-2 grid cells)
            interface_band = 0.1  # SDF threshold for interface region
            interface_region = np.abs(dfun) < interface_band
            velx_interface = velx * interface_region.astype(np.float32)
            vely_interface = vely * interface_region.astype(np.float32)
            print(f"  ✓ Computed interface velocities from bulk velocities (SDF band: {interface_band})")
        
        print(f"  ✓ Loaded {len(dfun)} timesteps")
        print(f"  ✓ Original shape: {dfun.shape}")
        
        # Apply downsampling if needed
        if downsample_factor > 1:
            from scipy.ndimage import zoom
            scale = 1.0 / downsample_factor
            # Downsample each field: (T, H, W) -> (T, H/factor, W/factor)
            dfun = zoom(dfun, (1, scale, scale), order=1)  # Bilinear
            temp = zoom(temp, (1, scale, scale), order=1)
            velx = zoom(velx, (1, scale, scale), order=1)
            vely = zoom(vely, (1, scale, scale), order=1)
            velx_interface = zoom(velx_interface, (1, scale, scale), order=1)
            vely_interface = zoom(vely_interface, (1, scale, scale), order=1)
            print(f"  ✓ Downsampled shape: {dfun.shape} ({downsample_factor}x)")
        
        # Extract wall temperature from filename
        filename = os.path.basename(data_file_path)
        if 'Twall_' in filename:
            heater_temp = float(filename.split('Twall_')[1].split('.')[0])
        elif 'inletVelScale' in filename:
            heater_temp = 103.0  # Default for flow boiling
        else:
            heater_temp = 103.0  # Default fallback
        
        print(f"  ✓ Heater temperature: {heater_temp}°C")
    
    return dfun, temp, velx, vely, velx_interface, vely_interface, heater_temp


def extract_channels(tensor: torch.Tensor, channel_indices: list) -> torch.Tensor:
    """Extract specific channels from a tensor based on channel indices."""
    return tensor[:, channel_indices, :, :]


# ============================================================================
# Physics Metrics Functions
# ============================================================================

def compute_velocity_divergence(velx: np.ndarray, vely: np.ndarray, 
                                 downsample_factor: int = 1) -> np.ndarray:
    """
    Compute velocity divergence (mass conservation check).
    
    For incompressible flow, ∇·V = ∂u/∂x + ∂v/∂y should be zero.
    
    Args:
        velx: (T, H, W) or (H, W) - x-component of velocity
        vely: (T, H, W) or (H, W) - y-component of velocity
        downsample_factor: Factor by which data was downsampled (adjusts dx/dy)
        
    Returns:
        divergence: Same shape as input - velocity divergence field
    """
    # Adjust grid spacing for downsampling
    # Original dx = dy = 1/32 for 512x512 grid
    dx = downsample_factor / 32
    dy = downsample_factor / 32
    
    # Standard finite difference
    if velx.ndim == 2:
        # Single frame: (H, W)
        dudx = np.gradient(velx, dx, axis=1)
        dvdy = np.gradient(vely, dy, axis=0)
    else:
        # Multiple frames: (T, H, W)
        dudx = np.gradient(velx, dx, axis=2)
        dvdy = np.gradient(vely, dy, axis=1)
    
    divergence = dudx + dvdy
    return divergence


def compute_interface_mask_massflux(velx_interface: np.ndarray, vely_interface: np.ndarray,
                                     threshold: float = 1e-2) -> np.ndarray:
    """
    Create a mask for interface cells where there is non-zero mass flux.
    
    The interface is where velx_interface or vely_interface have non-zero values,
    indicating active phase change (mass transfer) at the bubble interface.
    
    This is more physically meaningful for velocity analysis than SDF zero-crossing
    because:
    - SDF zero-crossing gives the geometric interface location
    - Mass flux detection gives where actual phase change is occurring
    - Velocity has discontinuity where there's mass transfer, not just at geometric interface
    
    Args:
        velx_interface: (T, H, W) or (H, W) - x-component of velocity at interface
        vely_interface: (T, H, W) or (H, W) - y-component of velocity at interface
        threshold: Minimum velocity magnitude to consider as interface (default: 1e-2)
        
    Returns:
        interface_mask: Boolean mask where True = interface cell with mass flux
    """
    # Cells with velocity magnitude above threshold are considered interface cells
    interface_mask = (np.abs(velx_interface) >= threshold) | (np.abs(vely_interface) >= threshold)
    
    return interface_mask


def compute_interface_mask_zero_crossing(sdf: np.ndarray) -> np.ndarray:
    """
    Create a mask for interface cells using zero-crossing detection.
    
    The interface is where SDF crosses zero - this is more accurate than
    using a threshold band. We find cells where SDF changes sign between
    adjacent cells.
    
    Args:
        sdf: (T, H, W) or (H, W) - signed distance function
        
    Returns:
        interface_mask: Boolean mask where True = interface cell
    """
    if sdf.ndim == 2:
        # Single frame: (H, W)
        # Check for sign change in x-direction
        sign_change_x = (sdf[:, :-1] * sdf[:, 1:]) < 0
        # Check for sign change in y-direction  
        sign_change_y = (sdf[:-1, :] * sdf[1:, :]) < 0
        
        # Mark cells on both sides of the crossing as interface cells
        sign_change_x_padded = np.zeros_like(sdf, dtype=bool)
        sign_change_x_padded[:, :-1] |= sign_change_x
        sign_change_x_padded[:, 1:] |= sign_change_x
        
        sign_change_y_padded = np.zeros_like(sdf, dtype=bool)
        sign_change_y_padded[:-1, :] |= sign_change_y
        sign_change_y_padded[1:, :] |= sign_change_y
        
        interface_mask = sign_change_x_padded | sign_change_y_padded
    else:
        # Multiple frames: (T, H, W)
        # Check for sign change in x-direction
        sign_change_x = (sdf[:, :, :-1] * sdf[:, :, 1:]) < 0
        # Check for sign change in y-direction
        sign_change_y = (sdf[:, :-1, :] * sdf[:, 1:, :]) < 0
        
        # Mark cells on both sides of the crossing as interface cells
        sign_change_x_padded = np.zeros_like(sdf, dtype=bool)
        sign_change_x_padded[:, :, :-1] |= sign_change_x
        sign_change_x_padded[:, :, 1:] |= sign_change_x
        
        sign_change_y_padded = np.zeros_like(sdf, dtype=bool)
        sign_change_y_padded[:, :-1, :] |= sign_change_y
        sign_change_y_padded[:, 1:, :] |= sign_change_y
        
        interface_mask = sign_change_x_padded | sign_change_y_padded
    
    return interface_mask


def compute_interface_temperature(sdf: np.ndarray, temperature: np.ndarray) -> Tuple[float, float]:
    """
    Compute interface temperature at the liquid-vapor interface.
    
    The interface is where dfun (SDF) crosses zero - the zero-level set that
    represents the exact boundary between liquid (SDF < 0) and vapor (SDF > 0).
    
    Method: Zero-Crossing Detection
    --------------------------------
    We find cells where the SDF changes sign between adjacent cells, which
    indicates the interface passes through that location. This is more 
    physically accurate than using an arbitrary threshold band.
    
    Args:
        sdf: (T, H, W) or (H, W) - signed distance function (dfun)
        temperature: (T, H, W) or (H, W) - temperature field
        
    Returns:
        mean_interface_temp: Mean temperature at interface
        std_interface_temp: Standard deviation of interface temperature
    """
    # Find cells where SDF crosses zero (sign changes between neighbors)
    # When sdf[i] * sdf[i+1] < 0, the sign changed → interface passes between them
    
    if sdf.ndim == 2:
        # Single frame: (H, W)
        # Check for sign change in x-direction
        sign_change_x = (sdf[:, :-1] * sdf[:, 1:]) < 0
        # Check for sign change in y-direction  
        sign_change_y = (sdf[:-1, :] * sdf[1:, :]) < 0
        
        # Mark cells on both sides of the crossing as interface cells
        sign_change_x_padded = np.zeros_like(sdf, dtype=bool)
        sign_change_x_padded[:, :-1] |= sign_change_x
        sign_change_x_padded[:, 1:] |= sign_change_x
        
        sign_change_y_padded = np.zeros_like(sdf, dtype=bool)
        sign_change_y_padded[:-1, :] |= sign_change_y
        sign_change_y_padded[1:, :] |= sign_change_y
        
        interface_mask = sign_change_x_padded | sign_change_y_padded
    else:
        # Multiple frames: (T, H, W)
        # Check for sign change in x-direction
        sign_change_x = (sdf[:, :, :-1] * sdf[:, :, 1:]) < 0
        # Check for sign change in y-direction
        sign_change_y = (sdf[:, :-1, :] * sdf[:, 1:, :]) < 0
        
        # Mark cells on both sides of the crossing as interface cells
        sign_change_x_padded = np.zeros_like(sdf, dtype=bool)
        sign_change_x_padded[:, :, :-1] |= sign_change_x
        sign_change_x_padded[:, :, 1:] |= sign_change_x
        
        sign_change_y_padded = np.zeros_like(sdf, dtype=bool)
        sign_change_y_padded[:, :-1, :] |= sign_change_y
        sign_change_y_padded[:, 1:, :] |= sign_change_y
        
        interface_mask = sign_change_x_padded | sign_change_y_padded
    
    # Extract temperatures at interface
    interface_temps = temperature[interface_mask]
    
    if len(interface_temps) == 0:
        return np.nan, np.nan
    
    mean_temp = np.mean(interface_temps)
    std_temp = np.std(interface_temps)
    
    return mean_temp, std_temp


def compute_heatflux(dfun: np.ndarray, temp: np.ndarray, heater_temp: float,
                     lc: float = 0.73e-3, thcl: float = 6.25e-2,
                     downsample_factor: int = 1) -> np.ndarray:
    """
    Calculate heat flux for FC-72 fluid.
    
    Reference: scripts/heatflux_inference.py
    
    Args:
        dfun: (T, H, W) np.ndarray - signed distance function
        temp: (T, H, W) np.ndarray - temperature field
        heater_temp: heater temperature in Celsius
        lc: characteristic length (0.73e-3 for FC-72)
        thcl: thermal conductivity of liquid (6.25e-2 for FC-72)
        downsample_factor: Factor by which data was downsampled
        
    Returns:
        hfluxes: (T,) np.ndarray - heat flux time series
    """
    # Get actual dimensions from input
    T_frames, H, W = dfun.shape
    
    # Adjust grid spacing for downsampling
    # Original dx = 1/32 for 512x512 grid
    # After downsampling by factor N, effective dx = N/32
    dx = downsample_factor / 32

    x_min, x_max = -8, 8
    y_min, y_max = 0, 16
    
    # Use actual resolution from input data
    x_centers = x_min + (np.arange(W) + 0.5) * dx
    y_centers = y_min + (np.arange(H) + 0.5) * dx

    x_grid, _ = np.meshgrid(x_centers, y_centers)

    heater_mask = (x_grid >= -5) & (x_grid <= 5)  # (H, W)
    heater_mask_3d = np.broadcast_to(heater_mask, (T_frames, H, W))  # (T, H, W)

    liquid_mask = dfun < 0  # (T, H, W)
    temp_fields = (heater_mask_3d & liquid_mask).astype(float) * (heater_temp - temp)  # (T, H, W)
    hflux_fields = thcl * (temp_fields / (dx * 0.5 * lc))
    hfluxes = hflux_fields[:, 0, :].mean(axis=1)

    return hfluxes


def compute_row_temperature(temperature: np.ndarray, row_indices: List[int]) -> Dict[int, Tuple[float, float]]:
    """
    Compute average temperature along specified rows.
    
    Reference: scripts/temperature_row_analysis.py
    
    Args:
        temperature: (T, H, W) np.ndarray - temperature field
        row_indices: List of row indices (y-coordinates) to analyze
        
    Returns:
        row_temps: Dict mapping row_index -> (mean_temp, std_temp)
    """
    row_temps = {}
    
    for row_y in row_indices:
        if row_y >= temperature.shape[1]:
            print(f"  ⚠️ Row {row_y} exceeds temperature field height {temperature.shape[1]}, skipping")
            continue
        
        # Extract temperature profiles for this row across all frames
        row_data = temperature[:, row_y, :]  # [T, W]
        
        # Average across frames and positions
        mean_temp = np.mean(row_data)
        std_temp = np.std(row_data)
        
        row_temps[row_y] = (mean_temp, std_temp)
    
    return row_temps


def compute_vorticity(velx: np.ndarray, vely: np.ndarray, 
                      downsample_factor: int = 1) -> np.ndarray:
    """
    Compute vorticity (curl of velocity) to check swirling/rotational flow patterns.
    
    Vorticity in 2D: ω = ∂v/∂x - ∂u/∂y
    
    Positive vorticity indicates counter-clockwise rotation.
    Negative vorticity indicates clockwise rotation.
    High vorticity magnitude indicates strong rotational flow (e.g., around bubbles).
    
    Args:
        velx: (T, H, W) or (H, W) - x-component of velocity (u)
        vely: (T, H, W) or (H, W) - y-component of velocity (v)
        downsample_factor: Factor by which data was downsampled (adjusts dx/dy)
        
    Returns:
        vorticity: Same shape as input - vorticity field (ω)
    """
    # Adjust grid spacing for downsampling
    # Original dx = dy = 1/32 for 512x512 grid
    dx = downsample_factor / 32
    dy = downsample_factor / 32
    
    if velx.ndim == 2:
        # Single frame: (H, W)
        dvdx = np.gradient(vely, dx, axis=1)  # ∂v/∂x
        dudy = np.gradient(velx, dy, axis=0)  # ∂u/∂y
    else:
        # Multiple frames: (T, H, W)
        dvdx = np.gradient(vely, dx, axis=2)  # ∂v/∂x
        dudy = np.gradient(velx, dy, axis=1)  # ∂u/∂y
    
    vorticity = dvdx - dudy
    return vorticity


def compute_region_masks(sdf: np.ndarray, 
                         velx: Optional[np.ndarray], 
                         vely: Optional[np.ndarray],
                         near_wall_rows: int = 16,
                         downsample_factor: int = 1) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create masks for different regions of the domain.
    
    Returns SEPARATE region masks for temperature and velocity analysis:
    - Temperature regions: Use SDF-based (zero-crossing) interface detection
    - Velocity regions: Use massflux-based interface detection (if velocity available)
    
    For Task 1 (temperature only), velocity arrays can be None and only
    temperature region masks will be meaningful.
    
    SDF Convention:
    - SDF < 0: liquid phase (outside bubble)
    - SDF > 0: vapor phase (inside bubble)
    - SDF = 0: bubble interface
    
    Regions for each field type:
    - Near-wall: Bottom rows of the domain (y < near_wall_rows)
    - Inside-bubble: Vapor region (SDF > 0) AND not at interface
    - Near-interface: Cells at the interface (SDF-based for temp, massflux-based for velocity)
    - Bulk-liquid: Liquid region (SDF < 0) AND not at interface AND not near wall
    
    Args:
        sdf: (T, H, W) - signed distance function
        velx: (T, H, W) - x-component of BULK velocity (for massflux threshold detection), can be None
        vely: (T, H, W) - y-component of BULK velocity (for massflux threshold detection), can be None
        near_wall_rows: Number of rows from bottom to consider as near-wall (at full resolution)
        downsample_factor: Factor by which data was downsampled
        
    Returns:
        Dict with 'temperature' and 'velocity' keys, each containing region masks
    """
    T, H, W = sdf.shape
    
    # Scale near_wall_rows for downsampling
    scaled_near_wall_rows = near_wall_rows // downsample_factor
    
    # Create y-coordinate array
    y_coords = np.arange(H)
    y_grid = np.broadcast_to(y_coords[None, :, None], (T, H, W))
    
    # Near-wall: bottom rows (y < scaled_near_wall_rows) - same for both
    near_wall_mask = y_grid < scaled_near_wall_rows
    
    # =========================================================================
    # SDF-based regions (for TEMPERATURE analysis)
    # =========================================================================
    interface_mask_sdf = compute_interface_mask_zero_crossing(sdf)
    not_interface_sdf = ~interface_mask_sdf
    
    # Inside-bubble (SDF-based): vapor region (SDF > 0) AND not at SDF interface
    inside_bubble_sdf = (sdf > 0) & not_interface_sdf
    
    # Bulk-liquid (SDF-based): liquid region (SDF < 0) AND not at SDF interface AND not near wall
    bulk_liquid_sdf = (sdf < 0) & not_interface_sdf & (y_grid >= scaled_near_wall_rows)
    
    temperature_masks = {
        'near_wall': near_wall_mask,
        'inside_bubble': inside_bubble_sdf,
        'near_interface': interface_mask_sdf,
        'bulk_liquid': bulk_liquid_sdf
    }
    
    # =========================================================================
    # Massflux-based regions (for VELOCITY analysis) - only if velocity available
    # =========================================================================
    has_velocity = velx is not None and vely is not None
    
    if has_velocity:
        interface_mask_massflux = compute_interface_mask_massflux(velx, vely)
        not_interface_massflux = ~interface_mask_massflux
        
        # Inside-bubble (massflux-based): vapor region (SDF > 0) AND not at massflux interface
        inside_bubble_massflux = (sdf > 0) & not_interface_massflux
        
        # Bulk-liquid (massflux-based): liquid region (SDF < 0) AND not at massflux interface AND not near wall
        bulk_liquid_massflux = (sdf < 0) & not_interface_massflux & (y_grid >= scaled_near_wall_rows)
        
        velocity_masks = {
            'near_wall': near_wall_mask,
            'inside_bubble': inside_bubble_massflux,
            'near_interface': interface_mask_massflux,
            'bulk_liquid': bulk_liquid_massflux
        }
    else:
        # No velocity data (Task 1) - velocity_masks will be None
        velocity_masks = None
    
    return {
        'temperature': temperature_masks,
        'velocity': velocity_masks
    }


def compute_region_errors(gt_field: np.ndarray, pred_field: np.ndarray,
                          region_masks: Dict[str, np.ndarray],
                          field_name: str = 'field') -> Dict[str, Dict[str, float]]:
    """
    Compute error metrics for each region of the domain.
    
    Args:
        gt_field: (T, H, W) - ground truth field
        pred_field: (T, H, W) - predicted field
        region_masks: Dict of region masks from compute_region_masks
        field_name: Name of the field for logging
        
    Returns:
        Dict of error metrics per region
    """
    region_errors = {}
    
    for region_name, mask in region_masks.items():
        gt_values = gt_field[mask]
        pred_values = pred_field[mask]
        
        if len(gt_values) == 0:
            region_errors[region_name] = {
                'mae': np.nan,
                'rmse': np.nan,
                'mean_gt': np.nan,
                'mean_pred': np.nan,
                'num_cells': 0
            }
            continue
        
        error = pred_values - gt_values
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        
        region_errors[region_name] = {
            'mae': mae,
            'rmse': rmse,
            'mean_gt': np.mean(gt_values),
            'mean_pred': np.mean(pred_values),
            'num_cells': len(gt_values)
        }
    
    return region_errors


# ============================================================================
# Inference Functions
# ============================================================================

def run_inference_batch(model, dataset, device='cuda', num_inference_steps=50,
                        max_samples=None, model_type='flow_matching',
                        start_idx=0, solver='heun', guidance_scale=1.0,
                        initial_state_mode='from_data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on dataset and return predictions and ground truth.
    
    Args:
        model: Trained model (Flow Matching or AR models)
        dataset: BulkFlow or BulkFlowAutoregressive dataset
        device: Device to run on
        num_inference_steps: Number of ODE integration steps
        max_samples: Maximum number of samples to process (None = all)
        model_type: Model type ('flow_matching', 'flow_matching_ar', 'flow_matching_ar_bootstrap',
                               'unet_ar')
        start_idx: Starting index in the dataset (for frame range support)
        solver: ODE solver for flow_matching_ar ('euler', 'heun', 'midpoint', 'rk4')
        guidance_scale: Classifier-free guidance scale for flow_matching_ar
        initial_state_mode: Initial state mode for AR models ('from_data', 'zeros', 'from_conditioning')
        
    Returns:
        gt_velx, gt_vely, gt_temp: Ground truth arrays (T, H, W)
        pred_velx, pred_vely, pred_temp: Predicted arrays (T, H, W)
    """
    model = model.to(device)
    model.eval()
    
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    # Find indices in target_names
    velx_idx = target_names.index('velx') if 'velx' in target_names else None
    vely_idx = target_names.index('vely') if 'vely' in target_names else None
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else None
    
    # Calculate sample range
    available_from_start = len(dataset) - start_idx
    if max_samples is None:
        num_samples = available_from_start
    else:
        num_samples = min(max_samples, available_from_start)
    
    is_autoregressive = model_type in ['flow_matching_ar', 'unet_ar']
    
    gt_velx_list, gt_vely_list, gt_temp_list = [], [], []
    pred_velx_list, pred_vely_list, pred_temp_list = [], [], []
    
    print(f"\n🔮 Running inference on {num_samples} samples (indices {start_idx} to {start_idx + num_samples - 1})...")
    if is_autoregressive:
        print(f"   Model: Autoregressive ({model_type})")
        print(f"   Inference mode: True autoregressive rollout (using own predictions)")
        if model_type == 'flow_matching_ar':
            print(f"   ODE Solver: {solver}, Guidance scale: {guidance_scale}")
            print(f"   Integration steps per frame: {num_inference_steps}")
        else:
            print(f"   Direct regression (single forward pass per frame)")
    elif model_type == 'unet':
        print(f"   Model: UNet (direct regression)")
    elif model_type == 'ffno':
        print(f"   Model: FFNO (Factorized Fourier Neural Operator)")
    elif model_type == 'bubble_ddpm':
        print(f"   Model: DDPM (diffusion)")
        print(f"   Timesteps: {model.ddpm.num_timesteps}")
    elif model_type == 've_sde':
        print(f"   Model: VE-SDE (score-based diffusion)")
        print(f"   σ_min: {model.sigma_min}, σ_max: {model.sigma_max}")
        print(f"   Sampling: {model.sampling_method}, steps: {num_inference_steps}")
    elif model_type == 'edm':
        print(f"   Model: EDM (EDM-style diffusion)")
        print(f"   Sampling steps: {model.num_sampling_steps}")
        print(f"   Solver: {model.default_solver}")
    elif model_type == 'diffusionpde':
        print(f"   Model: DiffusionPDE (unconditional joint + guided sampling)")
        print(f"   Sampling steps: {model.num_sampling_steps}")
        print(f"   Solver: {model.default_solver}")
    elif model_type == 'flow_matching_jit':
        print(f"   Model: JiT Flow Matching (Vision Transformer, data prediction)")
        print(f"   Integration steps: {num_inference_steps}, Solver: {model.default_solver}")
    else:
        print(f"   Model: Flow Matching")
    
    # For AR models, we need to track the previous output for autoregressive rollout
    prev_output = None
    
    # DiffusionPDE needs gradients for autograd-based guidance
    grad_context = torch.enable_grad() if model_type == 'diffusionpde' else torch.no_grad()
    with grad_context:
        for i in tqdm(range(num_samples), desc="Inference"):
            sample_idx = start_idx + i
            
            # Get sample data
            sample_data = dataset[sample_idx]
            
            if is_autoregressive:
                # AR dataset returns: (conditioning_t, prev_output_gt, target_t, [wall_temp])
                if dataset.return_wall_temp:
                    input_data, prev_output_gt, target_data, wall_temp = sample_data
                else:
                    input_data, prev_output_gt, target_data = sample_data
                
                input_batch = input_data.unsqueeze(0).to(device)  # [1, C, H, W]
                target_batch = target_data.unsqueeze(0).to(device)  # [1, C, H, W]
                
                conditioning = input_batch[:, conditioning_channels, :, :]  # [1, C_cond, H, W]
                target = target_batch[:, target_channels, :, :]  # [1, C_target, H, W]
                
                # Initialize prev_output for the first frame
                if prev_output is None:
                    if initial_state_mode == 'from_data':
                        # Use ground truth previous output
                        prev_output_batch = prev_output_gt.unsqueeze(0).to(device)
                        prev_output = prev_output_batch[:, target_channels, :, :]
                    elif hasattr(model, 'create_initial_state'):
                        target_shape = (1, len(target_channels), target.shape[2], target.shape[3])
                        prev_output = model.create_initial_state(
                            shape=target_shape,
                            device=device,
                            mode=initial_state_mode,
                            conditioning=conditioning if initial_state_mode == 'from_conditioning' else None
                        )
                    else:
                        # Fallback: use zeros
                        target_shape = (1, len(target_channels), target.shape[2], target.shape[3])
                        prev_output = torch.zeros(target_shape, device=device)
                
                # Run AR model inference
                if model_type == 'flow_matching_ar':
                    predicted = model.sample(
                        condition=conditioning,
                        prev_output=prev_output,
                        shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                        device=device,
                        num_integration_steps=num_inference_steps,
                        solver=solver,
                        guidance_scale=guidance_scale,
                    )
                elif model_type in ['unet_ar']:
                    # Direct regression (single forward pass)
                    predicted = model.sample(
                        condition=conditioning,
                        prev_output=prev_output,
                    )
                
                # Update prev_output for next frame (true autoregressive rollout)
                prev_output = predicted.clone()
                
            else:
                # Get file index for denormalization
                file_idx = dataset.get_file_index(sample_idx)
                
                if dataset.return_wall_temp:
                    input_data, target_data, wall_temp = sample_data
                else:
                    input_data, target_data = sample_data
                
                # Non-temporal model: input is [C, H, W]
                input_batch = input_data.unsqueeze(0).to(device)  # (1, 3, H, W)
                target_batch = target_data.unsqueeze(0).to(device)  # (1, 3, H, W)
                
                # Extract conditioning and target.
                # NB: BulkFlowHistory returns a pre-flattened
                # [W * raw_cond_channels_per_frame, H, W] window.  For
                # flow_matching_history we delegate to the model's own
                # ``extract_history_conditioning`` so that the task-specific
                # ``conditioning_channels`` are subset for every frame
                # (e.g. [0] for temperature-from-sdf → W channels; [0,1,2] for
                # velocity-from-interface → 3*W channels), matching what the
                # UNet was built for at training time.
                if model_type == 'flow_matching_history':
                    conditioning = model.extract_history_conditioning(input_batch)
                else:
                    conditioning = extract_channels(input_batch, conditioning_channels)
                target = extract_channels(target_batch, target_channels)
                
                # Run inference based on model type
                if model_type == 'unet':
                    # UNet: direct forward pass
                    predicted = model(conditioning)
                elif model_type == 'ffno':
                    # FFNO: direct forward pass (spectral method)
                    predicted = model(conditioning)
                elif model_type == 'bubble_ddpm':
                    # DDPM: reverse diffusion sampling
                    predicted = model.ddpm.p_sample_loop(
                        condition=conditioning,
                        shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                        device=device
                    )
                elif model_type == 've_sde':
                    # VE-SDE: Score-based sampling
                    predicted = model.ve_sde.sample(
                        condition=conditioning,
                        shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                        device=device,
                        num_steps=num_inference_steps,
                        method=model.sampling_method,
                        snr=model.snr
                    )
                elif model_type == 'edm':
                    # EDM: EDM-style sampling
                    predicted = model.edm.sample(
                        condition=conditioning,
                        shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                        device=device,
                        num_steps=model.num_sampling_steps,
                        solver=model.default_solver
                    )
                elif model_type == 'diffusionpde':
                    num_joint = model.num_joint_channels
                    predicted = model.diffusion_pde.sample_with_guidance(
                        observed_gt=conditioning,
                        shape=(1, num_joint, target.shape[2], target.shape[3]),
                        device=device,
                        num_steps=model.num_sampling_steps,
                        solver=model.default_solver,
                    )
                elif model_type == 'flow_matching_jit':
                    # JiT Flow Matching: ODE integration with data prediction
                    predicted = model.flow_matching.sample(
                        condition=conditioning,
                        shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                        device=device,
                        num_integration_steps=num_inference_steps,
                        solver=model.default_solver
                    )
                else:
                    # Flow Matching: ODE integration
                    predicted = model.flow_matching.sample(
                        condition=conditioning,
                        shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                        device=device,
                        num_integration_steps=num_inference_steps
                    )
            
            # Move to CPU and remove batch dimension
            target = target.squeeze(0).cpu()
            predicted = predicted.squeeze(0).cpu()
            
            # Denormalize ALL output fields (temperature, velocity)
            # Normalization is now always applied
            for j, field_name in enumerate(target_names):
                target[j] = dataset._denormalize_field(target[j], field_name)
                predicted[j] = dataset._denormalize_field(predicted[j], field_name)
            
            # Extract individual fields
            if velx_idx is not None:
                gt_velx_list.append(target[velx_idx].numpy())
                pred_velx_list.append(predicted[velx_idx].numpy())
            if vely_idx is not None:
                gt_vely_list.append(target[vely_idx].numpy())
                pred_vely_list.append(predicted[vely_idx].numpy())
            if temp_idx is not None:
                gt_temp_list.append(target[temp_idx].numpy())
                pred_temp_list.append(predicted[temp_idx].numpy())
    
    # Stack arrays
    gt_velx = np.stack(gt_velx_list, axis=0) if gt_velx_list else None
    gt_vely = np.stack(gt_vely_list, axis=0) if gt_vely_list else None
    gt_temp = np.stack(gt_temp_list, axis=0) if gt_temp_list else None
    pred_velx = np.stack(pred_velx_list, axis=0) if pred_velx_list else None
    pred_vely = np.stack(pred_vely_list, axis=0) if pred_vely_list else None
    pred_temp = np.stack(pred_temp_list, axis=0) if pred_temp_list else None
    
    print(f"✓ Inference complete! Shape: {gt_temp.shape if gt_temp is not None else 'N/A'}")
    
    # Debug: print first frame's mean GT temperature for alignment verification
    if gt_temp is not None and len(gt_temp) > 0:
        print(f"   📍 DEBUG: First frame GT temp mean = {gt_temp[0].mean():.4f}°C")
    
    return gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp


def run_ar_bootstrap_inference_batch(
    model, dataset, device='cuda', num_inference_steps=50,
    max_samples=None, start_idx=0, solver='heun',
    bootstrap_ablation=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run AR Bootstrap inference on dataset and return predictions and ground truth.
    
    This function handles the BulkFlowARBootstrap dataset which returns trajectory
    segments with conditioning history for bootstrap initialization.
    
    IMPORTANT: BulkFlowARBootstrap uses overlapping segments (sliding window).
    To get non-overlapping consecutive frames, we skip by rollout_length between segments.
    
    For each sample (trajectory segment):
    1. Bootstrap the initial state from conditioning history
    2. Run autoregressive rollout using model's own predictions
    3. Collect all frames from the rollout
    
    Args:
        model: Trained AR Bootstrap model (ConditionalFlowMatchingARBootstrapLightning)
        dataset: BulkFlowARBootstrap dataset
        device: Device to run on
        num_inference_steps: Number of ODE integration steps per frame
        max_samples: Maximum number of trajectory segments to process (None = all)
        start_idx: Starting segment index in the dataset
        solver: ODE solver ('euler', 'heun', 'midpoint', 'rk4')
        bootstrap_ablation: If set ('zeros' or 'mean_conditioning_naive'), replaces
            the learned history encoder output with a simpler initialization.
        
    Returns:
        gt_velx, gt_vely, gt_temp: Ground truth arrays (T, H, W) - non-overlapping frames
        pred_velx, pred_vely, pred_temp: Predicted arrays (T, H, W) - non-overlapping frames
    """
    model = model.to(device)
    model.eval()
    
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    # Find indices in target_names
    velx_idx = target_names.index('velx') if 'velx' in target_names else None
    vely_idx = target_names.index('vely') if 'vely' in target_names else None
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else None
    
    # Calculate sample range
    # max_samples here is the number of segments to process (not total frames)
    available_from_start = len(dataset) - start_idx
    if max_samples is None:
        num_segments = available_from_start
    else:
        num_segments = min(max_samples, available_from_start)
    
    gt_velx_list, gt_vely_list, gt_temp_list = [], [], []
    pred_velx_list, pred_vely_list, pred_temp_list = [], [], []
    
    rollout_length = dataset.rollout_length
    history_length = dataset.history_length
    
    # IMPORTANT: To get non-overlapping frames, we skip by rollout_length between segments
    # Segment i covers frames [i*rollout + offset, (i+1)*rollout + offset)
    # So we process segments at indices: start_idx, start_idx + rollout_length, start_idx + 2*rollout_length, ...
    segment_stride = rollout_length
    
    print(f"\n🚀 Running AR Bootstrap inference on {num_segments} trajectory segments...")
    print(f"   Model type: {getattr(model, 'task_name', 'ar_bootstrap')}")
    if bootstrap_ablation is not None:
        print(f"   Bootstrap ABLATION: '{bootstrap_ablation}' (replaces learned encoder)")
    else:
        print(f"   Bootstrap: Uses history encoder to infer initial state")
    print(f"   History length: {history_length} frames")
    print(f"   Rollout length: {rollout_length} frames per segment")
    print(f"   Segment stride: {segment_stride} (non-overlapping)")
    print(f"   Total frames to generate: {num_segments * rollout_length}")
    print(f"   Integration steps per frame: {num_inference_steps}")
    print(f"   ODE Solver: {solver}")
    
    total_frames_generated = 0
    
    with torch.no_grad():
        for seg_i in tqdm(range(num_segments), desc="AR Bootstrap Inference"):
            # Skip by rollout_length to get non-overlapping segments
            segment_idx = start_idx + seg_i * segment_stride
            
            # Check if segment is within bounds
            if segment_idx >= len(dataset):
                print(f"   ⚠️ Segment index {segment_idx} exceeds dataset length {len(dataset)}, stopping")
                break
            
            # Get trajectory segment data
            sample_data = dataset[segment_idx]
            if dataset.return_wall_temp:
                cond_hist, cond_seq, target_seq, wall_temp = sample_data
            else:
                cond_hist, cond_seq, target_seq = sample_data
            
            # Move to device and add batch dimension
            cond_hist = cond_hist.unsqueeze(0).to(device)      # [1, T_hist, C, H, W]
            cond_seq = cond_seq.unsqueeze(0).to(device)        # [1, L, C, H, W]
            target_seq = target_seq.unsqueeze(0).to(device)    # [1, L, C, H, W]
            
            # Extract relevant channels
            cond_hist_extracted = cond_hist[:, :, conditioning_channels, :, :]
            cond_seq_extracted = cond_seq[:, :, conditioning_channels, :, :]
            target_seq_extracted = target_seq[:, :, target_channels, :, :]
            
            B, T_hist, C_cond, H, W = cond_hist_extracted.shape
            _, L, _, _, _ = cond_seq_extracted.shape
            C_out = target_seq_extracted.shape[2]
            
            # Bootstrap initial state (learned encoder or ablation replacement)
            if bootstrap_ablation is not None:
                target_names_list = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
                conditioning_names_list = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
                prev_output = compute_bootstrap_ablation_state(
                    bootstrap_ablation, cond_hist_extracted, C_out, device,
                    target_names=target_names_list, conditioning_names=conditioning_names_list,
                )
            else:
                current_cond_0 = cond_seq_extracted[:, 0]  # [B, C_cond, H, W]
                prev_output = model.bootstrap_initial_state(
                    cond_hist_extracted, current_cond_0
                )
            
            for l in range(L):
                current_cond = cond_seq_extracted[:, l]  # [B, C_cond, H, W]
                target_l = target_seq_extracted[:, l]    # [B, C_out, H, W]
                
                # Create availability mask (0 for first frame = bootstrapped, 1 for rest)
                if l == 0:
                    availability_mask = torch.zeros(B, 1, H, W, device=device)
                else:
                    availability_mask = torch.ones(B, 1, H, W, device=device)
                
                # Generate prediction
                predicted = model.sample(
                    condition=current_cond,
                    prev_output=prev_output,
                    shape=(B, C_out, H, W),
                    device=device,
                    availability_mask=availability_mask,
                    num_integration_steps=num_inference_steps,
                    solver=solver,
                )
                
                # Move to CPU and remove batch dimension
                target_cpu = target_l.squeeze(0).cpu()
                predicted_cpu = predicted.squeeze(0).cpu()
                
                # Denormalize output fields
                for j, field_name in enumerate(target_names):
                    target_cpu[j] = dataset._denormalize_field(target_cpu[j], field_name)
                    predicted_cpu[j] = dataset._denormalize_field(predicted_cpu[j], field_name)
                
                # Extract individual fields
                if velx_idx is not None:
                    gt_velx_list.append(target_cpu[velx_idx].numpy())
                    pred_velx_list.append(predicted_cpu[velx_idx].numpy())
                if vely_idx is not None:
                    gt_vely_list.append(target_cpu[vely_idx].numpy())
                    pred_vely_list.append(predicted_cpu[vely_idx].numpy())
                if temp_idx is not None:
                    gt_temp_list.append(target_cpu[temp_idx].numpy())
                    pred_temp_list.append(predicted_cpu[temp_idx].numpy())
                
                # Update prev_output for next frame
                prev_output = predicted
                total_frames_generated += 1
    
    # Stack arrays
    gt_velx = np.stack(gt_velx_list, axis=0) if gt_velx_list else None
    gt_vely = np.stack(gt_vely_list, axis=0) if gt_vely_list else None
    gt_temp = np.stack(gt_temp_list, axis=0) if gt_temp_list else None
    pred_velx = np.stack(pred_velx_list, axis=0) if pred_velx_list else None
    pred_vely = np.stack(pred_vely_list, axis=0) if pred_vely_list else None
    pred_temp = np.stack(pred_temp_list, axis=0) if pred_temp_list else None
    
    print(f"✓ AR Bootstrap inference complete!")
    print(f"   Total frames generated: {total_frames_generated}")
    print(f"   Output shape: {gt_temp.shape if gt_temp is not None else 'N/A'}")
    
    # Debug: print first frame's mean GT temperature for alignment verification
    if gt_temp is not None and len(gt_temp) > 0:
        print(f"   📍 DEBUG: First frame GT temp mean = {gt_temp[0].mean():.4f}°C")
    
    return gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp


# ============================================================================
# Main Metrics Computation
# ============================================================================

def compute_all_physics_metrics(
    gt_velx: np.ndarray, gt_vely: np.ndarray, gt_temp: np.ndarray,
    pred_velx: np.ndarray, pred_vely: np.ndarray, pred_temp: np.ndarray,
    sdf: np.ndarray, 
    heater_temp: float,
    row_indices: List[int] = [0, 8, 16, 24, 32],
    downsample_factor: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Compute all physics metrics for ground truth and predictions.
    
    Uses TWO interface detection methods:
    - SDF-based (zero-crossing): For temperature-related analysis (geometric interface)
    - Massflux-based: For velocity-related analysis (uses BULK velocities with threshold)
    
    For Task 1 (temperature_from_sdf), velocity arrays can be None and velocity
    metrics will be skipped.
    
    Args:
        gt_velx, gt_vely, gt_temp: Ground truth arrays (T, H, W) - velocity can be None for Task 1
        pred_velx, pred_vely, pred_temp: Predicted arrays (T, H, W) - velocity can be None for Task 1
        sdf: Signed distance function (T, H, W)
        heater_temp: Heater temperature in Celsius
        row_indices: Row indices for temperature analysis (at full 512 resolution)
        downsample_factor: Factor by which data was downsampled
        
    Returns:
        Dictionary containing all metrics for both GT and predictions
    """
    metrics = {}
    num_frames = gt_temp.shape[0]
    H, W = gt_temp.shape[1], gt_temp.shape[2]
    
    # Check if velocity data is available (Task 2/3) or not (Task 1)
    has_velocity = (gt_velx is not None and gt_vely is not None and 
                    pred_velx is not None and pred_vely is not None)
    
    print("\n📊 Computing physics metrics...")
    print(f"   Total frames to process: {num_frames}")
    print(f"   Spatial resolution: {H}x{W}")
    print(f"   Has velocity data: {has_velocity}")
    if downsample_factor > 1:
        print(f"   Downsample factor: {downsample_factor}x")
    
    # Scale row indices for downsampled data
    scaled_row_indices = [r // downsample_factor for r in row_indices]
    if downsample_factor > 1:
        print(f"   Row indices (original): {row_indices}")
        print(f"   Row indices (scaled):   {scaled_row_indices}")
    
    # ========================================================================
    # 1. Velocity Divergence (Mass Conservation) - Only for Task 2/3
    # ========================================================================
    if has_velocity:
        print("\n  1️⃣  Computing velocity divergence (mass conservation)...")
        print("      Note: Interface excluded since velocity has discontinuity due to phase change")
        print("      Using MASSFLUX-based detection for velocity analysis (non-zero interface velocity)")
        
        # Compute interface masks
        # SDF-based (zero-crossing) for temperature analysis
        immersed_bdry_mask = compute_interface_mask_zero_crossing(sdf)
        
        # Massflux-based for velocity analysis
        # NOTE: Uses BULK velocities (gt_velx, gt_vely) with threshold, NOT interface velocities
        # This captures cells where velocity magnitude >= threshold (default 1e-2)
        interface_mask = compute_interface_mask_massflux(gt_velx, gt_vely)
        non_interface_mask = ~interface_mask
        
        # Process with progress indication
        gt_div_list = []
        pred_div_list = []
        for i in tqdm(range(num_frames), desc="      Divergence", ncols=80):
            gt_div_list.append(compute_velocity_divergence(gt_velx[i], gt_vely[i], downsample_factor))
            pred_div_list.append(compute_velocity_divergence(pred_velx[i], pred_vely[i], downsample_factor))
        
        gt_div = np.stack(gt_div_list)
        pred_div = np.stack(pred_div_list)
        
        # Metrics for ALL regions (reference)
        metrics['velocity_divergence'] = {
            'gt_mean': np.mean(np.abs(gt_div)),
            'gt_max': np.max(np.abs(gt_div)),
            'gt_std': np.std(gt_div),
            'pred_mean': np.mean(np.abs(pred_div)),
            'pred_max': np.max(np.abs(pred_div)),
            'pred_std': np.std(pred_div),
            'error_mean': np.mean(np.abs(gt_div - pred_div)),
            'error_max': np.max(np.abs(gt_div - pred_div)),
        }
        
        # Metrics EXCLUDING interface (more fair - interface has velocity discontinuity)
        gt_div_no_interface = gt_div[non_interface_mask]
        pred_div_no_interface = pred_div[non_interface_mask]
        
        metrics['velocity_divergence_excl_interface'] = {
            'gt_mean': np.mean(np.abs(gt_div_no_interface)),
            'gt_max': np.max(np.abs(gt_div_no_interface)),
            'gt_std': np.std(gt_div_no_interface),
            'pred_mean': np.mean(np.abs(pred_div_no_interface)),
            'pred_max': np.max(np.abs(pred_div_no_interface)),
            'pred_std': np.std(pred_div_no_interface),
            'error_mean': np.mean(np.abs(gt_div_no_interface - pred_div_no_interface)),
            'error_max': np.max(np.abs(gt_div_no_interface - pred_div_no_interface)),
            'num_cells': len(gt_div_no_interface),
        }
        
        print(f"\n      📊 Divergence (ALL regions for reference):")
        print(f"         GT:   mean|∇·V| = {metrics['velocity_divergence']['gt_mean']:.6f}, "
              f"max|∇·V| = {metrics['velocity_divergence']['gt_max']:.6f}")
        print(f"         Pred: mean|∇·V| = {metrics['velocity_divergence']['pred_mean']:.6f}, "
              f"max|∇·V| = {metrics['velocity_divergence']['pred_max']:.6f}")
        
        print(f"\n      📊 Divergence (EXCLUDING interface, non-zero massflux):")
        print(f"         GT:   mean|∇·V| = {metrics['velocity_divergence_excl_interface']['gt_mean']:.6f}, "
              f"max|∇·V| = {metrics['velocity_divergence_excl_interface']['gt_max']:.6f}")
        print(f"         Pred: mean|∇·V| = {metrics['velocity_divergence_excl_interface']['pred_mean']:.6f}, "
              f"max|∇·V| = {metrics['velocity_divergence_excl_interface']['pred_max']:.6f}")
    else:
        print("\n  1️⃣  Skipping velocity divergence (Task 1 - temperature only)")
    
    # ========================================================================
    # 2. Interface Temperature (uses SDF-based detection)
    # ========================================================================
    print("\n  2️⃣  Computing interface temperature...")
    print("      Using SDF-based (zero-crossing) detection for temperature analysis")
    print("      Finding cells where dfun crosses zero (immersed boundary)...")
    
    gt_interface_mean, gt_interface_std = compute_interface_temperature(sdf, gt_temp)
    pred_interface_mean, pred_interface_std = compute_interface_temperature(sdf, pred_temp)
    
    metrics['interface_temperature'] = {
        'gt_mean': gt_interface_mean,
        'gt_std': gt_interface_std,
        'pred_mean': pred_interface_mean,
        'pred_std': pred_interface_std,
        'error_mean': np.abs(gt_interface_mean - pred_interface_mean) if not np.isnan(gt_interface_mean) else np.nan,
    }
    
    print(f"      ✓ GT interface temp:   {gt_interface_mean:.2f} ± {gt_interface_std:.2f} °C")
    print(f"      ✓ Pred interface temp: {pred_interface_mean:.2f} ± {pred_interface_std:.2f} °C")
    
    # ========================================================================
    # 3. Wall Heat Flux
    # ========================================================================
    print("\n  3️⃣  Computing wall heat flux...")
    print("      Calculating heat flux for each frame...")
    
    # The heatflux function already processes all frames, but we show progress
    print("      Processing GT heat flux...")
    gt_hflux = compute_heatflux(sdf, gt_temp, heater_temp, downsample_factor=downsample_factor)
    print("      Processing Pred heat flux...")
    pred_hflux = compute_heatflux(sdf, pred_temp, heater_temp, downsample_factor=downsample_factor)
    
    metrics['wall_heat_flux'] = {
        'gt_mean': np.mean(gt_hflux),
        'gt_std': np.std(gt_hflux),
        'gt_min': np.min(gt_hflux),
        'gt_max': np.max(gt_hflux),
        'pred_mean': np.mean(pred_hflux),
        'pred_std': np.std(pred_hflux),
        'pred_min': np.min(pred_hflux),
        'pred_max': np.max(pred_hflux),
        'error_mean': np.mean(np.abs(gt_hflux - pred_hflux)),
        'error_rmse': np.sqrt(np.mean((gt_hflux - pred_hflux)**2)),
        'relative_error': np.mean(np.abs(gt_hflux - pred_hflux)) / np.mean(gt_hflux) * 100,
    }
    
    print(f"      ✓ GT heat flux:   {metrics['wall_heat_flux']['gt_mean']:.2f} ± {metrics['wall_heat_flux']['gt_std']:.2f}")
    print(f"      ✓ Pred heat flux: {metrics['wall_heat_flux']['pred_mean']:.2f} ± {metrics['wall_heat_flux']['pred_std']:.2f}")
    print(f"      ✓ Relative error: {metrics['wall_heat_flux']['relative_error']:.2f}%")
    
    # ========================================================================
    # 4. Row Temperature Analysis
    # ========================================================================
    print("\n  4️⃣  Computing row temperature analysis...")
    print(f"      Analyzing rows (scaled): {scaled_row_indices}")
    
    gt_row_temps = compute_row_temperature(gt_temp, scaled_row_indices)
    pred_row_temps = compute_row_temperature(pred_temp, scaled_row_indices)
    
    metrics['row_temperature'] = {}
    
    # Print as a formatted table
    print(f"\n      {'Row':<8} {'GT Mean (°C)':<15} {'GT Std (°C)':<15} {'Pred Mean (°C)':<15} {'Pred Std (°C)':<15} {'Error (°C)':<12}")
    print(f"      {'-'*8} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
    
    for row_y in row_indices:
        if row_y in gt_row_temps and row_y in pred_row_temps:
            gt_mean, gt_std = gt_row_temps[row_y]
            pred_mean, pred_std = pred_row_temps[row_y]
            error = np.abs(gt_mean - pred_mean)
            
            metrics['row_temperature'][f'row_{row_y}'] = {
                'gt_mean': gt_mean,
                'gt_std': gt_std,
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'error_mean': error,
            }
            
            print(f"      y={row_y:<5} {gt_mean:<15.3f} {gt_std:<15.3f} {pred_mean:<15.3f} {pred_std:<15.3f} {error:<12.3f}")
    
    print(f"      {'-'*8} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
    
    # ========================================================================
    # 5. Vorticity Analysis (uses Massflux-based detection) - Only for Task 2/3
    # ========================================================================
    if has_velocity:
        print("\n  5️⃣  Computing vorticity (rotational flow patterns)...")
        print("      Vorticity ω = ∂v/∂x - ∂u/∂y")
        print("      Using MASSFLUX-based detection for velocity analysis (non-zero interface velocity)")
        
        gt_vorticity_list = []
        pred_vorticity_list = []
        for i in tqdm(range(num_frames), desc="      Vorticity", ncols=80):
            gt_vorticity_list.append(compute_vorticity(gt_velx[i], gt_vely[i], downsample_factor))
            pred_vorticity_list.append(compute_vorticity(pred_velx[i], pred_vely[i], downsample_factor))
        
        gt_vorticity = np.stack(gt_vorticity_list)
        pred_vorticity = np.stack(pred_vorticity_list)
        
        # Use massflux-based interface mask (already computed for divergence)
        # interface_mask and non_interface_mask are already available from divergence calculation
        
        # Additional masks for bulk liquid
        bulk_liquid_mask = (sdf < 0) & non_interface_mask  # Liquid region, not at interface
        
        # Compute vorticity metrics for ALL regions (for reference)
        metrics['vorticity'] = {
            'gt_mean': np.mean(gt_vorticity),
            'gt_mean_abs': np.mean(np.abs(gt_vorticity)),
            'gt_max': np.max(np.abs(gt_vorticity)),
            'gt_std': np.std(gt_vorticity),
            'pred_mean': np.mean(pred_vorticity),
            'pred_mean_abs': np.mean(np.abs(pred_vorticity)),
            'pred_max': np.max(np.abs(pred_vorticity)),
            'pred_std': np.std(pred_vorticity),
            'error_mean': np.mean(np.abs(gt_vorticity - pred_vorticity)),
            'error_rmse': np.sqrt(np.mean((gt_vorticity - pred_vorticity)**2)),
        }
        
        # Compute vorticity metrics EXCLUDING interface region (using massflux-based detection)
        gt_vort_no_interface = gt_vorticity[non_interface_mask]
        pred_vort_no_interface = pred_vorticity[non_interface_mask]
        
        metrics['vorticity_excluding_interface'] = {
            'gt_mean': np.mean(gt_vort_no_interface),
            'gt_mean_abs': np.mean(np.abs(gt_vort_no_interface)),
            'gt_max': np.max(np.abs(gt_vort_no_interface)),
            'gt_std': np.std(gt_vort_no_interface),
            'pred_mean': np.mean(pred_vort_no_interface),
            'pred_mean_abs': np.mean(np.abs(pred_vort_no_interface)),
            'pred_max': np.max(np.abs(pred_vort_no_interface)),
            'pred_std': np.std(pred_vort_no_interface),
            'error_mean': np.mean(np.abs(gt_vort_no_interface - pred_vort_no_interface)),
            'error_rmse': np.sqrt(np.mean((gt_vort_no_interface - pred_vort_no_interface)**2)),
            'num_cells': len(gt_vort_no_interface),
        }
        
        # Compute vorticity metrics for BULK LIQUID only (liquid + not at interface)
        gt_vort_liquid = gt_vorticity[bulk_liquid_mask]
        pred_vort_liquid = pred_vorticity[bulk_liquid_mask]
        
        metrics['vorticity_bulk_liquid'] = {
            'gt_mean': np.mean(gt_vort_liquid),
            'gt_mean_abs': np.mean(np.abs(gt_vort_liquid)),
            'gt_max': np.max(np.abs(gt_vort_liquid)),
            'gt_std': np.std(gt_vort_liquid),
            'pred_mean': np.mean(pred_vort_liquid),
            'pred_mean_abs': np.mean(np.abs(pred_vort_liquid)),
            'pred_max': np.max(np.abs(pred_vort_liquid)),
            'pred_std': np.std(pred_vort_liquid),
            'error_mean': np.mean(np.abs(gt_vort_liquid - pred_vort_liquid)),
            'error_rmse': np.sqrt(np.mean((gt_vort_liquid - pred_vort_liquid)**2)),
            'num_cells': len(gt_vort_liquid),
        }
        
        print(f"\n      📊 Vorticity Results (ALL regions for reference):")
        print(f"         GT:   mean|ω| = {metrics['vorticity']['gt_mean_abs']:.6f}, max|ω| = {metrics['vorticity']['gt_max']:.6f}")
        print(f"         Pred: mean|ω| = {metrics['vorticity']['pred_mean_abs']:.6f}, max|ω| = {metrics['vorticity']['pred_max']:.6f}")
        print(f"         RMSE: {metrics['vorticity']['error_rmse']:.6f}")
        
        print(f"\n      📊 Vorticity Results (EXCLUDING interface, non-zero massflux):")
        print(f"         GT:   mean|ω| = {metrics['vorticity_excluding_interface']['gt_mean_abs']:.6f}, max|ω| = {metrics['vorticity_excluding_interface']['gt_max']:.6f}")
        print(f"         Pred: mean|ω| = {metrics['vorticity_excluding_interface']['pred_mean_abs']:.6f}, max|ω| = {metrics['vorticity_excluding_interface']['pred_max']:.6f}")
        print(f"         RMSE: {metrics['vorticity_excluding_interface']['error_rmse']:.6f}")
        
        print(f"\n      📊 Vorticity Results (BULK LIQUID only, SDF < 0 & not at interface):")
        print(f"         GT:   mean|ω| = {metrics['vorticity_bulk_liquid']['gt_mean_abs']:.6f}, max|ω| = {metrics['vorticity_bulk_liquid']['gt_max']:.6f}")
        print(f"         Pred: mean|ω| = {metrics['vorticity_bulk_liquid']['pred_mean_abs']:.6f}, max|ω| = {metrics['vorticity_bulk_liquid']['pred_max']:.6f}")
        print(f"         RMSE: {metrics['vorticity_bulk_liquid']['error_rmse']:.6f}")
    else:
        print("\n  5️⃣  Skipping vorticity (Task 1 - temperature only)")
    
    # ========================================================================
    # 6. Region-wise Error Analysis
    # ========================================================================
    print("\n  6️⃣  Computing region-wise error analysis...")
    print("      Temperature: SDF-based interface detection (geometric)")
    if has_velocity:
        print("      Velocity: Massflux-based interface detection (non-zero velocity)")
    print("      Regions: near-wall, inside-bubble, near-interface, bulk-liquid")
    
    # Create region masks (near_wall_rows is at full resolution, will be scaled inside)
    # Returns separate masks for temperature (SDF-based) and velocity (massflux-based)
    all_region_masks = compute_region_masks(sdf, gt_velx, gt_vely, 
                                            near_wall_rows=16, downsample_factor=downsample_factor)
    
    # Compute errors for temperature field using SDF-based masks
    print("      Computing temperature errors by region (SDF-based)...")
    temp_region_masks = all_region_masks['temperature']
    temp_region_errors = compute_region_errors(gt_temp, pred_temp, temp_region_masks, 'temperature')
    
    # Initialize region_errors dict
    metrics['region_errors'] = {
        'temperature': temp_region_errors,
    }
    
    # Compute errors for velocity fields using massflux-based masks - Only for Task 2/3
    if has_velocity:
        vel_region_masks = all_region_masks['velocity']
        
        print("      Computing velocity-x errors by region (massflux-based)...")
        velx_region_errors = compute_region_errors(gt_velx, pred_velx, vel_region_masks, 'velx')
        
        print("      Computing velocity-y errors by region (massflux-based)...")
        vely_region_errors = compute_region_errors(gt_vely, pred_vely, vel_region_masks, 'vely')
        
        metrics['region_errors']['velx'] = velx_region_errors
        metrics['region_errors']['vely'] = vely_region_errors
    
    # Print region-wise temperature errors as a table
    print(f"\n      {'Region':<18} {'# Cells':>12} {'GT Mean':>12} {'Pred Mean':>12} {'MAE':>10} {'RMSE':>10}")
    print(f"      {'-'*18} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    for region_name, errors in temp_region_errors.items():
        if errors['num_cells'] > 0:
            print(f"      {region_name:<18} {errors['num_cells']:>12} {errors['mean_gt']:>12.3f} "
                  f"{errors['mean_pred']:>12.3f} {errors['mae']:>10.4f} {errors['rmse']:>10.4f}")
        else:
            print(f"      {region_name:<18} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
    print(f"      {'-'*18} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    
    return metrics


def metrics_to_dataframe(metrics: Dict, data_file: str, checkpoint: str) -> pd.DataFrame:
    """
    Convert metrics dictionary to a pandas DataFrame for CSV export.
    
    Args:
        metrics: Dictionary containing all computed metrics
        data_file: Path to data file (for metadata)
        checkpoint: Path to checkpoint file (for metadata)
        
    Returns:
        DataFrame with metrics organized in rows
    """
    rows = []
    
    # Metadata
    rows.append({
        'Category': 'Metadata',
        'Metric': 'data_file',
        'GT_Value': os.path.basename(data_file),
        'Pred_Value': '',
        'Error': '',
        'Unit': ''
    })
    rows.append({
        'Category': 'Metadata',
        'Metric': 'checkpoint',
        'GT_Value': os.path.basename(checkpoint),
        'Pred_Value': '',
        'Error': '',
        'Unit': ''
    })
    
    # Check if velocity data is available (Task 2/3) or not (Task 1)
    has_velocity = 'velocity_divergence' in metrics
    
    # Velocity Divergence - for each region (Task 2/3 only)
    if has_velocity:
        divergence_regions = [
            ('velocity_divergence', 'Velocity Divergence (all regions)'),
            ('velocity_divergence_excl_interface', 'Velocity Divergence (excl. interface, non-zero massflux)')
        ]
        
        for region_key, category_name in divergence_regions:
            div_metrics = metrics[region_key]
            rows.append({
                'Category': category_name,
                'Metric': 'mean_abs_divergence',
                'GT_Value': f"{div_metrics['gt_mean']:.6f}",
                'Pred_Value': f"{div_metrics['pred_mean']:.6f}",
                'Error': f"{div_metrics['error_mean']:.6f}",
                'Unit': '[-]'  # Non-dimensional (1/L*)
            })
            rows.append({
                'Category': category_name,
                'Metric': 'max_abs_divergence',
                'GT_Value': f"{div_metrics['gt_max']:.6f}",
                'Pred_Value': f"{div_metrics['pred_max']:.6f}",
                'Error': f"{div_metrics['error_max']:.6f}",
                'Unit': '[-]'  # Non-dimensional (1/L*)
            })
            rows.append({
                'Category': category_name,
                'Metric': 'std_divergence',
                'GT_Value': f"{div_metrics['gt_std']:.6f}",
                'Pred_Value': f"{div_metrics['pred_std']:.6f}",
                'Error': '',
                'Unit': '[-]'  # Non-dimensional (1/L*)
            })
    
    # Immersed Boundary Temperature (where dfun crosses zero - SDF-based)
    interface_metrics = metrics['interface_temperature']
    rows.append({
        'Category': 'Immersed Boundary Temperature (SDF zero-crossing)',
        'Metric': 'mean_temperature',
        'GT_Value': f"{interface_metrics['gt_mean']:.3f}",
        'Pred_Value': f"{interface_metrics['pred_mean']:.3f}",
        'Error': f"{interface_metrics['error_mean']:.3f}" if not np.isnan(interface_metrics['error_mean']) else 'N/A',
        'Unit': '°C'
    })
    rows.append({
        'Category': 'Immersed Boundary Temperature (SDF zero-crossing)',
        'Metric': 'std_temperature',
        'GT_Value': f"{interface_metrics['gt_std']:.3f}",
        'Pred_Value': f"{interface_metrics['pred_std']:.3f}",
        'Error': '',
        'Unit': '°C'
    })
    
    # Wall Heat Flux
    hflux_metrics = metrics['wall_heat_flux']
    rows.append({
        'Category': 'Wall Heat Flux',
        'Metric': 'mean_heat_flux',
        'GT_Value': f"{hflux_metrics['gt_mean']:.2f}",
        'Pred_Value': f"{hflux_metrics['pred_mean']:.2f}",
        'Error': f"{hflux_metrics['error_mean']:.2f}",
        'Unit': 'W/m²'
    })
    rows.append({
        'Category': 'Wall Heat Flux',
        'Metric': 'std_heat_flux',
        'GT_Value': f"{hflux_metrics['gt_std']:.2f}",
        'Pred_Value': f"{hflux_metrics['pred_std']:.2f}",
        'Error': '',
        'Unit': 'W/m²'
    })
    rows.append({
        'Category': 'Wall Heat Flux',
        'Metric': 'rmse_heat_flux',
        'GT_Value': '',
        'Pred_Value': '',
        'Error': f"{hflux_metrics['error_rmse']:.2f}",
        'Unit': 'W/m²'
    })
    rows.append({
        'Category': 'Wall Heat Flux',
        'Metric': 'relative_error',
        'GT_Value': '',
        'Pred_Value': '',
        'Error': f"{hflux_metrics['relative_error']:.2f}",
        'Unit': '%'
    })
    
    # Row Temperature Analysis
    row_temp_metrics = metrics['row_temperature']
    for row_key, row_values in row_temp_metrics.items():
        row_y = row_key.replace('row_', '')
        rows.append({
            'Category': f'Row Temperature (y={row_y})',
            'Metric': 'mean_temperature',
            'GT_Value': f"{row_values['gt_mean']:.3f}",
            'Pred_Value': f"{row_values['pred_mean']:.3f}",
            'Error': f"{row_values['error_mean']:.3f}",
            'Unit': '°C'
        })
        rows.append({
            'Category': f'Row Temperature (y={row_y})',
            'Metric': 'std_temperature',
            'GT_Value': f"{row_values['gt_std']:.3f}",
            'Pred_Value': f"{row_values['pred_std']:.3f}",
            'Error': '',
            'Unit': '°C'
        })
    
    # Vorticity - for each region (Task 2/3 only)
    if has_velocity:
        vorticity_regions = [
            ('vorticity', 'Vorticity (all regions)'),
            ('vorticity_excluding_interface', 'Vorticity (excl. interface, non-zero massflux)'),
            ('vorticity_bulk_liquid', 'Vorticity (bulk liquid, SDF<0 & not interface)')
        ]
        
        for region_key, category_name in vorticity_regions:
            vort_metrics = metrics[region_key]
            rows.append({
                'Category': category_name,
                'Metric': 'mean_vorticity',
                'GT_Value': f"{vort_metrics['gt_mean']:.6f}",
                'Pred_Value': f"{vort_metrics['pred_mean']:.6f}",
                'Error': '',
                'Unit': '[-]'  # Non-dimensional (1/t*)
            })
            rows.append({
                'Category': category_name,
                'Metric': 'mean_abs_vorticity',
                'GT_Value': f"{vort_metrics['gt_mean_abs']:.6f}",
                'Pred_Value': f"{vort_metrics['pred_mean_abs']:.6f}",
                'Error': f"{vort_metrics['error_mean']:.6f}",
                'Unit': '[-]'  # Non-dimensional (1/t*)
            })
            rows.append({
                'Category': category_name,
                'Metric': 'max_abs_vorticity',
                'GT_Value': f"{vort_metrics['gt_max']:.6f}",
                'Pred_Value': f"{vort_metrics['pred_max']:.6f}",
                'Error': '',
                'Unit': '[-]'  # Non-dimensional (1/t*)
            })
            rows.append({
                'Category': category_name,
                'Metric': 'std_vorticity',
                'GT_Value': f"{vort_metrics['gt_std']:.6f}",
                'Pred_Value': f"{vort_metrics['pred_std']:.6f}",
                'Error': '',
                'Unit': '[-]'  # Non-dimensional (1/t*)
            })
            rows.append({
                'Category': category_name,
                'Metric': 'rmse_vorticity',
                'GT_Value': '',
                'Pred_Value': '',
                'Error': f"{vort_metrics['error_rmse']:.6f}",
                'Unit': '[-]'  # Non-dimensional (1/t*)
            })
    
    # Region-wise Error Analysis
    region_errors = metrics['region_errors']
    
    # Temperature by region
    for region_name, errors in region_errors['temperature'].items():
        if errors['num_cells'] > 0:
            rows.append({
                'Category': f'Region Error - Temperature ({region_name})',
                'Metric': 'mae',
                'GT_Value': f"{errors['mean_gt']:.3f}",
                'Pred_Value': f"{errors['mean_pred']:.3f}",
                'Error': f"{errors['mae']:.4f}",
                'Unit': '°C'
            })
            rows.append({
                'Category': f'Region Error - Temperature ({region_name})',
                'Metric': 'rmse',
                'GT_Value': '',
                'Pred_Value': '',
                'Error': f"{errors['rmse']:.4f}",
                'Unit': '°C'
            })
    
    # Velocity region errors (Task 2/3 only)
    if has_velocity and 'velx' in region_errors:
        # Velocity-x by region
        for region_name, errors in region_errors['velx'].items():
            if errors['num_cells'] > 0:
                rows.append({
                    'Category': f'Region Error - Velx ({region_name})',
                    'Metric': 'mae',
                    'GT_Value': f"{errors['mean_gt']:.6f}",
                    'Pred_Value': f"{errors['mean_pred']:.6f}",
                    'Error': f"{errors['mae']:.6f}",
                    'Unit': '[-]'  # Non-dimensional velocity
                })
                rows.append({
                    'Category': f'Region Error - Velx ({region_name})',
                    'Metric': 'rmse',
                    'GT_Value': '',
                    'Pred_Value': '',
                    'Error': f"{errors['rmse']:.6f}",
                    'Unit': '[-]'  # Non-dimensional velocity
                })
        
        # Velocity-y by region
        for region_name, errors in region_errors['vely'].items():
            if errors['num_cells'] > 0:
                rows.append({
                    'Category': f'Region Error - Vely ({region_name})',
                    'Metric': 'mae',
                    'GT_Value': f"{errors['mean_gt']:.6f}",
                    'Pred_Value': f"{errors['mean_pred']:.6f}",
                    'Error': f"{errors['mae']:.6f}",
                    'Unit': '[-]'  # Non-dimensional velocity
                })
                rows.append({
                    'Category': f'Region Error - Vely ({region_name})',
                    'Metric': 'rmse',
                    'GT_Value': '',
                    'Pred_Value': '',
                    'Error': f"{errors['rmse']:.6f}",
                    'Unit': '[-]'  # Non-dimensional velocity
                })
    
    df = pd.DataFrame(rows)
    return df


def metrics_to_simplified_dataframe(metrics: Dict, data_file: str, checkpoint: str) -> pd.DataFrame:
    """
    Convert metrics dictionary to a SIMPLIFIED pandas DataFrame for CSV export.
    Contains only the most important metrics from each category.
    
    Args:
        metrics: Dictionary containing all computed metrics
        data_file: Path to data file (for metadata)
        checkpoint: Path to checkpoint file (for metadata)
        
    Returns:
        DataFrame with key metrics organized in rows
    """
    rows = []
    
    # Check if velocity data is available (Task 2/3) or not (Task 1)
    has_velocity = 'velocity_divergence' in metrics
    
    # 1. Velocity Divergence (excl. interface only - the meaningful one) - Task 2/3 only
    if has_velocity:
        div_m = metrics['velocity_divergence_excl_interface']
        rows.append({
            'Metric': 'Velocity Divergence (excl. interface)',
            'GT': f"{div_m['gt_mean']:.6f}",
            'Pred': f"{div_m['pred_mean']:.6f}",
            'Error': f"{div_m['error_mean']:.6f}",
            'Unit': '[-]'
        })
    
    # 2. Interface Temperature
    int_m = metrics['interface_temperature']
    error_str = f"{int_m['error_mean']:.3f}" if not np.isnan(int_m['error_mean']) else 'N/A'
    rows.append({
        'Metric': 'Interface Temperature',
        'GT': f"{int_m['gt_mean']:.2f}",
        'Pred': f"{int_m['pred_mean']:.2f}",
        'Error': error_str,
        'Unit': '°C'
    })
    
    # 3. Wall Heat Flux
    hf_m = metrics['wall_heat_flux']
    rows.append({
        'Metric': 'Wall Heat Flux (mean)',
        'GT': f"{hf_m['gt_mean']:.2f}",
        'Pred': f"{hf_m['pred_mean']:.2f}",
        'Error': f"{hf_m['relative_error']:.2f}%",
        'Unit': 'W/m²'
    })
    
    # 4. Row Temperature - average error across all rows
    row_errors = [v['error_mean'] for v in metrics['row_temperature'].values()]
    avg_row_error = np.mean(row_errors) if row_errors else 0
    rows.append({
        'Metric': 'Row Temperature (avg error)',
        'GT': '-',
        'Pred': '-',
        'Error': f"{avg_row_error:.3f}",
        'Unit': '°C'
    })
    
    # 5. Vorticity (excl. interface) - Task 2/3 only
    if has_velocity:
        vort_m = metrics['vorticity_excluding_interface']
        rows.append({
            'Metric': 'Vorticity (excl. interface)',
            'GT': f"{vort_m['gt_mean_abs']:.6f}",
            'Pred': f"{vort_m['pred_mean_abs']:.6f}",
            'Error': f"{vort_m['error_rmse']:.6f}",
            'Unit': '[-]'
        })
    
    # 6. Region-wise Temperature Error (bulk liquid - most important)
    temp_errors = metrics['region_errors']['temperature']
    if 'bulk_liquid' in temp_errors and temp_errors['bulk_liquid']['num_cells'] > 0:
        rows.append({
            'Metric': 'Temperature MAE (bulk liquid)',
            'GT': f"{temp_errors['bulk_liquid']['mean_gt']:.2f}",
            'Pred': f"{temp_errors['bulk_liquid']['mean_pred']:.2f}",
            'Error': f"{temp_errors['bulk_liquid']['mae']:.4f}",
            'Unit': '°C'
        })
    
    # 7. Region-wise Velocity Error (bulk liquid) - Task 2/3 only
    if has_velocity and 'velx' in metrics['region_errors']:
        velx_errors = metrics['region_errors']['velx']
        vely_errors = metrics['region_errors']['vely']
        if 'bulk_liquid' in velx_errors and velx_errors['bulk_liquid']['num_cells'] > 0:
            # Average of velx and vely MAE
            avg_vel_mae = (velx_errors['bulk_liquid']['mae'] + vely_errors['bulk_liquid']['mae']) / 2
            rows.append({
                'Metric': 'Velocity MAE (bulk liquid, avg)',
                'GT': '-',
                'Pred': '-',
                'Error': f"{avg_vel_mae:.6f}",
                'Unit': '[-]'
            })
        
        # 8. Near-interface velocity error
        if 'near_interface' in velx_errors and velx_errors['near_interface']['num_cells'] > 0:
            avg_vel_mae_interface = (velx_errors['near_interface']['mae'] + vely_errors['near_interface']['mae']) / 2
            rows.append({
                'Metric': 'Velocity MAE (near interface, avg)',
                'GT': '-',
                'Pred': '-',
                'Error': f"{avg_vel_mae_interface:.6f}",
                'Unit': '[-]'
            })
    
    df = pd.DataFrame(rows)
    return df


def print_summary_table(metrics: Dict):
    """Print a formatted summary table of all metrics."""
    
    # Check if velocity data is available (Task 2/3) or not (Task 1)
    has_velocity = 'velocity_divergence' in metrics
    
    if has_velocity:
        task_label = "Velocity from Interface"
    else:
        task_label = "Temperature from SDF"
    
    print("\n" + "=" * 90)
    print(f"                    PHYSICS METRICS SUMMARY - {task_label}")
    print("=" * 90)
    
    # Velocity Divergence - Only for Task 2/3
    if has_velocity:
        print("\n📐 VELOCITY DIVERGENCE (Mass Conservation)")
        print("   For incompressible flow, ∇·V = ∂u/∂x + ∂v/∂y should be zero")
        print("   Interface excluded using MASSFLUX-based detection (non-zero interface velocity)")
        print("   Units: Non-dimensional (velocity and length are non-dimensionalized in BubbleML)")
        print("-" * 85)
        
        for region_key, region_label in [
            ('velocity_divergence', 'ALL REGIONS (reference)'),
            ('velocity_divergence_excl_interface', 'EXCLUDING INTERFACE (non-zero massflux)')
        ]:
            div_m = metrics[region_key]
            print(f"\n  📍 {region_label}")
            print(f"  {'Metric':<30} {'Ground Truth':>18} {'Prediction':>18} {'Error':>15}")
            print(f"  {'-'*30} {'-'*18} {'-'*18} {'-'*15}")
            print(f"  {'Mean |∇·V| [1/s]':<30} {div_m['gt_mean']:>18.6f} {div_m['pred_mean']:>18.6f} {div_m['error_mean']:>15.6f}")
            print(f"  {'Max |∇·V| [1/s]':<30} {div_m['gt_max']:>18.6f} {div_m['pred_max']:>18.6f} {div_m['error_max']:>15.6f}")
            print(f"  {'Std ∇·V [1/s]':<30} {div_m['gt_std']:>18.6f} {div_m['pred_std']:>18.6f} {'':>15}")
    
    # Immersed Boundary Temperature (SDF-based)
    print("\n🌡️  IMMERSED BOUNDARY TEMPERATURE (SDF zero-crossing)")
    print("   Uses SDF-based detection for temperature analysis")
    print("-" * 75)
    int_m = metrics['interface_temperature']
    print(f"  {'Metric':<30} {'Ground Truth':>18} {'Prediction':>18} {'Error':>15}")
    print(f"  {'-'*30} {'-'*18} {'-'*18} {'-'*15}")
    error_str = f"{int_m['error_mean']:.3f}" if not np.isnan(int_m['error_mean']) else "N/A"
    print(f"  {'Mean Temperature [°C]':<30} {int_m['gt_mean']:>18.3f} {int_m['pred_mean']:>18.3f} {error_str:>15}")
    print(f"  {'Std Temperature [°C]':<30} {int_m['gt_std']:>18.3f} {int_m['pred_std']:>18.3f} {'':>15}")
    
    # Wall Heat Flux
    print("\n🔥 WALL HEAT FLUX")
    print("-" * 75)
    hf_m = metrics['wall_heat_flux']
    print(f"  {'Metric':<30} {'Ground Truth':>18} {'Prediction':>18} {'Error':>15}")
    print(f"  {'-'*30} {'-'*18} {'-'*18} {'-'*15}")
    print(f"  {'Mean Heat Flux [W/m²]':<30} {hf_m['gt_mean']:>18.2f} {hf_m['pred_mean']:>18.2f} {hf_m['error_mean']:>15.2f}")
    print(f"  {'Std Heat Flux [W/m²]':<30} {hf_m['gt_std']:>18.2f} {hf_m['pred_std']:>18.2f} {'':>15}")
    print(f"  {'Min Heat Flux [W/m²]':<30} {hf_m['gt_min']:>18.2f} {hf_m['pred_min']:>18.2f} {'':>15}")
    print(f"  {'Max Heat Flux [W/m²]':<30} {hf_m['gt_max']:>18.2f} {hf_m['pred_max']:>18.2f} {'':>15}")
    print(f"  {'RMSE [W/m²]':<30} {'':>18} {'':>18} {hf_m['error_rmse']:>15.2f}")
    print(f"  {'Relative Error [%]':<30} {'':>18} {'':>18} {hf_m['relative_error']:>15.2f}")
    
    # Row Temperature Analysis - All rows in one table
    print("\n📊 ROW TEMPERATURE ANALYSIS (averaged across all frames and x-positions)")
    print("-" * 90)
    print(f"  {'Row (y)':<12} {'GT Mean [°C]':>15} {'GT Std [°C]':>15} {'Pred Mean [°C]':>15} {'Pred Std [°C]':>15} {'Error [°C]':>12}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
    
    for row_key, row_values in metrics['row_temperature'].items():
        row_y = row_key.replace('row_', '')
        print(f"  {'y = ' + row_y:<12} {row_values['gt_mean']:>15.3f} {row_values['gt_std']:>15.3f} "
              f"{row_values['pred_mean']:>15.3f} {row_values['pred_std']:>15.3f} {row_values['error_mean']:>12.3f}")
    
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
    
    # Compute average error across all rows
    row_errors = [v['error_mean'] for v in metrics['row_temperature'].values()]
    avg_row_error = np.mean(row_errors) if row_errors else 0
    print(f"  {'Average':<12} {'':>15} {'':>15} {'':>15} {'':>15} {avg_row_error:>12.3f}")
    
    # Vorticity Analysis - Only for Task 2/3
    if has_velocity:
        print("\n🌀 VORTICITY (Rotational Flow Patterns)")
        print("   Vorticity ω = ∂v/∂x - ∂u/∂y (positive = counter-clockwise)")
        print("   Interface excluded using MASSFLUX-based detection (non-zero interface velocity)")
        print("   Units: Non-dimensional (velocity and length are non-dimensionalized in BubbleML)")
        print("-" * 85)
        
        # Show vorticity for different regions
        for region_key, region_label in [
            ('vorticity', 'ALL REGIONS (reference)'),
            ('vorticity_excluding_interface', 'EXCLUDING INTERFACE (non-zero massflux)'),
            ('vorticity_bulk_liquid', 'BULK LIQUID ONLY (SDF<0 & not at interface)')
        ]:
            vort_m = metrics[region_key]
            print(f"\n  📍 {region_label}")
            print(f"  {'Metric':<30} {'Ground Truth':>18} {'Prediction':>18} {'Error':>15}")
            print(f"  {'-'*30} {'-'*18} {'-'*18} {'-'*15}")
            print(f"  {'Mean ω [1/s]':<30} {vort_m['gt_mean']:>18.6f} {vort_m['pred_mean']:>18.6f} {'':>15}")
            print(f"  {'Mean |ω| [1/s]':<30} {vort_m['gt_mean_abs']:>18.6f} {vort_m['pred_mean_abs']:>18.6f} {vort_m['error_mean']:>15.6f}")
            print(f"  {'Max |ω| [1/s]':<30} {vort_m['gt_max']:>18.6f} {vort_m['pred_max']:>18.6f} {'':>15}")
            print(f"  {'Std ω [1/s]':<30} {vort_m['gt_std']:>18.6f} {vort_m['pred_std']:>18.6f} {'':>15}")
            print(f"  {'RMSE [1/s]':<30} {'':>18} {'':>18} {vort_m['error_rmse']:>15.6f}")
    
    # Region-wise Error Analysis
    print("\n📍 REGION-WISE ERROR ANALYSIS")
    print("   SDF convention: SDF<0=liquid, SDF>0=vapor/bubble, SDF=0=interface")
    print("   Regions: near-wall (y<16), inside-bubble (SDF>0), near-interface, bulk-liquid (SDF<0 & not interface)")
    print("   Temperature regions: SDF-based interface detection (geometric)")
    if has_velocity:
        print("   Velocity regions: Massflux-based interface detection (non-zero velocity)")
        print("   Note: Temperature in °C (dimensional), Velocity in non-dimensional units [-]")
    else:
        print("   Note: Temperature in °C (dimensional)")
    print("-" * 95)
    
    # Temperature errors by region
    print("\n  🌡️  Temperature Errors by Region:")
    print(f"     {'Region':<20} {'# Cells':>12} {'GT Mean [°C]':>14} {'Pred Mean [°C]':>14} {'MAE [°C]':>12} {'RMSE [°C]':>12}")
    print(f"     {'-'*20} {'-'*12} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
    for region_name, errors in metrics['region_errors']['temperature'].items():
        if errors['num_cells'] > 0:
            print(f"     {region_name:<20} {errors['num_cells']:>12} {errors['mean_gt']:>14.3f} "
                  f"{errors['mean_pred']:>14.3f} {errors['mae']:>12.4f} {errors['rmse']:>12.4f}")
        else:
            print(f"     {region_name:<20} {'0':>12} {'N/A':>14} {'N/A':>14} {'N/A':>12} {'N/A':>12}")
    
    # Velocity errors by region - Only for Task 2/3
    if has_velocity:
        # Velocity-x errors by region (non-dimensional)
        print("\n  ➡️  Velocity-X Errors by Region (non-dimensional):")
        print(f"     {'Region':<20} {'# Cells':>12} {'GT Mean [-]':>14} {'Pred Mean [-]':>14} {'MAE [-]':>12} {'RMSE [-]':>12}")
        print(f"     {'-'*20} {'-'*12} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
        for region_name, errors in metrics['region_errors']['velx'].items():
            if errors['num_cells'] > 0:
                print(f"     {region_name:<20} {errors['num_cells']:>12} {errors['mean_gt']:>14.6f} "
                      f"{errors['mean_pred']:>14.6f} {errors['mae']:>12.6f} {errors['rmse']:>12.6f}")
            else:
                print(f"     {region_name:<20} {'0':>12} {'N/A':>14} {'N/A':>14} {'N/A':>12} {'N/A':>12}")
        
        # Velocity-y errors by region (non-dimensional)
        print("\n  ⬆️  Velocity-Y Errors by Region (non-dimensional):")
        print(f"     {'Region':<20} {'# Cells':>12} {'GT Mean [-]':>14} {'Pred Mean [-]':>14} {'MAE [-]':>12} {'RMSE [-]':>12}")
        print(f"     {'-'*20} {'-'*12} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
        for region_name, errors in metrics['region_errors']['vely'].items():
            if errors['num_cells'] > 0:
                print(f"     {region_name:<20} {errors['num_cells']:>12} {errors['mean_gt']:>14.6f} "
                      f"{errors['mean_pred']:>14.6f} {errors['mae']:>12.6f} {errors['rmse']:>12.6f}")
            else:
                print(f"     {region_name:<20} {'0':>12} {'N/A':>14} {'N/A':>14} {'N/A':>12} {'N/A':>12}")
    
    print("\n" + "=" * 95)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Physics Metrics for Task 1 (Temperature from SDF), Task 2 & Task 3 (Velocity from Interface)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths
    parser.add_argument('--checkpoint', type=str,
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_47096469/checkpoints/epoch=145-step=485742.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_noisy_velocity_from_interface_pb_subcooled_singlestep_none_47411585/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_noisy_velocity_from_interface_pb_subcooled_singlestep_none_47407293/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_47455112/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_noisy_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47455592/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47455411/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47635283/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47552213/checkpoints/epoch=28-step=192966.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/unet_ar_32_ar_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47551858/checkpoints/epoch=29-step=099810.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_pf_max3_decay_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47595777/checkpoints/epoch=09-step=008300.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47635054/checkpoints/epoch=28-step=024070.ckpt",
                        # default=None,
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820981/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820980/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_div_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820985/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_div_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820984/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_vort_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820992/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_vort_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820993/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_adv_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820994/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_adv_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820995/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_div_vort_adv_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47821002/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_div_vort_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47821001/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_div_vort_adv_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820993/checkpoints/last.ckpt",
                        # Level 1
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47841592/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47835444/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/unet_32_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47842727/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/unet_32_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47845919/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/bubble_ddpm_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849371/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/bubble_ddpm_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849378/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/ve_sde_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849739/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/ve_sde_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849759/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/ffno_m12_w64_l4_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849886/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/ffno_m12_w64_l4_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849889/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/edm_ch32_b2_s50_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47852082/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/edm_ch32_b2_s50_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47853227/checkpoints/last.ckpt",
                        
                        # Level 2 & 3
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47856390/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856403/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/unet_ar_32_ar_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47856451/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/unet_ar_32_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856472/checkpoints/last.ckpt",
                        
                        # fm_ablation_all
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47877420/checkpoints/last.ckpt",
                        # fm_ablation_skip
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47877434/checkpoints/last.ckpt",
                        # fm_ablation_adaptive
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47877548/checkpoints/last.ckpt",
                        # fm_ablation_attention
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47877559/checkpoints/last.ckpt",
                        # fm_ablation_baseline
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47877572/checkpoints/last.ckpt",
                        
                        # bootstrap ablation
                        # ablation_1: history_length=10, use_availability_mask=true, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861270/checkpoints/last.ckpt",
                        # ablation_2: history_length=5, use_availability_mask=true, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist5_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861275/checkpoints/last.ckpt",
                        # ablation_3: history_length=20, use_availability_mask=true, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861276/checkpoints/last.ckpt",
                        # ablation_7: history_length=40, use_availability_mask=true, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist40_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861282/checkpoints/last.ckpt",
                        # ablation_4: history_length=10, use_availability_mask=false, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861277/checkpoints/last.ckpt",
                        # ablation_5: history_length=10, use_availability_mask=true, training_strategy=teacher_forcing
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861278/checkpoints/last.ckpt",
                        # ablation_6: history_length=10, use_availability_mask=true, training_strategy=scheduled_sampling
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861279/checkpoints/last.ckpt",

                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_32_256_2_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_48594826/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/diffusionpde_ch32_b2_s50_zobs1.0_zpde0.5_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50073800/checkpoints/last.ckpt",
                        help='Path to model checkpoint')
    parser.add_argument('--data-file', type=str,
                        default="/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5",
                        help='Path to HDF5 data file')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./ICML/test_DiffusionPDE',
                        help='Output directory for CSV and plots')
    parser.add_argument('--output-filename', type=str, 
                        default='physics_metrics_task2_test.csv',
                        # default='physics_metrics_task2_test.csv',
                        # default='physics_metrics_task2_47820981.csv',
                        # default='physics_metrics_task2_47820980.csv',
                        # default='physics_metrics_task2_47820985.csv',
                        # default='physics_metrics_task2_47820984.csv',
                        # default='physics_metrics_task2_47820992.csv',
                        # default='physics_metrics_task2_47820993.csv',
                        # default='physics_metrics_task2_47820994.csv',
                        # default='physics_metrics_task2_47820995.csv',
                        # default='physics_metrics_task2_47821002.csv',
                        # default='physics_metrics_task2_47821001.csv',
                        
                        # bootstrap ablation
                        # ablation_1: history_length=10, use_availability_mask=true, training_strategy=push_forward
                        # default='bootstrap_ablation_1_10_true_pf_47861270.csv',
                        # ablation_2: history_length=5, use_availability_mask=true, training_strategy=push_forward
                        # default='bootstrap_ablation_2_5_true_pf_47861275.csv',
                        # ablation_3: history_length=20, use_availability_mask=true, training_strategy=push_forward
                        # default='bootstrap_ablation_3_20_true_pf_47861276.csv',
                        # ablation_7: history_length=40, use_availability_mask=true, training_strategy=push_forward
                        # default='bootstrap_ablation_7_40_true_pf_47861282.csv',
                        # ablation_4: history_length=10, use_availability_mask=false, training_strategy=push_forward
                        # default='bootstrap_ablation_4_10_false_pf_47861277.csv',
                        # ablation_5: history_length=10, use_availability_mask=true, training_strategy=teacher_forcing
                        # default='bootstrap_ablation_5_10_true_tf_47861278.csv',
                        # ablation_6: history_length=10, use_availability_mask=true, training_strategy=scheduled_sampling
                        # default='bootstrap_ablation_6_10_true_ss_47861279.csv',
                        
                        # Level 1
                        # default='flow_matching_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47841592.csv',
                        # default='flow_matching_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47835444.csv',
                        # default='unet_32_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47842727.csv',
                        # default='unet_32_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47845919.csv',
                        # default='bubble_ddpm_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849371.csv',
                        # default='bubble_ddpm_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849378.csv',
                        # default='ve_sde_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849739.csv',
                        # default='ve_sde_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849759.csv',
                        # default='ffno_m12_w64_l4_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849886.csv',
                        # default='ffno_m12_w64_l4_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849889.csv',
                        # default='edm_ch32_b2_s50_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47852082.csv',
                        # default='edm_ch32_b2_s50_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47853227.csv',
                        
                        # Level 2 & 3
                        # default='flow_matching_ar_32_256_2_False_ar_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47856390.csv',
                        # default='flow_matching_ar_32_256_2_False_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856403.csv',
                        # default='unet_ar_32_ar_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47856451.csv',
                        # default='unet_ar_32_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856472.csv',
                        
                        # fm_ablation_all
                        # default='fm_ablation_all_47877420.csv',
                        # fm_ablation_skip
                        # default='fm_ablation_skip_47877434.csv',
                        # fm_ablation_adaptive
                        # default='fm_ablation_adaptive_47877548.csv',
                        # fm_ablation_attention
                        # default='fm_ablation_attention_47877559.csv',
                        # fm_ablation_baseline
                        # default='fm_ablation_baseline_47877572.csv',
                        
                        help='Output CSV filename')
    
    # Model type
    parser.add_argument('--model-type', type=str, default='auto',
                        choices=['flow_matching', 'flow_matching_jit', 'flow_matching_ar', 'flow_matching_ar_bootstrap',
                                'edm_ar_bootstrap', 'unet', 'unet_ar', 'bubble_ddpm', 've_sde', 'ffno', 'edm', 'diffusionpde', 'auto'],
                        help='Model type: flow_matching, flow_matching_jit, flow_matching_ar, flow_matching_ar_bootstrap, '
                             'edm_ar_bootstrap, unet, unet_ar, bubble_ddpm, ve_sde, ffno, edm, diffusionpde, or auto (detect from path)')
    parser.add_argument('--norm-mode', type=str, default='all',
                        choices=['none', 'all', 'temperature_only'],
                        help='Normalization mode: none, all (default), or temperature_only. Must match training setting.')
    
    # Inference parameters
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='Number of ODE integration steps')
    parser.add_argument('--start-time', type=int, default=100,
                        help='Starting timestep for analysis')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (None = all)')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Downsampling factor for fast prototyping (1=full res, 4=128x128)')
    
    # Frame range for quick testing
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Starting frame index for metrics (0-based, for quick testing)')
    parser.add_argument('--frame-end', type=int, default=10,
                        help='Ending frame index (exclusive) for metrics (None = all frames)')
    
    # Temporal model specific arguments
    parser.add_argument('--history-length', type=int, default=10,
                        help='Number of historical frames for temporal/bootstrap models')
    parser.add_argument('--history-stride', type=int, default=1,
                        help='Stride between history frames for bootstrap (1=consecutive, 2=every other). '
                             'Auto-detected from checkpoint if available.')
    parser.add_argument('--temporal-stride', type=int, default=2,
                        help='Stride between frames for temporal model')
    parser.add_argument('--temporal-hidden-dim', type=int, default=32,
                        help='S4 hidden dimension for temporal model')
    parser.add_argument('--temporal-d-state', type=int, default=16,
                        help='S4 state dimension for temporal model')
    parser.add_argument('--temporal-n-layers', type=int, default=2,
                        help='Number of S4 layers for temporal model')
    
    # AR Bootstrap specific arguments
    parser.add_argument('--rollout-length', type=int, default=5,
                        help='Number of frames in rollout segment for AR Bootstrap model')
    parser.add_argument('--history-encoder-type', type=str, default='temporal_mixer',
                        choices=['conv3d', 'temporal_mixer', 'attention'],
                        help='History encoder type for AR Bootstrap: conv3d (expressive) or temporal_mixer (fast)')
    parser.add_argument('--history-encoder-hidden', type=int, default=32,
                        help='Hidden channels for history encoder in AR Bootstrap model')
    
    # Attention history encoder arguments (only used when --history-encoder-type=attention)
    parser.add_argument('--attention-encoder-embed-dim', type=int, default=256,
                        help='Transformer embed dim for attention history encoder')
    parser.add_argument('--attention-encoder-num-heads', type=int, default=8,
                        help='Number of attention heads for attention history encoder')
    parser.add_argument('--attention-encoder-depth', type=int, default=4,
                        help='Number of spatial-temporal block pairs for attention history encoder')
    parser.add_argument('--attention-encoder-patch-size', type=int, default=8,
                        help='Patch size for attention history encoder')
    parser.add_argument('--attention-encoder-mlp-ratio', type=float, default=4.0,
                        help='FFN expansion ratio for attention history encoder')
    parser.add_argument('--attention-encoder-dropout', type=float, default=0.0,
                        help='Dropout for attention history encoder')
    parser.add_argument('--attention-encoder-output-head', type=str, default='linear',
                        choices=['linear', 'conv_decoder'],
                        help='Output head type for attention history encoder')
    parser.add_argument('--attention-encoder-max-history-length', type=int, default=50,
                        help='Max history length for learned temporal positional embeddings')
    
    # Autoregressive model specific arguments
    parser.add_argument('--solver', type=str, default='rk4',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver for flow_matching_ar model')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                        help='Classifier-free guidance scale for flow_matching_ar (1.0 = no guidance)')
    parser.add_argument('--initial-state-mode', type=str, default='from_data',
                        choices=['from_data', 'zeros', 'from_conditioning', 'small_noise'],
                        help='How to initialize prev_output for AR models: '
                             'from_data (use GT), zeros, from_conditioning, small_noise')
    parser.add_argument('--residual-prediction', action='store_true', default=False,
                        help='Enable residual/delta prediction mode for AR models')
    
    # Bootstrap ablation (for flow_matching_ar_bootstrap / edm_ar_bootstrap)
    parser.add_argument('--bootstrap-ablation', type=str, default=None,
                        choices=['zeros', 'mean_conditioning_naive'],
                        help='Bootstrap ablation mode for AR bootstrap models. '
                             'Replaces the learned history encoder output with a simpler initialization: '
                             'zeros: use all-zero initial state. '
                             'mean_conditioning_naive: average conditioning history and map interface vel to bulk vel.')
    
    # UNet AR specific arguments
    parser.add_argument('--init-features', type=int, default=32,
                        help='Initial features for UNet AR model')
    
    # VE-SDE specific arguments
    parser.add_argument('--sigma-min', type=float, default=0.01,
                        help='Minimum noise level for VE-SDE model')
    parser.add_argument('--sigma-max', type=float, default=1.0,
                        help='Maximum noise level for VE-SDE model (use 1.0 for normalized data)')
    parser.add_argument('--sampling-method', type=str, default='pc',
                        choices=['pc', 'ode', 'euler'],
                        help='Sampling method for VE-SDE: pc (Predictor-Corrector), ode, or euler')
    parser.add_argument('--snr', type=float, default=0.16,
                        help='Signal-to-noise ratio for Langevin corrector in VE-SDE PC sampling')
    
    # Temperature normalization
    parser.add_argument('--normalize-temperature', action='store_true', default=False,
                        help='Enable temperature normalization (default: True)')
    parser.add_argument('--no-normalize-temperature', dest='normalize_temperature', action='store_false',
                        help='Disable temperature normalization')
    parser.add_argument('--normalization-stats', type=str, default=None,
                        help='Path to normalization_stats.json file (overrides auto-detection from checkpoint directory)')
    
    # Row analysis
    parser.add_argument('--row-indices', type=str, default='0,8,16,24,32',
                        help='Comma-separated list of row indices for temperature analysis')
    
    # Task / noise options
    parser.add_argument('--task', type=str, default='auto',
                        choices=['temperature_from_sdf', 'velocity_from_interface', 'noisy_velocity_from_interface', 'auto'],
                        help='Task: temperature_from_sdf (Task 1), velocity_from_interface (Task 2), '
                             'noisy_velocity_from_interface (Task 3), or auto (detect from checkpoint path)')
    parser.add_argument('--use-clean-inputs', action='store_true', default=False,
                        help='For Task 3: use clean inputs instead of noisy (physics fidelity check)')
    
    # Noise parameters (override YAML config for Task 3 noisy inference)
    parser.add_argument('--noise-type', type=str, default='gaussian',
                        choices=['gaussian', 'simple', 'optical_flow', 'complex'],
                        help='Noise type: gaussian/simple or optical_flow/complex (overrides YAML)')
    parser.add_argument('--sdf-noise-std', type=float, default=2,
                        help='SDF noise std (for gaussian noise, overrides YAML)')
    parser.add_argument('--vel-noise-std', type=float, default=1,
                        help='Velocity noise std (for gaussian noise, overrides YAML)')
    # Optical flow noise parameters
    parser.add_argument('--sdf-gradient-scale', type=float, default=None,
                        help='SDF gradient scale (for optical_flow noise, overrides YAML)')
    parser.add_argument('--vel-base-noise-std', type=float, default=None,
                        help='Velocity base noise std (for optical_flow noise, overrides YAML)')
    parser.add_argument('--vel-scale-factor', type=float, default=None,
                        help='Velocity scale factor (for optical_flow noise, overrides YAML)')
    parser.add_argument('--correlation-length', type=float, default=None,
                        help='Spatial correlation length (for optical_flow noise, overrides YAML)')
    
    # Random seed for reproducibility
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility. If set, inference will be deterministic. '
                             'Same seed + same model = identical results every run.')
    parser.add_argument('--no-seed', dest='seed', action='store_const', const=None,
                        help='Disable seed (default behavior, stochastic inference)')

    
    args = parser.parse_args()
    
    # Set random seed for reproducibility if specified
    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        # For full determinism on CUDA (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"🎲 Random seed set to {args.seed} (deterministic mode)")
    
    # Parse row indices
    row_indices = [int(x.strip()) for x in args.row_indices.split(',')]
    
    # Auto-detect task from checkpoint path if not specified
    if args.task == 'auto':
        checkpoint_lower = args.checkpoint.lower()
        if 'temperature_from_sdf' in checkpoint_lower:
            args.task = 'temperature_from_sdf'
            print("🔍 Auto-detected Task 1 (temperature_from_sdf) from checkpoint path")
        elif 'noisy_velocity_from_interface' in checkpoint_lower:
            args.task = 'noisy_velocity_from_interface'
            print("🔍 Auto-detected Task 3 (noisy_velocity_from_interface) from checkpoint path")
        elif 'velocity_from_interface' in checkpoint_lower:
            args.task = 'velocity_from_interface'
            print("🔍 Auto-detected Task 2 (velocity_from_interface) from checkpoint path")
        else:
            # Default fallback
            args.task = 'velocity_from_interface'
            print("⚠️  Could not auto-detect task, defaulting to velocity_from_interface")
    
    # Determine task title
    if args.task == 'temperature_from_sdf':
        task_title = "Task 1: Temperature from SDF"
    elif args.task == 'noisy_velocity_from_interface':
        if args.use_clean_inputs:
            task_title = "Task 3: Noisy Velocity from Interface (CLEAN inputs - physics fidelity)"
        else:
            task_title = "Task 3: Noisy Velocity from Interface (NOISY inputs - deployment)"
    else:
        task_title = "Task 2: Velocity from Interface (clean inputs)"

    # Auto-detect model type from checkpoint path
    model_type = args.model_type
    if model_type == 'auto':
        checkpoint_lower = args.checkpoint.lower()
        if 'unet_ar' in checkpoint_lower:
            print("🔍 Auto-detected UNet AR model from checkpoint path")
            model_type = 'unet_ar'
        elif 'unet' in checkpoint_lower and 'unet_ar' not in checkpoint_lower:
            print("🔍 Auto-detected UNet model from checkpoint path")
            model_type = 'unet'
        elif 'ffno' in checkpoint_lower:
            print("🔍 Auto-detected FFNO model from checkpoint path")
            model_type = 'ffno'
        elif 'edm_ar_bootstrap' in checkpoint_lower:
            print("🔍 Auto-detected EDM AR Bootstrap model from checkpoint path")
            model_type = 'edm_ar_bootstrap'
        elif 'flow_matching_ar_bootstrap' in checkpoint_lower or 'ar_bootstrap' in checkpoint_lower:
            print("🔍 Auto-detected Flow Matching AR Bootstrap model from checkpoint path")
            model_type = 'flow_matching_ar_bootstrap'
        elif 'flow_matching_ar' in checkpoint_lower:
            print("🔍 Auto-detected Flow Matching AR model from checkpoint path")
            model_type = 'flow_matching_ar'
        elif 'bubble_ddpm' in checkpoint_lower or 'ddpm' in checkpoint_lower:
            print("🔍 Auto-detected DDPM model from checkpoint path")
            model_type = 'bubble_ddpm'
        elif 've_sde' in checkpoint_lower or 'vesde' in checkpoint_lower:
            print("🔍 Auto-detected VE-SDE model from checkpoint path")
            model_type = 've_sde'
        elif 'diffusionpde' in checkpoint_lower:
            print("🔍 Auto-detected DiffusionPDE model from checkpoint path")
            model_type = 'diffusionpde'
        elif 'edm' in checkpoint_lower:
            print("🔍 Auto-detected EDM model from checkpoint path")
            model_type = 'edm'
        elif 'flow_matching_jit' in checkpoint_lower:
            print("🔍 Auto-detected data-prediction Flow Matching (flow_matching_jit) from checkpoint path")
            model_type = 'flow_matching_jit'
        elif 'flow_matching' in checkpoint_lower:
            print("🔍 Auto-detected Flow Matching model from checkpoint path")
            model_type = 'flow_matching'
        else:
            print("⚠️  Could not auto-detect model type, defaulting to flow_matching")
            model_type = 'flow_matching'
    
    is_autoregressive = model_type in ['flow_matching_ar', 'unet_ar']
    is_ar_bootstrap = (model_type in ['flow_matching_ar_bootstrap', 'edm_ar_bootstrap'])
    
    print(f"🔬 Physics Metrics Evaluation - {task_title}")
    print("=" * 70)
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Data file:        {args.data_file}")
    print(f"Model type:       {model_type}")
    print(f"Inference steps:  {args.num_inference_steps}")
    print(f"Start time:       {args.start_time}")
    print(f"Max samples:      {args.max_samples if args.max_samples else 'All'}")
    if args.downsample_factor > 1:
        print(f"Downsample:       {args.downsample_factor}x (512→{512 // args.downsample_factor})")
    # Frame range info
    frame_range_str = f"{args.frame_start} to {args.frame_end if args.frame_end else 'end'}"
    print(f"Frame range:      {frame_range_str}")
    print(f"Row indices:      {row_indices}")
    if args.task == 'noisy_velocity_from_interface':
        print(f"Input type:       {'CLEAN (physics fidelity)' if args.use_clean_inputs else 'NOISY (deployment)'}")
    if is_autoregressive:
        print(f"🔄 Autoregressive Settings:")
        print(f"   Initial state mode: {args.initial_state_mode}")
        if model_type == 'flow_matching_ar':
            print(f"   ODE Solver: {args.solver}")
            print(f"   Guidance scale: {args.guidance_scale}")
        print(f"   Residual prediction: {args.residual_prediction}")
    if is_ar_bootstrap:
        print(f"🚀 AR Bootstrap Settings:")
        print(f"   History length: {args.history_length} frames (for bootstrap)")
        print(f"   Rollout length: {args.rollout_length} frames per segment")
        print(f"   History encoder type: {args.history_encoder_type}")
        print(f"   ODE Solver: {args.solver}")
        print(f"   Bootstrap: Uses history encoder to infer initial state")
    if args.seed is not None:
        print(f"Random seed:      {args.seed} (deterministic)")
    print(f"Output dir:       {args.output_dir}")
    print("=" * 70)
    
    try:
        # Step 1: Load task configuration
        task_cfg = load_task_config(args.task)
        
        # Step 2: Create model configuration
        if model_type == 'flow_matching_ar':
            model_cfg = DictConfig({
                'name': 'flow_matching_ar',
                'in_channels': 9,  # target (3) + conditioning (3) + prev_output (3)
                'out_channels': 3,
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'use_attention': False,
                'dropout': 0.1,
                'num_integration_steps': args.num_inference_steps,
                'conditioning_strategy': 'none',
                'temp_min': 55.0,
                'temp_max': 120.0,
                'residual_prediction': args.residual_prediction,
                'inference': {
                    'solver': args.solver,
                    'guidance_scale': args.guidance_scale,
                },
                # Physics-informed losses (only used during training, not inference)
                'auxiliary_losses': {
                    'spectral_enabled': False,
                    'gradient_enabled': False,
                    'divergence_enabled': False,
                    'vorticity_enabled': False,
                    'advection_enabled': False,
                },
            })
        elif model_type == 'unet_ar':
            model_cfg = DictConfig({
                'name': 'unet_ar',
                'in_channels': 6,  # conditioning (3) + prev_output (3)
                'out_channels': 3,
                'init_features': args.init_features,
                'conditioning_strategy': 'none',
                'temp_min': 55.0,
                'temp_max': 120.0,
                'residual_prediction': args.residual_prediction,
            })
        elif model_type == 'flow_matching_ar_bootstrap':
            # AR Bootstrap: in_channels = target (3) + conditioning (3) + prev_output (3) + availability_mask (1)
            model_cfg = DictConfig({
                'name': 'flow_matching_ar_bootstrap',
                'in_channels': 10,  # target (3) + conditioning (3) + prev_output (3) + mask (1)
                'out_channels': 3,
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'use_attention': False,
                'dropout': 0.1,
                'num_integration_steps': args.num_inference_steps,
                'temp_min': 55.0,
                'temp_max': 120.0,
                # Bootstrap specific
                'history_length': args.history_length,
                'rollout_length': args.rollout_length,
                'use_availability_mask': True,
                'history_encoder_type': args.history_encoder_type,
                'history_encoder_hidden': args.history_encoder_hidden,
                'attention_encoder_embed_dim': args.attention_encoder_embed_dim,
                'attention_encoder_num_heads': args.attention_encoder_num_heads,
                'attention_encoder_depth': args.attention_encoder_depth,
                'attention_encoder_patch_size': args.attention_encoder_patch_size,
                'attention_encoder_mlp_ratio': args.attention_encoder_mlp_ratio,
                'attention_encoder_dropout': args.attention_encoder_dropout,
                'attention_encoder_output_head': args.attention_encoder_output_head,
                'attention_encoder_max_history_length': args.attention_encoder_max_history_length,
                'bootstrap_loss_weight': 1.0,
                'ar_loss_weight': 1.0,
                'bootstrap_state_loss_weight': 0.5,
                'inference': {
                    'solver': args.solver,
                },
            })
        elif model_type == 'edm_ar_bootstrap':
            model_cfg = DictConfig({
                'name': 'edm_ar_bootstrap',
                'base_resolution': 512,
                'model_channels': 32,
                'channel_mult': [1, 2, 4],
                'channel_mult_emb': 4,
                'num_blocks': 2,
                'dropout': 0.10,
                'sigma_min': 0.002,
                'sigma_max': 80,
                'sigma_data': 0.5,
                'rho': 7,
                'embedding_type': 'positional',
                'channel_mult_noise': 1,
                'encoder_type': 'standard',
                'decoder_type': 'standard',
                'resample_filter': [1, 1],
                'use_fp16': False,
                'num_sampling_steps': args.num_inference_steps,
                'temp_min': 55.0,
                'temp_max': 120.0,
                'history_length': args.history_length,
                'rollout_length': args.rollout_length,
                'use_availability_mask': False,
                'history_encoder_type': args.history_encoder_type,
                'history_encoder_hidden': args.history_encoder_hidden,
                'history_encoder_blocks': 3,
                'temporal_mixer_spatial_conv': True,
                'temporal_mixer_temporal_weights': True,
                'attention_encoder_embed_dim': args.attention_encoder_embed_dim,
                'attention_encoder_num_heads': args.attention_encoder_num_heads,
                'attention_encoder_depth': args.attention_encoder_depth,
                'attention_encoder_patch_size': args.attention_encoder_patch_size,
                'attention_encoder_mlp_ratio': args.attention_encoder_mlp_ratio,
                'attention_encoder_dropout': args.attention_encoder_dropout,
                'attention_encoder_output_head': args.attention_encoder_output_head,
                'attention_encoder_max_history_length': args.attention_encoder_max_history_length,
                'conditioning_strategy': 'none',
                'inference': {
                    'solver': args.solver,
                },
            })
        elif model_type == 'unet':
            # UNet: frame-to-frame direct regression
            # in_channels = num_conditioning, out_channels = num_target
            model_cfg = DictConfig({
                'name': 'unet',
                'init_features': 32,
                'temp_min': 55.0,
                'temp_max': 120.0,
            })
        elif model_type == 'ffno':
            # FFNO: frame-to-frame Fourier Neural Operator
            # in_channels = num_conditioning, out_channels = num_target
            model_cfg = DictConfig({
                'name': 'ffno',
                'modes': 12,
                'width': 64,
                'n_layers': 4,
                'dropout': 0.0,
                'use_fork': False,
                'fourier_mode': 'full',
                'temp_min': 55.0,
                'temp_max': 120.0,
            })
        elif model_type == 'bubble_ddpm':
            # DDPM: frame-to-frame diffusion model
            # in_channels = num_target (noisy) + num_conditioning
            # out_channels = num_target (noise prediction)
            num_cond = len(task_cfg.conditioning_channels)
            num_target = len(task_cfg.target_channels)
            in_channels = num_target + num_cond
            out_channels = num_target
            
            model_cfg = DictConfig({
                'name': 'bubble_ddpm',
                'in_channels': in_channels,
                'out_channels': out_channels,
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'use_attention': False,
                'dropout': 0.1,
                'num_timesteps': 1000,
                'beta_start': 1e-4,
                'beta_end': 2e-2,
                'num_inference_steps': 1000,
                'temp_min': 55.0,
                'temp_max': 120.0,
            })
        elif model_type == 've_sde':
            # VE-SDE: frame-to-frame score-based diffusion
            # in_channels = num_target (noisy) + num_conditioning
            # out_channels = num_target (noise prediction)
            num_cond = len(task_cfg.conditioning_channels)
            num_target = len(task_cfg.target_channels)
            in_channels = num_target + num_cond
            out_channels = num_target
            
            model_cfg = DictConfig({
                'name': 've_sde',
                'in_channels': in_channels,
                'out_channels': out_channels,
                'base_channels': 32,
                'sigma_embed_dim': 256,
                'num_res_blocks': 2,
                'use_attention': False,
                'dropout': 0.1,
                'sigma_min': getattr(args, 'sigma_min', 0.01),
                'sigma_max': getattr(args, 'sigma_max', 1.0),
                'num_sampling_steps': args.num_inference_steps,
                'sampling_method': getattr(args, 'sampling_method', 'pc'),
                'snr': getattr(args, 'snr', 0.16),
                'conditioning_strategy': 'none',
                'temp_min': 55.0,
                'temp_max': 120.0,
                'num_inference_steps': args.num_inference_steps,
            })
        elif model_type == 'edm':
            # EDM: frame-to-frame EDM-style diffusion
            # img_resolution is computed dynamically from base_resolution / downsample_factor
            # Configured to match flow_matching for fair comparison (32→64→128 channels)
            # downsample_factor comes from args (which should match data config)
            model_cfg = DictConfig({
                'name': 'edm',
                'base_resolution': 512,  # BubbleML dataset base resolution
                'downsample_factor': args.downsample_factor,  # From data config via args
                'model_channels': 32,    # Matches flow_matching base_channels
                'channel_mult': [1, 2, 4],  # 32→64→128 (matches flow_matching)
                'channel_mult_emb': 4,
                'num_blocks': 2,         # Matches flow_matching num_res_blocks
                # attn_resolutions computed dynamically based on img_resolution
                'dropout': 0.10,
                'sigma_min': 0.002,
                'sigma_max': 80,
                'sigma_data': 0.5,
                'rho': 7,
                'embedding_type': 'positional',
                'channel_mult_noise': 1,
                'encoder_type': 'standard',
                'decoder_type': 'standard',
                'resample_filter': [1, 1],
                'use_fp16': False,
                'num_sampling_steps': args.num_inference_steps,
                'solver': args.solver,
                'temp_min': 55.0,
                'temp_max': 120.0,
            })
        elif model_type == 'diffusionpde':
            model_cfg = DictConfig({
                'name': 'diffusionpde',
                'base_resolution': 512,
                'downsample_factor': args.downsample_factor,
                'model_channels': 32,
                'channel_mult': [1, 2, 4],
                'channel_mult_emb': 4,
                'num_blocks': 2,
                'dropout': 0.10,
                'sigma_min': 0.002,
                'sigma_max': 80,
                'sigma_data': 0.5,
                'rho': 7,
                'embedding_type': 'positional',
                'channel_mult_noise': 1,
                'encoder_type': 'standard',
                'decoder_type': 'standard',
                'resample_filter': [1, 1],
                'use_fp16': False,
                'num_sampling_steps': args.num_inference_steps,
                'solver': args.solver,
                'zeta_obs': 1.0,
                'zeta_pde': 0.5,
                'pde_start_fraction': 0.8,
                'pde_obs_decay': 0.1,
                'bulk_sdf_threshold': 0.05,
                'temp_min': 55.0,
                'temp_max': 120.0,
            })
        elif model_type == 'flow_matching_jit':
            # JiT Vision Transformer with data prediction + velocity loss
            model_cfg = DictConfig({
                'name': 'flow_matching_jit',
                'img_size': 128,
                'patch_size': 8,
                'hidden_size': 384,
                'depth': 8,
                'num_heads': 6,
                'mlp_ratio': 4.0,
                'bottleneck_dim': 64,
                'dropout': 0.0,
                'P_mean': -0.8,
                'P_std': 0.8,
                'noise_scale': 1.0,
                't_eps': 1e-5,
                'num_integration_steps': args.num_inference_steps,
                'temp_min': 55.0,
                'temp_max': 120.0,
                'inference': {
                    'solver': getattr(args, 'solver', 'heun'),
                    'guidance_scale': 1.0,
                }
            })
        else:
            # Compute channels dynamically from task_cfg
            num_cond = len(task_cfg.conditioning_channels)
            num_target = len(task_cfg.target_channels)
            # in_channels = num_target (current state) + num_conditioning
            in_channels = num_target + num_cond
            out_channels = num_target
            
            model_cfg = DictConfig({
                'name': 'flow_matching',
                'in_channels': in_channels,
                'out_channels': out_channels,
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'attention_type': 'bottleneck',  # 'none', 'bottleneck'
                'adaptive_scale': False,
                'skip_scale': False,
                'dropout': 0.1,
                'num_integration_steps': args.num_inference_steps,
                'temp_min': 55.0,
                'temp_max': 120.0,
                'inference': {
                    'solver': getattr(args, 'solver', 'heun'),
                    'guidance_scale': 1.0,
                }
            })
        
        optim_cfg = DictConfig({'name': 'adamw', 'lr': 0.001})
        scheduler_cfg = DictConfig({'name': 'cosine'})
        
        # Step 2.5: Load normalization statistics
        # Priority: 1) Explicit file path, 2) Checkpoint directory, 3) Compute from data
        normalization_stats = None
        
        # Option 1: Load from explicitly provided file (via --normalization-stats)
        if args.normalization_stats and os.path.exists(args.normalization_stats):
            print(f"\n📊 Loading normalization stats from provided file: {args.normalization_stats}")
            with open(args.normalization_stats, 'r') as f:
                normalization_stats = json.load(f)
            print(f"   ✓ Loaded normalization stats:")
            print(f"      Temperature: [{normalization_stats['temperature']['min']:.2f}, {normalization_stats['temperature']['max']:.2f}]°C")
            print(f"      Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
            print(f"      SDF scale: {normalization_stats['sdf']['scale']:.4f}")
        elif args.normalization_stats:
            # User provided path but file doesn't exist
            print(f"\n⚠️  WARNING: Normalization stats file not found: {args.normalization_stats}")
            print(f"   📊 Falling back to checkpoint directory or computing from data...")
        
        # Option 2: Try to load from checkpoint directory (if not already loaded)
        if normalization_stats is None:
            checkpoint_dir = os.path.dirname(args.checkpoint)
            if "checkpoints" in checkpoint_dir:
                checkpoint_dir = os.path.dirname(checkpoint_dir)  # Go up one level
            stats_file = os.path.join(checkpoint_dir, "normalization_stats.json")

            if os.path.exists(stats_file):
                print(f"\n📊 Loading normalization stats from training: {stats_file}")
                with open(stats_file, 'r') as f:
                    normalization_stats = json.load(f)
                print(f"   ✓ Loaded training normalization stats:")
                print(f"      Temperature: [{normalization_stats['temperature']['min']:.2f}, {normalization_stats['temperature']['max']:.2f}]°C")
                print(f"      Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
                print(f"      SDF scale: {normalization_stats['sdf']['scale']:.4f}")
        
        # Option 3: Fallback: compute from inference file (WARNING: may not match training!)
        if normalization_stats is None:
            print(f"\n⚠️  WARNING: normalization_stats.json not found")
            print(f"   📊 Computing normalization stats from inference file (may not match training!)...")
            print(f"   💡 For accurate results, provide --normalization-stats or ensure normalization_stats.json exists in checkpoint directory")
            normalization_stats = compute_normalization_stats(
                filenames=[args.data_file],
                start_time=args.start_time,
                verbose=True
            )
        
        # Step 3: Load model
        model = load_model_from_checkpoint(
            args.checkpoint, model_cfg, optim_cfg, scheduler_cfg, task_cfg,
            model_type=model_type,
            normalization_stats=normalization_stats,
            norm_mode=args.norm_mode
        )
        
        # Sync bootstrap args from checkpoint so dataset matches training config
        if model_type in ('flow_matching_ar_bootstrap', 'edm_ar_bootstrap'):
            if hasattr(model, 'history_length') and model.history_length != args.history_length:
                print(f"   ⚠️  Overriding --history-length {args.history_length} → {model.history_length} (from checkpoint)")
                args.history_length = model.history_length
            if hasattr(model, 'history_stride') and model.history_stride != args.history_stride:
                print(f"   ⚠️  Overriding --history-stride {args.history_stride} → {model.history_stride} (from checkpoint)")
                args.history_stride = model.history_stride
            if hasattr(model, 'rollout_length') and model.rollout_length != args.rollout_length:
                print(f"   ⚠️  Overriding --rollout-length {args.rollout_length} → {model.rollout_length} (from checkpoint)")
                args.rollout_length = model.rollout_length
        
        # Step 3.5: Extract noise configuration for Task 3
        noise_cfg = None
        if args.task == 'noisy_velocity_from_interface' and 'noise_cfg' in task_cfg:
            noise_cfg = dict(task_cfg.noise_cfg)
            if args.use_clean_inputs:
                # Disable noise for clean-input physics evaluation
                noise_cfg['enabled'] = False
            else:
                # Override noise parameters from CLI if specified
                if args.noise_type is not None:
                    noise_cfg['noise_type'] = args.noise_type
                if args.sdf_noise_std is not None:
                    noise_cfg['sdf_noise_std'] = args.sdf_noise_std
                if args.vel_noise_std is not None:
                    noise_cfg['vel_noise_std'] = args.vel_noise_std
                # Optical flow specific parameters
                if args.sdf_gradient_scale is not None:
                    noise_cfg['sdf_gradient_scale'] = args.sdf_gradient_scale
                if args.vel_base_noise_std is not None:
                    noise_cfg['vel_base_noise_std'] = args.vel_base_noise_std
                if args.vel_scale_factor is not None:
                    noise_cfg['vel_scale_factor'] = args.vel_scale_factor
                if args.correlation_length is not None:
                    noise_cfg['correlation_length'] = args.correlation_length
                
                # Log the effective noise configuration
                noise_type = noise_cfg.get('noise_type', 'optical_flow')
                print(f"\n🔊 Noise configuration (effective):")
                print(f"   Type: {noise_type}")
                if noise_type in ['gaussian', 'simple']:
                    print(f"   SDF noise std: {noise_cfg.get('sdf_noise_std', 0.1)}")
                    print(f"   Vel noise std: {noise_cfg.get('vel_noise_std', 0.05)}")
                else:
                    print(f"   SDF noise std: {noise_cfg.get('sdf_noise_std', 0.1)}")
                    print(f"   SDF gradient scale: {noise_cfg.get('sdf_gradient_scale', 0.3)}")
                    print(f"   Vel base noise std: {noise_cfg.get('vel_base_noise_std', 0.05)}")
                    print(f"   Vel scale factor: {noise_cfg.get('vel_scale_factor', 0.15)}")
                    print(f"   Correlation length: {noise_cfg.get('correlation_length', 3.0)}")

        # Step 4: Load dataset (with or without noise)
        # Use SAME normalization stats as training for consistency!
        dataset = load_dataset(
            args.data_file,
            output_fields=['temperature', 'velx', 'vely'],
            start_time=args.start_time,
            normalize_temperature=args.normalize_temperature,
            return_wall_temp=False,
            noise_cfg=noise_cfg,
            use_clean_inputs=args.use_clean_inputs,
            is_temporal=False,
            is_autoregressive=is_autoregressive,
            is_ar_bootstrap=is_ar_bootstrap,
            history_length=args.history_length if is_ar_bootstrap else 1,
            temporal_stride=1,
            history_stride=args.history_stride if is_ar_bootstrap else 1,
            rollout_length=args.rollout_length if is_ar_bootstrap else 5,
            downsample_factor=args.downsample_factor,
            normalization_stats=normalization_stats,
            norm_mode=args.norm_mode
        )
        
        # Step 5: Load ground truth data for SDF and heater temperature
        # Note: Interface velocities not needed - massflux detection uses bulk velocities
        sdf_gt, _, _, _, _, _, heater_temp = load_ground_truth_data(
            args.data_file, args.start_time, args.downsample_factor
        )
        
        # Step 5.5: Determine actual number of samples to process based on frame range
        frame_start = args.frame_start
        total_available = len(dataset)
        frame_end = args.frame_end if args.frame_end is not None else total_available
        
        # For AR Bootstrap, frame_start/frame_end refer to actual FRAMES we want to evaluate
        # We need to calculate how many segments to process to get those frames
        if is_ar_bootstrap:
            rollout_length = dataset.rollout_length
            desired_num_frames = frame_end - frame_start
            
            # Calculate number of segments needed to cover the desired frames
            # ceil division: num_segments = ceil(desired_num_frames / rollout_length)
            num_segments_needed = (desired_num_frames + rollout_length - 1) // rollout_length
            
            # Check bounds
            if frame_start >= total_available:
                raise ValueError(f"frame_start ({frame_start}) >= total available segments ({total_available})")
            
            num_segments_needed = min(num_segments_needed, total_available - frame_start)
            actual_num_frames = num_segments_needed * rollout_length
            
            # Apply max_samples if specified (limits number of segments)
            if args.max_samples is not None:
                num_segments_needed = min(num_segments_needed, args.max_samples)
                actual_num_frames = num_segments_needed * rollout_length
            
            # Debug: verify timestep alignment for AR Bootstrap
            # BulkFlowARBootstrap: segment_idx i → segment_start = effective_start_time + i
            # With effective_start_time = max(start_time, history_length) = max(100, 10) = 100
            # So segment_idx = frame_start → segment_start = 100 + frame_start = HDF5 timestep
            effective_start = max(args.start_time, args.history_length)
            first_hdf5_timestep = effective_start + frame_start
            last_hdf5_timestep = first_hdf5_timestep + actual_num_frames - 1
            
            print(f"\n📋 AR Bootstrap: Desired frames [{frame_start}:{frame_end}] ({desired_num_frames} frames)")
            print(f"   Segments to process: {num_segments_needed} (each has {rollout_length} frames)")
            print(f"   Actual frames generated: {actual_num_frames}")
            print(f"   📍 Starting segment index: {frame_start}")
            print(f"   📍 HDF5 timesteps: {first_hdf5_timestep} to {last_hdf5_timestep}")
            print(f"   📍 SDF slice: [{frame_start}:{frame_start + actual_num_frames}] → HDF5 timesteps {args.start_time + frame_start} to {args.start_time + frame_start + actual_num_frames - 1}")
            
            # For SDF, slice to match actual frames generated
            sdf_gt = sdf_gt[frame_start:frame_start + actual_num_frames]
            
            # Store for later use
            num_frames_to_process = num_segments_needed  # This is segment count for AR Bootstrap
        else:
            frame_end = min(frame_end, total_available)
            
            if frame_start >= total_available:
                raise ValueError(f"frame_start ({frame_start}) >= total available frames ({total_available})")
            
            num_frames_to_process = frame_end - frame_start
            print(f"\n📋 Frame range: [{frame_start}:{frame_end}] ({num_frames_to_process} frames)")
            
            # Apply max_samples on top of frame range if specified
            if args.max_samples is not None:
                num_frames_to_process = min(num_frames_to_process, args.max_samples)
                frame_end = frame_start + num_frames_to_process
                print(f"   Limited by max_samples to {num_frames_to_process} frames")
            
            # Slice ground truth SDF to match frame range
            sdf_gt = sdf_gt[frame_start:frame_end]
        
        # Step 6: Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n🖥️  Using device: {device}")
        
        # Step 7: Run inference on the specified frame range
        if is_ar_bootstrap:
            # Use specialized AR Bootstrap inference function
            gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp = run_ar_bootstrap_inference_batch(
                model, dataset, device, args.num_inference_steps,
                max_samples=num_frames_to_process,
                start_idx=frame_start,
                solver=args.solver,
                bootstrap_ablation=args.bootstrap_ablation,
            )
            # Ensure SDF matches the actual number of frames generated
            actual_frames = gt_temp.shape[0] if gt_temp is not None else 0
            if sdf_gt.shape[0] > actual_frames:
                sdf_gt = sdf_gt[:actual_frames]
            elif sdf_gt.shape[0] < actual_frames:
                print(f"⚠️ Warning: SDF frames ({sdf_gt.shape[0]}) < generated frames ({actual_frames})")
                # Pad or truncate as needed
                gt_velx = gt_velx[:sdf_gt.shape[0]]
                gt_vely = gt_vely[:sdf_gt.shape[0]]
                gt_temp = gt_temp[:sdf_gt.shape[0]]
                pred_velx = pred_velx[:sdf_gt.shape[0]]
                pred_vely = pred_vely[:sdf_gt.shape[0]]
                pred_temp = pred_temp[:sdf_gt.shape[0]]
        else:
            # Both BulkFlowAutoregressive and other datasets use the same indexing:
            # timestep = idx + effective_start_time
            # where effective_start_time = max(start_time, ...)
            # So no adjustment is needed - use frame_start directly
            
            # Debug: verify timestep alignment
            # Both datasets: timestep = idx + effective_start_time = idx + max(start_time, 1) ≈ idx + start_time
            # SDF: sdf_gt[idx] = HDF5 timestep (start_time + idx)
            # These should match when using the same idx
            effective_start = max(args.start_time, 1)  # BulkFlowAutoregressive uses max(start_time, 1)
            first_hdf5_timestep = frame_start + effective_start
            last_hdf5_timestep = (frame_start + num_frames_to_process - 1) + effective_start
            print(f"   📍 Dataset indices: {frame_start} to {frame_start + num_frames_to_process - 1}")
            print(f"   📍 HDF5 timesteps: {first_hdf5_timestep} to {last_hdf5_timestep}")
            print(f"   📍 SDF slice corresponds to HDF5 timesteps: {args.start_time + frame_start} to {args.start_time + frame_end - 1}")
            
            gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp = run_inference_batch(
                model, dataset, device, args.num_inference_steps, 
                max_samples=num_frames_to_process,
                model_type=model_type,
                start_idx=frame_start,  # No adjustment needed - same indexing as other datasets
                solver=args.solver,
                guidance_scale=args.guidance_scale,
                initial_state_mode=args.initial_state_mode
            )
        
        # Debug: print SDF first frame stats for alignment verification
        print(f"\n📍 DEBUG: SDF first frame mean = {sdf_gt[0].mean():.4f}")
        print(f"📍 DEBUG: SDF shape = {sdf_gt.shape}, GT temp shape = {gt_temp.shape if gt_temp is not None else 'N/A'}")
        
        # Step 8: Compute all physics metrics
        # Uses massflux-based detection (bulk velocities) for velocity, SDF-based for temperature
        metrics = compute_all_physics_metrics(
            gt_velx, gt_vely, gt_temp,
            pred_velx, pred_vely, pred_temp,
            sdf_gt,
            heater_temp,
            row_indices,
            downsample_factor=args.downsample_factor
        )
        
        # Step 9: Print summary table
        print_summary_table(metrics)
        
        # Step 10: Create DataFrames and save
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, args.output_filename)
        
        # Full detailed CSV
        df = metrics_to_dataframe(metrics, args.data_file, args.checkpoint)
        df.to_csv(csv_path, index=False)
        
        # Simplified CSV (key metrics only)
        simplified_filename = args.output_filename.replace('.csv', '_simplified.csv')
        simplified_csv_path = os.path.join(args.output_dir, simplified_filename)
        df_simplified = metrics_to_simplified_dataframe(metrics, args.data_file, args.checkpoint)
        df_simplified.to_csv(simplified_csv_path, index=False)
        
        # Print the SIMPLIFIED CSV table to terminal (more useful for quick review)
        print("\n" + "=" * 80)
        print("📋 SIMPLIFIED METRICS (key values only)")
        print("=" * 80)
        print(df_simplified.to_string(index=False))
        print("=" * 80)
        
        # Also print full CSV path info
        print(f"\n✅ Full physics metrics saved to: {csv_path}")
        print(f"✅ Simplified metrics saved to: {simplified_csv_path}")
        print(f"\n🎉 Physics metrics evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

