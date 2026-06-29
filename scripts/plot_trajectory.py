#!/usr/bin/env python3
"""
Trajectory Visualization Script.

Runs inference and generates trajectory comparison plots and GIFs 
using the BubbleML temperature colormap.

Supports multiple models:
- DiffusionPDE (joint diffusion + guided sampling)
- HistoryFM (history-window conditional flow matching)
- Flow Matching AR Bootstrap (autoregressive with history encoder)
- Flow Matching (frame-to-frame OT-CFM)
- DDPM (Denoising Diffusion Probabilistic Models)

Usage:
    # Run with default checkpoints (includes HB-ARFM, HistoryFM, DiffusionPDE when paths exist)
    python scripts/plot_trajectory.py --num-frames 50 --plot-num-frames 6 --plot-stride 2
    # (columns are t=0,2,4,6,8,10; same as --plot-frame-step 2)
    
    # Custom checkpoints and solver
    python scripts/plot_trajectory.py \\
        --diffusionpde-ckpt /path/to/diffusionpde.ckpt \\
        --history-fm-ckpt /path/to/history_fm.ckpt \\
        --ar-bootstrap-ckpt /path/to/ar_bootstrap.ckpt \\
        --flow-matching-ckpt /path/to/flow_matching.ckpt \\
        --ddpm-ckpt /path/to/ddpm.ckpt \\
        --num-frames 50 --solver heun
    
    # Available solvers: euler, heun (default), midpoint, rk4

"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bubblefusion.models.diffusionpde import DiffusionPDELightning
from bubblefusion.models.flow_matching_history import ConditionalFlowMatchingHistoryLightning
from bubblefusion.models.flow_matching_ar_bootstrap import ConditionalFlowMatchingARBootstrapLightning
from bubblefusion.models.flow_matching import ConditionalFlowMatchingLightning
from bubblefusion.models.ddpm import BubbleDDPMLightning
from bubblefusion.models.unet_ar import UNetARLightning
from bubblefusion.data.bubbleml import (
    BulkFlow,
    BulkFlowARBootstrap,
    BulkFlowAutoregressive,
    BulkFlowHistory,
    compute_normalization_stats,
)

_LOG_ROOT = "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs"


# =============================================================================
# CUSTOM TEMPERATURE COLORMAP (from Bubbleformer)
# https://github.com/HPCForge/Bubbleformer/blob/main/bubbleformer/plot/plotting.py
# =============================================================================

def temp_cmap():
    """Custom temperature colormap matching BubbleML visualization style."""
    temp_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                   0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                   '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                   '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                   '#FC1407', '#FB0007']
    colors = list(zip(temp_ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
    return cmap


def extract_wall_temp_from_filepath(filepath: str) -> float:
    """
    Extract wall temperature from data file path.
    Expected format: .../Twall_XX.hdf5 or .../Twall_XX.YY.hdf5
    
    Args:
        filepath: Path to HDF5 data file
        
    Returns:
        Wall temperature in Celsius as a float
    """
    import os
    basename = os.path.basename(filepath)
    match = re.search(r'Twall[_-]?(\d+(?:\.\d+)?)', basename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        print(f"⚠️  Could not extract wall temperature from: {filepath}")
        print(f"   Using default: 96.0°C")
        return 96.0


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_task_config(task_name: str = 'velocity_from_interface') -> DictConfig:
    """Load task configuration from YAML file."""
    config_path = Path(__file__).parent.parent / 'bubblefusion' / 'config' / 'task_cfg' / f'{task_name}.yaml'
    
    if config_path.exists():
        task_cfg = OmegaConf.load(config_path)
        print(f"✓ Loaded task config: {task_name}")
        return task_cfg
    else:
        return DictConfig({
            'name': 'velocity_from_interface',
            'conditioning_channels': [0, 1, 2],
            'conditioning_names': ['sdf', 'velx_interface', 'vely_interface'],
            'target_channels': [1, 2, 0],
            'target_names': ['velx', 'vely', 'temperature']
        })


def extract_hparams_from_checkpoint(checkpoint_path: str) -> dict:
    """Extract hyperparameters from a Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint.get('hyper_parameters', {})


def load_diffusionpde_model(
    checkpoint_path: str,
    normalization_stats: dict,
    task_cfg: DictConfig,
    device: str = 'cuda',
) -> DiffusionPDELightning:
    """Load DiffusionPDE model from checkpoint."""
    print(f"\n📦 Loading DiffusionPDE model...")
    downsample_factor = normalization_stats.get('downsample_factor', 4)
    model_cfg = DictConfig({
        'name': 'diffusionpde',
        'base_resolution': 512,
        'downsample_factor': downsample_factor,
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
        'num_sampling_steps': 50,
        'solver': 'heun',
        'zeta_obs': 1.0,
        'zeta_pde': 0.5,
        'pde_start_fraction': 0.8,
        'pde_obs_decay': 0.1,
        'bulk_sdf_threshold': 0.05,
        'temp_min': 55.0,
        'temp_max': 120.0,
    })
    optim_cfg = DictConfig({'name': 'adamw', 'lr': 0.001})
    scheduler_cfg = DictConfig({'name': 'cosine'})
    model = DiffusionPDELightning.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        task_cfg=task_cfg,
        normalization_stats=normalization_stats,
        strict=False,
    )
    model = model.to(device).eval()
    print(f"✓ DiffusionPDE loaded")
    return model


def load_flow_matching_history_model(
    checkpoint_path: str,
    normalization_stats: dict,
    device: str = 'cuda',
) -> ConditionalFlowMatchingHistoryLightning:
    """Load HistoryFM (history-window flow matching) from checkpoint."""
    print(f"\n📦 Loading HistoryFM model...")
    model = ConditionalFlowMatchingHistoryLightning.load_from_checkpoint(
        checkpoint_path,
        normalization_stats=normalization_stats,
        strict=False,
    )
    model = model.to(device).eval()
    print(f"✓ HistoryFM loaded (history_window={model.history_window})")
    return model


def load_ar_bootstrap_model(
    checkpoint_path: str,
    normalization_stats: dict,
    task_cfg: DictConfig,
    device: str = 'cuda'
) -> ConditionalFlowMatchingARBootstrapLightning:
    """Load Flow Matching AR Bootstrap model from checkpoint."""
    print(f"\n📦 Loading AR Bootstrap model...")
    
    hparams = extract_hparams_from_checkpoint(checkpoint_path)
    
    if 'model_cfg' in hparams:
        model_cfg = DictConfig(hparams['model_cfg'])
    else:
        # Extract parameters from checkpoint path if possible
        base_channels = 64 if '_64_' in checkpoint_path else 32
        hist_match = re.search(r'hist(\d+)', checkpoint_path)
        history_length = int(hist_match.group(1)) if hist_match else 10
        roll_match = re.search(r'roll(\d+)', checkpoint_path)
        rollout_length = int(roll_match.group(1)) if roll_match else 5
        
        model_cfg = DictConfig({
            # Model architecture
            'base_channels': base_channels,
            'time_embed_dim': 256,
            'num_res_blocks': 2,
            'use_attention': False,
            'dropout': 0.1,
            
            # Flow matching parameters
            'num_integration_steps': 50,
            'noise_scale': 1.0,
            
            # Bootstrap configuration
            'history_length': history_length,
            'rollout_length': rollout_length,
            'use_availability_mask': True,
            
            # History encoder configuration
            'history_encoder_type': 'temporal_mixer',
            'history_encoder_hidden': 32,
            'history_encoder_blocks': 3,
            'temporal_mixer_spatial_conv': True,
            'temporal_mixer_temporal_weights': True,
            
            # Loss configuration
            'bootstrap_loss_weight': 1.0,
            'ar_loss_weight': 1.0,
            'bootstrap_state_loss_weight': 0.5,
            
            # AR temporal decay
            'ar_temporal_decay': {'enabled': False, 'gamma': 0.85},
            
            # Auxiliary losses
            'auxiliary_losses': {
                'spectral_enabled': False,
                'spectral_weight': 0.1,
                'spectral_high_freq_weight': 2.0,
                'spectral_freq_threshold': 0.3,
                'gradient_enabled': False,
                'gradient_weight': 0.1,
            },
            
            # Inference configuration
            'inference': {'solver': 'heun', 'guidance_scale': 1.0},
            
            # Temperature (deprecated but needed for compatibility)
            'temp_min': 55.0,
            'temp_max': 120.0,
            
            # Wall temperature conditioning
            'conditioning_strategy': 'none',
            
            # Scheduled sampling (disabled)
            'scheduled_sampling': {
                'enabled': True,
                'schedule_type': 'linear',
                'warmup_epochs': 5,
                'transition_epochs': 20,
                'min_teacher_ratio': 0.2,
                'exponential_decay_rate': 0.95,
                'sigmoid_k': 5.0,
                'sampling_steps': 10,
            },
            
            # Push forward
            'push_forward': {
                'enabled': False,
                'warmup_epochs': 3,
                'max_push_steps': 3,
                'step_increase_epochs': 10,
                'sampling_steps': 15,
                'loss_on_all_pushed': True,
                'detach_pushed': False,
            },
        })
    
    optim_cfg = DictConfig(hparams.get('optim_cfg', {'name': 'adamw', 'lr': 1e-4}))
    scheduler_cfg = DictConfig(hparams.get('scheduler_cfg', {'name': 'cosine'}))
    
    model = ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        task_cfg=task_cfg,
        normalization_stats=normalization_stats,
        strict=False
    )
    
    model = model.to(device).eval()
    print(f"✓ AR Bootstrap loaded (history={model.history_length})")
    return model


def load_flow_matching_model(
    checkpoint_path: str,
    normalization_stats: dict,
    task_cfg: DictConfig,
    device: str = 'cuda'
) -> ConditionalFlowMatchingLightning:
    """Load Flow Matching model from checkpoint."""
    print(f"\n📦 Loading Flow Matching model...")
    
    hparams = extract_hparams_from_checkpoint(checkpoint_path)
    
    if 'model_cfg' in hparams:
        model_cfg = DictConfig(hparams['model_cfg'])
    else:
        base_channels = 64 if '_64_' in checkpoint_path or 'ch64' in checkpoint_path else 32
        model_cfg = DictConfig({
            'base_channels': base_channels,
            'time_embed_dim': 320,
            'num_res_blocks': 2,
            'attention_type': 'none',
            'dropout': 0.0,
            'adaptive_scale': False,
            'skip_scale': False,
            'num_integration_steps': 50,
            'inference': {'solver': 'heun'},
        })
    
    optim_cfg = DictConfig(hparams.get('optim_cfg', {'name': 'adamw', 'lr': 1e-4}))
    scheduler_cfg = DictConfig(hparams.get('scheduler_cfg', {'name': 'cosine'}))
    
    model = ConditionalFlowMatchingLightning.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        task_cfg=task_cfg,
        normalization_stats=normalization_stats,
        strict=False
    )
    
    model = model.to(device).eval()
    print(f"✓ Flow Matching loaded")
    return model


def load_ddpm_model(
    checkpoint_path: str,
    normalization_stats: dict,
    task_cfg: DictConfig,
    device: str = 'cuda'
) -> BubbleDDPMLightning:
    """Load DDPM model from checkpoint."""
    print(f"\n📦 Loading DDPM model...")
    
    hparams = extract_hparams_from_checkpoint(checkpoint_path)
    
    if 'model_cfg' in hparams:
        model_cfg = DictConfig(hparams['model_cfg'])
    else:
        base_channels = 64 if '_64_' in checkpoint_path or 'ch64' in checkpoint_path else 32
        model_cfg = DictConfig({
            'base_channels': base_channels,
            'time_embed_dim': 256,
            'num_res_blocks': 2,
            'use_attention': True,
            'dropout': 0.0,
            'num_timesteps': 1000,
            'beta_start': 1e-4,
            'beta_end': 2e-2,
            'num_inference_steps': 1000,
        })
    
    optim_cfg = DictConfig(hparams.get('optim_cfg', {'name': 'adamw', 'lr': 1e-4}))
    scheduler_cfg = DictConfig(hparams.get('scheduler_cfg', {'name': 'cosine'}))
    
    model = BubbleDDPMLightning.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        task_cfg=task_cfg,
        normalization_stats=normalization_stats,
        strict=False
    )
    
    model = model.to(device).eval()
    print(f"✓ DDPM loaded (timesteps={model.ddpm.num_timesteps})")
    return model


def load_unet_ar_model(
    checkpoint_path: str,
    normalization_stats: dict,
    task_cfg: DictConfig,
    device: str = 'cuda'
) -> UNetARLightning:
    """Load UNet AR model from checkpoint."""
    print(f"\n📦 Loading UNet AR model...")
    
    hparams = extract_hparams_from_checkpoint(checkpoint_path)
    
    if 'model_cfg' in hparams:
        model_cfg = DictConfig(hparams['model_cfg'])
    else:
        # Extract init_features from checkpoint path if possible
        feat_match = re.search(r'unet_ar_(\d+)', checkpoint_path)
        init_features = int(feat_match.group(1)) if feat_match else 32
        
        model_cfg = DictConfig({
            'init_features': init_features,
            'residual_prediction': False,
            'conditioning_strategy': 'none',
            'scheduled_sampling': {'enabled': False},
            'auxiliary_losses': {
                'spectral_enabled': False,
                'gradient_enabled': False,
            },
        })
    
    optim_cfg = DictConfig(hparams.get('optim_cfg', {'name': 'adamw', 'lr': 1e-4}))
    scheduler_cfg = DictConfig(hparams.get('scheduler_cfg', {'name': 'cosine'}))
    
    model = UNetARLightning.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        task_cfg=task_cfg,
        normalization_stats=normalization_stats,
        strict=False
    )
    
    model = model.to(device).eval()
    print(f"✓ UNet AR loaded (init_features={model.unet.encoder1[0].out_channels})")
    return model


# =============================================================================
# INFERENCE
# =============================================================================

def run_diffusionpde_inference(
    model: DiffusionPDELightning,
    dataset: BulkFlow,
    start_idx: int,
    num_frames: int,
    device: str = 'cuda',
    num_sampling_steps: int = 50,
    solver: str = 'heun',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run frame-by-frame inference with DiffusionPDE (guided sampling uses autograd)."""
    print(f"\n🔄 Running DiffusionPDE inference ({num_frames} frames, solver={solver})...")
    predictions = []
    ground_truth = []
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    num_joint = model.num_joint_channels
    dev = torch.device(device)

    for i in tqdm(range(num_frames)):
        idx = min(start_idx + i, len(dataset) - 1)
        input_data, target_data = dataset[idx]
        input_data = input_data.unsqueeze(0).to(device)
        conditioning = input_data[:, conditioning_channels, :, :]
        B, _, H, W = conditioning.shape
        target = target_data[target_channels].unsqueeze(0).to(device)

        with torch.enable_grad():
            pred = model.diffusion_pde.sample_with_guidance(
                observed_gt=conditioning,
                shape=(B, num_joint, H, W),
                device=dev,
                num_steps=num_sampling_steps,
                solver=solver,
            )
        predictions.append(pred.squeeze(0).cpu())
        ground_truth.append(target.squeeze(0).cpu())

    return torch.stack(predictions), torch.stack(ground_truth)


def run_flow_matching_history_inference(
    model: ConditionalFlowMatchingHistoryLightning,
    dataset: BulkFlowHistory,
    start_idx: int,
    num_frames: int,
    device: str = 'cuda',
    num_integration_steps: int = 50,
    solver: str = 'heun',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run frame-by-frame inference with HistoryFM (sliding conditioning window)."""
    print(f"\n🔄 Running HistoryFM inference ({num_frames} frames, solver={solver})...")
    predictions = []
    ground_truth = []
    target_channels = model.target_channels
    C_out = len(target_channels)

    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            idx = min(start_idx + i, len(dataset) - 1)
            input_data, target_data = dataset[idx]
            input_data = input_data.unsqueeze(0).to(device)
            gt_slice = target_data[target_channels]
            H, W = gt_slice.shape[1], gt_slice.shape[2]
            pred = model.flow_matching.sample(
                condition=input_data,
                shape=(1, C_out, H, W),
                device=device,
                num_integration_steps=num_integration_steps,
                solver=solver,
            )
            predictions.append(pred.squeeze(0).cpu())
            ground_truth.append(gt_slice)

    return torch.stack(predictions), torch.stack(ground_truth)


def run_ar_bootstrap_inference(
    model: ConditionalFlowMatchingARBootstrapLightning,
    dataset: BulkFlowARBootstrap,
    start_idx: int,
    num_frames: int,
    device: str = 'cuda',
    num_integration_steps: int = 50,
    solver: str = 'heun',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run autoregressive inference with AR Bootstrap model."""
    print(f"\n🔄 Running AR Bootstrap inference ({num_frames} frames, solver={solver})...")
    
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    sample_data = dataset[start_idx]
    cond_hist, cond_seq, target_seq = sample_data[:3]
    
    cond_hist = cond_hist.unsqueeze(0).to(device)
    cond_seq = cond_seq.unsqueeze(0).to(device)
    target_seq = target_seq.unsqueeze(0).to(device)
    
    cond_hist_extracted = cond_hist[:, :, conditioning_channels, :, :]
    cond_seq_extracted = cond_seq[:, :, conditioning_channels, :, :]
    target_seq_extracted = target_seq[:, :, target_channels, :, :]
    
    B, T_hist, C_cond, H, W = cond_hist_extracted.shape
    _, L, _, _, _ = cond_seq_extracted.shape
    C_out = len(target_channels)
    
    current_cond_0 = cond_seq_extracted[:, 0]
    bootstrapped_state = model.bootstrap_initial_state(cond_hist_extracted, current_cond_0)
    
    predictions = []
    ground_truth = []
    prev_output = bootstrapped_state
    
    frames_to_process = min(num_frames, L)
    for l in range(frames_to_process):
        current_cond = cond_seq_extracted[:, l]
        availability_mask = torch.zeros(B, 1, H, W, device=device) if l == 0 else torch.ones(B, 1, H, W, device=device)
        
        with torch.no_grad():
            predicted = model.sample(
                condition=current_cond,
                prev_output=prev_output,
                shape=(B, C_out, H, W),
                device=device,
                availability_mask=availability_mask,
                num_integration_steps=num_integration_steps,
                solver=solver,
            )
        predictions.append(predicted.squeeze(0).cpu())
        ground_truth.append(target_seq_extracted[:, l].squeeze(0).cpu())
        prev_output = predicted
    
    if num_frames > L:
        effective_start_time = dataset.effective_start_time
        base_timestep = start_idx + effective_start_time
        
        for frame_idx in range(L, num_frames):
            current_timestep = base_timestep + frame_idx
            if current_timestep >= dataset.traj_lens[0] - 1:
                break
            
            cond_frame = dataset._get_conditioning_frame(0, current_timestep)
            target_frame = dataset._get_output_frame(0, current_timestep)
            
            cond_frame = cond_frame[conditioning_channels].unsqueeze(0).to(device)
            target_frame = target_frame[target_channels]
            
            availability_mask = torch.ones(B, 1, H, W, device=device)
            
            with torch.no_grad():
                predicted = model.sample(
                    condition=cond_frame,
                    prev_output=prev_output,
                    shape=(B, C_out, H, W),
                    device=device,
                    availability_mask=availability_mask,
                    num_integration_steps=num_integration_steps,
                    solver=solver,
                )
            predictions.append(predicted.squeeze(0).cpu())
            ground_truth.append(target_frame)
            prev_output = predicted
    
    return torch.stack(predictions), torch.stack(ground_truth)


def run_flow_matching_inference(
    model: ConditionalFlowMatchingLightning,
    dataset: BulkFlow,
    start_idx: int,
    num_frames: int,
    device: str = 'cuda',
    num_integration_steps: int = 50,
    solver: str = 'heun',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run frame-by-frame inference with Flow Matching model."""
    print(f"\n🔄 Running Flow Matching inference ({num_frames} frames, solver={solver})...")
    
    predictions = []
    ground_truth = []
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            idx = min(start_idx + i, len(dataset) - 1)
            input_data, target_data = dataset[idx]
            input_data = input_data.unsqueeze(0).to(device)
            
            conditioning = input_data[:, conditioning_channels, :, :]
            B, C_cond, H, W = conditioning.shape
            C_out = len(target_channels)
            
            pred = model.flow_matching.sample(
                condition=conditioning,
                shape=(B, C_out, H, W),
                device=device,
                num_integration_steps=num_integration_steps,
                solver=solver
            )
            predictions.append(pred.squeeze(0).cpu())
            
            gt = target_data[target_channels]
            ground_truth.append(gt)
    
    return torch.stack(predictions), torch.stack(ground_truth)


def run_ddpm_inference(
    model: BubbleDDPMLightning,
    dataset: BulkFlow,
    start_idx: int,
    num_frames: int,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run frame-by-frame inference with DDPM model."""
    print(f"\n🔄 Running DDPM inference ({num_frames} frames)...")
    
    predictions = []
    ground_truth = []
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            idx = min(start_idx + i, len(dataset) - 1)
            input_data, target_data = dataset[idx]
            input_data = input_data.unsqueeze(0).to(device)
            
            conditioning = input_data[:, conditioning_channels, :, :]
            B, C_cond, H, W = conditioning.shape
            C_out = len(target_channels)
            
            pred = model.ddpm.p_sample_loop(
                condition=conditioning,
                shape=(B, C_out, H, W),
                device=device
            )
            predictions.append(pred.squeeze(0).cpu())
            
            gt = target_data[target_channels]
            ground_truth.append(gt)
    
    return torch.stack(predictions), torch.stack(ground_truth)


def run_unet_ar_inference(
    model: UNetARLightning,
    dataset: BulkFlowAutoregressive,
    start_idx: int,
    num_frames: int,
    device: str = 'cuda',
    initial_state_mode: str = 'zeros',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run autoregressive inference with UNet AR model."""
    print(f"\n🔄 Running UNet AR inference ({num_frames} frames, init={initial_state_mode})...")
    
    predictions = []
    ground_truth = []
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    # Get first sample to initialize
    sample_data = dataset[start_idx]
    input_data, prev_output_data, target_data = sample_data[:3]
    
    # Create initial state
    input_batch = input_data.unsqueeze(0).to(device)
    conditioning = input_batch[:, conditioning_channels, :, :]
    B, C_cond, H, W = conditioning.shape
    C_out = len(target_channels)
    
    if initial_state_mode == 'zeros':
        prev_output = torch.zeros(B, C_out, H, W, device=device)
    elif initial_state_mode == 'small_noise':
        prev_output = torch.randn(B, C_out, H, W, device=device) * 0.01
    elif initial_state_mode == 'from_conditioning':
        prev_output = torch.zeros(B, C_out, H, W, device=device)
        if C_cond >= 3 and C_out >= 2:
            prev_output[:, 0, :, :] = conditioning[:, 1, :, :]  # velx
            prev_output[:, 1, :, :] = conditioning[:, 2, :, :]  # vely
    else:
        prev_output = torch.zeros(B, C_out, H, W, device=device)
    
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            idx = min(start_idx + i, len(dataset) - 1)
            sample_data = dataset[idx]
            input_data, _, target_data = sample_data[:3]
            
            input_batch = input_data.unsqueeze(0).to(device)
            conditioning = input_batch[:, conditioning_channels, :, :]
            
            # Run inference
            pred = model.sample(
                condition=conditioning,
                prev_output=prev_output,
            )
            
            predictions.append(pred.squeeze(0).cpu())
            gt = target_data[target_channels]
            ground_truth.append(gt)
            
            # Update prev_output for autoregressive rollout
            prev_output = pred
    
    return torch.stack(predictions), torch.stack(ground_truth)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_trajectory_comparison(
    trajectories: Dict[str, torch.Tensor],
    output_path: str,
    channel_idx: int = 2,
    num_frames_to_show: int = 6,
    frame_step: int = None,
    bulk_temp: float = 48.3,
    heater_temp: float = 114.7,
    vmin: float = 0.0,
    wspace: float = 0.05,
    hspace: float = 0.05,
):
    """Plot trajectory comparison with custom temperature colormap.
    
    Args:
        trajectories: Dict mapping model names to trajectory tensors.
                     First entry should be 'Ground Truth'.
        output_path: Path to save the plot.
        channel_idx: Channel to plot (2 = temperature for velocity_from_interface task).
        num_frames_to_show: Number of columns (time indices) in the grid.
        frame_step: Stride between columns (e.g. 10 → t=0,10,20,30,40,50 for 6 frames).
                    If None, indices are chosen to span the trajectory evenly.
        bulk_temp: Bulk temperature for denormalization.
        heater_temp: Heater temperature for denormalization.
        vmin: Minimum value for colormap (0-1 range). Set to 0.02-0.05.
        wspace: Width spacing between subplots (0=touching, 0.2=default).
        hspace: Height spacing between subplots (0=touching, 0.2=loose).
    """
    # Get all trajectory lengths and find minimum
    T = min(traj.shape[0] for traj in trajectories.values())
    num_models = len(trajectories)
    
    if frame_step is not None:
        if frame_step < 1:
            raise ValueError(f'frame_step must be >= 1, got {frame_step}')
        frame_indices = [i * frame_step for i in range(num_frames_to_show)]
        max_idx = frame_indices[-1]
        if max_idx >= T:
            raise ValueError(
                f'Need trajectory length >= {max_idx + 1} for '
                f'{num_frames_to_show} frames at stride {frame_step} '
                f'(indices {frame_indices}), but shortest trajectory has T={T}. '
                f'Increase --num-frames to at least {(num_frames_to_show - 1) * frame_step + 1}.'
            )
    else:
        step = max(1, T // num_frames_to_show)
        frame_indices = list(range(0, T, step))[:num_frames_to_show]
    
    print(f"   Plotting {len(frame_indices)} frames: {frame_indices}")
    
    # Increased figure size for paper quality
    fig, axes = plt.subplots(num_models, len(frame_indices), 
                              figsize=(3.0 * len(frame_indices), 3.0 * num_models))
    
    if len(frame_indices) == 1:
        axes = axes.reshape(num_models, 1)
    if num_models == 1:
        axes = axes.reshape(1, -1)
    
    cmap = temp_cmap()
    
    def prep_frame(traj, t):
        data = traj[t, channel_idx].numpy()
        data_denorm = (data + 1) / 2 * (heater_temp - bulk_temp) + bulk_temp
        data_for_cmap = (data_denorm - bulk_temp) / (heater_temp - bulk_temp)
        return np.clip(data_for_cmap, 0, 1)
    
    im = None
    for row_idx, (model_name, traj) in enumerate(trajectories.items()):
        for i, t in enumerate(frame_indices):
            im = axes[row_idx, i].imshow(prep_frame(traj, t), cmap=cmap, vmin=vmin, vmax=1, origin='lower')
            if row_idx == 0:
                # Larger font for paper - frame titles
                axes[row_idx, i].set_title(f't={t}', fontsize=16, fontweight='bold')
            axes[row_idx, i].axis('off')
        
        # Add vertical model name label on the left side
        # Rotation=90 makes text read from bottom to top
        # Place it at the center of the first subplot row
        ax = axes[row_idx, 0]
        ax.text(-0.15, 0.5, model_name, 
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                rotation=90,  # Rotate 90 degrees (bottom to top)
                ha='center', va='center')
    
    # Adjust spacing - leave more room on the left for vertical labels
    fig.subplots_adjust(left=0.08, right=0.90, wspace=wspace, hspace=hspace)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Temperature (°C)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Adjust tick positions to account for vmin
    tick_positions = [vmin, 0.25, 0.5, 0.75, 1.0]
    tick_positions = [t for t in tick_positions if t >= vmin]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f'{bulk_temp + t*(heater_temp-bulk_temp):.0f}' for t in tick_positions])
    
    plt.suptitle('Temperature Field Trajectory Comparison', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_gif(
    trajectory: torch.Tensor, 
    output_path: str, 
    channel_idx: int = 2, 
    title: str = '',
    bulk_temp: float = 48.3, 
    heater_temp: float = 114.7,
    fps: int = 10,
    vmin: float = 0.0,
):
    """Create GIF from trajectory using BubbleML temperature colormap."""
    T, C, H, W = trajectory.shape
    data = trajectory[:, channel_idx].numpy()
    
    data_denorm = (data + 1) / 2 * (heater_temp - bulk_temp) + bulk_temp
    data_for_cmap = (data_denorm - bulk_temp) / (heater_temp - bulk_temp)
    data_for_cmap = np.clip(data_for_cmap, 0, 1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = temp_cmap()
    im = ax.imshow(data_for_cmap[0], cmap=cmap, vmin=vmin, vmax=1, origin='lower')
    ax.set_title(f'{title} - Frame 0')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)')
    tick_positions = [vmin, 0.25, 0.5, 0.75, 1.0]
    tick_positions = [t for t in tick_positions if t >= vmin]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f'{bulk_temp + t*(heater_temp-bulk_temp):.0f}' for t in tick_positions])
    
    def update(frame):
        im.set_data(data_for_cmap[frame])
        ax.set_title(f'{title} - Frame {frame}')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000//fps, blit=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"✓ Saved GIF: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Trajectory Visualization')
    
    # Model checkpoints (None = skip model, provide path to include)
    parser.add_argument(
        '--diffusionpde-ckpt',
        type=str,
        default=f"{_LOG_ROOT}/diffusionpde_ch32_b2_s50_zobs1.0_zpde0.5_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50073800/checkpoints/last.ckpt",
        help='Path to DiffusionPDE checkpoint (None to skip)',
    )
    parser.add_argument(
        '--history-fm-ckpt',
        type=str,
        # default=f"{_LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50271127/checkpoints/last.ckpt",
        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52452934/checkpoints/epoch=09-step=016640.ckpt",
        help='Path to HistoryFM (flow_matching_history) checkpoint (None to skip)',
    )
    parser.add_argument('--ar-bootstrap-ckpt', type=str, 
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861279/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47955165/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861270/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_25/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist5_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861275/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861276/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist40_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861282/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861277/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861278/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861279/checkpoints/epoch=41-step=034860.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/checkpoints/epoch=07-step=013280.ckpt",
                        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist64_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52453282/checkpoints/epoch=09-step=016600.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/checkpoints/epoch=07-step=013280.ckpt",
                        help='Path to AR Bootstrap checkpoint (None to skip)')
    parser.add_argument('--flow-matching-ckpt', type=str, 
                        default=None,
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47835444/checkpoints/last.ckpt",
                        help='Path to Flow Matching checkpoint (None to skip)')
    parser.add_argument('--ddpm-ckpt', type=str, default=None,
                        help='Path to DDPM checkpoint (None to skip)')
    parser.add_argument('--unet-ar-ckpt', type=str,
                        # default=None,
                        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/unet_ar_32_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856472/checkpoints/last.ckpt",
                        help='Path to UNet AR checkpoint (None to skip)')
    
    # Data settings
    parser.add_argument('--data-path', type=str,
                        default="/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5")
    parser.add_argument('--norm-stats', type=str,
                        default="/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json")
    parser.add_argument('--task', type=str, default='velocity_from_interface')
    parser.add_argument('--num-frames', type=int, default=40)
    parser.add_argument('--start-idx', type=int, default=300)
    parser.add_argument('--output-dir', type=str, default='./ICML/CamReady/Figure5_Trajectory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-sampling-steps', type=int, default=50)
    parser.add_argument('--solver', type=str, default='rk4',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver for flow matching models (euler, heun, midpoint, rk4)')
    parser.add_argument('--downsample-factor', type=int, default=4)
    
    # Trajectory plot settings
    parser.add_argument('--plot-num-frames', type=int, default=6,
                        help='Number of columns (time snapshots) in trajectory_comparison_frame<start-idx>_stride<plot-frame-step>.png')
    parser.add_argument(
        '--plot-frame-step', '--plot-stride',
        type=int,
        default=2,
        dest='plot_frame_step',
        metavar='S',
        help='Stride between PNG columns (e.g. 10 → t=0,10,20,30,40,50 with --plot-num-frames 6). '
             'Requires --num-frames >= (plot-num-frames - 1) * stride + 1. Does not affect GIFs.',
    )
    parser.add_argument('--channel', type=int, default=2, choices=[0, 1, 2],
                        help='Channel to plot: 0=velx, 1=vely, 2=temperature (default)')
    
    # GIF settings
    parser.add_argument('--no-gif', action='store_true',
                        help='Skip GIF generation')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for GIF')
    
    # Colormap settings
    parser.add_argument('--vmin', type=float, default=0.02,
                        help='Min value for colormap (0-1).')
    
    # Spacing settings
    parser.add_argument('--wspace', type=float, default=0.05,
                        help='Width spacing between subplots (0=touching, 0.2=loose)')
    parser.add_argument('--hspace', type=float, default=0.05,
                        help='Height spacing between subplots (0=touching, 0.2=loose)')
    
    # History length for AR Bootstrap model
    parser.add_argument('--history-length', type=int, default=10,
                        help='History length for AR Bootstrap model')
    parser.add_argument(
        '--history-fm-window',
        type=int,
        default=None,
        help='History window W for HistoryFM dataset (default: read from checkpoint)',
    )
    
    # Initial state for UNet AR: zeros, small_noise, or from_conditioning
    parser.add_argument('--initial-state-mode', type=str, default='from_conditioning',
                        choices=['zeros', 'small_noise', 'from_conditioning'],
                        help='Initial state for AR models: from_conditioning uses interface velocity')
    
    # Random seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    if args.plot_frame_step < 1:
        parser.error('--plot-frame-step / --plot-stride must be >= 1')
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TRAJECTORY VISUALIZATION")
    print("=" * 60)
    
    # Extract wall/heater temperature from data filename
    heater_temp = extract_wall_temp_from_filepath(args.data_path)
    bulk_temp = 48.3  # FC72 bulk temperature for subcooled pool boiling
    print(f"\n🌡️  Temperature range: {bulk_temp}°C (bulk) to {heater_temp}°C (heater)")
    
    # Load configs
    task_cfg = load_task_config(args.task)
    
    if args.norm_stats and Path(args.norm_stats).exists():
        with open(args.norm_stats, 'r') as f:
            normalization_stats = json.load(f)
    else:
        normalization_stats = compute_normalization_stats([args.data_path], start_time=100)
    normalization_stats['downsample_factor'] = args.downsample_factor
    
    # Create datasets
    print(f"\n📂 Creating datasets from {args.data_path}")
    dataset_ff = BulkFlow(
        filenames=[args.data_path],
        output_fields=['temperature', 'velx', 'vely'],
        start_time=100,
        normalization_stats=normalization_stats,
        downsample_factor=args.downsample_factor,
    )
    
    dataset_ar_bootstrap = BulkFlowARBootstrap(
        filenames=[args.data_path],
        output_fields=['temperature', 'velx', 'vely'],
        start_time=100,
        history_length=args.history_length,
        rollout_length=5,
        normalization_stats=normalization_stats,
        downsample_factor=args.downsample_factor,
    )
    
    # Create autoregressive dataset for UNet AR
    dataset_ar = BulkFlowAutoregressive(
        filenames=[args.data_path],
        output_fields=['temperature', 'velx', 'vely'],
        start_time=100,
        normalization_stats=normalization_stats,
        downsample_factor=args.downsample_factor,
    )

    history_fm_window = args.history_fm_window
    if history_fm_window is None and args.history_fm_ckpt and Path(args.history_fm_ckpt).exists():
        hp = extract_hparams_from_checkpoint(args.history_fm_ckpt)
        mc = hp.get('model_cfg')
        if mc is not None:
            if isinstance(mc, dict):
                history_fm_window = int(mc.get('history_window', 10))
            else:
                history_fm_window = int(DictConfig(mc).get('history_window', 10))
    if history_fm_window is None:
        history_fm_window = 10

    dataset_history = BulkFlowHistory(
        filenames=[args.data_path],
        output_fields=['temperature', 'velx', 'vely'],
        start_time=100,
        history_window=history_fm_window,
        normalization_stats=normalization_stats,
        downsample_factor=args.downsample_factor,
    )
    
    # Dictionary to store all trajectories
    trajectories = {}
    gt_traj = None
    
    # Define which models to process
    # Format: (ckpt_arg, name, model_type, dataset_type)
    # dataset_type: 'ff' = frame-to-frame, 'ar_bootstrap' = AR Bootstrap, 'ar' = simple AR
    # NOTE: AR Bootstrap is first so it appears right after Ground Truth in the plot
    model_configs = [
        ('ar_bootstrap_ckpt', 'HB-ARFM (Ours)', 'ar_bootstrap', 'ar_bootstrap'),
        ('history_fm_ckpt', 'HistoryFM', 'flow_matching_history', 'history'),
        ('diffusionpde_ckpt', 'DiffusionPDE', 'diffusionpde', 'ff'),
        ('flow_matching_ckpt', 'Flow Matching', 'flow_matching', 'ff'),
        ('ddpm_ckpt', 'DDPM', 'ddpm', 'ff'),
        ('unet_ar_ckpt', 'UNet', 'unet_ar', 'ar'),
    ]
    
    # Process each model type - only if checkpoint is provided and exists
    for ckpt_arg, model_name, model_type, dataset_type in model_configs:
        ckpt_path = getattr(args, ckpt_arg.replace('-', '_'), None)
        
        # Skip if checkpoint is None or doesn't exist
        if ckpt_path is None:
            print(f"\n⏭️  Skipping {model_name} (no checkpoint provided)")
            continue
        if not Path(ckpt_path).exists():
            print(f"\n⚠️  Skipping {model_name} (checkpoint not found: {ckpt_path})")
            continue
            
        # Run inference
        print(f"\n⚡ Running {model_name} inference...")
        
        if model_type == 'diffusionpde':
            model = load_diffusionpde_model(ckpt_path, normalization_stats, task_cfg, args.device)
            dpde_solver = args.solver if args.solver in ('euler', 'heun') else 'heun'
            traj, gt = run_diffusionpde_inference(
                model, dataset_ff, args.start_idx, args.num_frames,
                args.device, args.num_sampling_steps, dpde_solver,
            )
            if gt_traj is None:
                gt_traj = gt
                trajectories['Ground Truth'] = gt_traj

        elif model_type == 'flow_matching_history':
            model = load_flow_matching_history_model(ckpt_path, normalization_stats, args.device)
            traj, gt = run_flow_matching_history_inference(
                model, dataset_history, args.start_idx, args.num_frames,
                args.device, args.num_sampling_steps, args.solver,
            )
            if gt_traj is None:
                gt_traj = gt
                trajectories['Ground Truth'] = gt_traj
                
        elif model_type == 'ar_bootstrap':
            model = load_ar_bootstrap_model(ckpt_path, normalization_stats, task_cfg, args.device)
            traj, gt = run_ar_bootstrap_inference(
                model, dataset_ar_bootstrap, args.start_idx, args.num_frames,
                args.device, args.num_sampling_steps, args.solver
            )
            if gt_traj is None:
                gt_traj = gt
                trajectories['Ground Truth'] = gt_traj
                
        elif model_type == 'flow_matching':
            model = load_flow_matching_model(ckpt_path, normalization_stats, task_cfg, args.device)
            traj, gt = run_flow_matching_inference(
                model, dataset_ff, args.start_idx, args.num_frames,
                args.device, args.num_sampling_steps, args.solver
            )
            if gt_traj is None:
                gt_traj = gt
                trajectories['Ground Truth'] = gt_traj
                
        elif model_type == 'ddpm':
            model = load_ddpm_model(ckpt_path, normalization_stats, task_cfg, args.device)
            traj, gt = run_ddpm_inference(
                model, dataset_ff, args.start_idx, args.num_frames,
                args.device
            )
            if gt_traj is None:
                gt_traj = gt
                trajectories['Ground Truth'] = gt_traj
                
        elif model_type == 'unet_ar':
            model = load_unet_ar_model(ckpt_path, normalization_stats, task_cfg, args.device)
            traj, gt = run_unet_ar_inference(
                model, dataset_ar, args.start_idx, args.num_frames,
                args.device, initial_state_mode=args.initial_state_mode
            )
            if gt_traj is None:
                gt_traj = gt
                trajectories['Ground Truth'] = gt_traj
        
        trajectories[model_name] = traj
    
    # Ensure Ground Truth is first in the dictionary
    if 'Ground Truth' in trajectories:
        ordered_trajectories = {'Ground Truth': trajectories.pop('Ground Truth')}
        ordered_trajectories.update(trajectories)
        trajectories = ordered_trajectories
    
    # Check we have trajectories to plot
    if len(trajectories) < 2:
        print("\n⚠️  Need at least 2 trajectories to create comparison plot.")
        print("   Provide at least one valid model checkpoint.")
        return
    
    # Create trajectory comparison plot
    print("\n📊 Creating trajectory comparison plot...")
    plot_trajectory_comparison(
        trajectories,
        str(output_dir / f'trajectory_comparison_frame{args.start_idx}_stride{args.plot_frame_step}.png'),
        channel_idx=args.channel,
        num_frames_to_show=args.plot_num_frames,
        frame_step=args.plot_frame_step,
        bulk_temp=bulk_temp,
        heater_temp=heater_temp,
        vmin=args.vmin,
        wspace=args.wspace,
        hspace=args.hspace,
    )
    
    # Create GIFs
    if not args.no_gif:
        print("\n🎬 Creating GIFs...")
        for model_name, traj in trajectories.items():
            # Create safe filename
            safe_name = model_name.lower().replace(' ', '_')
            gif_path = output_dir / f'{safe_name}_temperature.gif'
            create_gif(traj, str(gif_path), 2, model_name, 
                      bulk_temp=bulk_temp, heater_temp=heater_temp,
                      fps=args.fps, vmin=args.vmin)
    
    print(f"\n✓ All outputs saved to {output_dir}")
    print(f"   Models compared: {list(trajectories.keys())}")


if __name__ == '__main__':
    main()
