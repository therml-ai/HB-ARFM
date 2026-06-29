#!/usr/bin/env python3
"""
Comprehensive inference script for Task 1, Task 2 & Task 3.

Task 1: temperature_from_sdf - Predict temperature from SDF only
Task 2: velocity_from_interface - Predict velocity + temperature from SDF + interface velocity  
Task 3: noisy_velocity_from_interface - Same as Task 2 but with noisy inputs

Supports:
- Flow Matching models (frame-to-frame and autoregressive)
- VE-SDE Score-Based models
- Both clean and noisy input inference for Task 3

Usage:
    # Task 1 (temperature from SDF):
    python scripts/comprehensive_inference_task123.py --task temperature_from_sdf ...
    
    # Task 2 (clean inputs):
    python scripts/comprehensive_inference_task123.py --task velocity_from_interface ...
    
    # Task 3 (noisy inputs):
    python scripts/comprehensive_inference_task123.py --task noisy_velocity_from_interface ...
    
    # Task 3 with clean inputs (physics fidelity check):
    python scripts/comprehensive_inference_task123.py --task noisy_velocity_from_interface --use-clean-inputs ...
    
    # Auto-find checkpoint by task name:
    python scripts/comprehensive_inference_task123.py --find-checkpoint ./logs --task temperature_from_sdf
    
Features:
- Single-channel (Task 1) and multi-channel (Task 2/3) prediction
- Support for Task 1, Task 2 and Task 3
- Clean/noisy input inference for Task 3 (matches dual validation setup)
- Command line arguments for easy use
- Automatic checkpoint discovery by task name
- Auto-detection of task from checkpoint path
- Default paths for convenience  
- Comprehensive metrics analysis
- Batch inference capabilities
- Adaptive visualization (SDF only or full velocity field)
- Error handling and validation
- Task-aware model loading with task_cfg
"""

import sys
import os
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, PowerNorm
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import argparse
from PIL import Image
from pathlib import Path

# Ensure the project root (which contains the `bubblefusion` package) is on the
# import path so this script runs regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bubblefusion.models.flow_matching import ConditionalFlowMatchingLightning
from bubblefusion.models.flow_matching_jit import ConditionalFlowMatchingJiTLightning
from bubblefusion.models.flow_matching_ar import ConditionalFlowMatchingARLightning
from bubblefusion.models.flow_matching_ar_bootstrap import ConditionalFlowMatchingARBootstrapLightning
from bubblefusion.models.unet import UNetLightning
from bubblefusion.models.unet_ar import UNetARLightning
from bubblefusion.models.ve_sde import ScoreBasedVESDELightning
from bubblefusion.models.ddpm import BubbleDDPMLightning
from bubblefusion.models.ffno import FFNOLightning
from bubblefusion.models.edm import EDMLightning
from bubblefusion.models.edm_ar_bootstrap import EDMARBootstrapLightning
from bubblefusion.models.diffusionpde import DiffusionPDELightning
from bubblefusion.models.flow_matching_history import ConditionalFlowMatchingHistoryLightning
from bubblefusion.data import BulkFlow, BulkFlowAutoregressive
from bubblefusion.data.bubbleml import compute_normalization_stats, BulkFlowARBootstrap, BulkFlowHistory
import re


# =============================================================================
# CUSTOM TEMPERATURE COLORMAP (from Bubbleformer)
# https://github.com/HPCForge/Bubbleformer/blob/main/bubbleformer/plot/plotting.py
# =============================================================================

def temp_cmap():
    """Custom temperature colormap matching BubbleML visualization style.
    
    This colormap expects data in [0, 1] range where:
    - 0 = bulk temperature (e.g., 48.3°C for subcooled pool boiling)
    - 1 = heater temperature (e.g., 114.7°C)
    
    When using with physical temperature values in °C, first convert:
        data_for_cmap = (temp_celsius - bulk_temp) / (heater_temp - bulk_temp)
        data_for_cmap = np.clip(data_for_cmap, 0, 1)
    """
    temp_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                   0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                   '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                   '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                   '#FC1407', '#FB0007']
    colors = list(zip(temp_ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('temperature_colormap', colors)
    return cmap


def sdf_cmap():
    """Custom SDF colormap matching BubbleML visualization style.
    
    Diverging colormap centered at 0:
    - Blue: negative values (inside bubble)
    - White: zero (interface)
    - Red: positive values (liquid phase)
    
    Reference: https://github.com/HPCForge/Bubbleformer/blob/main/bubbleformer/plot/plotting.py
    """
    ranges = [0.0, 0.49, 0.51, 1.0]
    color_codes = ["blue", "white", "white", "red"]
    colors = list(zip(ranges, color_codes))
    cmap = LinearSegmentedColormap.from_list('sdf_colormap', colors)
    return cmap


def smooth_data(x, y, window_size=None):
    """
    Smooth data using Savitzky-Golay filter or moving average.
    Compatible with scripts/plot_rel_l2_metrics.py
    """
    if len(y) < 3:
        return y
    
    # Auto-determine window size (should be odd and less than data length)
    if window_size is None:
        window_size = min(51, len(y) // 4 * 2 + 1)  # Approximately 1/4 of data, made odd
        if window_size < 5:
            window_size = 5  # minimum for polyorder=3
    
    # Ensure window_size is odd and valid
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, len(y))
    
    # For savgol_filter, polyorder must be < window_length
    # We use polyorder=3, so need window_size >= 4, but odd so >= 5
    polyorder = min(3, window_size - 1)
    if polyorder < 1:
        return y
    
    # Try to use Savitzky-Golay filter for better smoothing
    try:
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(y, window_size, polyorder)
        return smoothed
    except (ImportError, ValueError):
        # Fall back to moving average if scipy is not available or params invalid
        pass
    
    # Simple moving average fallback
    smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='same')
    return smoothed


def extract_wall_temp_from_filepath(filepath: str) -> float:
    """
    Extract wall temperature from data file path.
    Expected format: .../Twall_XX.hdf5 or .../Twall_XX.YY.hdf5
    
    Args:
        filepath: Path to HDF5 data file
        
    Returns:
        Wall temperature in Celsius as a float
    """
    basename = os.path.basename(filepath)
    match = re.search(r'Twall[_-]?(\d+(?:\.\d+)?)', basename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        print(f"⚠️  Could not extract wall temperature from: {filepath}")
        print(f"   Using default: 96.0°C")
        return 96.0


def compute_colorbar_ranges(trajectory_results, target_names, conditioning_names, 
                            wall_temp=96.0, temp_min=None):
    """
    Pre-compute fixed colorbar ranges across all frames for consistent visualization.
    
    Args:
        trajectory_results: List of (input_data, target, predicted) tuples
        target_names: List of target field names (e.g., ['velx', 'vely', 'temperature'])
        conditioning_names: List of conditioning field names (e.g., ['sdf', 'velx_interface', 'vely_interface'])
        wall_temp: Wall temperature in Celsius (from filename)
        temp_min: Minimum temperature for colorbar. If None, auto-detect from data.
        
    Returns:
        Dictionary with colorbar ranges for each field type
    """
    # Initialize ranges - temperature will be computed from data if temp_min is None
    ranges = {
        'temperature': [float('inf'), float('-inf')],  # Will be computed from data
        'velx': [float('inf'), float('-inf')],
        'vely': [float('inf'), float('-inf')],
        'velx_interface': [float('inf'), float('-inf')],
        'vely_interface': [float('inf'), float('-inf')],
        'sdf': [float('inf'), float('-inf')],
        'velocity_magnitude': [0, float('-inf')],  # For target/predicted velocity magnitude
        'velocity_magnitude_interface': [0, float('-inf')],  # For interface velocity magnitude
    }
    
    # Iterate through all frames to find global min/max
    for input_data, target, predicted in trajectory_results:
        # Process conditioning inputs
        for i, name in enumerate(conditioning_names):
            if name in ranges:
                data = input_data[i].numpy()
                ranges[name][0] = min(ranges[name][0], float(data.min()))
                ranges[name][1] = max(ranges[name][1], float(data.max()))
        
        # Compute interface velocity magnitude range
        velx_interface_idx = None
        vely_interface_idx = None
        for i, name in enumerate(conditioning_names):
            if name == 'velx_interface':
                velx_interface_idx = i
            elif name == 'vely_interface':
                vely_interface_idx = i
        
        if velx_interface_idx is not None and vely_interface_idx is not None:
            velx_interface = input_data[velx_interface_idx].numpy()
            vely_interface = input_data[vely_interface_idx].numpy()
            interface_mag = np.sqrt(velx_interface**2 + vely_interface**2)
            ranges['velocity_magnitude_interface'][1] = max(
                ranges['velocity_magnitude_interface'][1],
                float(interface_mag.max())
            )
        
        # Process target fields only (ground truth)
        # Using only ground truth ensures colorbar stays physically meaningful
        # and predictions that are way off will visually saturate
        for i, name in enumerate(target_names):
            if name in ranges:
                target_data = target[i].numpy()
                ranges[name][0] = min(ranges[name][0], float(target_data.min()))
                ranges[name][1] = max(ranges[name][1], float(target_data.max()))
        
        # Compute velocity magnitude range
        velx_idx = None
        vely_idx = None
        for i, name in enumerate(target_names):
            if name == 'velx':
                velx_idx = i
            elif name == 'vely':
                vely_idx = i
        
        if velx_idx is not None and vely_idx is not None:
            target_velx = target[velx_idx].numpy()
            target_vely = target[vely_idx].numpy()
            
            target_mag = np.sqrt(target_velx**2 + target_vely**2)
            
            # Use only ground truth for velocity magnitude range
            ranges['velocity_magnitude'][1] = max(
                ranges['velocity_magnitude'][1], 
                float(target_mag.max())
            )
    
    # For temperature: use wall_temp as max, and either provided temp_min or detected min
    if 'temperature' in ranges:
        detected_min = ranges['temperature'][0]
        detected_max = ranges['temperature'][1]
        
        # Use provided temp_min if given, otherwise use detected min (with small margin)
        if temp_min is not None:
            final_temp_min = temp_min
        else:
            # Round down to nearest integer for cleaner colorbar
            final_temp_min = float(np.floor(detected_min))
        
        # Use wall_temp as max (physics-based) but ensure it covers the data
        final_temp_max = max(wall_temp, detected_max)
        
        ranges['temperature'] = [final_temp_min, final_temp_max]
        
        print(f"   🌡️  Temperature range: detected [{detected_min:.2f}, {detected_max:.2f}]°C")
        print(f"   🌡️  Temperature colorbar: [{final_temp_min:.1f}, {final_temp_max:.1f}]°C")
    
    # Convert lists to tuples and handle edge cases
    result = {}
    for key, value in ranges.items():
        if isinstance(value, list):
            if value[0] == float('inf') or value[1] == float('-inf'):
                result[key] = None  # No valid range found
            else:
                # Add small padding for better visualization
                if key in ['velx', 'vely', 'velx_interface', 'vely_interface']:
                    # Make symmetric around zero for divergent colormaps
                    max_abs = max(abs(value[0]), abs(value[1]))
                    result[key] = (-max_abs, max_abs)
                else:
                    result[key] = tuple(value)
        else:
            result[key] = value
    
    print(f"\n📊 Computed Fixed Colorbar Ranges:")
    for key, value in result.items():
        if value is not None:
            print(f"   {key}: [{value[0]:.3f}, {value[1]:.3f}]")
    
    return result


def find_checkpoint_by_task(log_dir: str, task_name: str = 'velocity_from_interface'):
    """
    Find the latest checkpoint for a given task in the log directory.
    
    Searches for directories containing the task name and returns the latest checkpoint.
    
    Args:
        log_dir: Base log directory to search
        task_name: Task name to search for (e.g., 'velocity_from_interface')
        
    Returns:
        Path to the latest checkpoint, or None if not found
    """
    print(f"🔍 Searching for checkpoints in: {log_dir}")
    print(f"   Task pattern: *{task_name}*")
    
    # Search patterns
    patterns = [
        os.path.join(log_dir, f"*{task_name}*", "checkpoints", "*.ckpt"),
        os.path.join(log_dir, f"*{task_name}*", "lightning_logs", "**", "checkpoints", "*.ckpt"),
        os.path.join(log_dir, f"*{task_name}*", "*.ckpt"),
        os.path.join(log_dir, "**", f"*{task_name}*", "checkpoints", "*.ckpt"),
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        all_checkpoints.extend(found)
    
    if not all_checkpoints:
        print(f"   ❌ No checkpoints found for task: {task_name}")
        return None
    
    # Sort by modification time (newest first)
    all_checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    # Prefer 'last.ckpt' if available
    last_ckpts = [c for c in all_checkpoints if 'last.ckpt' in c]
    if last_ckpts:
        checkpoint = last_ckpts[0]
    else:
        checkpoint = all_checkpoints[0]
    
    print(f"   ✓ Found checkpoint: {checkpoint}")
    print(f"   📅 Modified: {Path(checkpoint).stat().st_mtime}")
    
    return checkpoint


def find_all_task_checkpoints(log_dir: str, task_name: str = 'velocity_from_interface'):
    """
    Find all training runs for a given task.
    
    Returns:
        List of (run_dir, checkpoint_path, modified_time) tuples
    """
    print(f"🔍 Listing all runs for task: {task_name}")
    
    # Find all directories matching the task name
    pattern = os.path.join(log_dir, f"*{task_name}*")
    run_dirs = glob.glob(pattern)
    
    results = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
            
        # Find checkpoints in this run
        ckpt_patterns = [
            os.path.join(run_dir, "checkpoints", "last.ckpt"),
            os.path.join(run_dir, "checkpoints", "*.ckpt"),
            os.path.join(run_dir, "*.ckpt"),
        ]
        
        for pattern in ckpt_patterns:
            ckpts = glob.glob(pattern)
            if ckpts:
                # Sort by mtime and take the latest
                ckpts.sort(key=os.path.getmtime, reverse=True)
                ckpt = ckpts[0]
                mtime = os.path.getmtime(ckpt)
                results.append((run_dir, ckpt, mtime))
                break
    
    # Sort by modification time
    results.sort(key=lambda x: x[2], reverse=True)
    
    if results:
        print(f"   Found {len(results)} training runs:")
        for i, (run_dir, ckpt, mtime) in enumerate(results[:5]):  # Show top 5
            print(f"   {i+1}. {os.path.basename(run_dir)}")
            print(f"      Checkpoint: {os.path.basename(ckpt)}")
    else:
        print(f"   ❌ No runs found for task: {task_name}")
    
    return results


def load_task_config(task_name: str = 'velocity_from_interface'):
    """Load task configuration from YAML file."""
    config_path = os.path.join(
        PROJECT_ROOT,
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
                               task_cfg: DictConfig = None, model_type: str = 'flow_matching',
                               normalization_stats: dict = None, norm_mode: str = 'all'):
    """Load the trained model from checkpoint with task config.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_cfg: Model configuration
        optim_cfg: Optimizer configuration
        scheduler_cfg: Scheduler configuration
        task_cfg: Task configuration
        model_type: Model type ('flow_matching', 'flow_matching_ar', 'flow_matching_ar_bootstrap',
                                'edm_ar_bootstrap', 'unet_ar', 've_sde', 'flow_matching_jit', or 'bubble_ddpm')
        normalization_stats: Normalization statistics for accurate denormalization during inference
        norm_mode: Normalization mode ('none', 'all', 'temperature_only')
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if model_type == 've_sde':
        print(f"📦 Loading VE-SDE Score-Based model...")
        model = ScoreBasedVESDELightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats
        )
        print(f"✓ VE-SDE model loaded successfully!")
        print(f"   σ_min: {model.sigma_min}, σ_max: {model.sigma_max}")
        print(f"   Sampling method: {model.sampling_method}")
    elif model_type == 'bubble_ddpm':
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
    elif model_type == 'flow_matching_ar':
        print(f"📦 Loading Autoregressive Flow Matching model...")
        model = ConditionalFlowMatchingARLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False  # Allow missing/extra keys (e.g., gradient_loss buffers from training)
        )
        print(f"✓ Autoregressive Flow Matching model loaded successfully!")
        print(f"   Note: Conditions on previous timestep output")
        print(f"   Residual prediction: {model.residual_prediction}")
        print(f"   Default solver: {model.default_solver}")
        print(f"   Default guidance: {model.default_guidance_scale}")
        print(f"   Single sample: uses ground truth previous output")
        print(f"   Trajectory: uses autoregressive rollout (own predictions)")
    elif model_type == 'flow_matching_ar_bootstrap':
        print(f"📦 Loading Autoregressive Flow Matching with Bootstrap model...")
        # Let Lightning use the checkpoint's saved hyperparameters (model_cfg,
        # optim_cfg, scheduler_cfg, task_cfg) so the architecture exactly matches
        # what was trained. Only normalization_stats needs to be passed since it
        # was excluded from save_hyperparameters().
        model = ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
            checkpoint_path,
            normalization_stats=normalization_stats,
            strict=True,
        )
        print(f"✓ Autoregressive Flow Matching with Bootstrap model loaded successfully!")
        print(f"   Bootstrap: Uses history encoder to infer initial state")
        print(f"   History length: {model.history_length} frames (from checkpoint)")
        print(f"   Rollout length: {model.rollout_length} frames (from checkpoint)")
        print(f"   Use availability mask: {model.use_availability_mask} (from checkpoint)")
        print(f"   Default solver: {model.default_solver}")
        print(f"   Single sample: uses bootstrapped or provided initial state")
        print(f"   Trajectory: bootstrap + autoregressive rollout")
    elif model_type == 'edm_ar_bootstrap':
        print(f"Loading Autoregressive EDM with Bootstrap model...")
        model = EDMARBootstrapLightning.load_from_checkpoint(
            checkpoint_path,
            normalization_stats=normalization_stats,
            strict=True,
        )
        print(f"Autoregressive EDM with Bootstrap model loaded successfully!")
        print(f"   Bootstrap: Uses history encoder to infer initial state")
        print(f"   History length: {model.history_length} frames (from checkpoint)")
        print(f"   Rollout length: {model.rollout_length} frames (from checkpoint)")
        print(f"   Use availability mask: {model.use_availability_mask} (from checkpoint)")
        print(f"   Default solver: {model.default_solver}")
        print(f"   Sampling steps: {model.num_sampling_steps}")
        print(f"   Single sample: uses bootstrapped or provided initial state")
        print(f"   Trajectory: bootstrap + autoregressive rollout (EDM diffusion)")
    elif model_type == 'unet_ar':
        print(f"📦 Loading Autoregressive UNet model (direct regression)...")
        model = UNetARLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False  # Allow missing/extra keys
        )
        print(f"✓ Autoregressive UNet model loaded successfully!")
        print(f"   Note: Direct regression (no diffusion/flow matching)")
        print(f"   Residual prediction: {model.residual_prediction}")
        print(f"   Single forward pass (fast inference)")
        print(f"   Single sample: uses ground truth previous output")
        print(f"   Trajectory: uses autoregressive rollout (own predictions)")
    elif model_type == 'unet':
        print(f"📦 Loading UNet model (direct regression)...")
        model = UNetLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            task_cfg=task_cfg,
            normalization_stats=normalization_stats,
            strict=False  # Allow missing/extra keys
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
            strict=False  # Allow missing/extra keys
        )
        print(f"✓ FFNO model loaded successfully!")
        print(f"   Note: Spectral method (Fourier Neural Operator)")
        print(f"   Modes: {model_cfg.get('modes', 12)}, Width: {model_cfg.get('width', 64)}")
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
            strict=False  # Allow missing/extra keys
        )
        print(f"✓ EDM model loaded successfully!")
        print(f"   Note: EDM-style diffusion (NeurIPS 2024 baseline)")
        print(f"   Model channels: {model_cfg.get('model_channels', 128)}")
        print(f"   Sampling steps: {model.num_sampling_steps}")
        print(f"   Solver: {model.default_solver}")
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
        print(f"   Solver: {model.default_solver}")
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
        print(f"   Hidden size: {model.model_cfg.get('hidden_size', 384)}, "
              f"Depth: {model.model_cfg.get('depth', 8)}, "
              f"Heads: {model.model_cfg.get('num_heads', 6)}")
        print(f"   Patch size: {model.model_cfg.get('patch_size', 8)}, "
              f"Solver: {model.default_solver}")
    elif model_type == 'flow_matching_history':
        print(f"📦 Loading History-Window Flow Matching model...")
        # Let Lightning restore model_cfg, optim_cfg, scheduler_cfg, task_cfg
        # from checkpoint hparams. Only normalization_stats needs to be provided
        # since it was excluded from save_hyperparameters().
        model = ConditionalFlowMatchingHistoryLightning.load_from_checkpoint(
            checkpoint_path,
            normalization_stats=normalization_stats,
            strict=False
        )
        print(f"✓ History-Window Flow Matching model loaded successfully!")
        print(f"   History window: {model.history_window} (from checkpoint)")
        print(f"   Solver: {model.default_solver}")
    else:
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
    
    model.eval()
    print(f"   Task: {model.task_name}")
    print(f"   Conditioning channels: {model.conditioning_channels}")
    print(f"   Target channels: {model.target_channels}")
    return model


def load_dataset(data_file_path: str, output_fields=None, start_time=100, 
                 return_wall_temp=False,
                 noise_cfg=None, use_clean_inputs=False,
                 is_temporal=False, history_length=50, temporal_stride=1,
                 history_stride=1,
                 is_autoregressive=False,
                 is_ar_bootstrap=False, rollout_length=5,
                 is_history_model=False, history_window=10,
                 downsample_factor=1,
                 scheduled_sampling=False,
                 normalization_stats=None,
                 norm_mode='all',
                 # Legacy parameter - ignored
                 normalize_temperature=True):
    """Load the BulkFlow, BulkFlowAutoregressive, or BulkFlowARBootstrap dataset.
    
    All fields are normalized according to NORMALIZATION_REQUIREMENTS.md:
    - SDF: Zero-preserving normalization
    - Velocity: Unified scale normalization
    - Temperature: Tanh normalization to [-1, 1]
    
    Args:
        data_file_path: Path to HDF5 data file
        output_fields: Output field names
        start_time: Starting timestep
        return_wall_temp: Whether to return wall temperature
        noise_cfg: Noise configuration dict (for Task 3)
        use_clean_inputs: If True, disable noise even if noise_cfg is provided (for Task 3 clean inference)
        is_temporal: Unused (legacy parameter)
        history_length: Number of historical frames (for AR bootstrap models)
        temporal_stride: Unused (legacy parameter)
        history_stride: Stride between history frames for AR bootstrap (1=consecutive)
        is_autoregressive: If True, use BulkFlowAutoregressive for AR models
        is_ar_bootstrap: If True, use BulkFlowARBootstrap for AR Bootstrap models
        rollout_length: Rollout length for AR Bootstrap models
        downsample_factor: Factor to downsample spatial resolution (1 = no downsampling)
        scheduled_sampling: For AR models, whether to return extra context (always False for inference)
        normalization_stats: Pre-computed normalization statistics (if None, computed from data_file)
    """
    print(f"Loading data from: {data_file_path}")
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    if output_fields is None:
        output_fields = ['temperature', 'velx', 'vely']
    
    # For Task 3 clean inference, disable noise
    effective_noise_cfg = None
    if noise_cfg is not None and not use_clean_inputs:
        effective_noise_cfg = noise_cfg
        print(f"   🔊 Noise will be applied to inputs (Task 3 noisy inference)")
    elif noise_cfg is not None and use_clean_inputs:
        print(f"   ✨ Noise disabled (Task 3 clean inference - physics fidelity check)")
    elif noise_cfg is None:
        print(f"   ✨ No noise (Task 2 or clean inference)")
    
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
        print(f"      Returns: (conditioning_t, output_{{t-1}}, output_t)")
        if scheduled_sampling:
            print(f"      Extra context: (conditioning_{{t-1}}, output_{{t-2}})")
        
        dataset = BulkFlowAutoregressive(
            filenames=[data_file_path],
            output_fields=output_fields,
            start_time=start_time,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=effective_noise_cfg,
            downsample_factor=downsample_factor,
            scheduled_sampling=scheduled_sampling,
            norm_mode=norm_mode
        )
        print(f"✓ Autoregressive dataset loaded: {len(dataset)} samples")
        print(f"   Each sample: conditioning [C, H, W], prev_output [C, H, W] -> output [C, H, W]")
    elif is_history_model:
        print(f"   📊 Using HISTORY-WINDOW dataset:")
        print(f"      History window: {history_window}")
        
        dataset = BulkFlowHistory(
            filenames=[data_file_path],
            output_fields=output_fields,
            start_time=start_time,
            history_window=history_window,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=effective_noise_cfg,
            downsample_factor=downsample_factor,
            norm_mode=norm_mode
        )
        print(f"✓ History-window dataset loaded: {len(dataset)} samples")
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


def extract_channels(tensor: torch.Tensor, channel_indices: list) -> torch.Tensor:
    """Extract specific channels from a tensor based on channel indices."""
    return tensor[:, channel_indices, :, :]


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

        # Map interface velocity -> bulk velocity (hand-crafted projection)
        if 'velx_interface' in conditioning_names and 'velx' in target_names:
            src = conditioning_names.index('velx_interface')
            dst = target_names.index('velx')
            prev_output[:, dst, :, :] = mean_cond[:, src, :, :]
        if 'vely_interface' in conditioning_names and 'vely' in target_names:
            src = conditioning_names.index('vely_interface')
            dst = target_names.index('vely')
            prev_output[:, dst, :, :] = mean_cond[:, src, :, :]
        # Temperature stays at 0 (mean of normalised data)
        return prev_output

    raise ValueError(
        f"Unknown bootstrap_ablation mode: '{ablation_mode}'. "
        "Use 'zeros' or 'mean_conditioning_naive'."
    )


def run_inference_on_sample(model, dataset, sample_idx=0, device='cuda', 
                            num_inference_steps=50, model_type='flow_matching',
                            initial_state_mode='from_data', history_length=10,
                            bootstrap_ablation=None):
    """Run inference on a single sample for multi-channel prediction.
    
    Args:
        model: Trained model (Flow Matching, AR, or VE-SDE)
        dataset: BulkFlow or BulkFlowAutoregressive dataset
        sample_idx: Sample index to process
        device: Device to run on
        num_inference_steps: Number of integration/sampling steps
        model_type: Type of model ('flow_matching', 'flow_matching_ar', 
                                   'flow_matching_ar_bootstrap', 'unet_ar', 'flow_matching_jit', or 've_sde')
        initial_state_mode: Mode for initializing prev_output for AR models.
            - 'from_data': Use ground truth from dataset (default)
            - 'from_history_mean': Use mean of previous N ground truth frames
            - 'zeros': Use zeros (neutral state for normalized data)
            - 'small_noise': Small random noise around zero
            - 'from_conditioning': Derive from conditioning inputs
        history_length: Number of previous frames to average for 'from_history_mean' mode (default: 10)
        bootstrap_ablation: If set ('zeros' or 'mean_conditioning_naive'), replaces the learned
            history encoder output for AR bootstrap models with a simpler initialization.
    """
    
    if sample_idx >= len(dataset):
        sample_idx = len(dataset) - 1
        print(f"⚠️  Sample index too large, using {sample_idx}")
    
    # Get file index for denormalization (not available for AR dataset, use 0)
    if hasattr(dataset, 'get_file_index'):
        file_idx = dataset.get_file_index(sample_idx)
    else:
        file_idx = 0
    
    # Extract conditioning channels based on task config
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    # Handle different model types
    is_autoregressive = (model_type in ['flow_matching_ar', 'unet_ar'])
    is_ar_bootstrap = (model_type in ['flow_matching_ar_bootstrap', 'edm_ar_bootstrap'])
    
    # Get a sample from the dataset
    sample_data = dataset[sample_idx]
    
    if is_ar_bootstrap:
        # Bootstrap AR model: dataset returns (cond_history, cond_sequence, target_sequence)
        if dataset.return_wall_temp:
            cond_hist, cond_seq, target_seq, wall_temp = sample_data
        else:
            cond_hist, cond_seq, target_seq = sample_data
        
        # Move to device
        cond_hist = cond_hist.unsqueeze(0).to(device)    # [1, T_hist, C, H, W]
        cond_seq = cond_seq.unsqueeze(0).to(device)      # [1, L, C, H, W]
        target_seq = target_seq.unsqueeze(0).to(device)  # [1, L, C, H, W]
        
        # Extract relevant channels
        cond_hist_extracted = cond_hist[:, :, conditioning_channels, :, :]
        cond_seq_extracted = cond_seq[:, :, conditioning_channels, :, :]
        target_seq_extracted = target_seq[:, :, target_channels, :, :]
        
        # Use first frame for single sample inference
        conditioning = cond_seq_extracted[:, 0]  # [1, C_cond, H, W]
        target = target_seq_extracted[:, 0]      # [1, C_target, H, W]
        
        B, _, H, W = conditioning.shape
        C_out = target.shape[1]

        # Bootstrap initial state (learned encoder or ablation replacement)
        if bootstrap_ablation is not None:
            target_names_list = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
            conditioning_names_list = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
            prev_output = compute_bootstrap_ablation_state(
                bootstrap_ablation, cond_hist_extracted, C_out, device,
                target_names=target_names_list, conditioning_names=conditioning_names_list,
            )
            print(f"   🚀 AR BOOTSTRAP model inference (bootstrap_ablation='{bootstrap_ablation}'):")
        else:
            current_cond = conditioning
            prev_output = model.bootstrap_initial_state(cond_hist_extracted, current_cond)
            print(f"   🚀 AR BOOTSTRAP model inference (learned encoder):")

        availability_mask = torch.zeros(B, 1, H, W, device=device)  # Bootstrap mode
        
        print(f"   Condition history shape: {cond_hist_extracted.shape}")  # [1, T_hist, C_cond, H, W]
        print(f"   Conditioning shape: {conditioning.shape}")  # [1, C_cond, H, W]
        print(f"   Initial state shape: {prev_output.shape}")  # [1, C_target, H, W]
        print(f"   Target shape: {target.shape}")  # [1, C_target, H, W]
        
        input_data_viz = conditioning.squeeze(0).cpu()
    elif is_autoregressive:
        # Autoregressive model: dataset returns (conditioning_t, prev_output, target_t)
        if dataset.return_wall_temp:
            input_data, prev_output_data, target_data, wall_temp = sample_data
        else:
            input_data, prev_output_data, target_data = sample_data
        
        input_batch = input_data.unsqueeze(0).to(device)  # [1, C, H, W]
        prev_output_batch = prev_output_data.unsqueeze(0).to(device)  # [1, C_out, H, W]
        target_batch = target_data.unsqueeze(0).to(device)  # [1, C_out, H, W]
        
        conditioning = input_batch[:, conditioning_channels, :, :]  # [1, num_cond, H, W]
        target = target_batch[:, target_channels, :, :]  # [1, num_target, H, W]
        
        # Create prev_output based on initial_state_mode
        target_shape = (1, len(target_channels), target.shape[2], target.shape[3])
        
        if initial_state_mode == 'from_data':
            # Use ground truth previous output (teacher forcing)
            prev_output = prev_output_batch[:, target_channels, :, :]  # [1, num_target, H, W]
            print(f"   🔄 AUTOREGRESSIVE model inference (initial_state_mode='{initial_state_mode}'):")
            print(f"   Using ground truth previous output (teacher forcing)")
        elif initial_state_mode == 'from_history_mean':
            # Use mean of previous N ground truth frames as initial state
            first_idx = max(0, sample_idx - history_length + 1)
            actual_length = sample_idx - first_idx + 1
            history_frames = []
            for hist_i in range(first_idx, sample_idx + 1):
                hist_sample = dataset[hist_i]
                if dataset.return_wall_temp:
                    _, hist_prev, _, _ = hist_sample
                else:
                    _, hist_prev, _ = hist_sample
                history_frames.append(hist_prev[target_channels])  # [C_target, H, W]
            history_stack = torch.stack(history_frames, dim=0)  # [N, C_target, H, W]
            prev_output = history_stack.mean(dim=0, keepdim=True).to(device)  # [1, C_target, H, W]
            print(f"   🔄 AUTOREGRESSIVE model inference (initial_state_mode='{initial_state_mode}'):")
            print(f"   Using mean of {actual_length} previous frames (requested {history_length})")
        else:
            # Create initial state without ground truth
            if hasattr(model, 'create_initial_state'):
                prev_output = model.create_initial_state(
                    shape=target_shape,
                    device=device,
                    mode=initial_state_mode,
                    conditioning=conditioning if initial_state_mode == 'from_conditioning' else None
                )
                print(f"   🔄 AUTOREGRESSIVE model inference (initial_state_mode='{initial_state_mode}'):")
                print(f"   Created initial state via model.create_initial_state()")
            else:
                # Fallback for models without create_initial_state
                if initial_state_mode == 'zeros':
                    prev_output = torch.zeros(target_shape, device=device)
                elif initial_state_mode == 'small_noise':
                    prev_output = torch.randn(target_shape, device=device) * 0.01
                elif initial_state_mode == 'from_conditioning':
                    prev_output = torch.zeros(target_shape, device=device)
                    # Use interface velocity as initial bulk velocity estimate
                    if conditioning.shape[1] >= 3 and target_shape[1] >= 2:
                        prev_output[:, 0, :, :] = conditioning[:, 1, :, :]  # velx
                        prev_output[:, 1, :, :] = conditioning[:, 2, :, :]  # vely
                else:
                    print(f"   ⚠️ Unknown initial_state_mode '{initial_state_mode}', using zeros")
                    prev_output = torch.zeros(target_shape, device=device)
                print(f"   🔄 AUTOREGRESSIVE model inference (initial_state_mode='{initial_state_mode}'):")
                print(f"   Created initial state (fallback)")
        
        print(f"   Conditioning shape: {conditioning.shape}")  # [1, C_cond, H, W]
        print(f"   Previous output shape: {prev_output.shape}")  # [1, C_target, H, W]
        print(f"   Target shape: {target.shape}")  # [1, C_target, H, W]
        
        input_data_viz = input_batch.squeeze(0).cpu()
    else:
        # Non-temporal model: input is [C, H, W]
        if dataset.return_wall_temp:
            input_data, target_data, wall_temp = sample_data
        else:
            input_data, target_data = sample_data
            
        input_batch = input_data.unsqueeze(0).to(device)  # [1, C, H, W]
        target_batch = target_data.unsqueeze(0).to(device)  # [1, C, H, W]
        
        if model_type == 'flow_matching_history':
            # For history model, input_batch is the full flattened history window
            # (W * C_cond channels) -- use it directly as conditioning
            conditioning = input_batch  # [1, W*C_cond, H, W]
        else:
            conditioning = input_batch[:, conditioning_channels, :, :]  # [1, num_cond, H, W]
        target = target_batch[:, target_channels, :, :]  # [1, num_target, H, W]
        
        print(f"   Input shape: {input_batch.shape}")
        print(f"   Conditioning shape: {conditioning.shape}")
        print(f"   Target shape: {target.shape}")
        
        input_data_viz = input_batch.squeeze(0).cpu()
    
    # Run inference based on model type
    # DiffusionPDE needs gradients for autograd-based guidance -- must run outside torch.no_grad()
    if model_type == 'diffusionpde':
        sampling_steps = getattr(model, 'num_sampling_steps', 50)
        solver = getattr(model, 'default_solver', 'heun')
        num_joint = model.num_joint_channels
        print(f"🔬 Running DiffusionPDE guided sampling with {sampling_steps} steps (solver={solver})...")
        predicted = model.diffusion_pde.sample_with_guidance(
            observed_gt=conditioning,
            shape=(1, num_joint, target.shape[2], target.shape[3]),
            device=device,
            num_steps=sampling_steps,
            solver=solver,
        )
    else:
      with torch.no_grad():
        if model_type == 've_sde':
            print(f"🔊 Running VE-SDE sampling with {num_inference_steps} steps...")
            predicted = model.ve_sde.sample(
                condition=conditioning,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device,
                num_steps=num_inference_steps,
                method=model.sampling_method,
                snr=model.snr
            )
        elif model_type == 'flow_matching_ar':
            print(f"🔄 Running Autoregressive Flow Matching sampling with {num_inference_steps} ODE steps...")
            print(f"   Using ground truth previous output (teacher forcing for single sample)")
            predicted = model.sample(
                condition=conditioning,
                prev_output=prev_output,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device,
                num_integration_steps=num_inference_steps,
            )
        elif model_type == 'flow_matching_ar_bootstrap':
            print(f"🚀 Running AR Bootstrap Flow Matching sampling with {num_inference_steps} ODE steps...")
            print(f"   Using bootstrapped initial state from conditioning history")
            predicted = model.sample(
                condition=conditioning,
                prev_output=prev_output,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device,
                availability_mask=availability_mask,
                num_integration_steps=num_inference_steps,
            )
        elif model_type == 'edm_ar_bootstrap':
            sampling_steps = getattr(model, 'num_sampling_steps', 50)
            solver = getattr(model, 'default_solver', 'heun')
            print(f"🚀 Running AR Bootstrap EDM sampling with {sampling_steps} steps (solver={solver})...")
            print(f"   Using bootstrapped initial state from conditioning history")
            predicted = model.sample(
                condition=conditioning,
                prev_output=prev_output,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device,
                availability_mask=availability_mask,
                num_steps=sampling_steps,
                solver=solver,
            )
        elif model_type == 'unet_ar':
            print(f"🔄 Running Autoregressive UNet inference (single forward pass)...")
            print(f"   Using ground truth previous output (teacher forcing for single sample)")
            predicted = model.sample(
                condition=conditioning,
                prev_output=prev_output,
            )
        elif model_type == 'unet':
            print(f"📦 Running UNet inference (single forward pass)...")
            predicted = model(conditioning)
        elif model_type == 'ffno':
            print(f"📦 Running FFNO inference (single forward pass)...")
            predicted = model(conditioning)
        elif model_type == 'bubble_ddpm':
            ddpm_steps = getattr(model, 'num_inference_steps', model.ddpm.num_timesteps)
            print(f"🎲 Running DDPM sampling with {ddpm_steps} diffusion steps...")
            predicted = model.ddpm.p_sample_loop(
                condition=conditioning,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device
            )
        elif model_type == 'edm':
            sampling_steps = getattr(model, 'num_sampling_steps', 50)
            solver = getattr(model, 'default_solver', 'heun')
            print(f"🔬 Running EDM sampling with {sampling_steps} steps (solver={solver})...")
            predicted = model.edm.sample(
                condition=conditioning,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device,
                num_steps=sampling_steps,
                solver=solver
            )
        elif model_type == 'flow_matching_jit':
            solver = getattr(model, 'default_solver', 'heun')
            print(f"🔮 Running JiT Flow Matching sampling with {num_inference_steps} ODE steps (solver={solver})...")
            predicted = model.flow_matching.sample(
                condition=conditioning,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device,
                num_integration_steps=num_inference_steps,
                solver=solver
            )
        elif model_type == 'flow_matching_history':
            solver = getattr(model, 'default_solver', 'heun')
            print(f"📚 Running History-Window Flow Matching sampling with {num_inference_steps} ODE steps (solver={solver})...")
            predicted = model.flow_matching.sample(
                condition=conditioning,  # [1, W*C_cond, H, W]
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device,
                num_integration_steps=num_inference_steps,
                solver=solver
            )
        else:
            # Regular flow matching (frame-to-frame)
            solver = getattr(model, 'default_solver', 'euler')
            print(f"🌊 Running Flow Matching sampling with {num_inference_steps} ODE steps (solver={solver})...")
            predicted = model.flow_matching.sample(
                condition=conditioning,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device,
                num_integration_steps=num_inference_steps,
                solver=solver
            )
    
    # Move back to CPU and remove batch dimension
    target = target.squeeze(0).cpu()
    predicted = predicted.squeeze(0).cpu()
    
    # Denormalize ALL output fields (temperature, velocity)
    # Normalization is now always applied, so we always denormalize
    print("🔄 Denormalizing output fields...")
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    
    for i, field_name in enumerate(target_names):
        target[i] = dataset._denormalize_field(target[i], field_name)
        predicted[i] = dataset._denormalize_field(predicted[i], field_name)
        
        if field_name == 'temperature':
            print(f"   Temperature (target) range: [{target[i].min():.1f}, {target[i].max():.1f}]°C")
            print(f"   Temperature (pred) range: [{predicted[i].min():.1f}, {predicted[i].max():.1f}]°C")
        else:
            print(f"   {field_name} (target) range: [{target[i].min():.4f}, {target[i].max():.4f}]")
            print(f"   {field_name} (pred) range: [{predicted[i].min():.4f}, {predicted[i].max():.4f}]")
    
    # Denormalize input fields (SDF, interface velocity) for visualization
    print("🔄 Denormalizing input fields...")
    conditioning_names = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
    for i, field_name in enumerate(conditioning_names):
        input_data_viz[i] = dataset._denormalize_field(input_data_viz[i], field_name)
        if field_name == 'sdf':
            print(f"   {field_name} (input) range: [{input_data_viz[i].min():.4f}, {input_data_viz[i].max():.4f}]")
        else:
            print(f"   {field_name} (input) range: [{input_data_viz[i].min():.4f}, {input_data_viz[i].max():.4f}]")
    
    return input_data_viz, target, predicted


def run_autoregressive_trajectory(model, dataset, start_sample_idx, num_frames,
                                   device='cuda', num_inference_steps=50,
                                   solver='heun', guidance_scale=1.0, model_type='flow_matching_ar',
                                   initial_state_mode='from_data', history_length=10, **kwargs):
    """
    Run true autoregressive inference for a trajectory (AR models only).
    
    Instead of using ground truth previous outputs (teacher forcing),
    this uses the model's own predictions autoregressively.
    
    Args:
        model: Trained AR model (ConditionalFlowMatchingARLightning or UNetARLightning)
        dataset: BulkFlowAutoregressive dataset
        start_sample_idx: Starting sample index (provides initial state)
        num_frames: Number of frames to generate
        device: Device to run on
        num_inference_steps: Number of ODE integration steps per frame (flow_matching_ar only)
        solver: ODE solver - 'euler', 'heun', 'midpoint', 'rk4' (flow_matching_ar only)
        guidance_scale: Classifier-free guidance scale (flow_matching_ar only)
        model_type: Model type ('flow_matching_ar' or 'unet_ar')
        initial_state_mode: Mode for initializing prev_output when ground truth is not available.
            - 'from_data': Use ground truth from dataset (default, requires GT)
            - 'from_history_mean': Use mean of previous N ground truth frames (N = history_length)
            - 'zeros': Use zeros (neutral state for normalized data, no GT needed)
            - 'small_noise': Small random noise around zero (no GT needed)
            - 'from_conditioning': Derive from conditioning inputs (no GT needed)
        history_length: Number of previous frames to average for 'from_history_mean' mode (default: 10)
        
    Returns:
        List of tuples: [(input_data, target, predicted), ...]
    """
    print(f"\n🔄 Running Autoregressive Trajectory Generation")
    print(f"   Model type: {model_type}")
    print(f"   Start index: {start_sample_idx}")
    print(f"   Num frames: {num_frames}")
    if model_type == 'flow_matching_ar':
        print(f"   Integration steps per frame: {num_inference_steps}")
        print(f"   ODE Solver: {solver}")
        print(f"   Guidance scale: {guidance_scale}")
    elif model_type in ['unet_ar']:
        print(f"   Direct regression (single forward pass per frame)")
    
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    # Get file index for denormalization
    file_idx = 0  # AR dataset doesn't have get_file_index
    
    results = []
    
    # Get initial sample for the starting state
    sample_data = dataset[start_sample_idx]
    if dataset.return_wall_temp:
        input_data, prev_output_data, target_data, wall_temp = sample_data
    else:
        input_data, prev_output_data, target_data = sample_data
    
    # Determine initial state based on mode
    print(f"   Initial state mode: {initial_state_mode}")
    
    if initial_state_mode == 'from_data':
        # Use ground truth previous output as initial state (default behavior)
        prev_output = prev_output_data.unsqueeze(0).to(device)  # [1, C, H, W]
        prev_output = prev_output[:, target_channels, :, :]  # Extract target channels
        print(f"   Using ground truth for initial state (shape: {prev_output.shape})")
    elif initial_state_mode == 'from_history_mean':
        # Use mean of previous N ground truth frames as initial state
        first_idx = max(0, start_sample_idx - history_length + 1)
        actual_length = start_sample_idx - first_idx + 1
        history_frames = []
        for hist_i in range(first_idx, start_sample_idx + 1):
            hist_sample = dataset[hist_i]
            if dataset.return_wall_temp:
                _, hist_prev, _, _ = hist_sample
            else:
                _, hist_prev, _ = hist_sample
            history_frames.append(hist_prev[target_channels])  # [C_target, H, W]
        history_stack = torch.stack(history_frames, dim=0)  # [N, C_target, H, W]
        prev_output = history_stack.mean(dim=0, keepdim=True).to(device)  # [1, C_target, H, W]
        print(f"   Using mean of {actual_length} previous frames (requested {history_length}) as initial state (shape: {prev_output.shape})")
    else:
        # Create initial state without ground truth using model's create_initial_state method
        # Get shape from target_data
        input_batch = input_data.unsqueeze(0).to(device)
        conditioning = input_batch[:, conditioning_channels, :, :]
        target_shape = (1, len(target_channels), target_data.shape[1], target_data.shape[2])
        
        if hasattr(model, 'create_initial_state'):
            prev_output = model.create_initial_state(
                shape=target_shape,
                device=device,
                mode=initial_state_mode,
                conditioning=conditioning if initial_state_mode == 'from_conditioning' else None
            )
            print(f"   Created initial state with mode='{initial_state_mode}' (shape: {prev_output.shape})")
        else:
            # Fallback for models that don't have create_initial_state
            if initial_state_mode == 'zeros':
                prev_output = torch.zeros(target_shape, device=device)
            elif initial_state_mode == 'small_noise':
                prev_output = torch.randn(target_shape, device=device) * 0.01
            elif initial_state_mode == 'from_conditioning':
                prev_output = torch.zeros(target_shape, device=device)
                # Use interface velocity as initial bulk velocity estimate
                if conditioning.shape[1] >= 3 and target_shape[1] >= 2:
                    prev_output[:, 0, :, :] = conditioning[:, 1, :, :]  # velx
                    prev_output[:, 1, :, :] = conditioning[:, 2, :, :]  # vely
            else:
                print(f"   ⚠️ Unknown initial_state_mode '{initial_state_mode}', using zeros")
                prev_output = torch.zeros(target_shape, device=device)
            print(f"   Created initial state (fallback) with mode='{initial_state_mode}' (shape: {prev_output.shape})")
    
    for frame_idx in range(num_frames):
        current_idx = start_sample_idx + frame_idx
        
        if current_idx >= len(dataset):
            print(f"   ⚠️ Reached end of dataset at frame {frame_idx}")
            break
        
        # Get current sample (for conditioning and ground truth target)
        sample_data = dataset[current_idx]
        if dataset.return_wall_temp:
            input_data, gt_prev_output, target_data, wall_temp = sample_data
        else:
            input_data, gt_prev_output, target_data = sample_data
        
        input_batch = input_data.unsqueeze(0).to(device)  # [1, C, H, W]
        target_batch = target_data.unsqueeze(0).to(device)  # [1, C, H, W]
        
        conditioning = input_batch[:, conditioning_channels, :, :]  # [1, C_cond, H, W]
        target = target_batch[:, target_channels, :, :]  # [1, C_target, H, W]
        
        # Run inference with model's own previous prediction (or GT for first frame)
        # Use model.sample() which handles residual reconstruction automatically
        with torch.no_grad():
            if model_type in ['unet_ar']:
                # UNet AR: direct regression (single forward pass)
                predicted = model.sample(
                    condition=conditioning,
                    prev_output=prev_output,
                )
            else:
                # Flow matching AR: ODE integration
                predicted = model.sample(
                    condition=conditioning,
                    prev_output=prev_output,
                    shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                    device=device,
                    num_integration_steps=num_inference_steps,
                    solver=solver,
                    guidance_scale=guidance_scale,
                )
        
        # Move to CPU
        target_cpu = target.squeeze(0).cpu()
        predicted_cpu = predicted.squeeze(0).cpu()
        input_data_viz = input_batch.squeeze(0).cpu()
        
        # Denormalize ALL output fields (temperature, velocity)
        # Normalization is now always applied
        target_names_list = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
        for i, field_name in enumerate(target_names_list):
            target_cpu[i] = dataset._denormalize_field(target_cpu[i], field_name)
            predicted_cpu[i] = dataset._denormalize_field(predicted_cpu[i], field_name)

        # Denormalize input fields (SDF, interface velocity) for visualization
        conditioning_names_list = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
        for i, field_name in enumerate(conditioning_names_list):
            input_data_viz[i] = dataset._denormalize_field(input_data_viz[i], field_name)

        results.append((input_data_viz, target_cpu, predicted_cpu))
        
        # Update prev_output for next frame (autoregressive)
        prev_output = predicted  # Use model's prediction, not ground truth
        
        if frame_idx % 10 == 0:
            print(f"   Frame {frame_idx + 1}/{num_frames} completed")
    
    print(f"   ✓ Generated {len(results)} frames autoregressively")
    return results


def run_bootstrap_trajectory(model, dataset, start_sample_idx, num_frames,
                              device='cuda', num_inference_steps=50,
                              solver='heun', bootstrap_ablation=None, **kwargs):
    """
    Run bootstrap + autoregressive inference for a trajectory.
    
    For the AR Bootstrap model, this uses the history encoder to bootstrap
    the initial state, then runs autoregressive rollout.
    
    Args:
        model: Trained AR Bootstrap model (ConditionalFlowMatchingARBootstrapLightning)
        dataset: BulkFlowARBootstrap dataset
        start_sample_idx: Starting sample index
        num_frames: Number of frames to generate
        device: Device to run on
        num_inference_steps: Number of ODE integration steps per frame
        solver: ODE solver - 'euler', 'heun', 'midpoint', 'rk4'
        bootstrap_ablation: If set ('zeros' or 'mean_conditioning_naive'), replaces
            the learned history encoder output with a simpler initialization.
        
    Returns:
        List of tuples: [(input_data, target, predicted), ...]
    """
    print(f"\n🚀 Running Bootstrap + AR Trajectory Generation")
    print(f"   Model type: flow_matching_ar_bootstrap")
    print(f"   Start index: {start_sample_idx}")
    print(f"   Num frames: {num_frames}")
    print(f"   Integration steps per frame: {num_inference_steps}")
    print(f"   ODE Solver: {solver}")
    
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    results = []
    
    # Get initial sample for bootstrap
    sample_data = dataset[start_sample_idx]
    if dataset.return_wall_temp:
        cond_hist, cond_seq, target_seq, wall_temp = sample_data
    else:
        cond_hist, cond_seq, target_seq = sample_data
    
    # Move to device
    cond_hist = cond_hist.unsqueeze(0).to(device)  # [1, T_hist, C, H, W]
    cond_seq = cond_seq.unsqueeze(0).to(device)    # [1, L, C, H, W]
    target_seq = target_seq.unsqueeze(0).to(device)  # [1, L, C, H, W]
    
    # Extract relevant channels
    cond_hist_extracted = cond_hist[:, :, conditioning_channels, :, :]
    cond_seq_extracted = cond_seq[:, :, conditioning_channels, :, :]
    target_seq_extracted = target_seq[:, :, target_channels, :, :]
    
    B, T_hist, C_cond, H, W = cond_hist_extracted.shape
    _, L, _, _, _ = cond_seq_extracted.shape
    C_out = target_seq_extracted.shape[2]
    
    print(f"   Condition history shape: {cond_hist_extracted.shape}")
    print(f"   Condition sequence shape: {cond_seq_extracted.shape}")
    print(f"   Target sequence shape: {target_seq_extracted.shape}")
    
    # Bootstrap initial state (learned encoder or ablation replacement)
    if bootstrap_ablation is not None:
        print(f"\n   🔧 Bootstrap ABLATION mode: '{bootstrap_ablation}'")
        target_names_list = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
        conditioning_names_list = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
        prev_output = compute_bootstrap_ablation_state(
            bootstrap_ablation, cond_hist_extracted, C_out, device,
            target_names=target_names_list, conditioning_names=conditioning_names_list,
        )
        print(f"   ✓ Ablation initial state shape: {prev_output.shape}")
    else:
        print(f"\n   🔧 Bootstrapping initial state from {T_hist} frames of history...")
        current_cond_0 = cond_seq_extracted[:, 0]  # [B, C_cond, H, W]
        prev_output = model.bootstrap_initial_state(
            cond_hist_extracted, current_cond_0
        )
        print(f"   ✓ Bootstrapped state shape: {prev_output.shape}")
    
    # Generate trajectory using model's sample_trajectory or manual rollout
    print(f"\n   🔄 Running autoregressive rollout for {min(num_frames, L)} frames...")
    
    # Process frames from the current segment
    frames_to_process = min(num_frames, L)
    
    for l in range(frames_to_process):
        current_cond = cond_seq_extracted[:, l]  # [B, C_cond, H, W]
        target_l = target_seq_extracted[:, l]    # [B, C_out, H, W]
        
        # Create availability mask (0 for first frame = bootstrapped, 1 for rest)
        if l == 0:
            availability_mask = torch.zeros(B, 1, H, W, device=device)
        else:
            availability_mask = torch.ones(B, 1, H, W, device=device)
        
        # Generate prediction
        with torch.no_grad():
            predicted = model.sample(
                condition=current_cond,
                prev_output=prev_output,
                shape=(B, C_out, H, W),
                device=device,
                availability_mask=availability_mask,
                num_integration_steps=num_inference_steps,
                solver=solver,
            )
        
        # Move to CPU for visualization
        target_cpu = target_l.squeeze(0).cpu()
        predicted_cpu = predicted.squeeze(0).cpu()
        input_data_viz = current_cond.squeeze(0).cpu()
        
        # Denormalize output fields (temperature, velocity)
        target_names_list = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
        for i, field_name in enumerate(target_names_list):
            target_cpu[i] = dataset._denormalize_field(target_cpu[i], field_name)
            predicted_cpu[i] = dataset._denormalize_field(predicted_cpu[i], field_name)
        
        # Denormalize conditioning fields (SDF, interface velocity)
        conditioning_names_list = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
        for i, field_name in enumerate(conditioning_names_list):
            input_data_viz[i] = dataset._denormalize_field(input_data_viz[i], field_name)
        
        results.append((input_data_viz, target_cpu, predicted_cpu))
        
        # Update prev_output for next frame
        prev_output = predicted
        
        if l % 5 == 0:
            print(f"   Frame {l + 1}/{frames_to_process} completed")
    
    # If we need more frames, load data directly from timesteps (not from overlapping samples)
    if num_frames > L:
        print(f"\n   ℹ️  Extending beyond initial {L}-frame segment.")
        print(f"   Loading continuous timesteps directly for frames {L+1}-{num_frames}...")
        print(f"   (AR chain continues using model's own predictions)")
        
        # Calculate the starting timestep of the initial segment
        # Based on how BulkFlowARBootstrap.__getitem__ works:
        # segment_start = local_idx + effective_start_time
        effective_start_time = dataset.effective_start_time
        base_timestep = start_sample_idx + effective_start_time
        
        # We've processed frames 0 to L-1 (timesteps base_timestep to base_timestep+L-1)
        # Now we need timesteps starting from base_timestep + L
        
        for frame_idx in range(L, num_frames):
            current_timestep = base_timestep + frame_idx
            
            # Check if we're within bounds
            if current_timestep >= dataset.traj_lens[0] - 1:
                print(f"   ⚠️ Reached end of trajectory at timestep {current_timestep}")
                break
            
            # Load conditioning and target directly from the HDF5 file
            # Using dataset's helper methods for proper normalization
            cond_frame = dataset._get_conditioning_frame(0, current_timestep)  # file_idx=0
            target_frame = dataset._get_output_frame(0, current_timestep)
            
            # Extract relevant channels
            cond_frame = cond_frame[conditioning_channels]
            target_frame = target_frame[target_channels]
            
            # Add batch dimension and move to device
            current_cond = cond_frame.unsqueeze(0).to(device)  # [1, C, H, W]
            target_l = target_frame.unsqueeze(0).to(device)
            
            availability_mask = torch.ones(B, 1, H, W, device=device)
            
            with torch.no_grad():
                predicted = model.sample(
                    condition=current_cond,
                    prev_output=prev_output,
                    shape=(B, C_out, H, W),
                    device=device,
                    availability_mask=availability_mask,
                    num_integration_steps=num_inference_steps,
                    solver=solver,
                )
            
            target_cpu = target_l.squeeze(0).cpu()
            predicted_cpu = predicted.squeeze(0).cpu()
            input_data_viz = current_cond.squeeze(0).cpu()
            
            # Denormalize output fields
            for i, field_name in enumerate(target_names_list):
                target_cpu[i] = dataset._denormalize_field(target_cpu[i], field_name)
                predicted_cpu[i] = dataset._denormalize_field(predicted_cpu[i], field_name)
            
            # Denormalize conditioning fields
            for i, field_name in enumerate(conditioning_names_list):
                input_data_viz[i] = dataset._denormalize_field(input_data_viz[i], field_name)
            
            results.append((input_data_viz, target_cpu, predicted_cpu))
            prev_output = predicted
            
            if len(results) % 10 == 0:
                print(f"   Frame {len(results)}/{num_frames} completed (timestep {current_timestep})")
    
    print(f"   ✓ Generated {len(results)} frames with bootstrap + AR rollout")
    return results


def compute_metrics(target, predicted, target_names):
    """Compute comprehensive evaluation metrics for multi-channel prediction."""
    
    metrics = {}
    
    # Overall metrics (across all channels)
    mse = torch.mean((target - predicted) ** 2)
    mae = torch.mean(torch.abs(target - predicted))
    rmse = torch.sqrt(mse)
    
    # Compute per-channel relative L2 first to calculate overall as average
    channel_rel_l2_values = []
    for i, name in enumerate(target_names):
        target_ch = target[i].flatten()
        pred_ch = predicted[i].flatten()
        ch_rel_l2 = torch.norm(target_ch - pred_ch) / (torch.norm(target_ch) + 1e-8)
        channel_rel_l2_values.append(ch_rel_l2)
    
    # Overall Relative L2: average of per-channel relative L2 values
    # This gives equal weight to each channel regardless of their absolute scales
    # (unlike the combined norm which is dominated by channels with larger absolute values like temperature)
    overall_rel_l2 = torch.mean(torch.stack(channel_rel_l2_values))
    
    # Also compute the combined norm version for reference (scale-dependent, dominated by temperature)
    relative_l2_combined = torch.norm(target - predicted) / (torch.norm(target) + 1e-8)
    
    metrics['Overall'] = {
        'MSE': mse.item(),
        'MAE': mae.item(),
        'RMSE': rmse.item(),
        'Relative_L2': overall_rel_l2.item(),  # Average of per-channel relative L2 (scale-invariant)
        'Relative_L2_Combined': relative_l2_combined.item()  # Combined norm (scale-dependent)
    }
    
    # Per-channel metrics
    for i, name in enumerate(target_names):
        target_ch = target[i].flatten()
        pred_ch = predicted[i].flatten()
        
        ch_mse = torch.mean((target_ch - pred_ch) ** 2)
        ch_mae = torch.mean(torch.abs(target_ch - pred_ch))
        ch_rmse = torch.sqrt(ch_mse)
        ch_rel_l2 = torch.norm(target_ch - pred_ch) / (torch.norm(target_ch) + 1e-8)
        
        # R-squared
        ss_res = torch.sum((target_ch - pred_ch) ** 2)
        ss_tot = torch.sum((target_ch - torch.mean(target_ch)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Correlation
        corr = torch.corrcoef(torch.stack([target_ch, pred_ch]))[0, 1]
        if torch.isnan(corr):
            corr = torch.tensor(0.0)
        
        metrics[name] = {
            'MSE': ch_mse.item(),
            'MAE': ch_mae.item(),
            'RMSE': ch_rmse.item(),
            'Relative_L2': ch_rel_l2.item(),
            'R2': r2.item() if isinstance(r2, torch.Tensor) else r2,
            'Correlation': corr.item(),
            'Target_Range': (target_ch.min().item(), target_ch.max().item()),
            'Pred_Range': (pred_ch.min().item(), pred_ch.max().item())
        }
    
    return metrics


def create_comprehensive_visualization(input_data, target, predicted, 
                                       target_names, conditioning_names,
                                       sample_idx=0, metrics=None, 
                                       save_dir="./inference_output",
                                       model_type='flow_matching',
                                       task_name='velocity_from_interface',
                                       use_clean_inputs=False,
                                       colorbar_ranges=None,
                                       use_bubbleml_cmap=False,
                                       bulk_temp=48.3,
                                       heater_temp=114.7,
                                       cmap_vmin=0.03,
                                       vel_vmin=0.1,
                                       vel_vmax=None,
                                       vel_interface_vmax=None,
                                       dpi=150):
    """
    Create comprehensive visualization for multi-channel prediction.
    
    Args:
        input_data: Conditioning input tensor [C, H, W]
        target: Ground truth target tensor [C, H, W]
        predicted: Model prediction tensor [C, H, W]
        target_names: List of target field names
        conditioning_names: List of conditioning field names
        sample_idx: Sample index for labeling
        metrics: Dictionary of computed metrics
        save_dir: Output directory
        model_type: Model type string
        task_name: Task name string
        use_clean_inputs: Whether clean inputs were used
        colorbar_ranges: Optional dict with fixed colorbar ranges for consistent GIF animation.
                        Keys: 'temperature', 'velx', 'vely', 'velx_interface', 'vely_interface', 
                              'sdf', 'velocity_magnitude'
                        Values: (vmin, vmax) tuples
        use_bubbleml_cmap: If True, use custom BubbleML temperature colormap for temperature fields
        bulk_temp: Bulk temperature in Celsius (default: 48.3°C for subcooled pool boiling)
        heater_temp: Heater temperature in Celsius (default: 114.7°C)
        cmap_vmin: Minimum value for BubbleML colormap (0-1 range). Set to 0.02-0.05 to hide noise.
        vel_vmin: Minimum value for velocity magnitude colormap. Set to 0.05-0.2 to hide background noise.
        vel_vmax: Fixed max value for velocity magnitude. If None, use colorbar_ranges or auto-detect.
        vel_interface_vmax: Fixed max value for interface velocity magnitude. If None, auto-detect.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_targets = len(target_names)
    num_cond = len(conditioning_names)
    
    # Determine if we have velocity fields for streamline plots
    has_velocity = 'velx' in target_names and 'vely' in target_names
    
    # Check if we have interface velocity components for magnitude calculation
    has_interface_velocity = 'velx_interface' in conditioning_names and 'vely_interface' in conditioning_names
    
    # Check if we have target velocity components for magnitude calculation
    has_target_velocity = 'velx' in target_names and 'vely' in target_names
    
    # Adaptive layout based on task:
    # - Task 1 (no velocity): 4 rows x max(num_targets, num_cond) columns (no streamlines)
    # - Task 2/3 (with velocity): 5 rows x num_targets columns
    # Add extra column for velocity magnitude if we have both components
    num_cond_display = num_cond + 1 if has_interface_velocity else num_cond
    num_targets_display = num_targets + 1 if has_target_velocity else num_targets
    num_cols = max(num_targets_display, num_cond_display)
    num_rows = 5 if has_velocity else 4
    
    # Create figure with adaptive size and constrained_layout for consistent spacing
    # constrained_layout is better than tight_layout for consistent frame sizes in GIFs
    fig = plt.figure(figsize=(5 * num_cols, 4 * num_rows + 0.5), constrained_layout=True)
    
    if model_type == 've_sde':
        model_name = "VE-SDE"
    elif model_type == 'flow_matching_ar':
        model_name = "Autoregressive Flow Matching"
    elif model_type == 'flow_matching_ar_bootstrap':
        model_name = "AR Flow Matching (Bootstrap)"
    elif model_type == 'edm_ar_bootstrap':
        model_name = "AR EDM (Bootstrap)"
    elif model_type == 'unet':
        model_name = "UNet"
    elif model_type == 'ffno':
        model_name = "FFNO"
    elif model_type == 'unet_ar':
        model_name = "Autoregressive UNet"
    elif model_type == 'edm':
        model_name = "EDM"
    elif model_type == 'diffusionpde':
        model_name = "DiffusionPDE"
    elif model_type == 'bubble_ddpm':
        model_name = "DDPM"
    else:
        model_name = "Flow Matching"
    
    # Determine task title
    if task_name == 'temperature_from_sdf':
        task_title = "Task 1: Temperature from SDF"
    elif task_name == 'noisy_velocity_from_interface':
        if use_clean_inputs:
            task_title = "Task 3: Noisy Velocity from Interface (Clean Inputs - Physics Fidelity)"
        else:
            task_title = "Task 3: Noisy Velocity from Interface (Noisy Inputs - Deployment)"
    else:
        task_title = "Task 2: Velocity from Interface"
    
    main_title = f'{task_title} ({model_name}) - Sample {sample_idx}'
    if metrics:
        main_title += f' (Overall Rel L2: {metrics["Overall"]["Relative_L2"]:.4f})'
    # Use y=1.0 with constrained_layout to place title at top
    fig.suptitle(main_title, fontsize=16)
    
    # Helper function to get colorbar range for a field
    def get_range(field_name):
        if colorbar_ranges and field_name in colorbar_ranges and colorbar_ranges[field_name] is not None:
            return colorbar_ranges[field_name]
        return None
    
    # === Row 1: Conditioning Inputs ===
    # Reorder: velx_interface, vely_interface, |vel_interface|, sdf
    # Store velocity components and SDF for reordered plotting
    velx_interface_data = None
    vely_interface_data = None
    sdf_data = None
    sdf_rng = None
    
    # First pass: collect data
    for i, name in enumerate(conditioning_names):
        data = input_data[i].numpy()
        if name == 'velx_interface':
            velx_interface_data = data
        elif name == 'vely_interface':
            vely_interface_data = data
        elif name == 'sdf':
            sdf_data = data
            sdf_rng = get_range(name)
    
    # Plot in reordered sequence: velx_interface, vely_interface, |vel_interface|, sdf
    plot_col = 1
    
    # Plot velx_interface
    if velx_interface_data is not None:
        ax = plt.subplot(num_rows, num_cols, plot_col)
        data = velx_interface_data
        rng = get_range('velx_interface')
        if use_bubbleml_cmap:
            norm = TwoSlopeNorm(vcenter=0, vmin=data.min(), vmax=data.max())
            im = ax.imshow(data, cmap='coolwarm', norm=norm, origin='lower')
        else:
            if rng:
                im = ax.imshow(data, cmap='RdBu', origin='lower', vmin=rng[0], vmax=rng[1])
            else:
                im = ax.imshow(data, cmap='RdBu', origin='lower')
        ax.set_title('Input: velx_interface', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plot_col += 1
    
    # Plot vely_interface
    if vely_interface_data is not None:
        ax = plt.subplot(num_rows, num_cols, plot_col)
        data = vely_interface_data
        rng = get_range('vely_interface')
        if use_bubbleml_cmap:
            norm = TwoSlopeNorm(vcenter=0, vmin=data.min(), vmax=data.max())
            im = ax.imshow(data, cmap='coolwarm', norm=norm, origin='lower')
        else:
            if rng:
                im = ax.imshow(data, cmap='RdBu', origin='lower', vmin=rng[0], vmax=rng[1])
            else:
                im = ax.imshow(data, cmap='RdBu', origin='lower')
        ax.set_title('Input: vely_interface', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plot_col += 1
    
    # Plot SDF (before |vel_interface|)
    if sdf_data is not None:
        ax = plt.subplot(num_rows, num_cols, plot_col)
        data = sdf_data
        rng = sdf_rng
        if use_bubbleml_cmap:
            # Use BubbleML SDF colormap with TwoSlopeNorm centered at 0
            norm = TwoSlopeNorm(vcenter=0, vmin=data.min(), vmax=data.max())
            im = ax.imshow(data, cmap='RdYlBu', norm=norm, origin='lower')
            # Add multiple white dotted contour lines to show SDF structure
            num_contours = 10
            contour_levels = np.linspace(data.min(), data.max(), num_contours)
            contour = ax.contour(data, levels=contour_levels, colors='white', 
                                 linewidths=0.8, linestyles='dotted', alpha=0.6)
            # Add the interface boundary (sdf=0) with a solid black line
            interface = ax.contour(data, levels=[0], colors='black', alpha=0.6, linewidths=1.5)
        else:
            if rng:
                im = ax.imshow(data, cmap='viridis', origin='lower', vmin=rng[0], vmax=rng[1])
            else:
                im = ax.imshow(data, cmap='viridis', origin='lower')
            # Add bubble interface contour
            contour = ax.contour(data, levels=[0], colors='white', linewidths=1)
        ax.set_title('Input: sdf', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plot_col += 1
    
    # Plot interface velocity magnitude (at the end)
    if has_interface_velocity and velx_interface_data is not None and vely_interface_data is not None:
        ax = plt.subplot(num_rows, num_cols, plot_col)
        vel_mag_interface = np.sqrt(velx_interface_data**2 + vely_interface_data**2)
        # Priority: 1) command line arg, 2) colorbar_ranges, 3) auto-detect
        if vel_interface_vmax is not None:
            vmax_interface = vel_interface_vmax
        else:
            interface_mag_range = get_range('velocity_magnitude_interface')
            if interface_mag_range is not None:
                vmax_interface = interface_mag_range[1]
            else:
                vmax_interface = vel_mag_interface.max() if vel_mag_interface.max() > 0 else 1
        # Use PowerNorm (gamma=0.5) + turbo colormap for better detail visibility
        norm = PowerNorm(gamma=0.5, vmin=vel_vmin, vmax=vmax_interface)
        im = ax.imshow(vel_mag_interface, cmap='turbo', norm=norm, origin='lower')
        ax.set_title(f'Input: |vel_interface|\n[{vel_vmin:.1f}, {vmax_interface:.2f}]', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # === Row 2: Target Fields ===
    target_velx_data = None
    target_vely_data = None
    
    for i, name in enumerate(target_names):
        ax = plt.subplot(num_rows, num_cols, num_cols + i + 1)
        data = target[i].numpy()
        rng = get_range(name)
        
        # Store velocity components for magnitude calculation
        if name == 'velx':
            target_velx_data = data
        elif name == 'vely':
            target_vely_data = data
        
        if name == 'temperature' and use_bubbleml_cmap:
            # Convert physical temperature (°C) to [0, 1] range for BubbleML colormap
            data_for_cmap = (data - bulk_temp) / (heater_temp - bulk_temp)
            data_for_cmap = np.clip(data_for_cmap, 0, 1)
            cmap = temp_cmap()
            im = ax.imshow(data_for_cmap, cmap=cmap, origin='lower', vmin=cmap_vmin, vmax=1.0)
            # Create custom colorbar with physical temperature labels
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            tick_positions = [cmap_vmin, 0.25, 0.5, 0.75, 1.0]
            tick_positions = [t for t in tick_positions if t >= cmap_vmin]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels([f'{bulk_temp + t*(heater_temp-bulk_temp):.0f}°C' for t in tick_positions])
        else:
            if name == 'temperature':
                cmap = 'coolwarm'
            else:
                cmap = 'RdBu'
            
            if rng:
                im = ax.imshow(data, cmap=cmap, origin='lower', vmin=rng[0], vmax=rng[1])
            else:
                im = ax.imshow(data, cmap=cmap, origin='lower')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        title = f'Target: {name}'
        if metrics and name in metrics:
            actual_rng = metrics[name]['Target_Range']
            title += f'\n[{actual_rng[0]:.2f}, {actual_rng[1]:.2f}]'
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    # Add target velocity magnitude plot
    if has_target_velocity and target_velx_data is not None and target_vely_data is not None:
        ax = plt.subplot(num_rows, num_cols, num_cols + num_targets + 1)
        target_vel_mag = np.sqrt(target_velx_data**2 + target_vely_data**2)
        # Priority: 1) command line arg, 2) colorbar_ranges, 3) auto-detect
        if vel_vmax is not None:
            vmax_vel = vel_vmax
        else:
            vel_mag_range = get_range('velocity_magnitude')
            if vel_mag_range is not None:
                vmax_vel = vel_mag_range[1]
            else:
                vmax_vel = target_vel_mag.max()
        # Use PowerNorm (gamma=0.5) to enhance low-mid range visibility
        norm = PowerNorm(gamma=0.5, vmin=vel_vmin, vmax=vmax_vel)
        im = ax.imshow(target_vel_mag, cmap='turbo', norm=norm, origin='lower')
        ax.set_title(f'Target: |vel|\n[{vel_vmin:.1f}, {vmax_vel:.2f}]', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # === Row 3: Predicted Fields ===
    pred_velx_data = None
    pred_vely_data = None
    
    for i, name in enumerate(target_names):
        ax = plt.subplot(num_rows, num_cols, 2 * num_cols + i + 1)
        data = predicted[i].numpy()
        rng = get_range(name)
        
        # Store velocity components for magnitude calculation
        if name == 'velx':
            pred_velx_data = data
        elif name == 'vely':
            pred_vely_data = data
        
        if name == 'temperature' and use_bubbleml_cmap:
            # Convert physical temperature (°C) to [0, 1] range for BubbleML colormap
            data_for_cmap = (data - bulk_temp) / (heater_temp - bulk_temp)
            data_for_cmap = np.clip(data_for_cmap, 0, 1)
            cmap = temp_cmap()
            im = ax.imshow(data_for_cmap, cmap=cmap, origin='lower', vmin=cmap_vmin, vmax=1.0)
            # Create custom colorbar with physical temperature labels
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            tick_positions = [cmap_vmin, 0.25, 0.5, 0.75, 1.0]
            tick_positions = [t for t in tick_positions if t >= cmap_vmin]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels([f'{bulk_temp + t*(heater_temp-bulk_temp):.0f}°C' for t in tick_positions])
        else:
            if name == 'temperature':
                cmap = 'coolwarm'
            else:
                cmap = 'RdBu'
            
            if rng:
                im = ax.imshow(data, cmap=cmap, origin='lower', vmin=rng[0], vmax=rng[1])
            else:
                im = ax.imshow(data, cmap=cmap, origin='lower')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        title = f'Predicted: {name}'
        if metrics and name in metrics:
            actual_rng = metrics[name]['Pred_Range']
            title += f'\n[{actual_rng[0]:.2f}, {actual_rng[1]:.2f}]'
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    # Add predicted velocity magnitude plot (use same vmax as target for comparison)
    if has_target_velocity and pred_velx_data is not None and pred_vely_data is not None:
        ax = plt.subplot(num_rows, num_cols, 2 * num_cols + num_targets + 1)
        pred_vel_mag = np.sqrt(pred_velx_data**2 + pred_vely_data**2)
        # Use same fixed vmax as target for fair comparison
        norm = PowerNorm(gamma=0.5, vmin=vel_vmin, vmax=vmax_vel)
        im = ax.imshow(pred_vel_mag, cmap='turbo', norm=norm, origin='lower')
        ax.set_title(f'Predicted: |vel|\n[{vel_vmin:.1f}, {vmax_vel:.2f}]', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # === Row 4: Difference Maps ===
    for i, name in enumerate(target_names):
        ax = plt.subplot(num_rows, num_cols, 3 * num_cols + i + 1)
        diff = target[i] - predicted[i]
        max_diff = torch.max(torch.abs(diff))
        im = ax.imshow(diff.numpy(), cmap='RdBu', vmin=-max_diff, vmax=max_diff, origin='lower')
        title = f'Diff: {name}'
        if metrics and name in metrics:
            title += f'\nRMSE: {metrics[name]["RMSE"]:.4f}, R²: {metrics[name]["R2"]:.3f}'
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add velocity magnitude difference plot
    if has_target_velocity and target_vel_mag is not None and pred_vel_mag is not None:
        ax = plt.subplot(num_rows, num_cols, 3 * num_cols + num_targets + 1)
        vel_mag_diff = target_vel_mag - pred_vel_mag
        max_mag_diff = np.abs(vel_mag_diff).max()
        im = ax.imshow(vel_mag_diff, cmap='RdBu', vmin=-max_mag_diff, vmax=max_mag_diff, origin='lower')
        rmse_mag = np.sqrt(np.mean(vel_mag_diff**2))
        ax.set_title(f'Diff: |vel|\nRMSE: {rmse_mag:.4f}', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # === Row 5: Streamline Plots (Combined Velocity Field) - Only for Task 2/3 ===
    if has_velocity:
        # Find velx and vely indices in target_names
        velx_idx = None
        vely_idx = None
        for i, name in enumerate(target_names):
            if name == 'velx':
                velx_idx = i
            elif name == 'vely':
                vely_idx = i
        
        if velx_idx is not None and vely_idx is not None:
            # Get velocity components
            target_velx = target[velx_idx].numpy()
            target_vely = target[vely_idx].numpy()
            pred_velx = predicted[velx_idx].numpy()
            pred_vely = predicted[vely_idx].numpy()
            
            # Create meshgrid for streamplot
            H, W = target_velx.shape
            Y, X = np.mgrid[0:H, 0:W]
            
            # Compute velocity magnitude for coloring
            target_vel_mag = np.sqrt(target_velx**2 + target_vely**2)
            pred_vel_mag = np.sqrt(pred_velx**2 + pred_vely**2)
            
            # Use fixed velocity magnitude range if provided, else compute per-frame
            vel_mag_range = get_range('velocity_magnitude')
            if vel_mag_range:
                vmax = vel_mag_range[1]
            else:
                vmax = max(target_vel_mag.max(), pred_vel_mag.max())
            
            # Target streamlines
            ax1 = plt.subplot(num_rows, num_cols, 4 * num_cols + 1)
            # Plot velocity magnitude as background
            im1 = ax1.imshow(target_vel_mag, cmap='plasma', origin='lower', vmin=0, vmax=vmax, alpha=0.7)
            # Add streamlines
            stream1 = ax1.streamplot(X, Y, target_velx, target_vely, 
                                      color='white', density=1.5, linewidth=0.8, 
                                      arrowsize=0.8, arrowstyle='->')
            ax1.set_title(f'Target: Velocity Streamlines\n|V| max: {target_vel_mag.max():.3f}', fontsize=11)
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='|V|')
            
            # Predicted streamlines
            ax2 = plt.subplot(num_rows, num_cols, 4 * num_cols + 2)
            im2 = ax2.imshow(pred_vel_mag, cmap='plasma', origin='lower', vmin=0, vmax=vmax, alpha=0.7)
            stream2 = ax2.streamplot(X, Y, pred_velx, pred_vely, 
                                      color='white', density=1.5, linewidth=0.8, 
                                      arrowsize=0.8, arrowstyle='->')
            ax2.set_title(f'Predicted: Velocity Streamlines\n|V| max: {pred_vel_mag.max():.3f}', fontsize=11)
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='|V|')
            
            # Difference in velocity magnitude
            ax3 = plt.subplot(num_rows, num_cols, 4 * num_cols + 3)
            vel_mag_diff = target_vel_mag - pred_vel_mag
            max_mag_diff = np.abs(vel_mag_diff).max()
            im3 = ax3.imshow(vel_mag_diff, cmap='RdBu', origin='lower', 
                             vmin=-max_mag_diff, vmax=max_mag_diff)
            ax3.set_title(f'Diff: Velocity Magnitude\nRMSE: {np.sqrt(np.mean(vel_mag_diff**2)):.4f}', fontsize=11)
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Δ|V|')
    
    # Note: tight_layout() is not needed when using constrained_layout=True
    
    # Use actual model type for filename suffix
    model_suffix = model_type.replace('_', '_')  # Keep underscores for consistency
    task_suffix = task_name.replace('_', '-')
    input_suffix = "clean" if use_clean_inputs else "noisy" if task_name == 'noisy_velocity_from_interface' else ""
    
    if input_suffix:
        filename = f'{task_suffix}_{input_suffix}_{model_suffix}_sample_{sample_idx}.png'
    else:
        filename = f'{task_suffix}_{model_suffix}_sample_{sample_idx}.png'
    
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✓ Visualization saved to: {output_path}")
    return output_path


def print_detailed_metrics(metrics, sample_idx, target_names):
    """Print detailed metrics in a formatted way."""
    
    print(f"\n" + "="*70)
    print(f"DETAILED METRICS FOR SAMPLE {sample_idx}")
    print("="*70)
    
    # Overall metrics
    overall = metrics['Overall']
    print(f"\n📊 OVERALL METRICS (all channels):")
    print(f"  MSE:          {overall['MSE']:.6f}")
    print(f"  MAE:          {overall['MAE']:.6f}")
    print(f"  RMSE:         {overall['RMSE']:.6f}")
    print(f"  Relative L2:  {overall['Relative_L2']:.6f}")
    
    # Per-channel metrics
    for name in target_names:
        if name in metrics:
            ch_metrics = metrics[name]
            print(f"\n📈 {name.upper()} METRICS:")
            print(f"  MSE:          {ch_metrics['MSE']:.6f}")
            print(f"  MAE:          {ch_metrics['MAE']:.6f}")
            print(f"  RMSE:         {ch_metrics['RMSE']:.6f}")
            print(f"  Relative L2:  {ch_metrics['Relative_L2']:.6f}")
            print(f"  R² Score:     {ch_metrics['R2']:.4f}")
            print(f"  Correlation:  {ch_metrics['Correlation']:.4f}")
            print(f"  Target Range: [{ch_metrics['Target_Range'][0]:.3f}, {ch_metrics['Target_Range'][1]:.3f}]")
            print(f"  Pred Range:   [{ch_metrics['Pred_Range'][0]:.3f}, {ch_metrics['Pred_Range'][1]:.3f}]")
            
            # Quality assessment
            r2 = ch_metrics['R2']
            if r2 > 0.9:
                quality = "Excellent ⭐⭐⭐"
            elif r2 > 0.7:
                quality = "Good ⭐⭐"
            elif r2 > 0.5:
                quality = "Fair ⭐"
            else:
                quality = "Poor"
            print(f"  Quality:      {quality}")


def generate_multiple_plots(model, dataset, sample_indices, target_names, conditioning_names,
                           device='cuda', num_inference_steps=50, save_dir="./inference_output",
                           model_type='flow_matching', task_name='velocity_from_interface',
                           use_clean_inputs=False, wall_temp=96.0, initial_state_mode='from_data',
                           use_bubbleml_cmap=False, bulk_temp=48.3, heater_temp=114.7, cmap_vmin=0.03, 
                           vel_vmin=0.1, vel_vmax=None, vel_interface_vmax=None, dpi=150,
                           history_length=10, bootstrap_ablation=None):
    """
    Generate plots for multiple sample indices with consistent colorbar ranges.
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        sample_indices: List of sample indices to process
        target_names: List of target field names
        conditioning_names: List of conditioning field names
        device: Device to run on
        num_inference_steps: Number of inference steps
        save_dir: Output directory
        model_type: Model type string
        task_name: Task name string
        use_clean_inputs: Whether clean inputs were used
        wall_temp: Wall temperature in Celsius (for temperature colorbar range)
        initial_state_mode: Mode for initializing prev_output for AR models
        use_bubbleml_cmap: If True, use custom BubbleML temperature colormap
        bulk_temp: Bulk temperature in Celsius (default: 48.3°C)
        heater_temp: Heater temperature in Celsius (default: 114.7°C)
        cmap_vmin: Minimum value for BubbleML colormap (0-1 range)
        vel_vmin: Minimum value for velocity magnitude colormap
        vel_vmax: Fixed max value for velocity magnitude (for cross-model comparison)
        vel_interface_vmax: Fixed max value for interface velocity magnitude
    """
    
    print(f"\n🎬 GENERATING PLOTS FOR MULTIPLE SAMPLES")
    print("-" * 50)
    
    # First pass: collect all results to compute colorbar ranges
    print("  📊 Pass 1: Running inference on all samples...")
    trajectory_results = []
    valid_sample_indices = []
    
    for i, sample_idx in enumerate(sample_indices):
        try:
            input_data, target, predicted = run_inference_on_sample(
                model, dataset, sample_idx, device, num_inference_steps, model_type,
                initial_state_mode=initial_state_mode, history_length=history_length,
                bootstrap_ablation=bootstrap_ablation
            )
            trajectory_results.append((input_data, target, predicted))
            valid_sample_indices.append(sample_idx)
            print(f"    ✓ Sample {sample_idx} ({i+1}/{len(sample_indices)})")
        except Exception as e:
            print(f"    ❌ Error processing sample {sample_idx}: {e}")
            continue
    
    if not trajectory_results:
        print("❌ No valid samples processed")
        return [], []
    
    # Compute fixed colorbar ranges across all samples
    colorbar_ranges = compute_colorbar_ranges(
        trajectory_results, target_names, conditioning_names,
        wall_temp=wall_temp  # temp_min auto-detected from data
    )
    
    # Second pass: generate plots with fixed colorbar ranges
    print("\n  📊 Pass 2: Generating plots with fixed colorbar ranges...")
    plot_paths = []
    all_metrics = []
    
    for i, (sample_idx, (input_data, target, predicted)) in enumerate(zip(valid_sample_indices, trajectory_results)):
        try:
            # Compute metrics
            metrics = compute_metrics(target, predicted, target_names)
            all_metrics.append(metrics)
            
            # Create visualization with fixed colorbar ranges
            plot_path = create_comprehensive_visualization(
                input_data, target, predicted, target_names, conditioning_names,
                sample_idx, metrics, save_dir, model_type, task_name, use_clean_inputs,
                colorbar_ranges=colorbar_ranges,
                use_bubbleml_cmap=use_bubbleml_cmap, bulk_temp=bulk_temp,
                heater_temp=heater_temp, cmap_vmin=cmap_vmin, vel_vmin=vel_vmin,
                vel_vmax=vel_vmax, vel_interface_vmax=vel_interface_vmax, dpi=dpi
            )
            plot_paths.append(plot_path)
            
            print(f"    ✓ Sample {sample_idx}: Overall Rel L2 = {metrics['Overall']['Relative_L2']:.4f}")
            
        except Exception as e:
            print(f"    ❌ Error creating visualization for sample {sample_idx}: {e}")
            continue
    
    return plot_paths, all_metrics


def create_gif_from_plots(plot_paths, gif_path, duration=1000, loop=0):
    """Create an animated GIF from a list of plot image paths.
    
    All images are resized to the same dimensions (max width x max height)
    to ensure consistent frame sizes and prevent shifting in the animation.
    """
    
    if not plot_paths:
        print("❌ No plot paths provided for GIF creation")
        return None
    
    print(f"\n🎬 CREATING GIF ANIMATION")
    print("-" * 30)
    
    try:
        # First pass: load images and find maximum dimensions
        raw_images = []
        for i, plot_path in enumerate(plot_paths):
            if os.path.exists(plot_path):
                img = Image.open(plot_path)
                raw_images.append(img)
                print(f"  ✓ Loaded image {i+1}/{len(plot_paths)}")
        
        if not raw_images:
            print("❌ No valid images found")
            return None
        
        # Find the maximum dimensions across all images
        max_width = max(img.width for img in raw_images)
        max_height = max(img.height for img in raw_images)
        print(f"  📐 Target frame size: {max_width} x {max_height}")
        
        # Second pass: resize all images to the same dimensions
        # Paste each image onto a white canvas of the target size (centered)
        images = []
        for i, img in enumerate(raw_images):
            if img.width != max_width or img.height != max_height:
                # Create a white canvas of the target size
                canvas = Image.new('RGB', (max_width, max_height), (255, 255, 255))
                # Calculate position to center the image
                x_offset = (max_width - img.width) // 2
                y_offset = (max_height - img.height) // 2
                # Paste the original image onto the canvas
                canvas.paste(img, (x_offset, y_offset))
                images.append(canvas)
            else:
                # Convert to RGB if needed (for consistency)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
        
        print(f"  ✓ All {len(images)} frames resized to {max_width} x {max_height}")
        
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )
        
        print(f"✅ GIF created: {gif_path}")
        return gif_path
        
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")
        return None


def print_summary_metrics(all_metrics, sample_indices, target_names):
    """Print summary statistics across all processed samples."""
    
    if not all_metrics:
        return
    
    print(f"\n📊 SUMMARY METRICS ACROSS {len(all_metrics)} SAMPLES")
    print("=" * 70)
    
    # Overall metrics
    rel_l2_scores = [m['Overall']['Relative_L2'] for m in all_metrics]
    print(f"\n🎯 OVERALL RELATIVE L2:")
    print(f"  Mean: {np.mean(rel_l2_scores):.4f} ± {np.std(rel_l2_scores):.4f}")
    print(f"  Range: [{np.min(rel_l2_scores):.4f}, {np.max(rel_l2_scores):.4f}]")
    
    # Per-channel summary
    for name in target_names:
        r2_scores = [m[name]['R2'] for m in all_metrics if name in m]
        rmse_scores = [m[name]['RMSE'] for m in all_metrics if name in m]
        
        print(f"\n📈 {name.upper()}:")
        print(f"  R² Mean:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        print(f"  RMSE Mean: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        
        # Quality distribution
        excellent = sum(1 for r2 in r2_scores if r2 > 0.9)
        good = sum(1 for r2 in r2_scores if 0.7 < r2 <= 0.9)
        fair = sum(1 for r2 in r2_scores if 0.5 < r2 <= 0.7)
        poor = sum(1 for r2 in r2_scores if r2 <= 0.5)
        total = len(r2_scores)
        print(f"  Quality: Excellent={excellent}/{total}, Good={good}/{total}, Fair={fair}/{total}, Poor={poor}/{total}")


def plot_metrics_trends(all_metrics, sample_indices, target_names, output_dir, 
                        model_type='flow_matching', task_name='velocity_from_interface',
                        use_clean_inputs=False):
    """
    Plot Rel L2 metrics trends for overall and per-channel metrics.
    
    Args:
        all_metrics: List of metric dictionaries from compute_metrics
        sample_indices: List of sample indices
        target_names: List of target field names (e.g., ['velx', 'vely', 'temperature'])
        output_dir: Directory to save plots
        model_type: Model type for filename
        task_name: Task name for filename
        use_clean_inputs: Whether clean inputs were used
    """
    if not all_metrics or len(all_metrics) == 0:
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare sample IDs (use sample_indices if available, otherwise use sequential)
    if len(sample_indices) == len(all_metrics):
        sample_ids = np.array(sample_indices)
    else:
        sample_ids = np.array(range(len(all_metrics)))
    
    # Sort by sample_id to ensure proper ordering
    sort_idx = np.argsort(sample_ids)
    sample_ids_sorted = sample_ids[sort_idx]
    
    # Helper function to plot a single metric
    def plot_single_metric(sample_ids, metric_values, metric_name, field_name="Overall"):
        """Plot a single metric with smoothed trend."""
        if len(metric_values) == 0:
            return None
        
        metric_values_sorted = np.array(metric_values)[sort_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Compute smoothed trend line
        smoothed_values = smooth_data(sample_ids_sorted, metric_values_sorted)
        
        # Plot smoothed trend line in the background
        ax.plot(sample_ids_sorted, smoothed_values, '-', linewidth=10, 
                alpha=0.4, color='#7CFC00', label='Smoothed Trend', zorder=1)
        
        # Plot the data points on top
        ax.plot(sample_ids_sorted, metric_values_sorted, 'o-', markersize=4, 
                linewidth=1.5, alpha=0.7, color='#2E86AB', label=metric_name, zorder=2)
        
        # Add statistics
        mean_val = np.mean(metric_values_sorted)
        std_val = np.std(metric_values_sorted)
        min_val = np.min(metric_values_sorted)
        max_val = np.max(metric_values_sorted)
        
        # Add mean line
        ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.4f}')
        
        # Add ±1 std lines
        ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, 
                   alpha=0.7, label=f'Mean ± Std: {mean_val:.4f} ± {std_val:.4f}')
        ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Labels and title
        ax.set_xlabel('Sample ID (Frame Number)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} vs Sample ID ({field_name})\n'
                     f'Mean: {mean_val:.4f} ± {std_val:.4f}, Range: [{min_val:.4f}, {max_val:.4f}]',
                     fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(loc='best', fontsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    # Create filename suffix
    task_suffix = task_name.replace('_', '-')
    input_suffix = "clean" if use_clean_inputs else "noisy" if task_name == 'noisy_velocity_from_interface' else ""
    if input_suffix:
        filename_prefix = f'{task_suffix}_{input_suffix}_{model_type}'
    else:
        filename_prefix = f'{task_suffix}_{model_type}'
    
    plots_created = []
    
    # Plot Overall Rel L2
    overall_rel_l2 = [m['Overall']['Relative_L2'] for m in all_metrics]
    fig = plot_single_metric(sample_ids, overall_rel_l2, 'Overall Rel L2', 'Overall')
    if fig:
        output_path = os.path.join(output_dir, f'{filename_prefix}_overall_rel_l2_trend.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plots_created.append(output_path)
        print(f"  ✓ Saved overall Rel L2 trend plot: {output_path}")
    
    # Plot per-channel Rel L2
    for name in target_names:
        if name in all_metrics[0]:  # Check if this channel exists in metrics
            channel_rel_l2 = [m[name]['Relative_L2'] for m in all_metrics if name in m]
            if len(channel_rel_l2) == len(sample_ids):
                fig = plot_single_metric(sample_ids, channel_rel_l2, f'{name.capitalize()} Rel L2', name)
                if fig:
                    output_path = os.path.join(output_dir, f'{filename_prefix}_{name}_rel_l2_trend.png')
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plots_created.append(output_path)
                    print(f"  ✓ Saved {name} Rel L2 trend plot: {output_path}")
    
    if plots_created:
        print(f"\n📈 Created {len(plots_created)} metric trend plots")
    
    return plots_created


def parse_sample_indices(samples_str, dataset_length):
    """Parse sample indices from string format.
    
    Supports:
    - Range: "0-100" -> [0, 1, ..., 100]
    - Comma-separated: "0,5,10,15" -> [0, 5, 10, 15]
    - Single index: "100" -> [100]
    - "all" -> all indices [0, 1, ..., dataset_length-1]
    """
    try:
        if samples_str.lower() == 'all':
            return list(range(dataset_length))
        elif '-' in samples_str and ',' not in samples_str:
            start, end = map(int, samples_str.split('-'))
            return list(range(start, min(end + 1, dataset_length)))
        elif ',' in samples_str:
            indices = [int(x.strip()) for x in samples_str.split(',')]
            return [idx for idx in indices if idx < dataset_length]
        else:
            idx = int(samples_str)
            return [idx] if idx < dataset_length else []
    except ValueError as e:
        print(f"❌ Error parsing indices: {e}")
        return []


def main():
    """Main inference function for Task 1, 2 & 3."""
    
    parser = argparse.ArgumentParser(
        description='Task 1, 2 & 3 Inference: Temperature from SDF (Task 1), Velocity + Temperature from Interface (Task 2/3)'
    )
    parser.add_argument('sample_idx', nargs='?', type=int, default=100,
                       help='Sample index for single inference')
    parser.add_argument('checkpoint_path', nargs='?', 
                        # default=None,
                        #  default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_32_256_2_False_ar_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820981/checkpoints/last.ckpt",
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
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/bubble_ddpm_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47848600checkpoints/last.ckpt",
                        # ICML
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
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861270/checkpoints/epoch=43-step=036520.ckpt",
                        # ablation_2: history_length=5, use_availability_mask=true, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist5_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861275/checkpoints/last.ckpt",
                        # ablation_3: history_length=20, use_availability_mask=true, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861276/checkpoints/last.ckpt",
                        # ablation_7: history_length=40, use_availability_mask=true, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist40_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861282/checkpoints/last.ckpt",
                        # ablation_4: history_length=10, use_availability_mask=false, training_strategy=push_forward
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861277/checkpoints/last.ckpt",
                        # ablation_5: history_length=10, use_availability_mask=true, training_strategy=teacher_forcing
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861278/checkpoints/epoch=22-step=019090.ckpt",
                        # ablation_6: history_length=10, use_availability_mask=true, training_strategy=scheduled_sampling
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861279/checkpoints/epoch=41-step=034860.ckpt",
                        # ablation_8: history_length=20, use_availability_mask=false, training_strategy=scheduled_sampling
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47955165/checkpoints/last.ckpt",
                        # ablation A: history_length=20, use_availability_mask=true, training_strategy=scheduled_sampling
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50221468/checkpoints/last.ckpt",
                        # ablation B: history_length=20, use_availability_mask=true, training_strategy=scheduled_sampling
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50221470/checkpoints/epoch=07-step=006640.ckpt",
                        # ablation C: history_length=10, use_availability_mask=false, training_strategy=scheduled_sampling
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50221555/checkpoints/epoch=09-step=008300.ckpt",
                        
                        
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_48673519/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_48671238/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_48672665/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_48673527/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_48729794/checkpoints/last.ckpt",
                        
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_48790696/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_norm_temperature_only_pb_subcooled_singlestep_none_ds4_48790722/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_norm_none_pb_subcooled_singlestep_none_ds4_48790727/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_loss_l1_pb_subcooled_singlestep_none_ds4_48790763/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_loss_huber_pb_subcooled_singlestep_none_ds4_48790772/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_loss__pb_subcooled_singlestep_none_ds4_48790781/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_loss_relative_l1_pb_subcooled_singlestep_none_ds4_48790787/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_jit_NA_NA_NA_NA_velocity_from_interface_loss_relative_l1_pb_subcooled_singlestep_none_ds4_48790794/checkpoints/last.ckpt",
                        
                        
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_NA_velocity_from_interface_norm_temperature_only_pb_subcooled_singlestep_none_ds4_48790816/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_32_256_2_NA_velocity_from_interface_loss_huber_pb_subcooled_singlestep_none_ds4_48790813/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/edm_NA_NA_NA_NA_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50009220/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/diffusionpde_ch32_b2_s50_zobs1.0_zpde0.5_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50073800/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/edm_ar_bootstrap_ch32_b2_hist10_roll5_tmix_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50223790/checkpoints/epoch=09-step=016600.ckpt",
                        
                        # Ab1: history_length=10, use_availability_mask=true, training_strategy=teacher_forcing
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230563/checkpoints/last.ckpt",
                        # Ab2: history_length=10, use_availability_mask=false, training_strategy=teacher_forcing
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230567/checkpoints/last.ckpt",
                        # Ab3: history_length=5, use_availability_mask=true, training_strategy=teacher_forcing
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist5_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230575/checkpoints/last.ckpt",
                        # Ab4: history_length=20, use_availability_mask=true, training_strategy=teacher_forcing
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230602/checkpoints/epoch=09-step=008300.ckpt",
                        # Ab5: history_length=20, use_availability_mask=false, training_strategy=teacher_forcing
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230606/checkpoints/epoch=09-step=008300.ckpt",
                        # ab6: history_length=30, use_availability_mask=true, training_strategy=scheduled_sampling
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist30_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230609/checkpoints/last.ckpt",
                        # ab7: history_length=40, use_availability_mask=true, training_strategy=scheduled_sampling
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist40_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230611/checkpoints/last.ckpt",
                        
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50271127/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/edm_ar_bootstrap_ch32_b2_hist10_roll5_attn_d256_L4_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50383372/checkpoints/last.ckpt",
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/unet_32_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47845919/checkpoints/last.ckpt",
                        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/ffno_m12_w64_l4_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849889/checkpoints/last.ckpt",
                        help='Path to model checkpoint (or use --find-checkpoint)')
    parser.add_argument('data_file_path', nargs='?',
                       default="/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5",
                    #    default="/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_117.hdf5",
                       help='Path to data file')
    parser.add_argument('--task', type=str, default='auto',
                       choices=['temperature_from_sdf', 'velocity_from_interface', 'noisy_velocity_from_interface', 'auto'],
                       help='Task name: temperature_from_sdf (Task 1), velocity_from_interface (Task 2), '
                            'noisy_velocity_from_interface (Task 3), or auto (detect from checkpoint path)')
    parser.add_argument('--use-clean-inputs', action='store_true',
                        default=True,
                       help='For Task 3: Use clean inputs instead of noisy (physics fidelity check)')
    
    # Inference specific arguments
    parser.add_argument('--num-inference-steps', type=int, default=50,
                       help='Number of integration/sampling steps (use 50 for flow_matching, 200+ for ve_sde)')
    parser.add_argument('--output-dir', 
                        # default='./ICML/Bootstrap_HE_test/ab5_10_true_tf/epoch22',
                        # default='./ICML/test_edm_ar_bootstrap/epoch9',
                        # default='./ICML/Bootstrap_HE_tf_test/ab5_20_false_tf/epoch9',
                        # default='./ICML/Boostrap_attention/10_true_tf/zeros',
                        # default='./ICML/Bootstrap_HE_test/abC_10_true_ss',
                        # default='./ICML/AR_Zero_mean_abaltion/mean',
                        default='./ICML/FFNO_test/',
                        # default='./loss_ablation_results/ar_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820981',
                        # default='./loss_ablation_results/ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820980',
                        # default='./loss_ablation_results/ar_ss_div_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820985',
                        # default='./loss_ablation_results/ar_ss_div_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820984',
                        # default='./loss_ablation_results/ar_ss_vort_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820992',
                        # default='./loss_ablation_results/ar_ss_vort_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820993',
                        # default='./loss_ablation_results/ar_ss_adv_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820994',
                        # default='./loss_ablation_results/ar_ss_adv_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820995',
                        # default='./loss_ablation_results/ar_ss_div_vort_adv_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47821002',
                        # default='./loss_ablation_results/ar_ss_div_vort_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47821001',
                        # default='./loss_ablation_results/ar_ss_div_vort_adv_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47820993',
                        
                        # bootstrap ablation
                        # ablation_1: history_length=10, use_availability_mask=true, training_strategy=push_forward
                        # default='./ICML/Level_4/bootstrap_ablation_1_10_true_pf_47861270',
                        # ablation_2: history_length=5, use_availability_mask=true, training_strategy=push_forward
                        # default='./ICML/Level_4/bootstrap_ablation_2_5_true_pf_47861275',
                        # ablation_3: history_length=20, use_availability_mask=true, training_strategy=push_forward
                        # default='./ICML/Level_4/bootstrap_ablation_3_20_true_pf_47861276',
                        # ablation_7: history_length=40, use_availability_mask=true, training_strategy=push_forward
                        # default='./ICML/Level_4/bootstrap_ablation_7_40_true_pf_47861282',
                        # ablation_4: history_length=10, use_availability_mask=false, training_strategy=push_forward
                        # default='./ICML/Level_4/bootstrap_ablation_4_10_false_pf_47861277',
                        # ablation_5: history_length=10, use_availability_mask=true, training_strategy=teacher_forcing
                        # default='./ICML/Level_4/bootstrap_ablation_5_10_true_tf_47861278',
                        # ablation_6: history_length=10, use_availability_mask=true, training_strategy=scheduled_sampling
                        # default='./ICML/Level_4/bootstrap_ablation_6_10_true_ss_47861279',
                        # ablation_8: history_length=20, use_availability_mask=false, training_strategy=scheduled_sampling
                        # default='./ICML/Level_4/bootstrap_ablation_8_20_false_ss_47955165',
                        
                        # Level 1
                        # default='./ICML/Level_1/Task_1/flow_matching_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47841592',
                        # default='./ICML/Level_1/Task_2/flow_matching_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47835444',
                        # default='./ICML/Level_1/Task_1/unet_32_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47842727',
                        # default='./ICML/Level_1/Task_2/unet_32_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47845919',
                        # default='./ICML/Level_1/Task_1/bubble_ddpm_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849371',
                        # default = './ICML/Level_1/Task_2/bubble_ddpm_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849378',
                        # default = './ICML/Level_1/Task_1/ve_sde_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849739',
                        # default = './ICML/Level_1/Task_2/ve_sde_32_256_2_False_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849759',
                        # default = './ICML/Level_1/Task_1/ffno_m12_w64_l4_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849886',
                        # default = './ICML/Level_1/Task_2/ffno_m12_w64_l4_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47849889',
                        # default = './ICML/Level_1/Task_1/edm_ch32_b2_s50_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47852082',
                        # default = './ICML/Level_1/Task_2/edm_ch32_b2_s50_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47853227',
                        
                        # Level 2
                        # default='./ICML/Level_2/Task_1/flow_matching_ar_32_256_2_False_ar_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47856390',
                        # default='./ICML/Level_2/Task_2/flow_matching_ar_32_256_2_False_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856403',
                        # default='./ICML/Level_2/Task_1/unet_ar_32_ar_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47856451',
                        # default='./ICML/Level_2/Task_2/unet_ar_32_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856472',
                        
                        # Level 3
                        # default='./ICML/Level_3/Task_1/flow_matching_ar_32_256_2_False_ar_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47856390',
                        # default='./ICML/Level_3/Task_2/flow_matching_ar_32_256_2_False_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856403',
                        # default='./ICML/Level_3/Task_1/unet_ar_32_ar_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47856451',
                        # default='./ICML/Level_3/Task_2/unet_ar_32_ar_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47856472',
                       
                        # fm_ablation_all
                        # default='./ICML/fm_ablation/fm_ablation_all_47877420',
                        # fm_ablation_skip
                        # default='./ICML/fm_ablation/fm_ablation_skip_47877434',
                        # fm_ablation_adaptive
                        # default='./ICML/fm_ablation/fm_ablation_adaptive_47877548',
                        # fm_ablation_attention
                        # default='./ICML/fm_ablation/fm_ablation_attention_47877559',
                        # fm_ablation_baseline
                        # default='./ICML/fm_ablation/fm_ablation_baseline_47877572',
                       help='Output directory for visualizations')
    parser.add_argument('--samples', type=str, default="100-120",
                       help='Sample range to process (e.g., "0-100", "0,5,10,15", or "all" for all frames). Overrides --gif-samples.')
    parser.add_argument('--generate-gif', action='store_true', default=False,
                       help='Generate GIF animation for multiple samples (requires --samples or --gif-samples)')
    parser.add_argument('--gif-samples', type=str, default='100-140',
                       help='Sample range for GIF (e.g., "0-20" or "0,5,10,15"). Used if --samples not specified.')
    parser.add_argument('--gif-duration', type=int, default=1000,
                       help='Duration per frame in milliseconds')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures (default: 150)')
    
    # BubbleML temperature colormap settings
    parser.add_argument('--use-bubbleml-cmap', action='store_true', default=True,
                       help='Use BubbleML custom temperature colormap instead of coolwarm')
    parser.add_argument('--vmin', type=float, default=0,
                       help='Min value for BubbleML colormap (0-1 range). Set to 0.02-0.05.')
    parser.add_argument('--vel-vmin', type=float, default=0,
                       help='Min value for velocity magnitude colormap. Set to 0.05-0.2.')
    parser.add_argument('--vel-vmax', type=float, default=None,
                       help='Fixed max value for velocity magnitude colormap. Set to a consistent value (e.g., 4.0) '
                            'for comparing across models. If None, auto-detect from ground truth data.')
    parser.add_argument('--vel-interface-vmax', type=float, default=None,
                       help='Fixed max value for interface velocity magnitude colormap. '
                            'If None, auto-detect from ground truth data.')
    parser.add_argument('--bulk-temp', type=float, default=None,
                       help='Bulk temperature in Celsius for colormap scaling. '
                            'If None, auto-detect from normalization_stats.json (default: ~48.3°C)')
    parser.add_argument('--heater-temp', type=float, default=None,
                       help='Heater temperature in Celsius for colormap scaling. '
                            'If None, auto-detect from normalization_stats.json (default: ~114.7°C)')
    
    parser.add_argument('--start-time', type=int, default=100,
                       help='Starting timestep for dataset')
    parser.add_argument('--downsample-factor', type=int, default=4,
                       help='Downsampling factor for fast prototyping (1=full res, 4=128x128)')
    parser.add_argument('--normalize-temperature', action='store_true', default=False)
    parser.add_argument('--no-normalize-temperature', dest='normalize_temperature',
                        action='store_false')
    parser.add_argument('--normalization-stats', type=str, default="/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json",
                       help='Path to normalization_stats.json file (overrides auto-detection from checkpoint directory)')
    parser.add_argument('--find-checkpoint', type=str, default=None, metavar='LOG_DIR',
                       help='Auto-find checkpoint in log directory by task name')
    parser.add_argument('--list-runs', type=str, default=None, metavar='LOG_DIR',
                       help='List all training runs for the specified task')
    parser.add_argument('--model-type', type=str, default='auto',
                       choices=['flow_matching', 'flow_matching_jit', 'flow_matching_history', 've_sde', 'flow_matching_ar', 'flow_matching_ar_bootstrap', 'edm_ar_bootstrap', 'unet', 'unet_ar', 'bubble_ddpm', 'ffno', 'edm', 'diffusionpde', 'auto'],
                       help='Model type: flow_matching, flow_matching_jit, flow_matching_history, flow_matching_ar, flow_matching_ar_bootstrap, edm_ar_bootstrap, unet, unet_ar, ve_sde, bubble_ddpm, ffno, edm, diffusionpde, or auto (detect from path)')
    parser.add_argument('--norm-mode', type=str, default='none',
                    choices=['none', 'all', 'temperature_only'],
                    help='Normalization mode: none, all (default), or temperature_only. Must match training setting.')
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
    
    
    # Temporal model specific arguments
    parser.add_argument('--history-length', type=int, default=10,
                       help='Number of historical frames for temporal model, bootstrap conditioning, '
                            'or from_history_mean initial state mode')
    parser.add_argument('--history-stride', type=int, default=1,
                       help='Stride between history frames for bootstrap (1=consecutive, 2=every other). '
                            'Auto-detected from checkpoint if available.')
    parser.add_argument('--rollout-length', type=int, default=5,
                       help='Number of frames in rollout segment for bootstrap model')
    parser.add_argument('--history-encoder-type', type=str, default='temporal_mixer',
                       choices=['conv3d', 'temporal_mixer', 'attention'],
                       help='History encoder type for bootstrap model: conv3d (expressive), temporal_mixer (fast), or attention (transformer)')
    
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
    
    parser.add_argument('--temporal-stride', type=int, default=2,
                       help='Stride between frames for temporal model')
    parser.add_argument('--temporal-hidden-dim', type=int, default=32,
                       help='S4 hidden dimension for temporal model')
    parser.add_argument('--temporal-d-state', type=int, default=16,
                       help='S4 state dimension for temporal model')
    parser.add_argument('--temporal-n-layers', type=int, default=2,
                       help='Number of S4 layers for temporal model')
    
    # Temporal consistency model specific arguments
    parser.add_argument('--temporal-consistency-weight', type=float, default=0.2,
                       help='Temporal consistency weight (λ) - overrides checkpoint if specified')
    parser.add_argument('--temporal-gradient-weight', type=float, default=0.1,
                       help='Temporal gradient weight - overrides checkpoint if specified')
    parser.add_argument('--temporal-loss-type', type=str, default='l1',
                       choices=['l1', 'l2', 'smooth_l1'],
                       help='Temporal loss type - overrides checkpoint if specified')
    parser.add_argument('--frame-gap', type=int, default=1,
                       help='Frame gap for temporal consistency - overrides checkpoint if specified')
    
    
    # AR model specific arguments
    parser.add_argument('--autoregressive-rollout', action='store_true', default=True,
                       help='For AR models: use true autoregressive rollout (model predictions) instead of teacher forcing')
    parser.add_argument('--solver', type=str, default='heun',
                       choices=['euler', 'heun', 'midpoint', 'rk4'],
                       help='ODE solver for flow matching (heun recommended for quality)')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                       help='Classifier-free guidance scale (1.0=none, 1.5-2.0=sharper outputs)')
    parser.add_argument('--initial-state-mode', type=str, default='from_history_mean',
                       choices=['from_data', 'from_history_mean', 'zeros', 'small_noise', 'from_conditioning'],
                       help='Initial state mode for AR models. '
                            'from_data: use ground truth prev_output from dataset (default). '
                            'from_history_mean: use mean of previous N ground truth frames (N = --history-length). '
                            'zeros: use zeros (neutral state for normalized data, no GT needed). '
                            'small_noise: small random noise (no GT needed). '
                            'from_conditioning: derive from SDF/interface velocity (no GT needed).')
    parser.add_argument('--residual-prediction', action='store_true', default=False,
                       help='Use residual/delta prediction mode (predict delta from previous state)')
    parser.add_argument('--no-residual-prediction', dest='residual_prediction', action='store_false',
                       help='Use absolute prediction mode (predict full state)')
    
    # Bootstrap ablation (for flow_matching_ar_bootstrap / edm_ar_bootstrap)
    parser.add_argument('--bootstrap-ablation', type=str, default=None,
                       choices=['zeros', 'mean_conditioning_naive'],
                       help='Bootstrap ablation mode for AR bootstrap models. '
                            'Replaces the learned history encoder output with a simpler initialization: '
                            'zeros: use all-zero initial state. '
                            'mean_conditioning_naive: average conditioning history and map interface vel to bulk vel.')
    
    # Random seed for reproducibility
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility. If set, inference will be deterministic. '
                            'Same seed + same model = identical results every run.')
    parser.add_argument('--no-seed', dest='seed', action='store_const', const=None,
                       help='Disable seed (default behavior, stochastic inference)')
    
    # VE-SDE specific arguments
    parser.add_argument('--sigma-min', type=float, default=0.01,
                       help='Minimum sigma for VE-SDE')
    parser.add_argument('--sigma-max', type=float, default=1.0,
                       help='Maximum sigma for VE-SDE')
    parser.add_argument('--sampling-method', type=str, default='pc',
                       choices=['pc', 'ode', 'euler'],
                       help='Sampling method for VE-SDE (ode is more stable than pc)')
    parser.add_argument('--snr', type=float, default=0.16,
                       help='SNR for VE-SDE PC sampling')
    
    args = parser.parse_args()
    
    # Handle --list-runs option
    if args.list_runs:
        find_all_task_checkpoints(args.list_runs, args.task)
        return
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint_path
    
    # Auto-detect model type from checkpoint path
    model_type = args.model_type
    if model_type == 'auto' or 've_sde' in checkpoint_path.lower() or 'vesde' in checkpoint_path.lower():
        # Auto-detect from checkpoint path
        if 've_sde' in checkpoint_path.lower() or 'vesde' in checkpoint_path.lower():
            print("🔍 Auto-detected VE-SDE model from checkpoint path")
            model_type = 've_sde'
        elif 'unet_ar' in checkpoint_path.lower():
            print("🔍 Auto-detected Autoregressive UNet model from checkpoint path")
            model_type = 'unet_ar'
        elif 'unet' in checkpoint_path.lower() and 'unet_ar' not in checkpoint_path.lower():
            print("🔍 Auto-detected UNet model from checkpoint path")
            model_type = 'unet'
        elif 'ffno' in checkpoint_path.lower():
            print("🔍 Auto-detected FFNO model from checkpoint path")
            model_type = 'ffno'
        elif 'edm_ar_bootstrap' in checkpoint_path.lower():
            print("🔍 Auto-detected EDM AR Bootstrap model from checkpoint path")
            model_type = 'edm_ar_bootstrap'
        elif 'flow_matching_ar_bootstrap' in checkpoint_path.lower() or 'ar_bootstrap' in checkpoint_path.lower():
            print("🔍 Auto-detected Autoregressive Flow Matching with Bootstrap model from checkpoint path")
            model_type = 'flow_matching_ar_bootstrap'
        elif 'flow_matching_ar' in checkpoint_path.lower() or '_ar_' in checkpoint_path.lower():
            print("🔍 Auto-detected Autoregressive Flow Matching model from checkpoint path")
            model_type = 'flow_matching_ar'
        elif 'bubble_ddpm' in checkpoint_path.lower() or 'ddpm' in checkpoint_path.lower():
            print("🔍 Auto-detected DDPM model from checkpoint path")
            model_type = 'bubble_ddpm'
        elif 'diffusionpde' in checkpoint_path.lower():
            print("🔍 Auto-detected DiffusionPDE model from checkpoint path")
            model_type = 'diffusionpde'
        elif 'edm' in checkpoint_path.lower():
            print("🔍 Auto-detected EDM model from checkpoint path")
            model_type = 'edm'
        elif 'flow_matching_improved' in checkpoint_path.lower():
            print("🔍 Auto-detected Improved Flow Matching model from checkpoint path")
            model_type = 'flow_matching_improved'
        elif 'flow_matching_jit' in checkpoint_path.lower():
            print("🔍 Auto-detected JiT Flow Matching model from checkpoint path")
            model_type = 'flow_matching_jit'
        elif 'flow_matching_history' in checkpoint_path.lower():
            print("🔍 Auto-detected History-Window Flow Matching model from checkpoint path")
            model_type = 'flow_matching_history'
        elif 'flow_matching' in checkpoint_path.lower():
            print("🔍 Auto-detected Flow Matching model from checkpoint path")
            model_type = 'flow_matching'
        else:
            # Default fallback
            print("⚠️  Could not auto-detect model type, defaulting to flow_matching")
            model_type = 'flow_matching'
    
    # Auto-detect task from checkpoint path if not specified
    if args.task == 'auto':
        if 'temperature_from_sdf' in checkpoint_path.lower():
            args.task = 'temperature_from_sdf'
            print("🔍 Auto-detected Task 1 (temperature_from_sdf) from checkpoint path")
        elif 'noisy_velocity_from_interface' in checkpoint_path.lower():
            args.task = 'noisy_velocity_from_interface'
            print("🔍 Auto-detected Task 3 (noisy_velocity_from_interface) from checkpoint path")
        elif 'velocity_from_interface' in checkpoint_path.lower():
            args.task = 'velocity_from_interface'
            print("🔍 Auto-detected Task 2 (velocity_from_interface) from checkpoint path")
        else:
            # Default fallback
            args.task = 'velocity_from_interface'
            print("⚠️  Could not auto-detect task, defaulting to velocity_from_interface")
    
    # Auto-find checkpoint if --find-checkpoint is specified
    if args.find_checkpoint:
        checkpoint_path = find_checkpoint_by_task(args.find_checkpoint, args.task)
        if checkpoint_path is None:
            print("❌ Could not find checkpoint. Please specify path directly.")
            sys.exit(1)
    elif checkpoint_path is None:
        # Default paths to try
        default_paths = [
            f"./logs/flow_matching_{args.task}/checkpoints/last.ckpt",
            f"./logs/*{args.task}*/checkpoints/last.ckpt",
        ]
        for pattern in default_paths:
            matches = glob.glob(pattern)
            if matches:
                checkpoint_path = matches[0]
                print(f"📍 Auto-detected checkpoint: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            print("❌ No checkpoint found. Use --find-checkpoint or specify path directly.")
            print("   Example: python scripts/comprehensive_inference_task123.py 100 /path/to/checkpoint.ckpt")
            print("   Example: python scripts/comprehensive_inference_task123.py --find-checkpoint ./logs --task noisy_velocity_from_interface")
            sys.exit(1)
    
    # Determine task title
    if args.task == 'temperature_from_sdf':
        task_title = "Task 1: Temperature from SDF"
    elif args.task == 'noisy_velocity_from_interface':
        if args.use_clean_inputs:
            task_title = "Task 3: Noisy Velocity from Interface (Clean Inputs - Physics Fidelity)"
        else:
            task_title = "Task 3: Noisy Velocity from Interface (Noisy Inputs - Deployment)"
    else:
        task_title = "Task 2: Velocity from Interface"
    
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
    
    print(f"🔬 {task_title}")
    print("=" * 60)
    print(f"Task:             {args.task}")
    print(f"Model type:       {model_type}")
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Data file:        {args.data_file_path}")
    print(f"Sample index:     {args.sample_idx}")
    print(f"Inference steps:  {args.num_inference_steps}")
    print(f"Start time:       {args.start_time}")
    print(f"Output dir:       {args.output_dir}")
    if args.seed is not None:
        print(f"Random seed:      {args.seed} (deterministic)")
    if args.downsample_factor > 1:
        print(f"Downsample:       {args.downsample_factor}x (512→{512 // args.downsample_factor})")
    if args.task == 'noisy_velocity_from_interface':
        print(f"Input type:       {'Clean (physics fidelity)' if args.use_clean_inputs else 'Noisy (deployment)'}")
    if model_type == 've_sde':
        print(f"🔊 VE-SDE Settings:")
        print(f"   σ_min: {args.sigma_min}, σ_max: {args.sigma_max}")
        print(f"   Sampling: {args.sampling_method}, SNR: {args.snr}")
    if model_type == 'flow_matching_ar':
        print(f"🔄 Autoregressive Flow Matching Settings:")
        print(f"   Conditions on: [conditioning_t, output_(t-1)]")
        print(f"   Residual prediction: {'ENABLED (predict delta)' if args.residual_prediction else 'DISABLED (predict absolute)'}")
        print(f"   ODE solver: {args.solver}")
        print(f"   Guidance scale: {args.guidance_scale}")
    if model_type == 'unet':
        print(f"📦 UNet Settings:")
        print(f"   Direct regression (single forward pass, no ODE)")
        print(f"   Frame-to-frame prediction (conditioning -> output)")
    if model_type == 'ffno':
        print(f"📦 FFNO Settings (Factorized Fourier Neural Operator):")
        print(f"   Spectral method (Fourier transform in feature space)")
        print(f"   Single forward pass (fast inference)")
        print(f"   Frame-to-frame prediction (conditioning -> output)")
    if model_type == 'unet_ar':
        print(f"🔄 Autoregressive UNet Settings:")
        print(f"   Conditions on: [conditioning_t, output_(t-1)]")
        print(f"   Residual prediction: {'ENABLED (predict delta)' if args.residual_prediction else 'DISABLED (predict absolute)'}")
        print(f"   Direct regression (single forward pass, no ODE)")
        print(f"   Single sample: uses ground truth previous output (teacher forcing)")
        if args.autoregressive_rollout:
            print(f"   GIF mode: TRUE autoregressive rollout (model's own predictions)")
        else:
            print(f"   GIF mode: uses ground truth previous outputs (teacher forcing)")
            print(f"   💡 Use --autoregressive-rollout for true AR inference")
    if model_type == 'flow_matching_ar_bootstrap':
        print(f"🚀 Autoregressive Flow Matching with Bootstrap Settings:")
        print(f"   Bootstrap: Infers initial state from conditioning history")
        print(f"   CLI --history-length: {args.history_length} (will be overridden by checkpoint if different)")
        print(f"   ODE solver: {args.solver}")
        print(f"   Availability mask: Tells model if prev_output is real or bootstrapped")
        print(f"   Trajectory: bootstrap + autoregressive rollout")
    if model_type == 'edm_ar_bootstrap':
        print(f"🚀 Autoregressive EDM with Bootstrap Settings:")
        print(f"   Bootstrap: Infers initial state from conditioning history")
        print(f"   CLI --history-length: {args.history_length} (will be overridden by checkpoint if different)")
        print(f"   EDM solver: {args.solver}")
        print(f"   Availability mask: Tells model if prev_output is real or bootstrapped")
        print(f"   Trajectory: bootstrap + autoregressive rollout (EDM diffusion)")
    print("=" * 60)
    
    try:
        # Load task configuration
        task_cfg = load_task_config(args.task)
        target_names = list(task_cfg.target_names)
        conditioning_names = list(task_cfg.conditioning_names)
        
        # Compute channels dynamically from task_cfg
        num_cond = len(task_cfg.conditioning_channels)
        num_target = len(task_cfg.target_channels)
        
        # Model configuration based on model_type
        if model_type == 've_sde':
            # VE-SDE Score-Based model (frame-to-frame diffusion)
            # in_channels = num_target (noisy state) + num_conditioning
            # out_channels = num_target (noise prediction)
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
                'sigma_min': args.sigma_min,
                'sigma_max': args.sigma_max,
                'num_sampling_steps': args.num_inference_steps,
                'sampling_method': args.sampling_method,
                'snr': args.snr,
                'conditioning_strategy': 'none',
                'temp_min': 55.0,
                'temp_max': 120.0,
                'num_inference_steps': args.num_inference_steps
            })
        elif model_type == 'flow_matching_ar':
            # Autoregressive model
            # in_channels = num_target (3) + num_conditioning (3) + num_target (3) = 9
            # (x_t + conditioning + prev_output)
            model_cfg = DictConfig({
                'name': 'flow_matching_ar',
                'in_channels': 9,  # x_t (3) + conditioning (3) + prev_output (3)
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
                'scheduled_sampling': {
                    'enabled': False,
                    'schedule_type': 'linear',
                    'warmup_epochs': 3,
                    'transition_epochs': 15,
                    'min_teacher_ratio': 0.0,
                },
                'auxiliary_losses': {
                    'spectral_enabled': False,
                    'gradient_enabled': False,
                    # Physics-informed losses (only used during training, not inference)
                    'divergence_enabled': False,
                    'vorticity_enabled': False,
                    'advection_enabled': False,
                }
            })
        elif model_type == 'flow_matching_ar_bootstrap':
            # For bootstrap models, we do NOT construct model_cfg here.
            # Instead, load_from_checkpoint uses the checkpoint's saved
            # hyperparameters directly (see the loading code above).
            # This avoids architecture mismatches between CLI defaults and
            # the actual training config (history_length, use_availability_mask, etc.).
            model_cfg = None
        elif model_type == 'edm_ar_bootstrap':
            # Same as flow_matching_ar_bootstrap: use checkpoint hyperparameters
            model_cfg = None
        elif model_type == 'unet_ar':
            # Autoregressive UNet model (direct regression, no diffusion)
            # in_channels = num_conditioning (3) + num_target (3) = 6
            model_cfg = DictConfig({
                'name': 'unet_ar',
                'init_features': 32,
                'conditioning_strategy': 'none',
                'temp_min': 55.0,
                'temp_max': 120.0,
                # Residual prediction mode
                'residual_prediction': args.residual_prediction,
                # Scheduled sampling config (for training, not used during inference)
                'scheduled_sampling': {
                    'enabled': False,
                    'schedule_type': 'linear',
                    'warmup_epochs': 5,
                    'transition_epochs': 40,
                    'min_teacher_ratio': 0.0,
                },
                # Auxiliary losses (not used for inference, but needed for loading)
                'auxiliary_losses': {
                    'spectral_enabled': False,
                    'gradient_enabled': False,
                }
            })
        elif model_type == 'unet':
            # UNet model (direct regression, frame-to-frame)
            # in_channels = num_conditioning, out_channels = num_target
            model_cfg = DictConfig({
                'name': 'unet',
                'init_features': 32,
                'temp_min': 55.0,
                'temp_max': 120.0,
            })
        elif model_type == 'ffno':
            # FFNO model (Factorized Fourier Neural Operator, frame-to-frame)
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
            # DDPM model (diffusion, frame-to-frame)
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
        elif model_type == 'edm':
            # EDM model (EDM-style diffusion, frame-to-frame)
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
                'solver': getattr(args, 'solver', 'heun'),
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
                'solver': getattr(args, 'solver', 'heun'),
                'zeta_obs': 1.0,
                'zeta_pde': 0.5,
                'pde_start_fraction': 0.8,
                'pde_obs_decay': 0.1,
                'bulk_sdf_threshold': 0.05,
                'temp_min': 55.0,
                'temp_max': 120.0,
            })
        elif model_type == 'flow_matching_history':
            # History-window flow matching: use checkpoint hparams if available,
            # otherwise construct a default config (same pattern as bootstrap models).
            model_cfg = None
        elif model_type == 'flow_matching_jit':
            # JiT Vision Transformer with data prediction + velocity loss
            model_cfg = DictConfig({
                'name': 'flow_matching_jit',
                'img_size': 128,
                'patch_size': 4,
                'hidden_size': 384,
                'depth': 8,
                'num_heads': 8,
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
                'attention_type': 'none',  # 'none', 'bottleneck'
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
        
        # Load normalization statistics
        # Priority: 1) Explicit file path, 2) Checkpoint directory, 3) Compute from data
        normalization_stats = None
        
        # Option 1: Load from explicitly provided file (via --normalization-stats)
        if args.normalization_stats and os.path.exists(args.normalization_stats):
            print(f"   📊 Loading normalization stats from provided file: {args.normalization_stats}")
            with open(args.normalization_stats, 'r') as f:
                normalization_stats = json.load(f)
            print(f"   ✓ Loaded normalization stats:")
            print(f"      Temperature: [{normalization_stats['temperature']['min']:.2f}, {normalization_stats['temperature']['max']:.2f}]°C")
            print(f"      Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
            print(f"      SDF scale: {normalization_stats['sdf']['scale']:.4f}")
        elif args.normalization_stats:
            # User provided path but file doesn't exist
            print(f"   ⚠️  WARNING: Normalization stats file not found: {args.normalization_stats}")
            print(f"   📊 Falling back to checkpoint directory or computing from data...")
        
        # Option 2: Try to load from checkpoint directory (if not already loaded)
        if normalization_stats is None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if "checkpoints" in checkpoint_dir:
                checkpoint_dir = os.path.dirname(checkpoint_dir)  # Go up one level
            stats_file = os.path.join(checkpoint_dir, "normalization_stats.json")

            if os.path.exists(stats_file):
                print(f"   📊 Loading normalization stats from training: {stats_file}")
                with open(stats_file, 'r') as f:
                    normalization_stats = json.load(f)
                print(f"   ✓ Loaded training normalization stats:")
                print(f"      Temperature: [{normalization_stats['temperature']['min']:.2f}, {normalization_stats['temperature']['max']:.2f}]°C")
                print(f"      Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
                print(f"      SDF scale: {normalization_stats['sdf']['scale']:.4f}")
        
        # Option 3: Fallback: compute from inference file (WARNING: may not match training!)
        if normalization_stats is None:
            from bubblefusion.data.bubbleml import compute_normalization_stats
            print(f"   ⚠️  WARNING: normalization_stats.json not found")
            print(f"   📊 Computing normalization stats from inference file (may not match training!)...")
            print(f"   💡 For accurate results, provide --normalization-stats or ensure normalization_stats.json exists in checkpoint directory")
            normalization_stats = compute_normalization_stats(
                filenames=[args.data_file_path],
                start_time=args.start_time,
                verbose=True
            )
        
        # Load model
        model = load_model_from_checkpoint(
            checkpoint_path, model_cfg, optim_cfg, scheduler_cfg, task_cfg,
            model_type=model_type,
            normalization_stats=normalization_stats,
            norm_mode=args.norm_mode
        )
        
        # For bootstrap models, sync args with the checkpoint's actual config
        # so the dataset uses the correct history_length, history_stride, and rollout_length.
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
        
        # Extract noise configuration for Task 3
        noise_cfg = None
        if args.task == 'noisy_velocity_from_interface' and 'noise_cfg' in task_cfg:
            noise_cfg = dict(task_cfg.noise_cfg)
            if args.use_clean_inputs:
                # Disable noise for clean inference
                noise_cfg['enabled'] = False
                print("   ✨ Noise disabled for clean inference")
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
                print(f"   🔊 Noise configuration (effective):")
                print(f"      Type: {noise_type}")
                if noise_type in ['gaussian', 'simple']:
                    print(f"      SDF noise std: {noise_cfg.get('sdf_noise_std', 0.1)}")
                    print(f"      Vel noise std: {noise_cfg.get('vel_noise_std', 0.05)}")
                else:
                    print(f"      SDF noise std: {noise_cfg.get('sdf_noise_std', 0.1)}")
                    print(f"      SDF gradient scale: {noise_cfg.get('sdf_gradient_scale', 0.3)}")
                    print(f"      Vel base noise std: {noise_cfg.get('vel_base_noise_std', 0.05)}")
                    print(f"      Vel scale factor: {noise_cfg.get('vel_scale_factor', 0.15)}")
                    print(f"      Correlation length: {noise_cfg.get('correlation_length', 3.0)}")
        
        # Load dataset based on model type
        is_autoregressive = (model_type in ['flow_matching_ar', 'unet_ar'])
        is_ar_bootstrap = (model_type in ['flow_matching_ar_bootstrap', 'edm_ar_bootstrap'])
        is_history_model = (model_type == 'flow_matching_history')
        dataset = load_dataset(
            args.data_file_path,
            output_fields=['temperature', 'velx', 'vely'],
            start_time=args.start_time,
            normalize_temperature=args.normalize_temperature,
            return_wall_temp=False,
            noise_cfg=noise_cfg,
            use_clean_inputs=args.use_clean_inputs,
            is_temporal=False,
            history_length=args.history_length if is_ar_bootstrap else 1,
            temporal_stride=1,
            history_stride=args.history_stride if is_ar_bootstrap else 1,
            is_autoregressive=is_autoregressive,
            is_ar_bootstrap=is_ar_bootstrap,
            rollout_length=args.rollout_length if is_ar_bootstrap else 5,
            is_history_model=is_history_model,
            history_window=model_cfg.get('history_window', 10) if model_cfg else 10,
            downsample_factor=args.downsample_factor,
            scheduled_sampling=False,
            normalization_stats=normalization_stats,
            norm_mode=args.norm_mode
        )
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {device}")
        model = model.to(device)
        
        # Determine which samples to process
        # Priority: --samples > --gif-samples (if --generate-gif) > single sample
        samples_to_process = None
        if args.samples is not None:
            samples_to_process = args.samples
            print(f"📋 Using --samples: {samples_to_process}")
        elif args.generate_gif:
            samples_to_process = args.gif_samples
            print(f"📋 Using --gif-samples: {samples_to_process}")
        
        if samples_to_process is not None:
            # Multiple sample processing mode
            sample_indices = parse_sample_indices(samples_to_process, len(dataset))
            
            if not sample_indices:
                print("❌ No valid sample indices")
                sys.exit(1)
            
            # Check if using autoregressive rollout mode
            use_ar_rollout = args.autoregressive_rollout and model_type in ['flow_matching_ar', 'unet_ar']
            use_bootstrap_rollout = model_type == 'flow_matching_ar_bootstrap'
            
            # Extract wall temperature for consistent colorbar ranges
            wall_temp = extract_wall_temp_from_filepath(args.data_file_path)
            print(f"\n🌡️  Wall temperature: {wall_temp}°C (from filename)")
            
            # Compute bulk_temp and heater_temp for BubbleML colormap
            # Priority: 1) Command line args, 2) Normalization stats
            if args.bulk_temp is not None:
                bulk_temp = args.bulk_temp
            elif normalization_stats and 'temperature' in normalization_stats:
                bulk_temp = normalization_stats['temperature']['min']
            else:
                bulk_temp = 48.3  # Default for subcooled pool boiling
            
            if args.heater_temp is not None:
                heater_temp = args.heater_temp
            elif normalization_stats and 'temperature' in normalization_stats:
                heater_temp = normalization_stats['temperature']['max']
            else:
                heater_temp = 114.7  # Default for subcooled pool boiling
            
            if args.use_bubbleml_cmap:
                print(f"   🎨 BubbleML colormap: [{bulk_temp:.1f}, {heater_temp:.1f}]°C, vmin={args.vmin}")
            
            if use_bootstrap_rollout:
                print(f"\n🎬 MULTI-SAMPLE MODE (Bootstrap + AR Rollout): Processing {len(sample_indices)} frames")
                print(f"   🚀 Using bootstrap to infer initial state from conditioning history")
                
                # For AR bootstrap, sample_indices[0] determines the starting position
                # in the trajectory, and len(sample_indices) determines how many
                # CONTINUOUS frames to generate (not overlapping rollouts)
                start_sample = sample_indices[0]
                num_frames_to_generate = len(sample_indices)
                
                # Calculate actual timesteps for user info
                effective_start_time = dataset.effective_start_time
                start_timestep = start_sample + effective_start_time
                end_timestep = start_timestep + num_frames_to_generate - 1
                print(f"   📍 Starting sample index: {start_sample} → Timestep {start_timestep}")
                print(f"   📍 Generating {num_frames_to_generate} continuous frames (timesteps {start_timestep}-{end_timestep})")
                
                # Run bootstrap + AR trajectory
                trajectory_results = run_bootstrap_trajectory(
                    model, dataset, start_sample, num_frames_to_generate,
                    device, args.num_inference_steps,
                    solver=args.solver,
                    bootstrap_ablation=args.bootstrap_ablation,
                )
                
                # Compute fixed colorbar ranges across all frames for consistent visualization
                colorbar_ranges = compute_colorbar_ranges(
                    trajectory_results, target_names, conditioning_names,
                    wall_temp=wall_temp  # temp_min auto-detected from data
                )
                
                # Generate plots from trajectory with fixed colorbar ranges
                plot_paths = []
                all_metrics = []
                
                for frame_idx, (input_data, target, predicted) in enumerate(trajectory_results):
                    sample_idx = sample_indices[0] + frame_idx
                    
                    metrics = compute_metrics(target, predicted, target_names)
                    all_metrics.append(metrics)
                    
                    plot_path = create_comprehensive_visualization(
                        input_data, target, predicted, target_names, conditioning_names,
                        sample_idx, metrics, args.output_dir, model_type, 
                        args.task, args.use_clean_inputs,
                        colorbar_ranges=colorbar_ranges,
                        use_bubbleml_cmap=args.use_bubbleml_cmap, bulk_temp=bulk_temp,
                        heater_temp=heater_temp, cmap_vmin=args.vmin, vel_vmin=args.vel_vmin,
                        vel_vmax=args.vel_vmax, vel_interface_vmax=args.vel_interface_vmax,
                        dpi=args.dpi
                    )
                    plot_paths.append(plot_path)
                    
                    print(f"    ✓ Frame {frame_idx + 1}/{len(trajectory_results)}: Overall Rel L2 = {metrics['Overall']['Relative_L2']:.4f}")
                
                # Update sample_indices to match what was actually generated
                sample_indices = list(range(sample_indices[0], sample_indices[0] + len(trajectory_results)))
            elif use_ar_rollout:
                print(f"\n🎬 MULTI-SAMPLE MODE (Autoregressive Rollout): Processing {len(sample_indices)} frames")
                print(f"   ⚠️ Using model's own predictions (not ground truth previous outputs)")
                
                # Run true autoregressive trajectory
                trajectory_results = run_autoregressive_trajectory(
                    model, dataset, sample_indices[0], len(sample_indices),
                    device, args.num_inference_steps,
                    solver=args.solver, guidance_scale=args.guidance_scale,
                    model_type=model_type,
                    initial_state_mode=args.initial_state_mode,
                    history_length=args.history_length
                )
                
                # Compute fixed colorbar ranges across all frames for consistent visualization
                colorbar_ranges = compute_colorbar_ranges(
                    trajectory_results, target_names, conditioning_names,
                    wall_temp=wall_temp  # temp_min auto-detected from data
                )
                
                # Generate plots from trajectory with fixed colorbar ranges
                plot_paths = []
                all_metrics = []
                
                for frame_idx, (input_data, target, predicted) in enumerate(trajectory_results):
                    sample_idx = sample_indices[0] + frame_idx
                    
                    metrics = compute_metrics(target, predicted, target_names)
                    all_metrics.append(metrics)
                    
                    plot_path = create_comprehensive_visualization(
                        input_data, target, predicted, target_names, conditioning_names,
                        sample_idx, metrics, args.output_dir, model_type, 
                        args.task, args.use_clean_inputs,
                        colorbar_ranges=colorbar_ranges,
                        use_bubbleml_cmap=args.use_bubbleml_cmap, bulk_temp=bulk_temp,
                        heater_temp=heater_temp, cmap_vmin=args.vmin, vel_vmin=args.vel_vmin,
                        vel_vmax=args.vel_vmax, vel_interface_vmax=args.vel_interface_vmax,
                        dpi=args.dpi
                    )
                    plot_paths.append(plot_path)
                    
                    print(f"    ✓ Frame {frame_idx + 1}/{len(trajectory_results)}: Overall Rel L2 = {metrics['Overall']['Relative_L2']:.4f}")
                
                # Update sample_indices to match what was actually generated
                sample_indices = list(range(sample_indices[0], sample_indices[0] + len(trajectory_results)))
            else:
                print(f"\n🎬 MULTI-SAMPLE MODE: Processing {len(sample_indices)} samples")
                if model_type in ['flow_matching_ar', 'unet_ar']:
                    print(f"   ℹ️ Using ground truth previous outputs (teacher forcing)")
                    print(f"   💡 Use --autoregressive-rollout for true AR inference")
                
                plot_paths, all_metrics = generate_multiple_plots(
                    model, dataset, sample_indices, target_names, conditioning_names,
                    device, args.num_inference_steps, args.output_dir, model_type,
                    args.task, args.use_clean_inputs,
                    wall_temp=wall_temp, initial_state_mode=args.initial_state_mode,
                    use_bubbleml_cmap=args.use_bubbleml_cmap, bulk_temp=bulk_temp,
                    heater_temp=heater_temp, cmap_vmin=args.vmin, vel_vmin=args.vel_vmin,
                    vel_vmax=args.vel_vmax, vel_interface_vmax=args.vel_interface_vmax,
                    dpi=args.dpi, history_length=args.history_length,
                    bootstrap_ablation=args.bootstrap_ablation
                )
            
            # Create GIF only if requested
            if args.generate_gif and plot_paths:
                task_suffix = args.task.replace('_', '-')
                input_suffix = "clean" if args.use_clean_inputs else "noisy" if args.task == 'noisy_velocity_from_interface' else ""
                rollout_suffix = "_rollout" if use_ar_rollout else ""
                if input_suffix:
                    gif_name = f'{task_suffix}_{input_suffix}_{model_type}{rollout_suffix}_animation.gif'
                else:
                    gif_name = f'{task_suffix}_{model_type}{rollout_suffix}_animation.gif'
                gif_path = os.path.join(args.output_dir, gif_name)
                create_gif_from_plots(plot_paths, gif_path, args.gif_duration)
            elif plot_paths:
                print(f"\n📊 Processed {len(plot_paths)} samples (GIF generation skipped)")
            
            # Print summary metrics
            if all_metrics:
                print_summary_metrics(all_metrics, sample_indices, target_names)
                
                # Plot metric trends
                print(f"\n📈 Generating metric trend plots...")
                plot_metrics_trends(
                    all_metrics, sample_indices, target_names, args.output_dir,
                    model_type=model_type, task_name=args.task,
                    use_clean_inputs=args.use_clean_inputs
                )
        else:
            # Single sample inference
            print(f"\n🎯 SINGLE SAMPLE INFERENCE")
            print("-" * 40)
            
            input_data, target, predicted = run_inference_on_sample(
                model, dataset, args.sample_idx, device, args.num_inference_steps,
                model_type=model_type, initial_state_mode=args.initial_state_mode,
                history_length=args.history_length,
                bootstrap_ablation=args.bootstrap_ablation
            )
            
            metrics = compute_metrics(target, predicted, target_names)
            print_detailed_metrics(metrics, args.sample_idx, target_names)
            
            create_comprehensive_visualization(
                input_data, target, predicted, target_names, conditioning_names,
                args.sample_idx, metrics, args.output_dir, model_type,
                args.task, args.use_clean_inputs,
                use_bubbleml_cmap=args.use_bubbleml_cmap, bulk_temp=bulk_temp,
                heater_temp=heater_temp, cmap_vmin=args.vmin, vel_vmin=args.vel_vmin,
                vel_vmax=args.vel_vmax, vel_interface_vmax=args.vel_interface_vmax,
                dpi=args.dpi
            )
        
        if args.task == 'temperature_from_sdf':
            task_name_display = "Task 1"
        elif args.task == 'noisy_velocity_from_interface':
            task_name_display = "Task 3"
        else:
            task_name_display = "Task 2"
        print(f"\n🎉 {task_name_display} inference completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
