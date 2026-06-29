#!/usr/bin/env python3
"""
Heat Flux Interpolation/Extrapolation Analysis Script

This script:
1. Loads all training and validation files from poolboiling_subcooled.yaml
2. Computes ground truth heat flux for all files (train and val)
3. Runs inference on validation files using the flow_matching_ar_bootstrap model
4. Computes predicted heat flux for validation files
5. Creates a publication-quality plot showing:
   - X-axis: Wall temperature (Twall)
   - Y-axis: Heat flux (W/m²)
   - Different symbols for GT and model predictions
   - Background regions: OOD (extrapolation) vs ID (interpolation)

Usage:
    python heatflux_interpolation_plot.py --checkpoint /path/to/checkpoint.ckpt
    python heatflux_interpolation_plot.py --start-frame 100 --end-frame 200
"""

import sys
import os
import argparse
import json
import numpy as np
import h5py as h5
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bubblefusion.models.flow_matching_ar_bootstrap import ConditionalFlowMatchingARBootstrapLightning
from bubblefusion.data.bubbleml import compute_normalization_stats, BulkFlowARBootstrap


# ============================================================================
# Data Configuration
# ============================================================================

# Training files (from poolboiling_subcooled.yaml)
TRAIN_TWALL_VALUES = [86, 88, 90, 92, 94, 98, 100, 102, 104, 106, 110, 112, 114, 116]

# Validation files (from poolboiling_subcooled.yaml)
VAL_TWALL_VALUES = [85, 96, 97, 107, 108, 117]
# VAL_TWALL_VALUES = [96]

# OOD (extrapolation) regions
OOD_TWALL_LOW = 85
OOD_TWALL_HIGH = 117

# Train range for determining ID/OOD
TRAIN_TWALL_MIN = min(TRAIN_TWALL_VALUES)  # 86
TRAIN_TWALL_MAX = max(TRAIN_TWALL_VALUES)  # 116


# ============================================================================
# Heat Flux Computation
# ============================================================================

def compute_heatflux(dfun: np.ndarray, temp: np.ndarray, heater_temp: float,
                     lc: float = 0.73e-3, thcl: float = 6.25e-2,
                     downsample_factor: int = 1) -> np.ndarray:
    """
    Calculate heat flux for FC-72 fluid.
    
    Reference: scripts/physics_metrics_task123.py
    
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
    T_frames, H, W = dfun.shape
    
    # Adjust grid spacing for downsampling
    dx = downsample_factor / 32

    x_min, x_max = -8, 8
    y_min, y_max = 0, 16
    
    x_centers = x_min + (np.arange(W) + 0.5) * dx
    y_centers = y_min + (np.arange(H) + 0.5) * dx

    x_grid, _ = np.meshgrid(x_centers, y_centers)

    heater_mask = (x_grid >= -5) & (x_grid <= 5)
    heater_mask_3d = np.broadcast_to(heater_mask, (T_frames, H, W))

    liquid_mask = dfun < 0
    temp_fields = (heater_mask_3d & liquid_mask).astype(float) * (heater_temp - temp)
    hflux_fields = thcl * (temp_fields / (dx * 0.5 * lc))
    hfluxes = hflux_fields[:, 0, :].mean(axis=1)

    return hfluxes


def extract_wall_temp(filepath: str) -> float:
    """Extract wall temperature from filename like Twall_96.hdf5."""
    match = re.search(r'Twall_(\d+)', filepath)
    if match:
        return float(match.group(1))
    raise ValueError(f"Cannot extract wall temperature from: {filepath}")


def compute_gt_heatflux_from_dataset(
    data_filepath: str, 
    normalization_stats: dict,
    start_time: int = 100,
    frame_start: int = 0,
    frame_end: int = 100,
    history_length: int = 10,
    rollout_length: int = 5,
    downsample_factor: int = 4,
) -> Tuple[np.ndarray, float]:
    """
    Compute GT heat flux using dataset (matching physics_metrics_task123.py).
    
    This uses the same approach as physics_metrics: GT temperature comes from
    the dataset's target_seq, which is normalized then denormalized.
    
    Frame indexing (matching physics_metrics_task123.py):
    - start_time: HDF5 offset (default 100, matches training)
    - frame_start/frame_end: Dataset INDICES
    
    Args:
        data_filepath: Path to HDF5 file
        normalization_stats: Normalization statistics
        start_time: HDF5 offset (default 100)
        frame_start: Starting dataset INDEX
        frame_end: Ending dataset INDEX
        history_length: History length for dataset
        rollout_length: Rollout length for dataset
        downsample_factor: Spatial downsampling
        
    Returns:
        gt_hflux: Ground truth heat flux array
        wall_temp: Wall temperature
    """
    # Create dataset (matching physics_metrics approach)
    dataset = BulkFlowARBootstrap(
        filenames=[data_filepath],
        output_fields=['temperature', 'velx', 'vely'],
        start_time=start_time,
        normalization_stats=normalization_stats,
        return_wall_temp=True,
        noise_cfg=None,
        history_length=history_length,
        rollout_length=rollout_length,
        downsample_factor=downsample_factor
    )
    
    # Load raw SDF from HDF5 (same as physics_metrics)
    dfun_raw, _, wall_temp = load_raw_data(
        data_filepath, start_time=start_time, downsample_factor=downsample_factor
    )
    
    # Get task config for channel ordering
    task_cfg = load_task_config('velocity_from_interface')
    target_channels = list(task_cfg.target_channels)
    target_names = list(task_cfg.target_names)
    temp_idx = target_names.index('temperature')
    
    # Calculate number of segments
    desired_num_frames = frame_end - frame_start
    num_segments = desired_num_frames // rollout_length
    
    # Collect GT temperatures from dataset (matching physics_metrics)
    gt_temp_list = []
    
    for seg_i in range(num_segments):
        segment_idx = frame_start + seg_i * rollout_length
        
        if segment_idx >= len(dataset):
            break
        
        sample_data = dataset[segment_idx]
        _, _, target_seq, _ = sample_data
        
        # Extract and denormalize each frame in the rollout
        for l in range(rollout_length):
            # CRITICAL: Reorder channels using target_channels (matching physics_metrics)
            target_frame = target_seq[l, target_channels, :, :].clone()
            
            # Denormalize (now channel order matches target_names)
            for j, field_name in enumerate(target_names):
                target_frame[j] = dataset._denormalize_field(target_frame[j], field_name)
            
            gt_temp_list.append(target_frame[temp_idx].numpy())
    
    # Stack GT temperature array
    gt_temp_array = np.stack(gt_temp_list, axis=0)
    
    # Slice SDF to match frames
    num_frames = len(gt_temp_list)
    dfun_for_hflux = dfun_raw[frame_start:frame_start + num_frames]
    
    # Compute GT heat flux
    gt_hflux = compute_heatflux(dfun_for_hflux, gt_temp_array, wall_temp, 
                                downsample_factor=downsample_factor)
    
    return gt_hflux, wall_temp


def load_raw_data(filepath: str, start_time: int = 100, downsample_factor: int = 4):
    """
    Load raw SDF and temperature data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        start_time: Starting timestep
        downsample_factor: Spatial downsampling factor
        
    Returns:
        sdf: (T, H, W) numpy array
        temperature: (T, H, W) numpy array
        wall_temp: Wall temperature from file
    """
    with h5.File(filepath, 'r') as f:
        # Get wall temperature
        if 'wallTemperature' in f.attrs:
            wall_temp = float(f.attrs['wallTemperature'])
        else:
            wall_temp = extract_wall_temp(filepath)
        
        # Load SDF (dfun)
        dfun = f['dfun'][start_time:]  # (T, H, W)
        temp = f['temperature'][start_time:]  # (T, H, W)
        
        # Apply downsampling
        if downsample_factor > 1:
            dfun = dfun[:, ::downsample_factor, ::downsample_factor]
            temp = temp[:, ::downsample_factor, ::downsample_factor]
    
    return dfun, temp, wall_temp


def load_task_config(task_name: str = 'velocity_from_interface') -> DictConfig:
    """Load task configuration from YAML file."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'bubblefusion', 'config', 'task_cfg', f'{task_name}.yaml'
    )
    
    if os.path.exists(config_path):
        task_cfg = OmegaConf.load(config_path)
        return task_cfg
    else:
        return DictConfig({
            'name': 'velocity_from_interface',
            'conditioning_channels': [0, 1, 2],
            'target_channels': [1, 2, 0],
            'target_names': ['velx', 'vely', 'temperature']
        })


def load_model(checkpoint_path: str, normalization_stats: dict, 
               task_cfg: DictConfig, device: str = 'cuda'):
    """Load the flow_matching_ar_bootstrap model."""
    print(f"\n🤖 Loading AR Bootstrap model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Model configuration (matching training config)
    model_cfg = DictConfig({
        'base_channels': 32,
        'time_embed_dim': 256,
        'num_res_blocks': 2,
        'use_attention': False,
        'attention_type': 'none',
        'dropout': 0.0,
        'history_length': 10,
        'rollout_length': 5,
        'use_availability_mask': True,
        'bootstrap_loss_weight': 1.0,
        'ar_loss_weight': 1.0,
        'bootstrap_state_loss_weight': 0.5,
        'history_encoder_type': 'temporal_mixer',
        'history_encoder_hidden': 32,
        'temporal_mixer_spatial_conv': True,
        'temporal_mixer_temporal_weights': True,
        'num_integration_steps': 50,
        'inference': {'solver': 'heun'},
        'scheduled_sampling': {'enabled': False},
    })
    
    optim_cfg = DictConfig({'name': 'adamw', 'lr': 1e-4, 'weight_decay': 1e-2})
    scheduler_cfg = DictConfig({'name': 'cosine'})
    
    model = ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        task_cfg=task_cfg,
        normalization_stats=normalization_stats,
        strict=False
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    return model


def run_inference_for_heatflux(
    model, 
    data_filepath: str, 
    normalization_stats: dict,
    start_time: int = 100,
    frame_start: int = 0,
    frame_end: int = 100,
    history_length: int = 10,
    rollout_length: int = 5,
    downsample_factor: int = 4,
    num_integration_steps: int = 50,
    solver: str = 'heun',
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run model inference on a validation file and compute heat flux.
    
    Frame indexing (matching physics_metrics_task123.py):
    - start_time: HDF5 offset for dataset initialization (default 100)
    - frame_start/frame_end: Dataset INDICES (not HDF5 frames)
    - dataset[frame_start] → HDF5 frame (start_time + frame_start)
    
    Important: 
    - GT uses RAW temperature from HDF5 (already in physical units)
    - Predictions are denormalized (model outputs normalized values)
    - Normalization stats are computed from TRAINING data only
    
    Args:
        model: Loaded model
        data_filepath: Path to HDF5 file
        normalization_stats: Normalization statistics
        start_time: HDF5 offset for dataset (default 100, matches training)
        frame_start: Starting dataset INDEX for heat flux computation
        frame_end: Ending dataset INDEX for heat flux computation
        history_length: History length for bootstrap
        rollout_length: Rollout length for AR
        downsample_factor: Spatial downsampling
        num_integration_steps: ODE integration steps
        solver: ODE solver
        device: Device to run on
        
    Returns:
        gt_hflux: Ground truth heat flux array
        pred_hflux: Predicted heat flux array
        wall_temp: Wall temperature
    """
    print(f"\n🔮 Running inference on: {os.path.basename(data_filepath)}")
    
    # Load dataset for inference (provides normalized inputs for the model)
    # start_time is the HDF5 offset (matches physics_metrics_task123.py)
    dataset = BulkFlowARBootstrap(
        filenames=[data_filepath],
        output_fields=['temperature', 'velx', 'vely'],
        start_time=start_time,
        normalization_stats=normalization_stats,
        return_wall_temp=True,
        noise_cfg=None,
        history_length=history_length,
        rollout_length=rollout_length,
        downsample_factor=downsample_factor
    )
    
    # Load RAW data from HDF5 for GT heat flux
    # Use same start_time as dataset for alignment
    dfun_raw, temp_raw, wall_temp = load_raw_data(
        data_filepath, start_time=start_time, downsample_factor=downsample_factor
    )
    
    # Get task config
    task_cfg = load_task_config('velocity_from_interface')
    conditioning_channels = list(task_cfg.conditioning_channels)
    target_channels = list(task_cfg.target_channels)
    target_names = list(task_cfg.target_names)
    temp_idx = target_names.index('temperature')
    
    # Calculate number of segments to process
    # frame_start/frame_end are dataset INDICES
    desired_num_frames = frame_end - frame_start
    num_segments = desired_num_frames // rollout_length
    
    # Calculate actual HDF5 frame numbers for debugging
    effective_start = max(start_time, history_length)
    first_hdf5_frame = effective_start + frame_start
    last_hdf5_frame = first_hdf5_frame + (num_segments * rollout_length) - 1
    
    print(f"   📍 Frame indexing (matching physics_metrics):")
    print(f"      Dataset start_time: {start_time}")
    print(f"      Frame range: [{frame_start}:{frame_end}] (dataset indices)")
    print(f"      HDF5 frames: {first_hdf5_frame} to {last_hdf5_frame}")
    print(f"   Processing {num_segments} segments ({num_segments * rollout_length} frames)...")
    
    # Collect BOTH GT and PREDICTION temperatures from dataset
    # (matching physics_metrics_task123.py approach)
    gt_temp_list = []
    pred_temp_list = []
    
    with torch.no_grad():
        for seg_i in tqdm(range(num_segments), desc="Inference"):
            # Use frame_start as the starting dataset index (matching physics_metrics)
            segment_idx = frame_start + seg_i * rollout_length
            
            if segment_idx >= len(dataset):
                break
            
            sample_data = dataset[segment_idx]
            cond_hist, cond_seq, target_seq, wt = sample_data
            
            cond_hist = cond_hist.unsqueeze(0).to(device)
            cond_seq = cond_seq.unsqueeze(0).to(device)
            target_seq = target_seq.unsqueeze(0).to(device)
            
            # Extract relevant channels (matching physics_metrics_task123.py)
            cond_hist_extracted = cond_hist[:, :, conditioning_channels, :, :]
            cond_seq_extracted = cond_seq[:, :, conditioning_channels, :, :]
            target_seq_extracted = target_seq[:, :, target_channels, :, :]  # CRITICAL: reorder to match target_names
            
            B, T_hist, C_cond, H, W = cond_hist_extracted.shape
            _, L, _, _, _ = cond_seq_extracted.shape
            C_out = target_seq_extracted.shape[2]
            
            # Bootstrap initial state
            current_cond_0 = cond_seq_extracted[:, 0]
            bootstrapped_state = model.bootstrap_initial_state(cond_hist_extracted, current_cond_0)
            
            prev_output = bootstrapped_state
            
            for l in range(L):
                current_cond = cond_seq_extracted[:, l]
                
                if l == 0:
                    availability_mask = torch.zeros(B, 1, H, W, device=device)
                else:
                    availability_mask = torch.ones(B, 1, H, W, device=device)
                
                predicted = model.sample(
                    condition=current_cond,
                    prev_output=prev_output,
                    shape=(B, C_out, H, W),
                    device=device,
                    availability_mask=availability_mask,
                    num_integration_steps=num_integration_steps,
                    solver=solver,
                )
                
                # Move prediction and GT to CPU
                predicted_cpu = predicted.squeeze(0).cpu()
                target_cpu = target_seq_extracted[0, l].cpu()  # Get GT for this frame (already reordered)
                
                # Debug: show first frame raw model output
                if seg_i == 0 and l == 0:
                    print(f"\n   🔍 Debug - First frame model output (NORMALIZED):")
                    print(f"      Temperature channel (idx={temp_idx}):")
                    print(f"      Range: [{predicted_cpu[temp_idx].min():.4f}, {predicted_cpu[temp_idx].max():.4f}]")
                    print(f"      Mean: {predicted_cpu[temp_idx].mean():.4f}")
                    print(f"      (Expected range for normalized temp: [-1, 1])")
                
                # Denormalize BOTH GT and PREDICTION (matching physics_metrics_task123.py)
                # Dataset returns normalized values, need to denormalize both
                for j, field_name in enumerate(target_names):
                    target_cpu[j] = dataset._denormalize_field(target_cpu[j], field_name)
                    predicted_cpu[j] = dataset._denormalize_field(predicted_cpu[j], field_name)
                
                # Debug: show first frame after denormalization
                if seg_i == 0 and l == 0:
                    print(f"\n   🔍 Debug - First frame model output (DENORMALIZED):")
                    print(f"      Temperature range: [{predicted_cpu[temp_idx].min():.2f}, {predicted_cpu[temp_idx].max():.2f}] °C")
                    print(f"      Temperature mean: {predicted_cpu[temp_idx].mean():.2f} °C")
                    print(f"   🔍 Debug - First frame GT (DENORMALIZED):")
                    print(f"      GT temp range: [{target_cpu[temp_idx].min():.2f}, {target_cpu[temp_idx].max():.2f}] °C")
                    print(f"      GT temp mean: {target_cpu[temp_idx].mean():.2f} °C")
                
                # Extract temperatures (matching physics_metrics approach)
                gt_temp_list.append(target_cpu[temp_idx].numpy())
                pred_temp_list.append(predicted_cpu[temp_idx].numpy())
                
                prev_output = predicted
    
    # Stack arrays
    gt_temp_array = np.stack(gt_temp_list, axis=0)
    pred_temp_array = np.stack(pred_temp_list, axis=0)
    
    # Use SDF from raw data (same as physics_metrics)
    # Slice by frame_start to match dataset indices
    num_frames = len(pred_temp_list)
    sdf_for_hflux = dfun_raw[frame_start:frame_start + num_frames]
    gt_temp_for_hflux = gt_temp_array  # Use dataset GT (matching physics_metrics)
    
    # Debug: Compare temperature statistics between GT and prediction
    print(f"\n   📊 Temperature Statistics (at wall row 0):")
    print(f"      GT temp range:   {gt_temp_for_hflux[:, 0, :].min():.2f} - {gt_temp_for_hflux[:, 0, :].max():.2f} °C")
    print(f"      GT temp mean:    {gt_temp_for_hflux[:, 0, :].mean():.2f} °C")
    print(f"      Pred temp range: {pred_temp_array[:, 0, :].min():.2f} - {pred_temp_array[:, 0, :].max():.2f} °C")
    print(f"      Pred temp mean:  {pred_temp_array[:, 0, :].mean():.2f} °C")
    print(f"      Wall temp:       {wall_temp:.2f} °C")
    print(f"      GT ΔT:           {wall_temp - gt_temp_for_hflux[:, 0, :].mean():.2f} °C")
    print(f"      Pred ΔT:         {wall_temp - pred_temp_array[:, 0, :].mean():.2f} °C")
    
    # Compute heat flux:
    # - GT: uses raw temperature from HDF5 (physical units)
    # - Pred: uses denormalized model output
    gt_hflux = compute_heatflux(sdf_for_hflux, gt_temp_for_hflux, wall_temp, 
                                downsample_factor=downsample_factor)
    pred_hflux = compute_heatflux(sdf_for_hflux, pred_temp_array, wall_temp, 
                                  downsample_factor=downsample_factor)
    
    print(f"   ✓ GT heat flux: {np.mean(gt_hflux):.2f} ± {np.std(gt_hflux):.2f} W/m²")
    print(f"   ✓ Pred heat flux: {np.mean(pred_hflux):.2f} ± {np.std(pred_hflux):.2f} W/m²")
    
    return gt_hflux, pred_hflux, wall_temp


def create_heatflux_plot(
    results: Dict[float, Dict],
    output_path: str,
    title: str = "Heat Flux vs Wall Temperature",
    figsize: Tuple[int, int] = (10, 7)
):
    """
    Create publication-quality heat flux plot.
    
    Args:
        results: Dictionary mapping wall_temp -> {gt_mean, gt_std, pred_mean, pred_std, is_val}
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
    """
    # Set up publication-quality plot style
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.titlesize': 20,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine x-axis range
    all_twalls = sorted(results.keys())
    x_min = min(all_twalls) - 2
    x_max = max(all_twalls) + 2
    
    # Add background regions for OOD (extrapolation) and ID (interpolation)
    # OOD low region (< TRAIN_TWALL_MIN)
    ax.axvspan(x_min, TRAIN_TWALL_MIN, alpha=0.2, color='salmon', 
               label='OOD (Extrapolation)')
    
    # OOD high region (> TRAIN_TWALL_MAX)
    ax.axvspan(TRAIN_TWALL_MAX, x_max, alpha=0.2, color='salmon')
    
    # ID region (interpolation)
    ax.axvspan(TRAIN_TWALL_MIN, TRAIN_TWALL_MAX, alpha=0.15, color='lightgreen',
               label='ID (Interpolation)')
    
    # Collect all data - one list for all GT, one for predictions
    all_gt_twalls = []
    all_gt_means = []
    all_gt_stds = []
    
    pred_twalls = []
    pred_means = []
    pred_stds = []
    
    for twall, data in sorted(results.items()):
        # All ground truth data (both training and validation)
        all_gt_twalls.append(twall)
        all_gt_means.append(data['gt_mean'])
        all_gt_stds.append(data['gt_std'])
        
        # Model predictions (only for validation files that were run through inference)
        if data['is_val'] and data['pred_mean'] is not None:
            pred_twalls.append(twall)
            pred_means.append(data['pred_mean'])
            pred_stds.append(data['pred_std'])
    
    # Sort GT data by temperature
    sorted_indices = np.argsort(all_gt_twalls)
    sorted_gt_twalls = np.array(all_gt_twalls)[sorted_indices]
    sorted_gt_means = np.array(all_gt_means)[sorted_indices]
    sorted_gt_stds = np.array(all_gt_stds)[sorted_indices]
    
    # Plot ALL ground truth data with ONE symbol (circles, blue)
    ax.errorbar(sorted_gt_twalls, sorted_gt_means, yerr=sorted_gt_stds, 
                fmt='o', color='#2166AC', markersize=10, capsize=5, capthick=2,
                linewidth=2, label='Ground Truth', markeredgecolor='black',
                markeredgewidth=1.5, zorder=3)
    
    # Connect GT points with a simple line
    ax.plot(sorted_gt_twalls, sorted_gt_means, '-', color='#2166AC', alpha=0.5, linewidth=2,
            zorder=2)  # No label - just connecting line
    
    # Plot model predictions with ONE symbol (diamonds, red/orange)
    if len(pred_twalls) > 0:
        ax.errorbar(pred_twalls, pred_means, yerr=pred_stds,
                    fmt='D', color='#D6604D', markersize=10, capsize=5, capthick=2,
                    linewidth=2, label='Model Prediction', markeredgecolor='black',
                    markeredgewidth=1.5, zorder=4)
    
    # Labels and title
    ax.set_xlabel('Wall Temperature (°C)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Heat Flux (W/m²)', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    
    # Legend - position in lower right corner
    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend: regions first (OOD, ID), then data (GT, Prediction)
    # Expected order: OOD (Extrapolation), ID (Interpolation), Ground Truth, Model Prediction
    ax.legend(handles, labels, loc='lower right', framealpha=0.95, fancybox=True, fontsize=13)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Figure saved to: {output_path}")
    
    # Also save as PDF for paper
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 PDF saved to: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Heat Flux Interpolation/Extrapolation Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str,
                        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861279/checkpoints/epoch=41-step=034860.ckpt",
                        help='Path to AR Bootstrap model checkpoint')
    
    # Data paths
    parser.add_argument('--data-home', type=str,
                        default="/share/crsp/lab/amowli/share/BubbleML_2",
                        help='Base directory for BubbleML data')
    
    # Normalization stats
    parser.add_argument('--norm-stats', type=str,
                        default="/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json",
                        help='Path to normalization stats JSON (default: auto-detect)')
    
    # Frame range for heat flux computation (matching physics_metrics_task123.py)
    # NOTE: These are DATASET INDICES, not HDF5 frame numbers!
    # To match physics_metrics --frame-start 100 --frame-end 105, use the same values here.
    # Actual HDF5 frame = start_time + frame_start (e.g., 100 + 100 = 200)
    parser.add_argument('--start-time', type=int, default=100,
                        help='HDF5 offset for dataset initialization (default 100, matches training)')
    parser.add_argument('--frame-start', type=int, default=100,
                        help='Starting dataset INDEX for heat flux computation (matches physics_metrics)')
    parser.add_argument('--frame-end', type=int, default=400,
                        help='Ending dataset INDEX for heat flux computation (matches physics_metrics)')
    
    # Model parameters
    parser.add_argument('--history-length', type=int, default=10,
                        help='History length for AR Bootstrap model')
    parser.add_argument('--rollout-length', type=int, default=5,
                        help='Rollout length for AR Bootstrap model')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Spatial downsampling factor')
    parser.add_argument('--num-integration-steps', type=int, default=50,
                        help='Number of ODE integration steps')
    parser.add_argument('--solver', type=str, default='rk4',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./ICML/heatflux_interpolation_100_400',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    
    # Skip inference (just compute GT and plot)
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip model inference (only compute GT heat flux)')
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build file paths
    data_subdir = "PoolBoiling-Subcooled-FC72-2D"
    train_files = [os.path.join(args.data_home, data_subdir, f"Twall_{t}.hdf5") 
                   for t in TRAIN_TWALL_VALUES]
    val_files = [os.path.join(args.data_home, data_subdir, f"Twall_{t}.hdf5") 
                 for t in VAL_TWALL_VALUES]
    
    print("=" * 70)
    print("Heat Flux Interpolation/Extrapolation Analysis")
    print("=" * 70)
    # Calculate actual HDF5 frame numbers for display
    effective_start = max(args.start_time, args.history_length)
    first_hdf5_frame = effective_start + args.frame_start
    last_hdf5_frame = effective_start + args.frame_end - 1
    print(f"\nConfiguration (matching physics_metrics_task123.py frame indexing):")
    print(f"  Dataset start_time: {args.start_time}")
    print(f"  Frame indices: {args.frame_start} to {args.frame_end} (dataset indices)")
    print(f"  HDF5 frames: {first_hdf5_frame} to {last_hdf5_frame}")
    print(f"  Downsample factor: {args.downsample_factor}")
    print(f"  Training files: {len(train_files)}")
    print(f"  Validation files: {len(val_files)}")
    print(f"  Output directory: {output_dir}")
    
    # Load normalization stats
    if args.norm_stats:
        norm_stats_path = args.norm_stats
    else:
        norm_stats_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'normalization_stats.json'
        )
    
    print(f"\n📊 Loading normalization stats from: {norm_stats_path}")
    with open(norm_stats_path, 'r') as f:
        normalization_stats = json.load(f)
    
    # Results storage
    results = {}
    
    # ========================================================================
    # Process Training Files (GT only - using dataset pipeline for consistency)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Computing Ground Truth Heat Flux for Training Files")
    print("=" * 70)
    
    for filepath in tqdm(train_files, desc="Training files"):
        if not os.path.exists(filepath):
            print(f"  ⚠️ File not found: {filepath}")
            continue
        
        try:
            # Use dataset approach for GT heat flux (matching physics_metrics_task123.py)
            # GT temperature comes from dataset (normalized -> denormalized)
            hflux, wall_temp = compute_gt_heatflux_from_dataset(
                filepath,
                normalization_stats=normalization_stats,
                start_time=args.start_time,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                history_length=args.history_length,
                rollout_length=args.rollout_length,
                downsample_factor=args.downsample_factor,
            )
            
            results[wall_temp] = {
                'gt_mean': float(np.mean(hflux)),
                'gt_std': float(np.std(hflux)),
                'pred_mean': None,
                'pred_std': None,
                'is_val': False,
                'filepath': filepath,
                'num_frames': len(hflux)
            }
            
        except Exception as e:
            print(f"  ❌ Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # Process Validation Files (GT + Inference)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Processing Validation Files")
    print("=" * 70)
    
    # Load model if doing inference
    model = None
    if not args.skip_inference:
        task_cfg = load_task_config('velocity_from_interface')
        model = load_model(args.checkpoint, normalization_stats, task_cfg, args.device)
    
    for filepath in val_files:
        if not os.path.exists(filepath):
            print(f"  ⚠️ File not found: {filepath}")
            continue
        
        try:
            wall_temp = extract_wall_temp(filepath)
            
            if args.skip_inference:
                # Only compute GT - use dataset approach (matching physics_metrics)
                # GT temperature comes from dataset (normalized -> denormalized)
                hflux, wt = compute_gt_heatflux_from_dataset(
                    filepath,
                    normalization_stats=normalization_stats,
                    start_time=args.start_time,
                    frame_start=args.frame_start,
                    frame_end=args.frame_end,
                    history_length=args.history_length,
                    rollout_length=args.rollout_length,
                    downsample_factor=args.downsample_factor,
                )
                
                results[wall_temp] = {
                    'gt_mean': float(np.mean(hflux)),
                    'gt_std': float(np.std(hflux)),
                    'pred_mean': None,
                    'pred_std': None,
                    'is_val': True,
                    'filepath': filepath,
                    'num_frames': len(hflux)
                }
            else:
                # Run inference
                # Frame indexing matches physics_metrics_task123.py
                gt_hflux, pred_hflux, wt = run_inference_for_heatflux(
                    model, filepath, normalization_stats,
                    start_time=args.start_time,
                    frame_start=args.frame_start,
                    frame_end=args.frame_end,
                    history_length=args.history_length,
                    rollout_length=args.rollout_length,
                    downsample_factor=args.downsample_factor,
                    num_integration_steps=args.num_integration_steps,
                    solver=args.solver,
                    device=args.device
                )
                
                results[wall_temp] = {
                    'gt_mean': float(np.mean(gt_hflux)),
                    'gt_std': float(np.std(gt_hflux)),
                    'pred_mean': float(np.mean(pred_hflux)),
                    'pred_std': float(np.std(pred_hflux)),
                    'is_val': True,
                    'filepath': filepath,
                    'num_frames': len(gt_hflux)
                }
                
        except Exception as e:
            print(f"  ❌ Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # Save Results
    # ========================================================================
    results_path = output_dir / 'heatflux_interpolation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    # ========================================================================
    # Create Plot
    # ========================================================================
    print("\n" + "=" * 70)
    print("Creating Heat Flux Plot")
    print("=" * 70)
    
    plot_path = output_dir / 'heatflux_interpolation_paper.png'
    # Show actual HDF5 frame numbers in title
    effective_start = max(args.start_time, args.history_length)
    first_hdf5 = effective_start + args.frame_start
    last_hdf5 = effective_start + args.frame_end - 1
    title = f"Heat Flux vs Wall Temperature\n(HDF5 Frames {first_hdf5}-{last_hdf5})"
    
    create_heatflux_plot(results, str(plot_path), title=title)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"\n{'Wall Temp (°C)':<15} {'GT Mean':<12} {'GT Std':<10} {'Pred Mean':<12} {'Pred Std':<10} {'Type':<8}")
    print("-" * 70)
    
    for twall in sorted(results.keys()):
        data = results[twall]
        gt_mean = f"{data['gt_mean']:.1f}"
        gt_std = f"{data['gt_std']:.1f}"
        pred_mean = f"{data['pred_mean']:.1f}" if data['pred_mean'] else "N/A"
        pred_std = f"{data['pred_std']:.1f}" if data['pred_std'] else "N/A"
        type_str = "VAL" if data['is_val'] else "TRAIN"
        
        # Mark OOD
        if twall < TRAIN_TWALL_MIN or twall > TRAIN_TWALL_MAX:
            type_str += " (OOD)"
        
        print(f"{twall:<15} {gt_mean:<12} {gt_std:<10} {pred_mean:<12} {pred_std:<10} {type_str:<8}")
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
