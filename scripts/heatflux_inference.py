#!/usr/bin/env python3
"""
Heat Flux Inference Script for Interpolation Analysis

This script runs inference using the flow_matching_ar_bootstrap model on all training
and validation files from the poolboiling_subcooled dataset. It computes heat flux
for both ground truth and predictions, then plots heat flux vs wall temperature
to visualize the model's interpolation ability.

The plot shows:
- One line for ground truth heat flux
- One line for predicted heat flux
- Different symbols for training files (circles) and validation files (triangles)

Usage:
    python heatflux_inference.py --checkpoint /path/to/checkpoint.ckpt
    python heatflux_inference.py --checkpoint /path/to/checkpoint.ckpt --frame-start 0 --frame-end 50
    python heatflux_inference.py --checkpoint /path/to/checkpoint.ckpt --output-dir ./ICML/heatflux_analysis
"""

import sys
import os
import argparse
import json
import numpy as np
import h5py as h5
import torch
import matplotlib.pyplot as plt
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
# Utility Functions
# ============================================================================

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
        return task_cfg
    else:
        print(f"⚠️  Task config not found: {config_path}")
        return DictConfig({
            'name': 'velocity_from_interface',
            'conditioning_channels': [0, 1, 2],
            'conditioning_names': ['sdf', 'velx_interface', 'vely_interface'],
            'target_channels': [1, 2, 0],
            'target_names': ['velx', 'vely', 'temperature']
        })


def extract_wall_temperature(filepath: str) -> float:
    """Extract wall temperature from filename like Twall_96.hdf5 -> 96.0"""
    filename = os.path.basename(filepath)
    match = re.search(r'Twall_(\d+)', filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not extract Twall from filename: {filename}")


def load_data_config(data_home: str = None) -> Tuple[List[str], List[str], str]:
    """Load data configuration and return train/val paths."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'bubblefusion', 'config', 'data_cfg', 'poolboiling_subcooled.yaml'
    )
    
    # Get data home from argument, environment, or use default
    if data_home is None:
        data_home = os.environ.get('DATA_HOME', '/share/crsp/lab/amowli/share/BubbleML_2')
    
    # Read the raw YAML file and manually replace ${data_home}
    with open(config_path, 'r') as f:
        yaml_content = f.read()
    
    # Replace the interpolation with actual path
    yaml_content = yaml_content.replace('${data_home}', data_home)
    
    # Now load with OmegaConf
    data_cfg = OmegaConf.create(yaml_content)
    
    # Extract paths (filter out commented paths which become None)
    train_paths = [p for p in data_cfg.train_paths if p is not None]
    val_paths = [p for p in data_cfg.val_paths if p is not None]
    
    return train_paths, val_paths, data_home


def load_ground_truth_sdf(data_file_path: str, start_time: int = 100, 
                          downsample_factor: int = 4) -> Tuple[np.ndarray, float]:
    """Load SDF and heater temperature from HDF5 file."""
    with h5.File(data_file_path, 'r') as f:
        dfun_full = f['dfun'][start_time:, :, :]  # (T, H, W)
        
        # Downsample if needed
        if downsample_factor > 1:
            dfun = dfun_full[:, ::downsample_factor, ::downsample_factor]
        else:
            dfun = dfun_full
    
    # Extract wall temperature from filename
    heater_temp = extract_wall_temperature(data_file_path)
    
    return dfun, heater_temp


def compute_heatflux(dfun: np.ndarray, temp: np.ndarray, heater_temp: float,
                     lc: float = 0.73e-3, thcl: float = 6.25e-2,
                     downsample_factor: int = 1) -> np.ndarray:
    """
    Calculate heat flux for FC-72 fluid.
    
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

    heater_mask = (x_grid >= -5) & (x_grid <= 5)  # (H, W)
    heater_mask_3d = np.broadcast_to(heater_mask, (T_frames, H, W))  # (T, H, W)

    liquid_mask = dfun < 0  # (T, H, W)
    temp_fields = (heater_mask_3d & liquid_mask).astype(float) * (heater_temp - temp)
    hflux_fields = thcl * (temp_fields / (dx * 0.5 * lc))
    hfluxes = hflux_fields[:, 0, :].mean(axis=1)

    return hfluxes


def load_model(checkpoint_path: str, model_cfg: DictConfig, 
               task_cfg: DictConfig, normalization_stats: dict):
    """Load the AR Bootstrap model from checkpoint."""
    print(f"\n🤖 Loading model from checkpoint: {checkpoint_path}")
    
    optim_cfg = DictConfig({'name': 'adamw', 'lr': 0.001})
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
    
    print(f"✓ AR Bootstrap model loaded successfully!")
    print(f"   History length: {model_cfg.get('history_length', 10)} frames")
    print(f"   Rollout length: {model_cfg.get('rollout_length', 5)} frames")
    
    model.eval()
    return model


def run_inference_single_file(
    model, 
    data_file: str, 
    normalization_stats: dict,
    frame_start: int,
    frame_end: int,
    start_time: int = 100,
    downsample_factor: int = 4,
    num_inference_steps: int = 50,
    history_length: int = 10,
    rollout_length: int = 5,
    solver: str = 'midpoint',
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run inference on a single data file and return GT/pred temperatures.
    
    Returns:
        gt_temp: Ground truth temperature (T, H, W)
        pred_temp: Predicted temperature (T, H, W)
        sdf: Signed distance function (T, H, W)
        heater_temp: Wall temperature in Celsius
    """
    print(f"\n📂 Processing: {os.path.basename(data_file)}")
    
    # Load dataset
    dataset = BulkFlowARBootstrap(
        filenames=[data_file],
        output_fields=['temperature', 'velx', 'vely'],
        start_time=start_time,
        normalization_stats=normalization_stats,
        return_wall_temp=False,
        noise_cfg=None,
        history_length=history_length,
        rollout_length=rollout_length,
        downsample_factor=downsample_factor
    )
    
    # Load ground truth SDF
    sdf_full, heater_temp = load_ground_truth_sdf(data_file, start_time, downsample_factor)
    
    # Get model configuration
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else 2
    
    # Calculate how many segments we need
    # Each segment produces rollout_length frames
    # To avoid overlap, we stride by rollout_length between segments
    total_segments = len(dataset)
    
    # Adjust frame range to segment indices
    # Each segment at index i covers frames [i, i+rollout_length)
    # But segments overlap, so to get non-overlapping frames, we need segment_stride = rollout_length
    segment_stride = rollout_length
    
    # Calculate segment range needed to cover frame_start to frame_end
    start_segment = frame_start // rollout_length
    end_segment = (frame_end + rollout_length - 1) // rollout_length
    
    # Ensure we don't exceed dataset bounds
    max_segments = (total_segments - start_segment * segment_stride) // segment_stride
    num_segments = min(end_segment - start_segment, max_segments)
    
    if num_segments <= 0:
        print(f"   ⚠️ Not enough data for requested frame range")
        num_segments = max(1, max_segments)
    
    print(f"   Frame range: {frame_start} to {frame_end}")
    print(f"   Segments to process: {num_segments}")
    print(f"   Total frames: {num_segments * rollout_length}")
    
    gt_temp_list = []
    pred_temp_list = []
    
    model = model.to(device)
    
    with torch.no_grad():
        for seg_i in tqdm(range(num_segments), desc="  Inference"):
            segment_idx = start_segment * segment_stride + seg_i * segment_stride
            
            if segment_idx >= len(dataset):
                print(f"   ⚠️ Segment index {segment_idx} exceeds dataset length {len(dataset)}")
                break
            
            sample_data = dataset[segment_idx]
            cond_hist, cond_seq, target_seq = sample_data
            
            # Move to device and add batch dimension
            cond_hist = cond_hist.unsqueeze(0).to(device)
            cond_seq = cond_seq.unsqueeze(0).to(device)
            target_seq = target_seq.unsqueeze(0).to(device)
            
            # Extract relevant channels
            cond_hist_extracted = cond_hist[:, :, conditioning_channels, :, :]
            cond_seq_extracted = cond_seq[:, :, conditioning_channels, :, :]
            target_seq_extracted = target_seq[:, :, target_channels, :, :]
            
            B, T_hist, C_cond, H, W = cond_hist_extracted.shape
            L = cond_seq_extracted.shape[1]
            C_out = target_seq_extracted.shape[2]
            
            # Bootstrap initial state
            current_cond_0 = cond_seq_extracted[:, 0]
            bootstrapped_state = model.bootstrap_initial_state(cond_hist_extracted, current_cond_0)
            
            # Run autoregressive rollout
            prev_output = bootstrapped_state
            
            for l in range(L):
                current_cond = cond_seq_extracted[:, l]
                target_l = target_seq_extracted[:, l]
                
                # Availability mask
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
                
                # Move to CPU and denormalize
                target_cpu = target_l.squeeze(0).cpu()
                predicted_cpu = predicted.squeeze(0).cpu()
                
                for j, field_name in enumerate(target_names):
                    target_cpu[j] = dataset._denormalize_field(target_cpu[j], field_name)
                    predicted_cpu[j] = dataset._denormalize_field(predicted_cpu[j], field_name)
                
                # Extract temperature
                gt_temp_list.append(target_cpu[temp_idx].numpy())
                pred_temp_list.append(predicted_cpu[temp_idx].numpy())
                
                # Update prev_output for next frame
                prev_output = predicted.clone()
    
    gt_temp = np.stack(gt_temp_list, axis=0)
    pred_temp = np.stack(pred_temp_list, axis=0)
    
    # Align SDF with inference output
    # The dataset outputs frames starting from start_time + history_length
    effective_start = start_time + history_length + frame_start
    sdf_start_idx = frame_start
    sdf_end_idx = sdf_start_idx + gt_temp.shape[0]
    
    if sdf_end_idx > sdf_full.shape[0]:
        sdf_end_idx = sdf_full.shape[0]
        gt_temp = gt_temp[:sdf_end_idx - sdf_start_idx]
        pred_temp = pred_temp[:sdf_end_idx - sdf_start_idx]
    
    sdf = sdf_full[sdf_start_idx:sdf_end_idx]
    
    print(f"   ✓ Inference complete: {gt_temp.shape[0]} frames")
    
    return gt_temp, pred_temp, sdf, heater_temp


def compute_gt_heatflux_only(
    data_file: str,
    frame_start: int,
    frame_end: int,
    start_time: int = 100,
    downsample_factor: int = 4,
) -> Tuple[float, float, float, int]:
    """
    Compute ground truth heat flux from data file without running inference.
    
    Args:
        data_file: Path to HDF5 file
        frame_start: Starting frame index (relative to start_time, 0-based)
        frame_end: Ending frame index (relative to start_time)
        start_time: Initial timestep to skip in HDF5 files (for transient)
        downsample_factor: Spatial downsampling factor
    
    Returns:
        gt_hflux_mean: Mean ground truth heat flux
        gt_hflux_std: Std of ground truth heat flux
        heater_temp: Wall temperature
        num_frames: Number of frames processed
    """
    print(f"\n📂 Loading GT from: {os.path.basename(data_file)}")
    
    # Load ground truth SDF and temperature from HDF5
    # Read from start_time, so indices are relative to start_time
    with h5.File(data_file, 'r') as f:
        dfun_full = f['dfun'][start_time:, :, :]  # (T, H, W), starting from start_time
        temp_full = f['temperature'][start_time:, :, :]  # (T, H, W)
        
        # Downsample if needed
        if downsample_factor > 1:
            dfun = dfun_full[:, ::downsample_factor, ::downsample_factor]
            temp = temp_full[:, ::downsample_factor, ::downsample_factor]
        else:
            dfun = dfun_full
            temp = temp_full
    
    # Extract wall temperature from filename
    heater_temp = extract_wall_temperature(data_file)
    
    # frame_start and frame_end are relative to start_time (0-based after start_time)
    # Use directly as indices since we already loaded data starting from start_time
    actual_end = min(frame_end, dfun.shape[0])
    dfun = dfun[frame_start:actual_end]
    temp = temp[frame_start:actual_end]
    
    print(f"   Using frames {frame_start} to {actual_end} (relative to start_time={start_time})")
    print(f"   Absolute HDF5 frames: {start_time + frame_start} to {start_time + actual_end}")
    
    # Compute heat flux
    gt_hflux = compute_heatflux(dfun, temp, heater_temp, downsample_factor=downsample_factor)
    
    print(f"   ✓ GT heat flux: {np.mean(gt_hflux):.2f}±{np.std(gt_hflux):.2f} W/m² ({len(gt_hflux)} frames)")
    
    return np.mean(gt_hflux), np.std(gt_hflux), heater_temp, len(gt_hflux)


def run_heatflux_analysis(
    model,
    train_files: List[str],
    val_files: List[str],
    normalization_stats: dict,
    frame_start: int,
    frame_end: int,
    start_time: int = 100,
    downsample_factor: int = 4,
    num_inference_steps: int = 50,
    history_length: int = 10,
    rollout_length: int = 5,
    solver: str = 'midpoint',
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    Run heat flux analysis:
    - Training files: only compute ground truth (no inference)
    - Validation files: run inference and compute both GT and prediction
    
    Returns:
        results: Dict mapping Twall -> {gt_hflux, pred_hflux (if val), is_val}
    """
    results = {}
    
    # Process training files - GT only (no inference)
    print(f"\n{'='*60}")
    print("📊 Processing TRAINING files (ground truth only)")
    print(f"{'='*60}")
    
    for data_file in train_files:
        try:
            gt_mean, gt_std, heater_temp, num_frames = compute_gt_heatflux_only(
                data_file, frame_start, frame_end, start_time, downsample_factor
            )
            
            results[heater_temp] = {
                'gt_hflux_mean': gt_mean,
                'gt_hflux_std': gt_std,
                'pred_hflux_mean': None,  # No prediction for training
                'pred_hflux_std': None,
                'is_val': False,
                'filepath': data_file,
                'num_frames': num_frames
            }
            
            print(f"   🔥 Twall={heater_temp}°C: GT={gt_mean:.2f}±{gt_std:.2f} W/m²")
            
        except Exception as e:
            print(f"   ❌ Error processing {data_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Process validation files - run inference
    print(f"\n{'='*60}")
    print("🔬 Processing VALIDATION files (with inference)")
    print(f"{'='*60}")
    
    for data_file in val_files:
        try:
            # First compute GT heat flux the same way as training (directly from HDF5)
            # This ensures consistency between training and validation GT
            gt_mean, gt_std, heater_temp, num_frames_gt = compute_gt_heatflux_only(
                data_file, frame_start, frame_end, start_time, downsample_factor
            )
            
            # Run inference for predictions only
            _, pred_temp, sdf, _ = run_inference_single_file(
                model, data_file, normalization_stats,
                frame_start, frame_end, start_time, downsample_factor,
                num_inference_steps, history_length, rollout_length, solver, device
            )
            
            # Compute predicted heat flux using the SDF from inference (aligned with pred_temp)
            pred_hflux = compute_heatflux(sdf, pred_temp, heater_temp,
                                          downsample_factor=downsample_factor)
            
            results[heater_temp] = {
                'gt_hflux_mean': gt_mean,
                'gt_hflux_std': gt_std,
                'pred_hflux_mean': np.mean(pred_hflux),
                'pred_hflux_std': np.std(pred_hflux),
                'is_val': True,
                'filepath': data_file,
                'num_frames': len(pred_hflux)
            }
            
            print(f"   🔥 Twall={heater_temp}°C: GT={gt_mean:.2f}±{gt_std:.2f}, "
                  f"Pred={np.mean(pred_hflux):.2f}±{np.std(pred_hflux):.2f} W/m²")
            
        except Exception as e:
            print(f"   ❌ Error processing {data_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def add_id_ood_background(ax, train_twalls: List[float], x_min: float, x_max: float):
    """
    Add colored background regions to distinguish ID (interpolation) vs OOD (extrapolation).
    
    Args:
        ax: matplotlib axis
        train_twalls: List of training wall temperatures (defines the ID region)
        x_min: Minimum x value of the plot
        x_max: Maximum x value of the plot
    """
    if not train_twalls:
        return
    
    # Training range defines ID region
    train_min = min(train_twalls)
    train_max = max(train_twalls)
    
    # Colors for regions (subtle, semi-transparent)
    OOD_COLOR = '#FFCCCC'  # Light red/pink for OOD (extrapolation)
    ID_COLOR = '#CCFFCC'   # Light green for ID (interpolation)
    ALPHA = 0.3
    
    # Left OOD region (extrapolation below training range)
    if x_min < train_min:
        ax.axvspan(x_min, train_min, alpha=ALPHA, color=OOD_COLOR, zorder=0, label='_nolegend_')
    
    # ID region (interpolation within training range)
    ax.axvspan(train_min, train_max, alpha=ALPHA, color=ID_COLOR, zorder=0, label='_nolegend_')
    
    # Right OOD region (extrapolation above training range)
    if x_max > train_max:
        ax.axvspan(train_max, x_max, alpha=ALPHA, color=OOD_COLOR, zorder=0, label='_nolegend_')


def plot_heatflux(results: Dict[str, Dict], output_path: str, title: str = None):
    """
    Create heat flux vs wall temperature plot (publication quality).
    
    Shows:
    - Training data: ground truth line with markers
    - Validation data: prediction dots with error bars
    - Background shading: green for ID (interpolation), pink for OOD (extrapolation)
    
    Args:
        results: Dict mapping Twall -> {gt_hflux_mean, pred_hflux_mean, is_val, ...}
        output_path: Path to save the plot
        title: Optional plot title
    """
    # Publication-quality font sizes
    TITLE_SIZE = 24
    LABEL_SIZE = 22
    TICK_SIZE = 18
    LEGEND_SIZE = 16
    
    # Sort by wall temperature
    twalls = sorted(results.keys())
    
    # Separate train and validation
    train_twalls = [t for t in twalls if not results[t]['is_val']]
    val_twalls = [t for t in twalls if results[t]['is_val']]
    
    # Extract values for training data (GT only)
    train_gt = [results[t]['gt_hflux_mean'] for t in train_twalls]
    train_gt_std = [results[t]['gt_hflux_std'] for t in train_twalls]
    
    # Extract values for validation data (GT and prediction)
    val_gt = [results[t]['gt_hflux_mean'] for t in val_twalls]
    val_gt_std = [results[t]['gt_hflux_std'] for t in val_twalls]
    val_pred = [results[t]['pred_hflux_mean'] for t in val_twalls]
    val_pred_std = [results[t]['pred_hflux_std'] for t in val_twalls]
    
    # All GT values for the ground truth line (train + val)
    all_gt = [results[t]['gt_hflux_mean'] for t in twalls]
    all_gt_std = [results[t]['gt_hflux_std'] for t in twalls]
    
    # Set reasonable y-limits with some padding
    all_values = all_gt + val_pred
    if all_values:
        y_min = min(all_values) * 0.9
        y_max = max(all_values) * 1.1
    
    # X-axis limits with padding
    x_min = min(twalls) - 1
    x_max = max(twalls) + 1
    
    # =========================================================================
    # Plot 1: With error bars (detailed view)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Add ID/OOD background shading first (behind everything)
    add_id_ood_background(ax, train_twalls, x_min, x_max)
    
    # Plot ground truth line connecting all points (train + val GT)
    ax.plot(twalls, all_gt, 'b-', linewidth=3, alpha=0.7, label='Ground Truth')
    
    # Plot training GT points with circles and error bars
    if train_twalls:
        ax.errorbar(train_twalls, train_gt, yerr=train_gt_std, 
                    fmt='o', markersize=14, capsize=6, capthick=2.5,
                    color='blue', markeredgecolor='darkblue', markeredgewidth=2,
                    label='GT (Training)', elinewidth=2, zorder=5)
    
    # Plot validation GT points (for reference on the line)
    if val_twalls:
        ax.errorbar(val_twalls, val_gt, yerr=val_gt_std,
                    fmt='o', markersize=14, capsize=6, capthick=2.5,
                    color='blue', markeredgecolor='darkblue', markeredgewidth=2,
                    elinewidth=2, zorder=5)  # No separate label - part of GT
    
    # Plot validation prediction dots with error bars (highlighted)
    if val_twalls:
        ax.errorbar(val_twalls, val_pred, yerr=val_pred_std,
                    fmt='^', markersize=18, capsize=8, capthick=3,
                    color='red', markeredgecolor='darkred', markeredgewidth=3,
                    label='Prediction (Validation)', elinewidth=3, zorder=10)
    
    # Labels and formatting
    ax.set_xlabel(r'Wall Temperature $T_{\mathrm{wall}}$ [°C]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel(r'Heat Flux $q$ [W/m²]', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_xlim(x_min, x_max)
    
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    else:
        ax.set_title('Heat Flux vs Wall Temperature', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    
    # Add region labels
    if train_twalls:
        train_min, train_max = min(train_twalls), max(train_twalls)
        # Add text annotations for regions
        ax.text(x_min + 0.5, y_max * 0.97, 'OOD\n(Extrap.)', fontsize=12, 
                ha='left', va='top', fontweight='bold', color='darkred', alpha=0.7)
        ax.text((train_min + train_max) / 2, y_max * 0.97, 'ID (Interpolation)', fontsize=12,
                ha='center', va='top', fontweight='bold', color='darkgreen', alpha=0.7)
        ax.text(x_max - 0.5, y_max * 0.97, 'OOD\n(Extrap.)', fontsize=12,
                ha='right', va='top', fontweight='bold', color='darkred', alpha=0.7)
    
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE, framealpha=0.95, 
              edgecolor='black', fancybox=False)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.tick_params(axis='both', labelsize=TICK_SIZE, width=2, length=8)
    
    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    if all_values:
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    plt.close()
    
    # =========================================================================
    # Plot 2: Clean version (GT line + validation prediction dots)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    
    # Add ID/OOD background shading
    add_id_ood_background(ax2, train_twalls, x_min, x_max)
    
    # Plot GT line with markers for all temps
    ax2.plot(twalls, all_gt, 'b-', linewidth=3.5, label='Ground Truth', 
             marker='o', markersize=12, markeredgecolor='darkblue', markeredgewidth=2)
    
    # Highlight validation prediction with larger triangles
    if val_twalls:
        ax2.scatter(val_twalls, val_pred,
                    c='red', s=400, marker='^', edgecolors='darkred', linewidths=3,
                    label='Prediction (Validation)', zorder=10)
    
    ax2.set_xlabel(r'Wall Temperature $T_{\mathrm{wall}}$ [°C]', fontsize=LABEL_SIZE, fontweight='bold')
    ax2.set_ylabel(r'Heat Flux $q$ [W/m²]', fontsize=LABEL_SIZE, fontweight='bold')
    ax2.set_xlim(x_min, x_max)
    ax2.set_title('Heat Flux vs Wall Temperature', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    
    # Add region labels
    if train_twalls:
        train_min, train_max = min(train_twalls), max(train_twalls)
        ax2.text(x_min + 0.5, y_max * 0.97, 'OOD', fontsize=12, 
                ha='left', va='top', fontweight='bold', color='darkred', alpha=0.7)
        ax2.text((train_min + train_max) / 2, y_max * 0.97, 'ID', fontsize=12,
                ha='center', va='top', fontweight='bold', color='darkgreen', alpha=0.7)
        ax2.text(x_max - 0.5, y_max * 0.97, 'OOD', fontsize=12,
                ha='right', va='top', fontweight='bold', color='darkred', alpha=0.7)
    
    ax2.legend(loc='lower right', fontsize=LEGEND_SIZE, framealpha=0.95,
               edgecolor='black', fancybox=False)
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.tick_params(axis='both', labelsize=TICK_SIZE, width=2, length=8)
    
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    
    if all_values:
        ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    output_path_clean = output_path.replace('.png', '_clean.png')
    plt.savefig(output_path_clean, dpi=300, bbox_inches='tight')
    print(f"📊 Clean plot saved to: {output_path_clean}")
    plt.close()
    
    # =========================================================================
    # Plot 3: Paper-ready (GT line + val prediction dots with error bars)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    # Add ID/OOD background shading
    add_id_ood_background(ax3, train_twalls, x_min, x_max)
    
    # Ground truth line
    ax3.plot(twalls, all_gt, 'b-', linewidth=2.5, alpha=0.8)
    
    # Training GT points
    if train_twalls:
        ax3.scatter(train_twalls, train_gt,
                    c='blue', s=150, marker='o', edgecolors='darkblue', linewidths=2,
                    label='Ground Truth (Train)', zorder=5)
    
    # Validation GT points (on the line)
    if val_twalls:
        ax3.scatter(val_twalls, val_gt,
                    c='blue', s=150, marker='o', edgecolors='darkblue', linewidths=2,
                    zorder=5)  # No separate label
    
    # Validation prediction dots with error bars
    if val_twalls:
        ax3.errorbar(val_twalls, val_pred, yerr=val_pred_std,
                     fmt='^', markersize=14, capsize=6, capthick=2.5,
                     color='red', markeredgecolor='darkred', markeredgewidth=3,
                     label='Prediction (Val)', elinewidth=2.5, zorder=10)
    
    ax3.set_xlabel(r'$T_{\mathrm{wall}}$ [°C]', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.set_ylabel(r'Heat Flux [W/m²]', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.set_xlim(x_min, x_max)
    
    # Add region labels (smaller for paper version)
    if train_twalls:
        train_min, train_max = min(train_twalls), max(train_twalls)
        ax3.text(x_min + 0.3, y_max * 0.97, 'OOD', fontsize=10, 
                ha='left', va='top', fontweight='bold', color='darkred', alpha=0.7)
        ax3.text((train_min + train_max) / 2, y_max * 0.97, 'ID', fontsize=10,
                ha='center', va='top', fontweight='bold', color='darkgreen', alpha=0.7)
        ax3.text(x_max - 0.3, y_max * 0.97, 'OOD', fontsize=10,
                ha='right', va='top', fontweight='bold', color='darkred', alpha=0.7)
    
    ax3.legend(loc='lower right', fontsize=LEGEND_SIZE-2, framealpha=0.95,
               edgecolor='black', fancybox=False)
    ax3.grid(True, alpha=0.3, linewidth=1.5)
    ax3.tick_params(axis='both', labelsize=TICK_SIZE, width=2, length=8)
    
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
    
    if all_values:
        ax3.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    output_path_paper = output_path.replace('.png', '_paper.png')
    plt.savefig(output_path_paper, dpi=300, bbox_inches='tight')
    print(f"📊 Paper-ready plot saved to: {output_path_paper}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Heat Flux Inference for Interpolation Analysis')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str,
                        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_pf_max3_pf_max3_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861270/checkpoints/last.ckpt",
                        help='Path to model checkpoint')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./ICML/heatflux_analysis_300_350',
                        help='Output directory for plots and results')
    parser.add_argument('--output-filename', type=str, default='heatflux_interpolation',
                        help='Base filename for outputs (without extension)')
    
    # Frame range
    parser.add_argument('--frame-start', type=int, default=300,
                        help='Starting frame index for heat flux averaging')
    parser.add_argument('--frame-end', type=int, default=350,
                        help='Ending frame index for heat flux averaging')
    
    # Model parameters
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='Number of ODE integration steps')
    parser.add_argument('--start-time', type=int, default=100,
                        help='Starting timestep in HDF5 files')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Downsampling factor (1=full res, 4=128x128)')
    parser.add_argument('--history-length', type=int, default=10,
                        help='History length for bootstrap model')
    parser.add_argument('--rollout-length', type=int, default=5,
                        help='Rollout length for AR segments')
    parser.add_argument('--solver', type=str, default='midpoint',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    
    # Additional options
    parser.add_argument('--normalization-stats', type=str, default="/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json",
                        help='Path to normalization_stats.json (auto-detect if not provided)')
    parser.add_argument('--data-home', type=str, default='/share/crsp/lab/amowli/share/BubbleML_2',
                        help='Root directory containing BubbleML data')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for the plot')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🔥 HEAT FLUX INFERENCE FOR INTERPOLATION ANALYSIS")
    print("=" * 80)
    
    # Step 1: Load task config
    task_cfg = load_task_config('velocity_from_interface')
    
    # Step 2: Load data config to get file paths
    train_paths, val_paths, data_home = load_data_config(args.data_home)
    
    print(f"\n📁 Data configuration:")
    print(f"   Data home: {data_home}")
    print(f"   Training files: {len(train_paths)}")
    for p in train_paths:
        print(f"      - {os.path.basename(p)}")
    print(f"   Validation files: {len(val_paths)}")
    for p in val_paths:
        print(f"      - {os.path.basename(p)}")
    
    # Filter to only existing files
    existing_train = [p for p in train_paths if os.path.exists(p)]
    existing_val = [p for p in val_paths if os.path.exists(p)]
    
    if not existing_train and not existing_val:
        print(f"\n❌ No data files found! Check DATA_HOME environment variable.")
        print(f"   Expected path: {data_home}")
        return
    
    print(f"\n✓ Found {len(existing_train)} training files and {len(existing_val)} validation files")
    
    # Step 3: Create model config
    model_cfg = DictConfig({
        'name': 'flow_matching_ar_bootstrap',
        'in_channels': 10,
        'out_channels': 3,
        'base_channels': 32,
        'time_embed_dim': 256,
        'num_res_blocks': 2,
        'use_attention': False,
        'dropout': 0.1,
        'num_integration_steps': args.num_inference_steps,
        'temp_min': 55.0,
        'temp_max': 120.0,
        'history_length': args.history_length,
        'rollout_length': args.rollout_length,
        'use_availability_mask': True,
        'history_encoder_type': 'temporal_mixer',
        'history_encoder_hidden': 32,
        'bootstrap_loss_weight': 1.0,
        'ar_loss_weight': 1.0,
        'bootstrap_state_loss_weight': 0.5,
        'inference': {'solver': args.solver},
    })
    
    # Step 4: Load normalization stats
    normalization_stats = None
    
    if args.normalization_stats and os.path.exists(args.normalization_stats):
        print(f"\n📊 Loading normalization stats from: {args.normalization_stats}")
        with open(args.normalization_stats, 'r') as f:
            normalization_stats = json.load(f)
    else:
        # Try checkpoint directory
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if "checkpoints" in checkpoint_dir:
            checkpoint_dir = os.path.dirname(checkpoint_dir)
        stats_file = os.path.join(checkpoint_dir, "normalization_stats.json")
        
        if os.path.exists(stats_file):
            print(f"\n📊 Loading normalization stats from training: {stats_file}")
            with open(stats_file, 'r') as f:
                normalization_stats = json.load(f)
        else:
            # Compute from training files
            print(f"\n📊 Computing normalization stats from training files...")
            all_files = existing_train + existing_val
            if all_files:
                normalization_stats = compute_normalization_stats(
                    filenames=all_files[:3],  # Use first 3 files to speed up
                    start_time=args.start_time,
                    verbose=True
                )
    
    if normalization_stats:
        print(f"   Temperature: [{normalization_stats['temperature']['min']:.2f}, "
              f"{normalization_stats['temperature']['max']:.2f}]°C")
        print(f"   Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
        print(f"   SDF scale: {normalization_stats['sdf']['scale']:.4f}")
    
    # Step 5: Load model
    model = load_model(args.checkpoint, model_cfg, task_cfg, normalization_stats)
    
    # Step 6: Run heat flux analysis
    # - Training files: GT only (no inference)
    # - Validation files: run inference
    
    print(f"\n{'='*80}")
    print(f"🔬 Heat Flux Analysis")
    print(f"   Training files (GT only): {len(existing_train)}")
    print(f"   Validation files (inference): {len(existing_val)}")
    print(f"   Frame range: {args.frame_start} to {args.frame_end}")
    print(f"{'='*80}")
    
    results = run_heatflux_analysis(
        model, existing_train, existing_val, normalization_stats,
        args.frame_start, args.frame_end,
        args.start_time, args.downsample_factor,
        args.num_inference_steps, args.history_length, args.rollout_length,
        args.solver, args.device
    )
    
    # Step 7: Save results to JSON
    results_file = os.path.join(args.output_dir, f"{args.output_filename}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Results saved to: {results_file}")
    
    # Step 8: Create plot
    plot_path = os.path.join(args.output_dir, f"{args.output_filename}.png")
    plot_heatflux(results, plot_path, title=args.title)
    
    # Step 9: Print summary table
    print(f"\n{'='*80}")
    print("📊 HEAT FLUX SUMMARY")
    print(f"{'='*80}")
    print(f"{'Twall [°C]':>12} {'Type':>10} {'GT HFlux':>15} {'Pred HFlux':>15} {'Rel Error':>12}")
    print(f"{'-'*12} {'-'*10} {'-'*15} {'-'*15} {'-'*12}")
    
    for twall in sorted(results.keys()):
        r = results[twall]
        file_type = "VAL" if r['is_val'] else "TRAIN"
        gt_str = f"{r['gt_hflux_mean']:>12.2f}±{r['gt_hflux_std']:>5.2f}"
        
        if r['pred_hflux_mean'] is not None:
            pred_str = f"{r['pred_hflux_mean']:>12.2f}±{r['pred_hflux_std']:>5.2f}"
            rel_error = abs(r['gt_hflux_mean'] - r['pred_hflux_mean']) / r['gt_hflux_mean'] * 100
            error_str = f"{rel_error:>11.2f}%"
        else:
            pred_str = "        N/A       "
            error_str = "        N/A "
        
        print(f"{twall:>12.0f} {file_type:>10} {gt_str} {pred_str} {error_str}")
    
    print(f"\n✅ Heat flux analysis complete!")
    print(f"   Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
