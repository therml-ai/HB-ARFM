#!/usr/bin/env python3
"""
Temperature Row Analysis Script - Compare ground truth vs model predictions along horizontal rows.

This script analyzes temperature profiles along horizontal rows (y-coordinates) of the temperature field.
For each selected row, it:
1. Extracts temperature values along the entire width (x-direction)
2. Averages these values across multiple frames
3. Plots ground truth vs model predictions for comparison

Usage:
    python temperature_row_analysis.py [checkpoint_path] [data_file_path] [--options]
    
Features:
- Row-wise temperature profile analysis
- Frame averaging for cleaner visualization
- Configurable row spacing and range
- Support for DDPM, Flow Matching, and UNet models
- Two plot outputs:
  * Individual row subplots with error bands (standard deviation)
  * Combined plot with all rows on same graph (no error bands)
"""

import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bubblefusion.models.unet import UNetLightning
from bubblefusion.models.flow_matching import ConditionalFlowMatchingLightning
from bubblefusion.models.flow_matching_ar_bootstrap import ConditionalFlowMatchingARBootstrapLightning
from bubblefusion.models.ddpm import BubbleDDPMLightning
from bubblefusion.data import BulkFlow
from bubblefusion.data.bubbleml import compute_normalization_stats, BulkFlowARBootstrap
from omegaconf import OmegaConf


def load_model_from_checkpoint(checkpoint_path: str, model_cfg: DictConfig, 
                             optim_cfg: DictConfig, scheduler_cfg: DictConfig,
                             task_cfg: DictConfig = None, normalization_stats: dict = None):
    """Load the trained model from checkpoint (auto-detects model type)."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Detect model type from config
    model_name = model_cfg.get('name', 'flow_matching').lower()
    
    # Load the appropriate model
    if model_name == 'unet':
        print(f"📦 Loading UNet model...")
        model = UNetLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg
        )
        model_type = 'unet'
    elif model_name == 'bubble_ddpm':
        print(f"📦 Loading DDPM model...")
        model = BubbleDDPMLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg
        )
        model_type = 'bubble_ddpm'
    elif model_name == 'flow_matching':
        print(f"📦 Loading Flow Matching model...")
        model = ConditionalFlowMatchingLightning.load_from_checkpoint(
            checkpoint_path,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg
        )
        model_type = 'flow_matching'
    elif model_name == 'flow_matching_ar_bootstrap':
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
        model_type = 'flow_matching_ar_bootstrap'
        print(f"   Bootstrap: Uses history encoder to infer initial state")
        print(f"   History length: {model_cfg.get('history_length', 10)} frames")
        print(f"   Rollout length: {model_cfg.get('rollout_length', 5)} frames")
        print(f"   Default solver: {model.default_solver}")

    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    model.eval()
    print(f"✓ {model_name.upper()} model loaded successfully!")
    return model, model_type


def load_dataset(data_file_path: str, output_fields=None, start_time=100, 
                normalize_temperature=True, return_wall_temp=False,
                is_ar_bootstrap=False, history_length=10, rollout_length=5,
                normalization_stats=None, noise_cfg=None, downsample_factor=1):
    """Load the BulkFlow or BulkFlowARBootstrap dataset."""
    print(f"Loading data from: {data_file_path}")
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    if output_fields is None:
        output_fields = ['temperature', 'velx', 'vely']
    
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
        print(f"      Rollout length: {rollout_length}")
        
        dataset = BulkFlowARBootstrap(
            filenames=[data_file_path],
            output_fields=output_fields,
            start_time=start_time,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=noise_cfg,
            history_length=history_length,
            rollout_length=rollout_length,
            downsample_factor=downsample_factor
        )
        print(f"✓ AR Bootstrap dataset loaded: {len(dataset)} samples")
        print(f"   Each sample: condition_history [T_hist, C, H, W], condition_seq [L, C, H, W], target_seq [L, C, H, W]")
    else:
        dataset = BulkFlow(
            filenames=[data_file_path],
            output_fields=output_fields,
            start_time=start_time,
            normalization_stats=normalization_stats,
            return_wall_temp=return_wall_temp,
            noise_cfg=noise_cfg,
            downsample_factor=downsample_factor
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples (from timestep {start_time} onwards)")
    
    return dataset


def run_inference_batch(model, dataset, frame_indices, device='cuda', 
                       num_inference_steps=50, model_type='bubble_ddpm', guidance_scale=1.0,
                       target_channel_idx=0, solver='heun'):
    """
    Run inference on a batch of frames and return predictions + ground truth.
    
    Args:
        model: Trained model
        dataset: BulkFlow or BulkFlowARBootstrap dataset
        frame_indices: List of frame indices to process
        device: Device to run on
        num_inference_steps: Number of inference steps
        model_type: 'bubble_ddpm', 'unet', 'flow_matching', 'flow_matching_spatial', 
                    'flow_matching_guidance', or 'flow_matching_ar_bootstrap'
        guidance_scale: Guidance scale for flow_matching_guidance (1.0=no guidance, >1.0=stronger conditioning)
        target_channel_idx: Index of target channel to extract for temperature analysis
        solver: ODE solver for flow matching models ('euler', 'heun', 'midpoint', 'rk4')
    
    Returns:
        ground_truth: [num_frames, H, W] - Ground truth temperature
        predictions: [num_frames, H, W] - Model predictions
    """
    ground_truth_list = []
    predictions_list = []
    
    is_ar_bootstrap = (model_type == 'flow_matching_ar_bootstrap')
    
    print(f"\n🔮 Running inference on {len(frame_indices)} frames...")
    if is_ar_bootstrap:
        print(f"   Model type: flow_matching_ar_bootstrap (bootstrap + autoregressive)")
        print(f"   Using bootstrapped initial state from conditioning history")
    
    for idx, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(dataset):
            print(f"⚠️  Frame {frame_idx} exceeds dataset length, skipping...")
            continue
        
        # Get sample from dataset
        sample_data = dataset[frame_idx]
        
        if is_ar_bootstrap:
            # AR Bootstrap model: dataset returns (cond_history, cond_sequence, target_sequence)
            if dataset.return_wall_temp:
                cond_hist, cond_seq, target_seq, wall_temp = sample_data
            else:
                cond_hist, cond_seq, target_seq = sample_data
            
            # Move to device
            cond_hist = cond_hist.unsqueeze(0).to(device)    # [1, T_hist, C, H, W]
            cond_seq = cond_seq.unsqueeze(0).to(device)      # [1, L, C, H, W]
            target_seq = target_seq.unsqueeze(0).to(device)  # [1, L, C, H, W]
            
            # Extract relevant channels from model
            conditioning_channels = model.conditioning_channels
            target_channels = model.target_channels
            
            cond_hist_extracted = cond_hist[:, :, conditioning_channels, :, :]
            cond_seq_extracted = cond_seq[:, :, conditioning_channels, :, :]
            target_seq_extracted = target_seq[:, :, target_channels, :, :]
            
            # Bootstrap initial state from history using first frame's conditioning
            current_cond = cond_seq_extracted[:, 0]  # [1, C_cond, H, W]
            bootstrapped_state = model.bootstrap_initial_state(cond_hist_extracted, current_cond)
            
            B, _, H, W = current_cond.shape
            C_out = target_seq_extracted.shape[2]
            availability_mask = torch.zeros(B, 1, H, W, device=device)  # Bootstrap mode
            
            # Generate prediction using bootstrap state
            with torch.no_grad():
                predicted = model.sample(
                    condition=current_cond,
                    prev_output=bootstrapped_state,
                    shape=(B, C_out, H, W),
                    device=device,
                    availability_mask=availability_mask,
                    num_integration_steps=num_inference_steps,
                    solver=solver,
                )
            
            # Get target (first frame of sequence)
            target = target_seq_extracted[:, 0]  # [1, C_out, H, W]
            
            # Extract temperature channel (target_channel_idx specifies which channel)
            # For Task 2/3: target_channels = [1, 2, 0] -> velx, vely, temperature
            # So temperature is at index 2 in the model output
            temp_idx = target_channel_idx
            target_temperature = target[:, temp_idx:temp_idx+1, :, :]
            predicted_temperature = predicted[:, temp_idx:temp_idx+1, :, :]
            
            # Move to CPU
            target_temperature = target_temperature.squeeze(0).cpu()
            predicted_temperature = predicted_temperature.squeeze(0).cpu()
            
            # Denormalize temperature
            target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
            temp_field_name = target_names[temp_idx] if temp_idx < len(target_names) else 'temperature'
            target_temperature = dataset._denormalize_field(target_temperature, temp_field_name)
            predicted_temperature = dataset._denormalize_field(predicted_temperature, temp_field_name)
            
        else:
            # Standard BulkFlow dataset
            # Get file index for denormalization
            file_idx = dataset.get_file_index(frame_idx)
            
            if dataset.return_wall_temp:
                input_data, target_data, wall_temp = sample_data
                wall_temp_batch = torch.tensor([wall_temp], dtype=torch.float32).to(device)
            else:
                input_data, target_data = sample_data
                wall_temp_batch = None
            
            # Extract SDF and temperature
            sdf = input_data[0:1, :, :]  # [1, H, W]
            target_temperature = target_data[0:1, :, :]  # [1, H, W]
            
            # Add batch dimension and move to device
            sdf_batch = sdf.unsqueeze(0).to(device)  # (1, 1, H, W)
            target_batch = target_temperature.unsqueeze(0).to(device)  # (1, 1, H, W)
            
            # Run inference based on model type
            with torch.no_grad():
                if model_type == 'bubble_ddpm':
                    predicted_temperature = model.ddpm.p_sample_loop(
                        condition=sdf_batch,
                        shape=(1, 1, sdf.shape[1], sdf.shape[2]),
                        device=device,
                        num_inference_steps=num_inference_steps
                    )
                    
                    # Apply inverse conditioning if needed
                    if model.conditioning_strategy == 'bias' and wall_temp_batch is not None:
                        bias = model.wall_temp_conditioner(wall_temp_batch)
                        predicted_temperature = predicted_temperature - bias
                    elif model.conditioning_strategy == 'film' and wall_temp_batch is not None:
                        if wall_temp_batch.dim() == 1:
                            wall_temp_batch = wall_temp_batch.unsqueeze(-1)
                        wall_temp_norm = (wall_temp_batch - 87.5) / 32.5
                        film_params = model.wall_temp_conditioner.film_net(wall_temp_norm)
                        gamma_raw, beta = torch.chunk(film_params, 2, dim=1)
                        gamma_range = model.wall_temp_conditioner.gamma_range
                        gamma = (1 - gamma_range) + 2 * gamma_range * torch.sigmoid(gamma_raw)
                        gamma = gamma.view(-1, 1, 1, 1)
                        beta = beta.view(-1, 1, 1, 1)
                        predicted_temperature = (predicted_temperature - beta) / (gamma + 1e-8)
                
                elif model_type == 'flow_matching':
                    predicted_temperature = model.flow_matching.sample(
                        condition=sdf_batch,
                        shape=(1, 1, sdf.shape[1], sdf.shape[2]),
                        device=device,
                        num_integration_steps=num_inference_steps
                    )
                    
                    # Apply inverse conditioning if needed
                    if model.conditioning_strategy == 'bias' and wall_temp_batch is not None:
                        bias = model.wall_temp_conditioner(wall_temp_batch)
                        predicted_temperature = predicted_temperature - bias
                    elif model.conditioning_strategy == 'film' and wall_temp_batch is not None:
                        if wall_temp_batch.dim() == 1:
                            wall_temp_batch = wall_temp_batch.unsqueeze(-1)
                        wall_temp_norm = (wall_temp_batch - 87.5) / 32.5
                        film_params = model.wall_temp_conditioner.film_net(wall_temp_norm)
                        gamma_raw, beta = torch.chunk(film_params, 2, dim=1)
                        gamma_range = model.wall_temp_conditioner.gamma_range
                        gamma = (1 - gamma_range) + 2 * gamma_range * torch.sigmoid(gamma_raw)
                        gamma = gamma.view(-1, 1, 1, 1)
                        beta = beta.view(-1, 1, 1, 1)
                        predicted_temperature = (predicted_temperature - beta) / (gamma + 1e-8)
                
                elif model_type == 'flow_matching_spatial':
                    # Flow Matching Spatial: ODE integration with wall temp as input
                    if wall_temp_batch is None:
                        raise ValueError("Flow Matching Spatial requires wall temperature!")
                    predicted_temperature = model.flow_matching.sample(
                        condition=sdf_batch,
                        wall_temp=wall_temp_batch,
                        shape=(1, 1, sdf.shape[1], sdf.shape[2]),
                        device=device,
                        num_integration_steps=num_inference_steps
                    )
                    # No inverse conditioning needed - wall temp is baked into the prediction
                
                elif model_type == 'flow_matching_guidance':
                    # Flow Matching Guidance: ODE integration with CFG
                    if wall_temp_batch is None:
                        raise ValueError("Flow Matching Guidance requires wall temperature!")
                    predicted_temperature = model.flow_matching.sample(
                        sdf=sdf_batch,
                        wall_temp=wall_temp_batch,
                        shape=(1, 1, sdf.shape[1], sdf.shape[2]),
                        device=device,
                        num_integration_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    )
                    # No inverse conditioning needed - wall temp is baked into the prediction
                
                else:  # model_type == 'unet'
                    predicted_temperature = model(sdf_batch)
                    
                    # Apply conditioning if needed
                    if model.conditioning_strategy == 'bias' and wall_temp_batch is not None:
                        bias = model.wall_temp_conditioner(wall_temp_batch)
                        predicted_temperature = predicted_temperature + bias
                    elif model.conditioning_strategy == 'film' and wall_temp_batch is not None:
                        predicted_temperature = model.wall_temp_conditioner(predicted_temperature, wall_temp_batch)
            
            # Move to CPU and remove batch dimension
            target_temperature = target_batch.squeeze(0).cpu()
            predicted_temperature = predicted_temperature.squeeze(0).cpu()
            
            # Denormalize temperatures if normalization was used
            if dataset.normalize_temperature:
                target_temperature = dataset._denormalize_temperature(target_temperature, file_idx)
                predicted_temperature = dataset._denormalize_temperature(predicted_temperature, file_idx)
        
        # Store results (remove channel dimension)
        ground_truth_list.append(target_temperature.squeeze(0).numpy())  # [H, W]
        predictions_list.append(predicted_temperature.squeeze(0).numpy())  # [H, W]
        
        if (idx + 1) % 10 == 0:
            print(f"  ✓ Processed {idx + 1}/{len(frame_indices)} frames")
    
    # Convert to numpy arrays [num_frames, H, W]
    ground_truth = np.stack(ground_truth_list, axis=0)
    predictions = np.stack(predictions_list, axis=0)
    
    print(f"✓ Inference complete! Shape: {ground_truth.shape}")
    
    return ground_truth, predictions


def setup_plot_style():
    """Set up publication-quality plot style (matching heatflux_interpolation_plot.py)."""
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


def analyze_temperature_rows(ground_truth, predictions, row_indices, save_path=None, log_scale=False):
    """
    Analyze temperature profiles along horizontal rows.
    
    Args:
        ground_truth: [num_frames, H, W] - Ground truth temperature
        predictions: [num_frames, H, W] - Model predictions
        row_indices: List of row indices (y-coordinates) to analyze
        save_path: Path to save the plot
        log_scale: If True, also create log-scale versions of the plots
    """
    # Set up publication-quality plot style
    setup_plot_style()
    
    num_frames, H, W = ground_truth.shape
    
    print(f"\n📊 Analyzing temperature profiles for rows: {row_indices}")
    
    # Create figure with subplots for each row
    num_rows = len(row_indices)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 3.5 * num_rows))
    
    if num_rows == 1:
        axes = [axes]
    
    # X-axis coordinates (pixel positions along width)
    x_coords = np.arange(W)
    
    # Store averaged data for combined plot
    gt_row_avgs = []
    pred_row_avgs = []
    
    # Process each row
    for idx, row_y in enumerate(row_indices):
        ax = axes[idx]
        
        # Extract temperature profiles for this row across all frames
        gt_row = ground_truth[:, row_y, :]  # [num_frames, W]
        pred_row = predictions[:, row_y, :]  # [num_frames, W]
        
        # Average across frames
        gt_row_avg = np.mean(gt_row, axis=0)  # [W]
        pred_row_avg = np.mean(pred_row, axis=0)  # [W]
        
        # Store for combined plot
        gt_row_avgs.append(gt_row_avg)
        pred_row_avgs.append(pred_row_avg)
        
        # Compute standard deviation for error bars (optional)
        gt_row_std = np.std(gt_row, axis=0)
        pred_row_std = np.std(pred_row, axis=0)
        
        # Plot ground truth and predictions
        ax.plot(x_coords, gt_row_avg, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(x_coords, pred_row_avg, 'r--', linewidth=2, label='Model Prediction', alpha=0.8)
        
        # Add shaded regions for standard deviation (optional)
        ax.fill_between(x_coords, gt_row_avg - gt_row_std, gt_row_avg + gt_row_std, 
                        color='blue', alpha=0.2)
        ax.fill_between(x_coords, pred_row_avg - pred_row_std, pred_row_avg + pred_row_std, 
                        color='red', alpha=0.2)
        
        # Compute error metrics
        mae = np.mean(np.abs(gt_row_avg - pred_row_avg))
        rmse = np.sqrt(np.mean((gt_row_avg - pred_row_avg) ** 2))
        
        # Set labels and title
        ax.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
        ax.set_title(f'Row y={row_y} (averaged over {num_frames} frames) | MAE: {mae:.3f}°C, RMSE: {rmse:.3f}°C', 
                    fontsize=18, fontweight='bold')
        ax.legend(loc='best', fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Print statistics
        print(f"  Row y={row_y}:")
        print(f"    GT range:   [{gt_row_avg.min():.2f}, {gt_row_avg.max():.2f}]°C")
        print(f"    Pred range: [{pred_row_avg.min():.2f}, {pred_row_avg.max():.2f}]°C")
        print(f"    MAE: {mae:.3f}°C, RMSE: {rmse:.3f}°C")
    
    # Overall title
    fig.suptitle(f'Temperature Profile Analysis (Averaged over {num_frames} frames)', 
                fontsize=20, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Temperature row analysis saved to: {save_path}")
    
    # Create combined plot with all rows on same graph
    fig_combined, fig_log, fig_paper_linear, fig_paper_log = create_combined_row_plot(
        x_coords, gt_row_avgs, pred_row_avgs, row_indices, num_frames, save_path, 
        log_scale=log_scale
    )
    
    return fig, fig_combined, fig_log, fig_paper_linear, fig_paper_log


def create_combined_row_plot(x_coords, gt_row_avgs, pred_row_avgs, row_indices, 
                             num_frames, original_save_path=None, log_scale=False):
    """
    Create a combined plot with all row averages on the same graph.
    
    Args:
        x_coords: X-axis coordinates (pixel positions)
        gt_row_avgs: List of ground truth row averages
        pred_row_avgs: List of prediction row averages
        row_indices: List of row indices (y-coordinates)
        num_frames: Number of frames averaged
        original_save_path: Original save path (to derive combined plot path)
        log_scale: If True, also create log-scale versions of the plots
    """
    print(f"\n📊 Creating combined plot with all rows...")
    
    # Create figure with 3 subplots (matching heatflux_interpolation_plot.py style)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 21))
    
    # Use a qualitative colormap with highly distinguishable colors
    # tab10 provides 10 distinct colors that are easy to differentiate
    # For more rows, we can use tab20 or a custom list
    num_rows = len(row_indices)
    if num_rows <= 10:
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(num_rows)]
    elif num_rows <= 20:
        cmap = plt.cm.tab20
        colors = [cmap(i) for i in range(num_rows)]
    else:
        # For many rows, use a combination of colormaps
        cmap = plt.cm.tab20
        colors = [cmap(i % 20) for i in range(num_rows)]
    
    # Plot 1: Ground truth for all rows
    for idx, (row_y, gt_row_avg) in enumerate(zip(row_indices, gt_row_avgs)):
        ax1.plot(x_coords, gt_row_avg, linewidth=2.5, label=f'Row y={row_y}', 
                color=colors[idx], alpha=0.9)
    
    ax1.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
    ax1.set_title(f'Ground Truth - All Rows (averaged over {num_frames} frames)', 
                 fontsize=18, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=13, ncol=2, framealpha=0.95, fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Plot 2: Predictions for all rows
    for idx, (row_y, pred_row_avg) in enumerate(zip(row_indices, pred_row_avgs)):
        ax2.plot(x_coords, pred_row_avg, linewidth=2.5, label=f'Row y={row_y}', 
                color=colors[idx], alpha=0.9)
    
    ax2.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
    ax2.set_title(f'Model Prediction - All Rows (averaged over {num_frames} frames)', 
                 fontsize=18, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=13, ncol=2, framealpha=0.95, fancybox=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Plot 3: Ground truth and predictions together
    # GT = solid line, Pred = dashed line (same color per row)
    for idx, (row_y, gt_row_avg, pred_row_avg) in enumerate(zip(row_indices, gt_row_avgs, pred_row_avgs)):
        # Ground truth - solid line, thicker
        ax3.plot(x_coords, gt_row_avg, linewidth=2.5, label=f'GT y={row_y}', 
                color=colors[idx], alpha=0.9, linestyle='-')
        # Prediction - dashed line (same color, slightly thinner)
        ax3.plot(x_coords, pred_row_avg, linewidth=2.0, label=f'Pred y={row_y}', 
                color=colors[idx], alpha=0.9, linestyle='--')
    
    ax3.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
    ax3.set_title(f'Ground Truth (solid) vs Model Prediction (dashed) - All Rows (averaged over {num_frames} frames)', 
                 fontsize=18, fontweight='bold')
    # Create a more organized legend with 2 columns: GT on left, Pred on right
    ax3.legend(loc='upper right', fontsize=13, ncol=2, columnspacing=1.0, framealpha=0.95, fancybox=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    
    # Overall title
    fig.suptitle(f'Combined Temperature Profile Comparison', 
                fontsize=20, fontweight='bold', y=0.998)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save combined plot
    if original_save_path:
        # Derive combined plot filename from original
        path_obj = Path(original_save_path)
        combined_save_path = path_obj.parent / f"{path_obj.stem}_combined{path_obj.suffix}"
        plt.savefig(combined_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Combined temperature row plot saved to: {combined_save_path}")
    
    # Create log-scale version if requested
    fig_log = None
    fig_paper_linear = None
    fig_paper_log = None
    if log_scale:
        fig_log = create_log_scale_plot(x_coords, gt_row_avgs, pred_row_avgs, row_indices,
                                        num_frames, original_save_path, colors)
        # Also create paper-ready narrow plots
        fig_paper_linear, fig_paper_log = create_paper_plots(
            x_coords, gt_row_avgs, pred_row_avgs, row_indices,
            num_frames, original_save_path, colors
        )
    
    return fig, fig_log, fig_paper_linear, fig_paper_log


def create_log_scale_plot(x_coords, gt_row_avgs, pred_row_avgs, row_indices,
                          num_frames, original_save_path, colors):
    """
    Create log-scale version of the combined temperature profile plot.
    
    For temperature data, we use a shifted log scale: log(T - T_min + offset)
    to better visualize relative differences across rows.
    
    Args:
        x_coords: X-axis coordinates (pixel positions)
        gt_row_avgs: List of ground truth row averages
        pred_row_avgs: List of prediction row averages
        row_indices: List of row indices (y-coordinates)
        num_frames: Number of frames averaged
        original_save_path: Original save path (to derive log plot path)
        colors: List of colors for each row (to match linear plot)
    """
    print(f"\n📊 Creating log-scale plot...")
    
    # Find the minimum temperature to use as baseline
    all_temps = np.concatenate([np.array(gt_row_avgs).flatten(), 
                                np.array(pred_row_avgs).flatten()])
    T_min = np.min(all_temps)
    T_max = np.max(all_temps)
    
    # Use offset to avoid log(0) - shift so minimum becomes 1
    offset = 1.0
    
    print(f"   Temperature range: [{T_min:.2f}, {T_max:.2f}]°C")
    print(f"   Log scale: log(T - {T_min:.2f} + {offset})")
    
    # Create figure with 3 subplots (same layout as linear, matching heatflux_interpolation_plot.py style)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 21))
    
    # Transform function for log scale
    def log_transform(temps):
        return np.log10(temps - T_min + offset)
    
    # Plot 1: Ground truth for all rows (log scale)
    for idx, (row_y, gt_row_avg) in enumerate(zip(row_indices, gt_row_avgs)):
        gt_log = log_transform(gt_row_avg)
        ax1.plot(x_coords, gt_log, linewidth=2.5, label=f'Row y={row_y}', 
                color=colors[idx], alpha=0.9)
    
    ax1.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
    ax1.set_ylabel(f'log₁₀(T - {T_min:.1f} + {offset})', fontsize=16, fontweight='bold')
    ax1.set_title(f'Ground Truth - All Rows [LOG SCALE] (averaged over {num_frames} frames)', 
                 fontsize=18, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=13, ncol=2, framealpha=0.95, fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Plot 2: Predictions for all rows (log scale)
    for idx, (row_y, pred_row_avg) in enumerate(zip(row_indices, pred_row_avgs)):
        pred_log = log_transform(pred_row_avg)
        ax2.plot(x_coords, pred_log, linewidth=2.5, label=f'Row y={row_y}', 
                color=colors[idx], alpha=0.9)
    
    ax2.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
    ax2.set_ylabel(f'log₁₀(T - {T_min:.1f} + {offset})', fontsize=16, fontweight='bold')
    ax2.set_title(f'Model Prediction - All Rows [LOG SCALE] (averaged over {num_frames} frames)', 
                 fontsize=18, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=13, ncol=2, framealpha=0.95, fancybox=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Plot 3: Ground truth and predictions together (log scale)
    for idx, (row_y, gt_row_avg, pred_row_avg) in enumerate(zip(row_indices, gt_row_avgs, pred_row_avgs)):
        gt_log = log_transform(gt_row_avg)
        pred_log = log_transform(pred_row_avg)
        # Ground truth - solid line
        ax3.plot(x_coords, gt_log, linewidth=2.5, label=f'GT y={row_y}', 
                color=colors[idx], alpha=0.9, linestyle='-')
        # Prediction - dashed line
        ax3.plot(x_coords, pred_log, linewidth=2.0, label=f'Pred y={row_y}', 
                color=colors[idx], alpha=0.9, linestyle='--')
    
    ax3.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
    ax3.set_ylabel(f'log₁₀(T - {T_min:.1f} + {offset})', fontsize=16, fontweight='bold')
    ax3.set_title(f'Ground Truth (solid) vs Model Prediction (dashed) - All Rows [LOG SCALE] (averaged over {num_frames} frames)', 
                 fontsize=18, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=13, ncol=2, columnspacing=1.0, framealpha=0.95, fancybox=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    
    # Overall title
    fig.suptitle(f'Combined Temperature Profile Comparison [LOG SCALE]', 
                fontsize=20, fontweight='bold', y=0.998)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save log-scale plot
    if original_save_path:
        path_obj = Path(original_save_path)
        log_save_path = path_obj.parent / f"{path_obj.stem}_combined_logscale{path_obj.suffix}"
        plt.savefig(log_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Log-scale temperature row plot saved to: {log_save_path}")
    
    return fig


def create_paper_plots(x_coords, gt_row_avgs, pred_row_avgs, row_indices,
                       num_frames, original_save_path, colors):
    """
    Create narrow paper-ready plots with just the GT vs Prediction comparison.
    
    Generates two separate PNG files:
    1. Linear scale version (narrow)
    2. Log scale version (narrow)
    
    These are designed to be placed side-by-side in a paper.
    
    Args:
        x_coords: X-axis coordinates (pixel positions)
        gt_row_avgs: List of ground truth row averages
        pred_row_avgs: List of prediction row averages
        row_indices: List of row indices (y-coordinates)
        num_frames: Number of frames averaged
        original_save_path: Original save path (to derive paper plot paths)
        colors: List of colors for each row (to match other plots)
    """
    print(f"\n📄 Creating paper-ready narrow plots...")
    
    # Find the minimum temperature for log scale
    all_temps = np.concatenate([np.array(gt_row_avgs).flatten(), 
                                np.array(pred_row_avgs).flatten()])
    T_min = np.min(all_temps)
    offset = 1.0
    
    def log_transform(temps):
        return np.log10(temps - T_min + offset)
    
    # Paper figure size
    fig_width = 10
    fig_height = 7
    
    # --- Linear scale plot (paper version) ---
    fig_linear = plt.figure(figsize=(fig_width, fig_height))
    ax_linear = fig_linear.add_subplot(111)
    
    for idx, (row_y, gt_row_avg, pred_row_avg) in enumerate(zip(row_indices, gt_row_avgs, pred_row_avgs)):
        # Ground truth - solid line
        ax_linear.plot(x_coords, gt_row_avg, linewidth=2, label=f'GT y={row_y}', 
                      color=colors[idx], alpha=0.9, linestyle='-')
        # Prediction - dashed line
        ax_linear.plot(x_coords, pred_row_avg, linewidth=2, label=f'Pred y={row_y}', 
                      color=colors[idx], alpha=0.9, linestyle='--')
    
    ax_linear.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
    ax_linear.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
    ax_linear.set_title(f'Ground Truth (solid) vs Model Prediction (dashed) - All Rows\n(averaged over {num_frames} frames)', 
                       fontsize=18, fontweight='bold', pad=15)
    # Legend inside plot, upper right corner, single column (vertical)
    ax_linear.legend(loc='upper right', fontsize=13, ncol=1, framealpha=0.95, fancybox=True)
    ax_linear.tick_params(axis='both', labelsize=14)
    ax_linear.grid(True, alpha=0.3, linestyle='--')
    ax_linear.set_axisbelow(True)
    
    fig_linear.tight_layout()
    
    # Save linear paper plot
    if original_save_path:
        path_obj = Path(original_save_path)
        linear_paper_path = path_obj.parent / f"{path_obj.stem}_paper_linear{path_obj.suffix}"
        fig_linear.savefig(linear_paper_path, dpi=300, bbox_inches='tight')
        print(f"✓ Paper-ready linear plot saved to: {linear_paper_path}")
    
    # --- Log scale plot (paper version) ---
    fig_log = plt.figure(figsize=(fig_width, fig_height))
    ax_log = fig_log.add_subplot(111)
    
    for idx, (row_y, gt_row_avg, pred_row_avg) in enumerate(zip(row_indices, gt_row_avgs, pred_row_avgs)):
        gt_log = log_transform(gt_row_avg)
        pred_log = log_transform(pred_row_avg)
        # Ground truth - solid line
        ax_log.plot(x_coords, gt_log, linewidth=2, label=f'GT y={row_y}', 
                   color=colors[idx], alpha=0.9, linestyle='-')
        # Prediction - dashed line
        ax_log.plot(x_coords, pred_log, linewidth=2, label=f'Pred y={row_y}', 
                   color=colors[idx], alpha=0.9, linestyle='--')
    
    ax_log.set_xlabel('X Position (pixels)', fontsize=16, fontweight='bold')
    ax_log.set_ylabel(f'log₁₀(T - {T_min:.1f} + {offset})', fontsize=16, fontweight='bold')
    ax_log.set_title(f'Ground Truth (solid) vs Model Prediction (dashed) - All Rows [LOG SCALE]\n(averaged over {num_frames} frames)', 
                    fontsize=18, fontweight='bold', pad=15)
    # Legend inside plot, upper right corner, single column (vertical)
    ax_log.legend(loc='upper right', fontsize=13, ncol=1, framealpha=0.95, fancybox=True)
    ax_log.tick_params(axis='both', labelsize=14)
    ax_log.grid(True, alpha=0.3, linestyle='--')
    ax_log.set_axisbelow(True)
    
    fig_log.tight_layout()
    
    # Save log paper plot
    if original_save_path:
        path_obj = Path(original_save_path)
        log_paper_path = path_obj.parent / f"{path_obj.stem}_paper_logscale{path_obj.suffix}"
        fig_log.savefig(log_paper_path, dpi=300, bbox_inches='tight')
        print(f"✓ Paper-ready log-scale plot saved to: {log_paper_path}")
    
    return fig_linear, fig_log


def load_task_config(task_name: str = 'velocity_from_interface'):
    """Load task configuration from YAML file."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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


def main():
    """Main function for temperature row analysis."""
    
    parser = argparse.ArgumentParser(
        description='Temperature Row Analysis - Compare ground truth vs model predictions along horizontal rows'
    )
    
    # Model and data paths
    parser.add_argument('checkpoint_path', nargs='?',
                        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_ss_velocity_from_interface_pb_subcooled_singlestep_none_ds4_47861279/checkpoints/epoch=41-step=034860.ckpt",
                       help='Path to model checkpoint')
    parser.add_argument('data_file_path', nargs='?',
                       default="/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5",
                       help='Path to data file')
    
    # Task configuration
    parser.add_argument('--task', type=str, default='auto',
                       choices=['temperature_from_sdf', 'velocity_from_interface', 'noisy_velocity_from_interface', 'auto'],
                       help='Task name: temperature_from_sdf (Task 1), velocity_from_interface (Task 2), '
                            'noisy_velocity_from_interface (Task 3), or auto (detect from checkpoint path)')
    
    # Frame selection
    parser.add_argument('--frame-start', type=int, default=100,
                       help='Starting frame index')
    parser.add_argument('--frame-end', type=int, default=400,
                       help='Ending frame index (None = all frames)')
    parser.add_argument('--frame-step', type=int, default=1,
                       help='Frame step size (1 = every frame)')
    
    # Row selection
    parser.add_argument('--row-start', type=int, default=0,
                       help='Starting row (y-coordinate)')
    parser.add_argument('--row-step', type=int, default=8,
                       help='Row step size (default: 8)')
    parser.add_argument('--row-end', type=int, default=16,
                       help='Ending row (y-coordinate, default: 64)')
    
    # Model configuration
    parser.add_argument('--model-type', default='auto',
                       choices=['auto', 'bubble_ddpm', 'flow_matching', 'flow_matching_spatial', 
                               'flow_matching_guidance', 'flow_matching_ar_bootstrap', 'unet'],
                       help='Model type (auto = detect from checkpoint path)')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                       help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                       help='Guidance scale for Flow Matching Guidance model (1.0=no guidance, >1.0=stronger conditioning)')
    parser.add_argument('--start-time', type=int, default=100,
                       help='Start time for dataset')
    parser.add_argument('--normalize-temperature', action='store_true', default=True,
                       help='Normalize temperature (default: True)')
    parser.add_argument('--no-normalize-temperature', dest='normalize_temperature', action='store_false',
                       help='Do not normalize temperature')
    parser.add_argument('--normalization-stats', type=str, 
                       default="/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json",
                       help='Path to normalization_stats.json file (overrides auto-detection from checkpoint directory)')
    parser.add_argument('--solver', type=str, default='rk4',
                       choices=['euler', 'heun', 'midpoint', 'rk4'],
                       help='ODE solver for flow matching models')
    
    # AR Bootstrap specific arguments
    parser.add_argument('--history-length', type=int, default=10,
                       help='History length for AR Bootstrap model (frames for bootstrap initialization)')
    parser.add_argument('--rollout-length', type=int, default=5,
                       help='Rollout length for AR Bootstrap model')
    parser.add_argument('--downsample-factor', type=int, default=4,
                       help='Downsample factor (1=full resolution, 4=128x128 for 512x512 data)')
    parser.add_argument('--target-channel-idx', type=int, default=2,
                       help='Index of target channel for temperature (default 2 for Task 2/3: [velx, vely, temp])')
    
    # Noise parameters (for Task 3)
    parser.add_argument('--use-clean-inputs', action='store_true', default=True,
                       help='Use clean inputs (no noise) for inference')
    parser.add_argument('--use-noisy-inputs', dest='use_clean_inputs', action='store_false',
                       help='Use noisy inputs for inference')
    
    # Output
    parser.add_argument('--output-dir', default='./ICML/temperature_row_analysis_100_400',
                       help='Output directory')
    parser.add_argument('--output-filename', default='temperature_row_comparison.png',
                       help='Output filename')
    parser.add_argument('--log-scale', action='store_true', default=False,
                       help='Generate additional log-scale plots for better differentiation of small values')
    
    args = parser.parse_args()
    
    # Auto-detect task from checkpoint path if needed
    if args.task == 'auto':
        if args.checkpoint_path is not None:
            if 'noisy_velocity_from_interface' in args.checkpoint_path.lower():
                args.task = 'noisy_velocity_from_interface'
            elif 'velocity_from_interface' in args.checkpoint_path.lower():
                args.task = 'velocity_from_interface'
            elif 'temperature_from_sdf' in args.checkpoint_path.lower():
                args.task = 'temperature_from_sdf'
            else:
                args.task = 'velocity_from_interface'  # Default to Task 2
        else:
            args.task = 'velocity_from_interface'
    
    # Auto-detect model type from checkpoint path if needed
    if args.model_type == 'auto' and args.checkpoint_path is not None:
        ckpt_lower = args.checkpoint_path.lower()
        if 'flow_matching_ar_bootstrap' in ckpt_lower:
            args.model_type = 'flow_matching_ar_bootstrap'
        elif 'flow_matching_spatial' in ckpt_lower:
            args.model_type = 'flow_matching_spatial'
        elif 'flow_matching_guidance' in ckpt_lower:
            args.model_type = 'flow_matching_guidance'
        elif 'flow_matching' in ckpt_lower:
            args.model_type = 'flow_matching'
        elif 'ddpm' in ckpt_lower:
            args.model_type = 'bubble_ddpm'
        elif 'unet' in ckpt_lower:
            args.model_type = 'unet'
        else:
            args.model_type = 'flow_matching'  # Default
    
    is_ar_bootstrap = (args.model_type == 'flow_matching_ar_bootstrap')
    
    # Print configuration
    print("🔬 Temperature Row Analysis")
    print("=" * 70)
    print(f"Checkpoint:       {args.checkpoint_path}")
    print(f"Data file:        {args.data_file_path}")
    print(f"Task:             {args.task}")
    print(f"Model type:       {args.model_type}")
    print(f"Inference steps:  {args.num_inference_steps}")
    print(f"ODE solver:       {args.solver}")
    print(f"Start time:       {args.start_time}")
    print(f"Frames:           [{args.frame_start}:{args.frame_end}:{args.frame_step}]")
    print(f"Rows:             [{args.row_start}:{args.row_end}:{args.row_step}]")
    if is_ar_bootstrap:
        print(f"History length:   {args.history_length}")
        print(f"Rollout length:   {args.rollout_length}")
        print(f"Downsample:       {args.downsample_factor}x")
        print(f"Target channel:   {args.target_channel_idx} (for temperature)")
    print(f"Use clean inputs: {args.use_clean_inputs}")
    print(f"Log scale plots:  {args.log_scale}")
    print(f"Output dir:       {args.output_dir}")
    print("=" * 70)
    
    try:
        # Load task configuration
        task_cfg = load_task_config(args.task)
        
        # Get noise configuration for Task 3 (use clean or noisy)
        noise_cfg = None
        if 'noise_cfg' in task_cfg and not args.use_clean_inputs:
            noise_cfg = dict(task_cfg.noise_cfg)
            print(f"   🔊 Using noisy inputs for inference")
        else:
            print(f"   ✨ Using clean inputs for inference")
        
        # Load normalization stats (priority: explicit file > checkpoint dir > compute from data)
        normalization_stats = None
        
        # Option 1: Load from explicitly provided file (via --normalization-stats)
        if args.normalization_stats and os.path.exists(args.normalization_stats):
            print(f"\n📊 Loading normalization stats from provided file: {args.normalization_stats}")
            with open(args.normalization_stats, 'r') as f:
                normalization_stats = json.load(f)
            print(f"   ✓ Loaded normalization stats:")
            print(f"   Temperature: [{normalization_stats['temperature']['min']:.1f}, {normalization_stats['temperature']['max']:.1f}]°C")
            print(f"   Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
            print(f"   SDF scale: {normalization_stats['sdf']['scale']:.4f}")
        
        # Option 2: Try to load from checkpoint directory
        if normalization_stats is None and args.checkpoint_path:
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            if "checkpoints" in checkpoint_dir:
                checkpoint_dir = os.path.dirname(checkpoint_dir)
            stats_file = os.path.join(checkpoint_dir, "normalization_stats.json")
            
            if os.path.exists(stats_file):
                print(f"\n📊 Loading normalization stats from checkpoint directory: {stats_file}")
                with open(stats_file, 'r') as f:
                    normalization_stats = json.load(f)
                print(f"   ✓ Loaded normalization stats:")
                print(f"   Temperature: [{normalization_stats['temperature']['min']:.1f}, {normalization_stats['temperature']['max']:.1f}]°C")
                print(f"   Velocity scale: {normalization_stats['unified_velocity_scale']:.4f}")
                print(f"   SDF scale: {normalization_stats['sdf']['scale']:.4f}")
        
        # Option 3: Fall back to computing from data file
        if normalization_stats is None:
            print(f"\n⚠️  WARNING: normalization_stats.json not found")
            print(f"📊 Computing normalization stats from inference file (may not match training!)...")
            print(f"💡 For accurate results, provide --normalization-stats or ensure normalization_stats.json exists in checkpoint directory")
            normalization_stats = compute_normalization_stats(
                filenames=[args.data_file_path],
                start_time=args.start_time,
                verbose=True
            )
        
        # Create model configuration based on model type
        if args.model_type == 'flow_matching_ar_bootstrap':
            # Get attention type from checkpoint path if available
            attention_type = 'none'
            if args.checkpoint_path:
                ckpt_lower = args.checkpoint_path.lower()
                if 'attention' in ckpt_lower or 'attn' in ckpt_lower:
                    attention_type = 'bottleneck'
            
            # Calculate in_channels: target (3) + conditioning (3) + prev_output (3) + availability_mask (1) = 10
            num_target_channels = len(task_cfg.target_channels)
            num_cond_channels = len(task_cfg.conditioning_channels)
            in_channels = num_target_channels + num_cond_channels + num_target_channels + 1  # +1 for mask
            
            model_cfg = DictConfig({
                'name': 'flow_matching_ar_bootstrap',
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'attention_type': attention_type,
                'use_attention': attention_type != 'none',
                'dropout': 0.0,
                'num_integration_steps': args.num_inference_steps,
                'history_length': args.history_length,
                'rollout_length': args.rollout_length,
                'use_availability_mask': True,
                'history_encoder_type': 'temporal_mixer',
                'history_encoder_hidden': 32,
                'temporal_mixer_spatial_conv': True,
                'temporal_mixer_temporal_weights': True,
                'bootstrap_loss_weight': 1.0,
                'ar_loss_weight': 1.0,
                'bootstrap_state_loss_weight': 0.5,
                'inference': {'solver': args.solver},
            })
        elif args.model_type == 'unet':
            model_cfg = DictConfig({
                'name': 'unet',
                'in_channels': 1,
                'out_channels': 1,
                'init_features': 32,
                'conditioning_strategy': 'none',
                'wall_temp_bias_hidden': 64,
                'wall_temp_film_hidden': 128,
                'wall_temp_film_gamma_range': 0.1,
                'temp_min': 55.0,
                'temp_max': 120.0
            })
        elif args.model_type == 'flow_matching_spatial':
            model_cfg = DictConfig({
                'name': 'flow_matching_spatial',
                'in_channels': 3,  # x_t + sdf + wall_temp_map
                'out_channels': 1,
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'use_attention': False,
                'dropout': 0.1,
                'num_integration_steps': 50,
                'temp_min': 55.0,
                'temp_max': 120.0
            })
        elif args.model_type == 'flow_matching_guidance':
            model_cfg = DictConfig({
                'name': 'flow_matching_guidance',
                'in_channels': 3,  # x_t + sdf + wall_temp_map
                'out_channels': 1,
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'use_attention': False,
                'dropout': 0.1,
                'num_integration_steps': 50,
                'cfg_dropout_sdf': 0.1,
                'guidance_scale': args.guidance_scale,
                'use_wall_temp_conditioning': True,
                'temp_min': 55.0,
                'temp_max': 120.0
            })
        elif args.model_type == 'flow_matching':
            model_cfg = DictConfig({
                'name': 'flow_matching',
                'in_channels': 2,
                'out_channels': 1,
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'use_attention': False,
                'dropout': 0.1,
                'num_integration_steps': 50,
                'conditioning_strategy': 'none',
                'wall_temp_bias_hidden': 64,
                'wall_temp_film_hidden': 128,
                'wall_temp_film_gamma_range': 0.75,
                'temp_min': 55.0,
                'temp_max': 120.0
            })
        else:  # bubble_ddpm
            model_cfg = DictConfig({
                'name': 'bubble_ddpm',
                'in_channels': 2,
                'out_channels': 1,
                'base_channels': 32,
                'time_embed_dim': 256,
                'num_res_blocks': 2,
                'use_attention': False,
                'dropout': 0.1,
                'num_timesteps': 1000,
                'beta_start': 0.00085,
                'beta_end': 0.012,
                'conditioning_strategy': 'bias',
                'wall_temp_bias_hidden': 64,
                'wall_temp_film_hidden': 128,
                'wall_temp_film_gamma_range': 0.75,
                'temp_min': 55.0,
                'temp_max': 120.0
            })
        
        optim_cfg = DictConfig({'name': 'adamw', 'lr': 0.001})
        scheduler_cfg = DictConfig({'name': 'cosine'})
        
        # Load model
        model, model_type = load_model_from_checkpoint(
            args.checkpoint_path, model_cfg, optim_cfg, scheduler_cfg,
            task_cfg=task_cfg, normalization_stats=normalization_stats
        )
        
        # Determine if wall temperature is needed
        # Spatial and guidance models always need wall temperature
        if model_type == 'flow_matching_spatial' or model_type == 'flow_matching_guidance':
            return_wall_temp = True
        else:
            conditioning_strategy = model_cfg.get('conditioning_strategy', 'none')
            return_wall_temp = conditioning_strategy != 'none'
        
        # Get output fields from task config
        # Map target_channels to field names
        output_fields = ['temperature', 'velx', 'vely']  # Dataset default order
        
        # Load dataset
        dataset = load_dataset(
            args.data_file_path,
            output_fields=output_fields,
            start_time=args.start_time,
            normalize_temperature=args.normalize_temperature,
            return_wall_temp=return_wall_temp,
            is_ar_bootstrap=is_ar_bootstrap,
            history_length=args.history_length,
            rollout_length=args.rollout_length,
            normalization_stats=normalization_stats,
            noise_cfg=noise_cfg,
            downsample_factor=args.downsample_factor
        )
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {device}")
        model = model.to(device)
        
        # Determine frame indices
        frame_end = args.frame_end if args.frame_end is not None else len(dataset)
        frame_indices = list(range(args.frame_start, frame_end, args.frame_step))
        
        print(f"\n📊 Processing {len(frame_indices)} frames...")
        
        # Run inference on all frames
        ground_truth, predictions = run_inference_batch(
            model, dataset, frame_indices, device, 
            num_inference_steps=args.num_inference_steps, 
            model_type=model_type, 
            guidance_scale=args.guidance_scale,
            target_channel_idx=args.target_channel_idx,
            solver=args.solver
        )
        
        # Determine row indices to analyze
        row_indices = list(range(args.row_start, args.row_end + 1, args.row_step))
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, args.output_filename)
        
        # Analyze temperature rows
        fig, fig_combined, fig_log, fig_paper_linear, fig_paper_log = analyze_temperature_rows(
            ground_truth, predictions, row_indices, save_path, log_scale=args.log_scale
        )
        
        print(f"\n🎉 Temperature row analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        if args.log_scale:
            print(f"📊 Log-scale plots also generated")
            print(f"📄 Paper-ready narrow plots also generated (linear + log scale)")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

