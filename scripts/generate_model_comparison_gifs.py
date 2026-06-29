#!/usr/bin/env python3
"""
Generate temperature and velocity magnitude GIFs for multiple models.

Supports:
- flow_matching_ar_bootstrap
- flow_matching_history
- diffusionpde

Each model produces two separate GIFs (temperature, velocity magnitude) with
the same colormaps used in comprehensive_inference_task123.py.

Usage:
    python scripts/generate_model_comparison_gifs.py \
        --data-file /path/to/data.hdf5 \
        --output-dir ./gifs_comparison \
        --samples 100-140
"""

import sys
import os
import json
import argparse
import re
from typing import Optional

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import matplotlib.gridspec as gridspec
from omegaconf import DictConfig, OmegaConf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubblefusion.models.flow_matching_ar_bootstrap import ConditionalFlowMatchingARBootstrapLightning
from bubblefusion.models.flow_matching_history import ConditionalFlowMatchingHistoryLightning
from bubblefusion.models.diffusionpde import DiffusionPDELightning
from bubblefusion.data.bubbleml import (
    compute_normalization_stats,
    BulkFlowARBootstrap,
    BulkFlowHistory,
    BulkFlow,
)


# ---------------------------------------------------------------------------
# Colormaps (same as comprehensive_inference_task123.py)
# ---------------------------------------------------------------------------

def temp_cmap():
    temp_ranges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.134, 0.167,
                   0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_codes = ['#0000FF', '#0443FF', '#0E7AFF', '#16B4FF', '#1FF1FF', '#21FFD3',
                   '#22FF9B', '#22FF67', '#22FF15', '#29FF06', '#45FF07', '#6DFF08',
                   '#9EFF09', '#D4FF0A', '#FEF30A', '#FEB709', '#FD7D08', '#FC4908',
                   '#FC1407', '#FB0007']
    colors = list(zip(temp_ranges, color_codes))
    return LinearSegmentedColormap.from_list('temperature_colormap', colors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_wall_temp_from_filepath(filepath: str) -> float:
    basename = os.path.basename(filepath)
    match = re.search(r'Twall[_-]?(\d+(?:\.\d+)?)', basename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 96.0


def parse_sample_range(samples_str: str, max_len: int):
    if '-' in samples_str:
        parts = samples_str.split('-')
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, min(end, max_len)))
    return [int(x) for x in samples_str.split(',') if int(x) < max_len]


def load_normalization_stats(norm_path, checkpoint_path, data_file_path, start_time):
    if norm_path and os.path.exists(norm_path):
        with open(norm_path, 'r') as f:
            return json.load(f)

    ckpt_dir = os.path.dirname(checkpoint_path)
    if 'checkpoints' in ckpt_dir:
        ckpt_dir = os.path.dirname(ckpt_dir)
    stats_file = os.path.join(ckpt_dir, 'normalization_stats.json')
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)

    return compute_normalization_stats(filenames=[data_file_path], start_time=start_time, verbose=True)


# ---------------------------------------------------------------------------
# Bootstrap ablation (from comprehensive_inference_task123.py)
# ---------------------------------------------------------------------------

def compute_bootstrap_ablation_state(ablation_mode, cond_hist_extracted, C_out,
                                     device, target_names=None, conditioning_names=None):
    B, T, C_cond, H, W = cond_hist_extracted.shape
    if ablation_mode == 'zeros':
        return torch.zeros(B, C_out, H, W, device=device)
    if ablation_mode == 'mean_conditioning_naive':
        prev_output = torch.zeros(B, C_out, H, W, device=device)
        mean_cond = cond_hist_extracted.mean(dim=1)
        if target_names is None:
            target_names = ['velx', 'vely', 'temperature']
        if conditioning_names is None:
            conditioning_names = ['sdf', 'velx_interface', 'vely_interface']
        if 'velx_interface' in conditioning_names and 'velx' in target_names:
            src = conditioning_names.index('velx_interface')
            dst = target_names.index('velx')
            prev_output[:, dst] = mean_cond[:, src]
        if 'vely_interface' in conditioning_names and 'vely' in target_names:
            src = conditioning_names.index('vely_interface')
            dst = target_names.index('vely')
            prev_output[:, dst] = mean_cond[:, src]
        return prev_output
    raise ValueError(f"Unknown bootstrap_ablation mode: '{ablation_mode}'")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ar_bootstrap_model(checkpoint_path, normalization_stats, device):
    model = ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
        checkpoint_path, normalization_stats=normalization_stats, strict=True,
    )
    model.eval()
    return model.to(device)


def load_history_model(checkpoint_path, normalization_stats, device):
    model = ConditionalFlowMatchingHistoryLightning.load_from_checkpoint(
        checkpoint_path, normalization_stats=normalization_stats, strict=False,
    )
    model.eval()
    return model.to(device)


def load_diffusionpde_model(checkpoint_path, normalization_stats, task_cfg, device, downsample_factor):
    num_cond = len(task_cfg.conditioning_channels)
    num_target = len(task_cfg.target_channels)
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
        checkpoint_path, model_cfg=model_cfg, optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg, task_cfg=task_cfg,
        normalization_stats=normalization_stats, strict=False,
    )
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Trajectory inference
# ---------------------------------------------------------------------------

def run_bootstrap_trajectory(model, dataset, start_idx, num_frames, device,
                             num_steps=50, solver='heun', bootstrap_ablation=None,
                             skip_denormalize=False):
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels

    sample = dataset[start_idx]
    cond_hist, cond_seq, target_seq = sample[:3]
    cond_hist = cond_hist.unsqueeze(0).to(device)
    cond_seq = cond_seq.unsqueeze(0).to(device)
    target_seq = target_seq.unsqueeze(0).to(device)

    cond_hist_e = cond_hist[:, :, conditioning_channels]
    cond_seq_e = cond_seq[:, :, conditioning_channels]
    target_seq_e = target_seq[:, :, target_channels]

    B, _, C_cond, H, W = cond_hist_e.shape
    L = cond_seq_e.shape[1]
    C_out = target_seq_e.shape[2]

    if bootstrap_ablation is not None:
        target_names_list = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
        cond_names_list = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
        prev = compute_bootstrap_ablation_state(
            bootstrap_ablation, cond_hist_e, C_out, device,
            target_names=target_names_list, conditioning_names=cond_names_list,
        )
    else:
        prev = model.bootstrap_initial_state(cond_hist_e, cond_seq_e[:, 0])

    results = []
    frames_to_process = min(num_frames, L)
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    cond_names = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))

    for l in range(frames_to_process):
        cond = cond_seq_e[:, l]
        tgt = target_seq_e[:, l]
        avail = torch.zeros(B, 1, H, W, device=device) if l == 0 else torch.ones(B, 1, H, W, device=device)
        with torch.no_grad():
            pred = model.sample(condition=cond, prev_output=prev,
                                shape=(B, C_out, H, W), device=device,
                                availability_mask=avail,
                                num_integration_steps=num_steps, solver=solver)
        tgt_cpu = tgt.squeeze(0).cpu()
        pred_cpu = pred.squeeze(0).cpu()
        inp_cpu = cond.squeeze(0).cpu()
        if not skip_denormalize:
            for i, fn in enumerate(target_names):
                tgt_cpu[i] = dataset._denormalize_field(tgt_cpu[i], fn)
                pred_cpu[i] = dataset._denormalize_field(pred_cpu[i], fn)
            for i, fn in enumerate(cond_names):
                inp_cpu[i] = dataset._denormalize_field(inp_cpu[i], fn)
        results.append((inp_cpu, tgt_cpu, pred_cpu))
        prev = pred

    if num_frames > L:
        effective_start = dataset.effective_start_time
        base_ts = start_idx + effective_start
        for frame_idx in range(L, num_frames):
            ts = base_ts + frame_idx
            if ts >= dataset.traj_lens[0] - 1:
                break
            cond_frame = dataset._get_conditioning_frame(0, ts)[conditioning_channels]
            tgt_frame = dataset._get_output_frame(0, ts)[target_channels]
            cond_t = cond_frame.unsqueeze(0).to(device)
            tgt_t = tgt_frame.unsqueeze(0).to(device)
            avail = torch.ones(B, 1, H, W, device=device)
            with torch.no_grad():
                pred = model.sample(condition=cond_t, prev_output=prev,
                                    shape=(B, C_out, H, W), device=device,
                                    availability_mask=avail,
                                    num_integration_steps=num_steps, solver=solver)
            tgt_cpu = tgt_t.squeeze(0).cpu()
            pred_cpu = pred.squeeze(0).cpu()
            inp_cpu = cond_t.squeeze(0).cpu()
            if not skip_denormalize:
                for i, fn in enumerate(target_names):
                    tgt_cpu[i] = dataset._denormalize_field(tgt_cpu[i], fn)
                    pred_cpu[i] = dataset._denormalize_field(pred_cpu[i], fn)
                for i, fn in enumerate(cond_names):
                    inp_cpu[i] = dataset._denormalize_field(inp_cpu[i], fn)
            results.append((inp_cpu, tgt_cpu, pred_cpu))
            prev = pred
    return results


def run_history_trajectory(model, dataset, sample_indices, device, num_steps=50, solver='heun',
                           skip_denormalize=False):
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    cond_names = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
    results = []
    for idx in sample_indices:
        if idx >= len(dataset):
            break
        inp_data, out_data = dataset[idx]
        conditioning = inp_data.unsqueeze(0).to(device)
        target = out_data[target_channels].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.flow_matching.sample(
                condition=conditioning,
                shape=(1, len(target_channels), target.shape[2], target.shape[3]),
                device=device, num_integration_steps=num_steps, solver=solver,
            )
        tgt_cpu = target.squeeze(0).cpu()
        pred_cpu = pred.squeeze(0).cpu()
        W = model.history_window
        n_cond = len(cond_names)
        inp_viz = inp_data[-(n_cond):].cpu().clone()
        if not skip_denormalize:
            for i, fn in enumerate(target_names):
                tgt_cpu[i] = dataset._denormalize_field(tgt_cpu[i], fn)
                pred_cpu[i] = dataset._denormalize_field(pred_cpu[i], fn)
            for i, fn in enumerate(cond_names):
                inp_viz[i] = dataset._denormalize_field(inp_viz[i], fn)
        results.append((inp_viz, tgt_cpu, pred_cpu))
    return results


def run_diffusionpde_trajectory(model, dataset, sample_indices, device, num_steps=50, solver='heun',
                                skip_denormalize=False):
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    cond_names = list(model.task_cfg.get('conditioning_names', ['sdf', 'velx_interface', 'vely_interface']))
    num_joint = model.num_joint_channels
    results = []
    for idx in sample_indices:
        if idx >= len(dataset):
            break
        inp_data, out_data = dataset[idx]
        inp_batch = inp_data.unsqueeze(0).to(device)
        conditioning = inp_batch[:, conditioning_channels]
        target = out_data[target_channels].unsqueeze(0).to(device)

        pred = model.diffusion_pde.sample_with_guidance(
            observed_gt=conditioning,
            shape=(1, num_joint, target.shape[2], target.shape[3]),
            device=device, num_steps=num_steps, solver=solver,
        )
        tgt_cpu = target.squeeze(0).cpu()
        pred_cpu = pred.squeeze(0).cpu()
        inp_viz = inp_batch.squeeze(0)[conditioning_channels].cpu().clone()
        if not skip_denormalize:
            for i, fn in enumerate(target_names):
                tgt_cpu[i] = dataset._denormalize_field(tgt_cpu[i], fn)
                pred_cpu[i] = dataset._denormalize_field(pred_cpu[i], fn)
            for i, fn in enumerate(cond_names):
                inp_viz[i] = dataset._denormalize_field(inp_viz[i], fn)
        results.append((inp_viz, tgt_cpu, pred_cpu))
    return results


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_temperature_frame(target, predicted, frame_idx, bulk_temp, heater_temp,
                             cmap_vmin, save_path, dpi=150, model_display_name='Predicted',
                             plot_normalized=False):
    """Render a single temperature comparison frame (target vs predicted)."""
    cmap = temp_cmap()

    tgt_data = target.numpy()
    pred_data = predicted.numpy()

    if plot_normalized:
        tgt_for_cmap = np.clip((tgt_data + 1.0) / 2.0, 0, 1)
        pred_for_cmap = np.clip((pred_data + 1.0) / 2.0, 0, 1)
        tick_positions = [cmap_vmin, 0.25, 0.5, 0.75, 1.0]
        tick_formatter = lambda x, _: f'{x * 2.0 - 1.0:.2f}'
    else:
        tgt_for_cmap = np.clip((tgt_data - bulk_temp) / (heater_temp - bulk_temp), 0, 1)
        pred_for_cmap = np.clip((pred_data - bulk_temp) / (heater_temp - bulk_temp), 0, 1)
        tick_positions = [cmap_vmin, 0.25, 0.5, 0.75, 1.0]
        tick_formatter = lambda x, _: f'{x * (heater_temp - bulk_temp) + bulk_temp:.0f}°C'

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(tgt_for_cmap, cmap=cmap, origin='lower', vmin=cmap_vmin, vmax=1.0)
    axes[0].set_title('Ground Truth', fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04,
                 ticks=tick_positions, format=tick_formatter)

    im1 = axes[1].imshow(pred_for_cmap, cmap=cmap, origin='lower', vmin=cmap_vmin, vmax=1.0)
    axes[1].set_title(f'{model_display_name}  |  Frame {frame_idx}', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04,
                 ticks=tick_positions, format=tick_formatter)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def render_velocity_mag_frame(target_velx, target_vely, pred_velx, pred_vely,
                              frame_idx, vel_vmax, save_path, dpi=150, model_display_name='Predicted',
                              plot_normalized=False):
    """Render a single velocity magnitude comparison frame (target vs predicted)."""
    tgt_mag = np.sqrt(target_velx.numpy() ** 2 + target_vely.numpy() ** 2)
    pred_mag = np.sqrt(pred_velx.numpy() ** 2 + pred_vely.numpy() ** 2)

    vmax = vel_vmax if vel_vmax is not None else max(tgt_mag.max(), pred_mag.max(), 1e-6)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    label_suffix = ' (norm.)' if plot_normalized else ''

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(tgt_mag, cmap='turbo', norm=norm, origin='lower')
    axes[0].set_title('Ground Truth', fontsize=11)
    axes[0].axis('off')
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    if plot_normalized:
        cb0.set_label('|v| (normalized)', fontsize=9)

    im1 = axes[1].imshow(pred_mag, cmap='turbo', norm=norm, origin='lower')
    axes[1].set_title(f'{model_display_name}  |  Frame {frame_idx}', fontsize=11)
    axes[1].axis('off')
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    if plot_normalized:
        cb1.set_label('|v| (normalized)', fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def create_gif(image_paths, gif_path, duration=500):
    if not image_paths:
        return
    images = []
    for p in image_paths:
        if os.path.exists(p):
            images.append(Image.open(p))
    if not images:
        return
    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)
    resized = []
    for im in images:
        if im.width != max_w or im.height != max_h:
            canvas = Image.new('RGB', (max_w, max_h), (255, 255, 255))
            canvas.paste(im, ((max_w - im.width) // 2, (max_h - im.height) // 2))
            resized.append(canvas)
        else:
            resized.append(im.convert('RGB') if im.mode != 'RGB' else im)
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    resized[0].save(gif_path, save_all=True, append_images=resized[1:],
                    duration=duration, loop=0, optimize=True)
    print(f"  GIF saved: {gif_path}")


# ---------------------------------------------------------------------------
# Combined grid frame renderer
# ---------------------------------------------------------------------------

def render_combined_frame(frame_idx, gt_temp, gt_velx, gt_vely,
                          model_preds, model_display_names,
                          bulk_temp, heater_temp, cmap_vmin, vel_vmax,
                          save_path, dpi=150, plot_normalized=False):
    """Render a combined comparison frame.

    Layout (2 rows x (1 + N_models) columns):
        Row 0: [GT temp] [model1 temp] [model2 temp] ... [colorbar]
        Row 1: [GT |vel|] [model1 |vel|] [model2 |vel|] ... [colorbar]

    Args:
        gt_temp: ground-truth temperature [H, W] numpy
        gt_velx, gt_vely: ground-truth velocity components [H, W] numpy
        model_preds: list of dicts with keys 'temp', 'velx', 'vely' (numpy)
        model_display_names: list of display-name strings (same order)
        plot_normalized: if True, data is in normalized space
    """
    n_models = len(model_preds)
    n_cols = 1 + n_models

    cmap_t = temp_cmap()

    if plot_normalized:
        gt_temp_cm = np.clip((gt_temp + 1.0) / 2.0, 0, 1)
        pred_temps_cm = [np.clip((mp['temp'] + 1.0) / 2.0, 0, 1) for mp in model_preds]
    else:
        gt_temp_cm = np.clip((gt_temp - bulk_temp) / (heater_temp - bulk_temp), 0, 1)
        pred_temps_cm = [np.clip((mp['temp'] - bulk_temp) / (heater_temp - bulk_temp), 0, 1) for mp in model_preds]

    gt_vmag = np.sqrt(gt_velx ** 2 + gt_vely ** 2)
    pred_vmags = [np.sqrt(mp['velx'] ** 2 + mp['vely'] ** 2) for mp in model_preds]

    vmax_vel = vel_vmax if vel_vmax is not None else max(
        gt_vmag.max(), max((v.max() for v in pred_vmags), default=1e-6), 1e-6)
    vel_norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax_vel)

    col_w = 3.0
    cbar_w = 0.35
    row_h = 3.0
    fig_w = col_w * n_cols + cbar_w
    fig_h = row_h * 2

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, n_cols + 1,
                           width_ratios=[1] * n_cols + [0.05],
                           wspace=0.05, hspace=0.18)

    # --- Row 0: Temperature ---
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(gt_temp_cm, cmap=cmap_t, origin='lower', vmin=cmap_vmin, vmax=1.0)
    ax.set_title('Ground Truth', fontsize=10)
    ax.axis('off')

    for j, (pcm, name) in enumerate(zip(pred_temps_cm, model_display_names)):
        ax = fig.add_subplot(gs[0, j + 1])
        im_t = ax.imshow(pcm, cmap=cmap_t, origin='lower', vmin=cmap_vmin, vmax=1.0)
        ax.set_title(f'{name}  |  Frame {frame_idx}', fontsize=10)
        ax.axis('off')

    cax_t = fig.add_subplot(gs[0, n_cols])
    cbar_t = fig.colorbar(im_t, cax=cax_t)
    tick_pos = [p for p in [cmap_vmin, 0.25, 0.5, 0.75, 1.0] if p >= cmap_vmin]
    cbar_t.set_ticks(tick_pos)
    if plot_normalized:
        cbar_t.set_ticklabels([f'{t * 2.0 - 1.0:.2f}' for t in tick_pos])
    else:
        cbar_t.set_ticklabels([f'{t * (heater_temp - bulk_temp) + bulk_temp:.0f}°C' for t in tick_pos])

    # --- Row 1: Velocity magnitude ---
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(gt_vmag, cmap='turbo', norm=vel_norm, origin='lower')
    ax.set_title('Ground Truth', fontsize=10)
    ax.axis('off')

    for j, (vmag, name) in enumerate(zip(pred_vmags, model_display_names)):
        ax = fig.add_subplot(gs[1, j + 1])
        im_v = ax.imshow(vmag, cmap='turbo', norm=vel_norm, origin='lower')
        ax.set_title(f'{name}  |  Frame {frame_idx}', fontsize=10)
        ax.axis('off')

    cax_v = fig.add_subplot(gs[1, n_cols])
    cb_v = fig.colorbar(im_v, cax=cax_v)
    if plot_normalized:
        cb_v.set_label('|v| (norm.)', fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-model GIF driver
# ---------------------------------------------------------------------------

def compute_global_vel_vmax(results, target_names):
    """Compute global velocity magnitude max across all frames for consistent colorbars."""
    velx_idx = target_names.index('velx') if 'velx' in target_names else None
    vely_idx = target_names.index('vely') if 'vely' in target_names else None
    if velx_idx is None or vely_idx is None:
        return None
    vmax = 0.0
    for _, tgt, _ in results:
        mag = np.sqrt(tgt[velx_idx].numpy() ** 2 + tgt[vely_idx].numpy() ** 2)
        vmax = max(vmax, float(mag.max()))
    return vmax if vmax > 0 else None


def generate_gifs_for_results(results, model_label, target_names,
                              output_dir, bulk_temp, heater_temp,
                              cmap_vmin, vel_vmax_override, dpi, gif_duration,
                              model_display_name='Predicted', plot_normalized=False):
    """Render frames and assemble temperature + velocity magnitude GIFs."""
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else None
    velx_idx = target_names.index('velx') if 'velx' in target_names else None
    vely_idx = target_names.index('vely') if 'vely' in target_names else None

    vel_vmax = vel_vmax_override if vel_vmax_override is not None else compute_global_vel_vmax(results, target_names)

    frames_dir = os.path.join(output_dir, model_label, 'frames')
    temp_paths = []
    vel_paths = []

    for i, (inp, tgt, pred) in enumerate(results):
        if temp_idx is not None:
            p = os.path.join(frames_dir, f'temp_{i:04d}.png')
            render_temperature_frame(tgt[temp_idx], pred[temp_idx], i,
                                     bulk_temp, heater_temp, cmap_vmin, p, dpi,
                                     model_display_name=model_display_name,
                                     plot_normalized=plot_normalized)
            temp_paths.append(p)
        if velx_idx is not None and vely_idx is not None:
            p = os.path.join(frames_dir, f'vel_{i:04d}.png')
            render_velocity_mag_frame(tgt[velx_idx], tgt[vely_idx],
                                      pred[velx_idx], pred[vely_idx],
                                      i, vel_vmax, p, dpi,
                                      model_display_name=model_display_name,
                                      plot_normalized=plot_normalized)
            vel_paths.append(p)

    if temp_paths:
        gif_p = os.path.join(output_dir, f'{model_label}_temperature.gif')
        create_gif(temp_paths, gif_p, gif_duration)
    if vel_paths:
        gif_p = os.path.join(output_dir, f'{model_label}_velocity_magnitude.gif')
        create_gif(vel_paths, gif_p, gif_duration)


# ---------------------------------------------------------------------------
# Single-entry inference (factored out so it works for both default + compare)
# ---------------------------------------------------------------------------

def run_model_inference(model_type, checkpoint_path, args, task_cfg,
                        norm_stats, device):
    """Load model + dataset for a (model_type, checkpoint) and run inference.

    Returns:
        (results, sample_indices) where:
            results: list of (inp, tgt, pred) frames
            sample_indices: list of dataset indices used
        or (None, None) if checkpoint is missing.
    """
    if not os.path.exists(checkpoint_path):
        print(f'  Checkpoint not found: {checkpoint_path}')
        return None, None

    if model_type == 'flow_matching_ar_bootstrap':
        model = load_ar_bootstrap_model(checkpoint_path, norm_stats, device)
        hist_len = getattr(model, 'history_length', 10)
        hist_stride = getattr(model, 'history_stride', 1)
        rollout_len = getattr(model, 'rollout_length', 5)
        print(f'  history_length={hist_len}, history_stride={hist_stride} '
              f'(spans {hist_len * hist_stride} timesteps), rollout_length={rollout_len}')

        dataset = BulkFlowARBootstrap(
            filenames=[args.data_file],
            output_fields=['temperature', 'velx', 'vely'],
            start_time=args.start_time,
            normalization_stats=norm_stats,
            history_length=hist_len,
            history_stride=hist_stride,
            rollout_length=rollout_len,
            downsample_factor=args.downsample_factor,
            norm_mode='all',
        )
        sample_indices = parse_sample_range(args.samples, len(dataset))
        print(f'  Dataset size: {len(dataset)}, frames: {len(sample_indices)}')

        results = run_bootstrap_trajectory(
            model, dataset, sample_indices[0], len(sample_indices),
            device, args.num_inference_steps, args.solver,
            skip_denormalize=args.plot_normalized,
        )

    elif model_type == 'flow_matching_history':
        model = load_history_model(checkpoint_path, norm_stats, device)
        hw = getattr(model, 'history_window', 10)
        hstride = int(getattr(model, 'history_stride', 1))
        print(f'  history_window={hw}, history_stride={hstride} '
              f'(spans {(hw - 1) * hstride + 1} timesteps)')

        dataset = BulkFlowHistory(
            filenames=[args.data_file],
            output_fields=['temperature', 'velx', 'vely'],
            start_time=args.start_time,
            history_window=hw,
            history_stride=hstride,
            normalization_stats=norm_stats,
            downsample_factor=args.downsample_factor,
            norm_mode='all',
        )
        sample_indices = parse_sample_range(args.samples, len(dataset))
        print(f'  Dataset size: {len(dataset)}, frames: {len(sample_indices)}')

        results = run_history_trajectory(
            model, dataset, sample_indices, device,
            args.num_inference_steps, args.solver,
            skip_denormalize=args.plot_normalized,
        )

    elif model_type == 'diffusionpde':
        model = load_diffusionpde_model(checkpoint_path, norm_stats, task_cfg,
                                        device, args.downsample_factor)

        dataset = BulkFlow(
            filenames=[args.data_file],
            output_fields=['temperature', 'velx', 'vely'],
            start_time=args.start_time,
            normalization_stats=norm_stats,
            downsample_factor=args.downsample_factor,
            norm_mode='all',
        )
        sample_indices = parse_sample_range(args.samples, len(dataset))
        print(f'  Dataset size: {len(dataset)}, frames: {len(sample_indices)}')

        results = run_diffusionpde_trajectory(
            model, dataset, sample_indices, device,
            args.num_inference_steps, args.solver,
            skip_denormalize=args.plot_normalized,
        )
    else:
        print(f'  Unknown model_type: {model_type}')
        return None, None

    del model
    torch.cuda.empty_cache()
    return results, sample_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    'flow_matching_ar_bootstrap': {
        'checkpoint': (
            '/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/checkpoints/epoch=07-step=013280.ckpt'
        ),
        'label': 'ar_bootstrap',
        'display_name': 'HB-ARFM',
    },
    'flow_matching_history': {
        'checkpoint': (
            '/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/'
            'flow_matching_history_default_velocity_from_interface_pb_subcooled_'
            'singlestep_none_ds4_50271127/checkpoints/last.ckpt'
        ),
        'label': 'history_window',
        'display_name': 'HistoryFM',
    },
    'diffusionpde': {
        'checkpoint': (
            '/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/'
            'diffusionpde_ch32_b2_s50_zobs1.0_zpde0.5_velocity_from_interface_'
            'pb_subcooled_singlestep_none_ds4_50073800/checkpoints/last.ckpt'
        ),
        'label': 'diffusionpde',
        'display_name': 'DiffusionPDE',
    },
}


# ---------------------------------------------------------------------------
# Multi-checkpoint comparison presets
# ---------------------------------------------------------------------------
# Each preset is a list of entries that will be rendered side-by-side as a
# single row in the combined GIF (GT | entry1 | entry2 | ...).
#
# Each entry is a dict with:
#   - 'model_type': one of MODEL_CONFIGS keys
#   - 'checkpoint': checkpoint path
#   - 'display_name': name shown on the column header
#   - (optional) 'label': filename-safe label for per-entry GIFs

COMPARISON_PRESETS = {
    'hist_length': [
        {
            'model_type': 'flow_matching_ar_bootstrap',
            'checkpoint': '/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/checkpoints/epoch=07-step=013280.ckpt',
            'display_name': 'HB-ARFM (hist=10, S=1)',
            'label': 'hb_arfm_hist10_s1',
        },
        {
            'model_type': 'flow_matching_ar_bootstrap',
            'checkpoint': '/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist64_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52453282/checkpoints/epoch=07-step=013280.ckpt',
            'display_name': 'HB-ARFM (hist=64, S=1)',
            'label': 'hb_arfm_hist64_s1',
        },
    ],
}


def _parse_compare_spec(spec: str) -> dict:
    """Parse a single ``--compare`` CLI entry.

    Format (3 colon-separated fields):
        ``<model_type>:<checkpoint_path>:<display_name>``

    Note: the checkpoint path may itself contain ':' characters (e.g. when
    referencing W&B-style step IDs like ``epoch=07-step=013280.ckpt``), so we
    split into exactly 3 fields from the left/right.
    """
    head, _, tail = spec.partition(':')
    model_type = head.strip()
    ckpt, _, display_name = tail.rpartition(':')
    ckpt = ckpt.strip()
    display_name = display_name.strip()
    if not model_type or not ckpt or not display_name:
        raise ValueError(
            f"Invalid --compare entry '{spec}'. Expected "
            f"'<model_type>:<checkpoint>:<display_name>'."
        )
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_type '{model_type}' in --compare entry. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    safe = display_name.lower()
    for ch in (' ', '(', ')', ',', '/', '=', '.'):
        safe = safe.replace(ch, '_')
    while '__' in safe:
        safe = safe.replace('__', '_')
    safe = safe.strip('_')
    return {
        'model_type': model_type,
        'checkpoint': ckpt,
        'display_name': display_name,
        'label': safe or model_type,
    }


def discover_epoch_checkpoints(model_type: str, ckpt_dir: str,
                               include_last: bool = True,
                               last_epoch: Optional[int] = None) -> list:
    """Auto-discover every ``epoch=*-step=*.ckpt`` (and optionally ``last.ckpt``)
    in a Lightning checkpoint directory and return them as comparison entries
    sorted by epoch.

    Args:
        model_type: One of MODEL_CONFIGS keys.
        ckpt_dir: Directory containing the .ckpt files.
        include_last: If True, append ``last.ckpt`` (if present) as the final entry.
        last_epoch: Optional epoch number to use for the ``last.ckpt`` display name.
                    If None, it's shown as "Last".

    Returns:
        List of entry dicts with the same shape as ``_parse_compare_spec`` output.
    """
    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    epoch_pattern = re.compile(r'epoch=(\d+)-step=(\d+)\.ckpt$')
    discovered = []
    for name in os.listdir(ckpt_dir):
        m = epoch_pattern.match(name)
        if m:
            ep = int(m.group(1))
            discovered.append((ep, name))
    discovered.sort(key=lambda x: x[0])

    entries = []
    for ep, name in discovered:
        entries.append({
            'model_type': model_type,
            'checkpoint': os.path.join(ckpt_dir, name),
            'display_name': f'Epoch {ep}',
            'label': f'epoch_{ep:02d}',
        })

    if include_last:
        last_path = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.exists(last_path):
            if last_epoch is not None:
                disp = f'Epoch {last_epoch} (last)'
                lbl = f'epoch_{last_epoch:02d}_last'
            else:
                disp = 'Last'
                lbl = 'last'
            entries.append({
                'model_type': model_type,
                'checkpoint': last_path,
                'display_name': disp,
                'label': lbl,
            })

    if not entries:
        raise ValueError(
            f"No epoch=*-step=*.ckpt (or last.ckpt) found in {ckpt_dir}"
        )
    return entries


def build_compare_entries(args) -> list:
    """Resolve --compare / --compare-preset / --compare-epochs CLI args
    into a flat ordered list of entry dicts.

    Order: preset → --compare-epochs → --compare.
    """
    entries = []
    if args.compare_preset:
        if args.compare_preset not in COMPARISON_PRESETS:
            raise ValueError(
                f"Unknown --compare-preset '{args.compare_preset}'. "
                f"Available: {list(COMPARISON_PRESETS.keys())}"
            )
        entries.extend(COMPARISON_PRESETS[args.compare_preset])
    if args.compare_epochs:
        model_type, ckpt_dir = args.compare_epochs
        entries.extend(
            discover_epoch_checkpoints(
                model_type, ckpt_dir,
                include_last=not args.compare_epochs_skip_last,
                last_epoch=args.last_epoch,
            )
        )
    if args.compare:
        for spec in args.compare:
            entries.append(_parse_compare_spec(spec))
    return entries


def run_compare(entries, args, task_cfg, norm_stats, target_names,
                bulk_temp, heater_temp, device):
    """Run inference for each compare entry and render a combined-row GIF."""
    print(f'\n{"=" * 60}')
    print(f'Multi-Checkpoint Comparison ({len(entries)} entries)')
    print(f'{"=" * 60}')

    all_results = []
    valid_entries = []
    for i, entry in enumerate(entries):
        print(f'\n[{i + 1}/{len(entries)}] {entry["display_name"]}')
        print(f'  Type: {entry["model_type"]}')
        print(f'  Checkpoint: {entry["checkpoint"]}')
        results, _ = run_model_inference(
            entry['model_type'], entry['checkpoint'],
            args, task_cfg, norm_stats, device,
        )
        if results is None:
            print(f'  Skipping (checkpoint missing).')
            continue
        all_results.append(results)
        valid_entries.append(entry)

        # Per-entry temp + velocity GIFs
        compare_dir = os.path.join(args.output_dir, 'compare_per_entry')
        generate_gifs_for_results(
            results, entry['label'], target_names,
            compare_dir, bulk_temp, heater_temp,
            args.cmap_vmin, args.vel_vmax, args.dpi, args.gif_duration,
            model_display_name=entry['display_name'],
            plot_normalized=args.plot_normalized,
        )

    if not all_results:
        print('  No valid entries to render combined comparison GIF.')
        return

    # Combined grid GIF (1 + N columns, 2 rows for temp/velocity)
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else None
    velx_idx = target_names.index('velx') if 'velx' in target_names else None
    vely_idx = target_names.index('vely') if 'vely' in target_names else None
    if temp_idx is None or velx_idx is None or vely_idx is None:
        print('  Combined grid skipped (missing temp/velx/vely in target_names).')
        return

    n_frames = min(len(r) for r in all_results)
    display_names = [e['display_name'] for e in valid_entries]

    # Global velocity vmax across all entries for consistent colorbars
    global_vel_vmax = args.vel_vmax
    if global_vel_vmax is None:
        global_vel_vmax = 0.0
        for res_list in all_results:
            for _, tgt, pred in res_list:
                for t in (tgt, pred):
                    mag = np.sqrt(t[velx_idx].numpy() ** 2 + t[vely_idx].numpy() ** 2)
                    global_vel_vmax = max(global_vel_vmax, float(mag.max()))
        if global_vel_vmax <= 0:
            global_vel_vmax = 1.0

    combined_dir = os.path.join(args.output_dir, 'compare_combined', 'frames')
    combined_paths = []

    for i in range(n_frames):
        # Use first entry's ground truth (they all share the same trajectory)
        _, gt, _ = all_results[0][i]
        gt_temp = gt[temp_idx].numpy()
        gt_vx = gt[velx_idx].numpy()
        gt_vy = gt[vely_idx].numpy()

        preds = []
        for res_list in all_results:
            _, _, pred = res_list[i]
            preds.append({
                'temp': pred[temp_idx].numpy(),
                'velx': pred[velx_idx].numpy(),
                'vely': pred[vely_idx].numpy(),
            })

        p = os.path.join(combined_dir, f'compare_{i:04d}.png')
        render_combined_frame(
            i, gt_temp, gt_vx, gt_vy,
            preds, display_names,
            bulk_temp, heater_temp, args.cmap_vmin, global_vel_vmax,
            p, args.dpi, plot_normalized=args.plot_normalized,
        )
        combined_paths.append(p)

    gif_p = os.path.join(args.output_dir, 'compare_combined.gif')
    create_gif(combined_paths, gif_p, args.gif_duration)


def main():
    parser = argparse.ArgumentParser(description='Generate temperature & velocity magnitude GIFs for multiple models')
    parser.add_argument('--data-file', type=str,
                        default='/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5',
                        help='Path to HDF5 data file')
    parser.add_argument('--output-dir', type=str, default='./ICML/gifs_model_comparison/new',
                        help='Output directory for GIFs')
    parser.add_argument('--samples', type=str, default='0-50',
                        help='Sample range (e.g., "100-140" or "0,5,10")')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='Number of ODE / denoising steps')
    parser.add_argument('--solver', type=str, default='rk4',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver')
    parser.add_argument('--start-time', type=int, default=800,
                        help='Starting timestep for dataset')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Spatial downsample factor')
    parser.add_argument('--normalization-stats', type=str,
                        default='/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json',
                        help='Path to normalization_stats.json')
    parser.add_argument('--bulk-temp', type=float, default=None,
                        help='Bulk temperature for colormap (auto-detected if None)')
    parser.add_argument('--heater-temp', type=float, default=None,
                        help='Heater temperature for colormap (auto-detected if None)')
    parser.add_argument('--cmap-vmin', type=float, default=0.012,
                        help='Temperature colormap vmin (0-1)')
    parser.add_argument('--vel-vmax', type=float, default=None,
                        help='Fixed velocity magnitude vmax (None=auto)')
    parser.add_argument('--gif-duration', type=int, default=100,
                        help='Duration per frame in ms')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for rendered frames')
    parser.add_argument('--models', nargs='+',
                        default=['flow_matching_ar_bootstrap', 'flow_matching_history', 'diffusionpde'],
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Which models to run')
    parser.add_argument('--seed', type=int, default=32,
                        help='Random seed for reproducibility')
    parser.add_argument('--combined-gif', action='store_true', default=False,
                        help='Also generate a combined grid GIF with all models side-by-side')
    parser.add_argument('--bootstrap-ablation-gif', action='store_true', default=False,
                        help='Generate a combined GIF comparing HB-ARFM (normal) vs zeros vs mean_conditioning_naive ablations')
    parser.add_argument('--plot-normalized', action='store_true', default=False,
                        help='Plot data in normalized space (skip denormalization). '
                             'Colorbars show normalized values instead of physical units.')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='Render a side-by-side row GIF comparing arbitrary checkpoints. '
                             'Each entry has the format '
                             '"<model_type>:<checkpoint_path>:<display_name>". '
                             'Multiple entries are placed in the same row '
                             '(GT | entry1 | entry2 | ...). '
                             f'Available model_types: {list(MODEL_CONFIGS.keys())}.')
    parser.add_argument('--compare-preset', type=str, default=None,
                        choices=list(COMPARISON_PRESETS.keys()),
                        help='Use a hardcoded comparison preset (see COMPARISON_PRESETS).')
    parser.add_argument('--compare-epochs', nargs=2, metavar=('MODEL_TYPE', 'CKPT_DIR'),
                        default=None,
                        help='Auto-discover every "epoch=*-step=*.ckpt" in CKPT_DIR '
                             '(plus last.ckpt by default), sort by epoch, and add them '
                             f'as comparison entries. MODEL_TYPE in {list(MODEL_CONFIGS.keys())}.')
    parser.add_argument('--last-epoch', type=int, default=None,
                        help='Epoch number to attach to last.ckpt in --compare-epochs '
                             '(e.g. 34 for the hist=64 run). Shown as "Epoch N (last)".')
    parser.add_argument('--compare-epochs-skip-last', action='store_true', default=False,
                        help='Do not include last.ckpt in --compare-epochs.')
    parser.add_argument('--skip-default-models', action='store_true', default=False,
                        help='Skip the per-model loop driven by --models (useful when '
                             'only --compare / --compare-preset / --compare-epochs is desired).')
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load task config
    task_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'bubblefusion', 'config',
                                 'task_cfg', 'velocity_from_interface.yaml')
    task_cfg = OmegaConf.load(task_cfg_path)
    target_names = list(task_cfg.target_names)
    conditioning_names = list(task_cfg.conditioning_names)
    print(f'Task: velocity_from_interface')
    print(f'  Target: {target_names}')
    print(f'  Conditioning: {conditioning_names}')

    # Normalization stats
    norm_stats = load_normalization_stats(
        args.normalization_stats,
        list(MODEL_CONFIGS.values())[0]['checkpoint'],
        args.data_file, args.start_time,
    )

    # Colormap params
    if args.bulk_temp is not None:
        bulk_temp = args.bulk_temp
    elif 'temperature' in norm_stats:
        bulk_temp = norm_stats['temperature']['min']
    else:
        bulk_temp = 48.3

    if args.heater_temp is not None:
        heater_temp = args.heater_temp
    elif 'temperature' in norm_stats:
        heater_temp = norm_stats['temperature']['max']
    else:
        heater_temp = 114.7

    if args.plot_normalized:
        print(f'Plotting in NORMALIZED space (skipping denormalization)')
        print(f'  Temperature colorbar: normalized [-1, 1]')
        print(f'  Velocity colorbar: normalized magnitude')
    else:
        print(f'Temperature colormap: [{bulk_temp:.1f}, {heater_temp:.1f}] C')

    os.makedirs(args.output_dir, exist_ok=True)

    # Storage for combined-gif (model_key -> results list)
    all_model_results = {}
    model_order = []

    # Process each default model
    if not args.skip_default_models:
        for model_key in args.models:
            cfg = MODEL_CONFIGS[model_key]
            ckpt = cfg['checkpoint']
            label = cfg['label']
            display_name = cfg.get('display_name', label)

            print(f'\n{"=" * 60}')
            print(f'Model: {model_key}  (label={label})')
            print(f'Checkpoint: {ckpt}')
            print(f'{"=" * 60}')

            results, _ = run_model_inference(
                model_key, ckpt, args, task_cfg, norm_stats, device,
            )
            if results is None:
                continue

            print(f'  Generated {len(results)} frames. Rendering GIFs...')

            generate_gifs_for_results(
                results, label, target_names,
                args.output_dir, bulk_temp, heater_temp,
                args.cmap_vmin, args.vel_vmax, args.dpi, args.gif_duration,
                model_display_name=display_name,
                plot_normalized=args.plot_normalized,
            )

            if args.combined_gif:
                all_model_results[model_key] = results
                model_order.append(model_key)

    # ------------------------------------------------------------------
    # Combined grid GIF (all models side by side)
    # ------------------------------------------------------------------
    if args.combined_gif and len(all_model_results) >= 1:
        print(f'\nRendering combined grid GIF...')
        temp_idx = target_names.index('temperature') if 'temperature' in target_names else None
        velx_idx = target_names.index('velx') if 'velx' in target_names else None
        vely_idx = target_names.index('vely') if 'vely' in target_names else None

        if temp_idx is not None and velx_idx is not None and vely_idx is not None:
            # Use the shortest result list to determine frame count
            n_frames = min(len(r) for r in all_model_results.values())
            display_names = [MODEL_CONFIGS[k].get('display_name', k) for k in model_order]

            # Compute global vel_vmax across all models & frames
            global_vel_vmax = args.vel_vmax
            if global_vel_vmax is None:
                global_vel_vmax = 0.0
                for res_list in all_model_results.values():
                    for _, tgt, pred in res_list:
                        for t in (tgt, pred):
                            mag = np.sqrt(t[velx_idx].numpy() ** 2 + t[vely_idx].numpy() ** 2)
                            global_vel_vmax = max(global_vel_vmax, float(mag.max()))
                if global_vel_vmax <= 0:
                    global_vel_vmax = 1.0

            combined_dir = os.path.join(args.output_dir, 'combined', 'frames')
            combined_paths = []

            for i in range(n_frames):
                # Ground truth from first model (shared across all)
                first_key = model_order[0]
                _, gt, _ = all_model_results[first_key][i]
                gt_temp = gt[temp_idx].numpy()
                gt_vx = gt[velx_idx].numpy()
                gt_vy = gt[vely_idx].numpy()

                preds = []
                for mk in model_order:
                    _, _, pred = all_model_results[mk][i]
                    preds.append({
                        'temp': pred[temp_idx].numpy(),
                        'velx': pred[velx_idx].numpy(),
                        'vely': pred[vely_idx].numpy(),
                    })

                p = os.path.join(combined_dir, f'combined_{i:04d}.png')
                render_combined_frame(
                    i, gt_temp, gt_vx, gt_vy,
                    preds, display_names,
                    bulk_temp, heater_temp, args.cmap_vmin, global_vel_vmax,
                    p, args.dpi, plot_normalized=args.plot_normalized,
                )
                combined_paths.append(p)

            gif_p = os.path.join(args.output_dir, 'combined_all_models.gif')
            create_gif(combined_paths, gif_p, args.gif_duration)

    # ------------------------------------------------------------------
    # Bootstrap ablation GIF: GT | HB-ARFM | HB-ARFM(Zeros) | HB-ARFM(Mean Cond.)
    # ------------------------------------------------------------------
    if args.bootstrap_ablation_gif:
        ar_cfg = MODEL_CONFIGS.get('flow_matching_ar_bootstrap')
        ar_ckpt = ar_cfg['checkpoint'] if ar_cfg else None
        if ar_ckpt and os.path.exists(ar_ckpt):
            print(f'\n{"=" * 60}')
            print('Bootstrap Ablation GIF')
            print(f'{"=" * 60}')

            model = load_ar_bootstrap_model(ar_ckpt, norm_stats, device)
            hist_len = getattr(model, 'history_length', 10)
            hist_stride = getattr(model, 'history_stride', 1)
            rollout_len = getattr(model, 'rollout_length', 5)
            print(f'  history_length={hist_len}, history_stride={hist_stride} '
                  f'(spans {hist_len * hist_stride} timesteps), rollout_length={rollout_len}')

            dataset = BulkFlowARBootstrap(
                filenames=[args.data_file],
                output_fields=['temperature', 'velx', 'vely'],
                start_time=args.start_time,
                normalization_stats=norm_stats,
                history_length=hist_len,
                history_stride=hist_stride,
                rollout_length=rollout_len,
                downsample_factor=args.downsample_factor,
                norm_mode='all',
            )
            sample_indices = parse_sample_range(args.samples, len(dataset))
            n_frames_abl = len(sample_indices)
            start_abl = sample_indices[0]

            ablation_variants = [
                (None, 'HB-ARFM'),
                ('zeros', 'HB-ARFM (Zeros)'),
                ('mean_conditioning_naive', 'HB-ARFM (Mean Cond.)'),
            ]

            abl_results = {}
            for abl_mode, abl_name in ablation_variants:
                print(f'  Running: {abl_name} ...')
                res = run_bootstrap_trajectory(
                    model, dataset, start_abl, n_frames_abl,
                    device, args.num_inference_steps, args.solver,
                    bootstrap_ablation=abl_mode,
                    skip_denormalize=args.plot_normalized,
                )
                abl_results[abl_name] = res
                # Also save per-variant GIFs
                safe_label = abl_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
                generate_gifs_for_results(
                    res, safe_label, target_names,
                    args.output_dir, bulk_temp, heater_temp,
                    args.cmap_vmin, args.vel_vmax, args.dpi, args.gif_duration,
                    model_display_name=abl_name,
                    plot_normalized=args.plot_normalized,
                )

            del model
            torch.cuda.empty_cache()

            # Render combined ablation grid
            temp_idx = target_names.index('temperature') if 'temperature' in target_names else None
            velx_idx = target_names.index('velx') if 'velx' in target_names else None
            vely_idx = target_names.index('vely') if 'vely' in target_names else None

            if temp_idx is not None and velx_idx is not None and vely_idx is not None:
                abl_names_ordered = [n for _, n in ablation_variants]
                n_frames_combined = min(len(r) for r in abl_results.values())

                global_vel_vmax_abl = args.vel_vmax
                if global_vel_vmax_abl is None:
                    global_vel_vmax_abl = 0.0
                    for res_list in abl_results.values():
                        for _, tgt, pred in res_list:
                            for t in (tgt, pred):
                                mag = np.sqrt(t[velx_idx].numpy() ** 2 + t[vely_idx].numpy() ** 2)
                                global_vel_vmax_abl = max(global_vel_vmax_abl, float(mag.max()))
                    if global_vel_vmax_abl <= 0:
                        global_vel_vmax_abl = 1.0

                abl_frames_dir = os.path.join(args.output_dir, 'bootstrap_ablation', 'frames')
                abl_paths = []

                for i in range(n_frames_combined):
                    _, gt, _ = abl_results[abl_names_ordered[0]][i]
                    gt_temp = gt[temp_idx].numpy()
                    gt_vx = gt[velx_idx].numpy()
                    gt_vy = gt[vely_idx].numpy()

                    preds = []
                    for an in abl_names_ordered:
                        _, _, pred = abl_results[an][i]
                        preds.append({
                            'temp': pred[temp_idx].numpy(),
                            'velx': pred[velx_idx].numpy(),
                            'vely': pred[vely_idx].numpy(),
                        })

                    p = os.path.join(abl_frames_dir, f'abl_{i:04d}.png')
                    render_combined_frame(
                        i, gt_temp, gt_vx, gt_vy,
                        preds, abl_names_ordered,
                        bulk_temp, heater_temp, args.cmap_vmin, global_vel_vmax_abl,
                        p, args.dpi, plot_normalized=args.plot_normalized,
                    )
                    abl_paths.append(p)

                gif_p = os.path.join(args.output_dir, 'bootstrap_ablation_combined.gif')
                create_gif(abl_paths, gif_p, args.gif_duration)
        else:
            print('  AR bootstrap checkpoint not found, skipping ablation GIF.')

    # ------------------------------------------------------------------
    # Multi-checkpoint comparison (--compare / --compare-preset)
    # ------------------------------------------------------------------
    compare_entries = build_compare_entries(args)
    if compare_entries:
        run_compare(
            compare_entries, args, task_cfg, norm_stats, target_names,
            bulk_temp, heater_temp, device,
        )

    print(f'\nAll GIFs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
