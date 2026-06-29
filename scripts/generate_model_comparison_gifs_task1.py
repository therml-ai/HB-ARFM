#!/usr/bin/env python3
"""
Generate temperature comparison GIFs for multiple models on Task 1
(temperature_from_sdf).

For Task 1 the only target is temperature and the only conditioning is the SDF.
Each combined frame is laid out as a single row:

    [ SDF GT ] [ Temp GT ] [ Model 1 pred T ] [ Model 2 pred T ] ...

Supported model types (auto-loaded from the same registry used by
``scripts/inference_metrics_task1.py``):
    flow_matching_ar_bootstrap   (HB-ARFM)
    flow_matching                (Flow Matching)
    flow_matching_history        (HistoryFM)
    ve_sde                       (VE-SDE)
    diffusionpde                 (DiffusionPDE)
    bubble_ddpm                  (DDPM)
    unet                         (U-Net)
    ffno                         (FFNO)

Usage:
    python scripts/generate_model_comparison_gifs_task1.py \
        --data-file /path/to/data.hdf5 \
        --output-dir ./gifs_task1 \
        --samples 100-140

    # Only a subset of models
    python scripts/generate_model_comparison_gifs_task1.py \
        --models hbarfm fm dpde --samples 100-140
"""

import sys
import os
import json
import argparse
import contextlib
import io
from collections import OrderedDict
from typing import Optional, List

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.gridspec as gridspec
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse model registry, checkpoint paths and config helpers from the
# task-1 metrics script so the GIFs always use the same models/checkpoints.
from scripts.inference_metrics_task1 import (
    MODEL_NAMES,
    CHECKPOINTS,
    MODEL_REGISTRY,
    _MODEL_ORDER,
    build_model_cfg,
    load_normalization_stats,
)

from scripts.physics_metrics_task123 import (
    load_task_config,
    load_model_from_checkpoint,
    load_dataset,
    extract_channels,
)


# ---------------------------------------------------------------------------
# Colormaps (kept in sync with scripts/generate_model_comparison_gifs.py)
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
# Sample range parser (same semantics as the task-2 GIF script)
# ---------------------------------------------------------------------------

def parse_sample_range(samples_str: str, max_len: int) -> List[int]:
    if '-' in samples_str:
        parts = samples_str.split('-')
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, min(end, max_len)))
    return [int(x) for x in samples_str.split(',') if int(x) < max_len]


# ---------------------------------------------------------------------------
# Per-frame trajectory inference (returns SDF + GT temp + predicted temp)
# ---------------------------------------------------------------------------

def _extract_latest_sdf_for_viz(input_data: torch.Tensor,
                                conditioning: torch.Tensor,
                                model_type: str) -> torch.Tensor:
    """Pull the SDF channel used for the *current* prediction frame.

    - Non-history models: conditioning has shape ``[1, len(cond_channels), H, W]``
      and for temperature_from_sdf that's ``[1, 1, H, W]`` (just SDF).
    - History models: conditioning has shape ``[1, W * len(cond_channels), H, W]``.
      The most recent frame's SDF is the *last* extracted channel.
    """
    cond = conditioning.squeeze(0).cpu()  # [C, H, W]
    return cond[-1].clone()


def run_standard_trajectory_task1(model, dataset, sample_indices, device,
                                  num_steps: int, solver: str, model_type: str,
                                  skip_denormalize: bool = False):
    """Inference loop for non-AR-bootstrap models.

    Handles flow_matching, flow_matching_history, unet, ffno, bubble_ddpm,
    ve_sde, diffusionpde — same dispatch logic as
    :func:`scripts.physics_metrics_task123.run_inference_batch` but per-frame
    so we can return (sdf, gt_temp, pred_temp) tuples for visualisation.

    Returns:
        List of (sdf [H, W], gt_temp [H, W], pred_temp [H, W]) torch tensors.
    """
    target_names = list(model.task_cfg.get('target_names', ['temperature']))
    cond_names = list(model.task_cfg.get('conditioning_names', ['sdf']))
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else 0

    results = []

    # DiffusionPDE uses autograd-based guidance, others run in no_grad
    grad_context = torch.enable_grad() if model_type == 'diffusionpde' else torch.no_grad()
    with grad_context:
        for idx in tqdm(sample_indices, desc=f"  {model_type}"):
            if idx >= len(dataset):
                break
            sample_data = dataset[idx]
            if getattr(dataset, 'return_wall_temp', False):
                input_data, target_data, _ = sample_data
            else:
                input_data, target_data = sample_data

            input_batch = input_data.unsqueeze(0).to(device)
            target_batch = target_data.unsqueeze(0).to(device)

            if model_type == 'flow_matching_history':
                conditioning = model.extract_history_conditioning(input_batch)
            else:
                conditioning = extract_channels(input_batch, conditioning_channels)
            target = extract_channels(target_batch, target_channels)

            B = 1
            H = target.shape[2]
            W = target.shape[3]
            n_target = len(target_channels)

            if model_type == 'unet':
                predicted = model(conditioning)
            elif model_type == 'ffno':
                predicted = model(conditioning)
            elif model_type == 'bubble_ddpm':
                predicted = model.ddpm.p_sample_loop(
                    condition=conditioning,
                    shape=(B, n_target, H, W),
                    device=device,
                )
            elif model_type == 've_sde':
                predicted = model.ve_sde.sample(
                    condition=conditioning,
                    shape=(B, n_target, H, W),
                    device=device,
                    num_steps=num_steps,
                    method=getattr(model, 'sampling_method', 'pc'),
                    snr=getattr(model, 'snr', 0.16),
                )
            elif model_type == 'edm':
                predicted = model.edm.sample(
                    condition=conditioning,
                    shape=(B, n_target, H, W),
                    device=device,
                    num_steps=model.num_sampling_steps,
                    solver=model.default_solver,
                )
            elif model_type == 'diffusionpde':
                num_joint = model.num_joint_channels
                predicted = model.diffusion_pde.sample_with_guidance(
                    observed_gt=conditioning,
                    shape=(B, num_joint, H, W),
                    device=device,
                    num_steps=model.num_sampling_steps,
                    solver=model.default_solver,
                )
            elif model_type == 'flow_matching_history':
                predicted = model.flow_matching.sample(
                    condition=conditioning,
                    shape=(B, n_target, H, W),
                    device=device,
                    num_integration_steps=num_steps,
                    solver=solver,
                )
            elif model_type == 'flow_matching':
                predicted = model.flow_matching.sample(
                    condition=conditioning,
                    shape=(B, n_target, H, W),
                    device=device,
                    num_integration_steps=num_steps,
                )
            elif model_type == 'flow_matching_jit':
                predicted = model.flow_matching.sample(
                    condition=conditioning,
                    shape=(B, n_target, H, W),
                    device=device,
                    num_integration_steps=num_steps,
                    solver=model.default_solver,
                )
            else:
                raise ValueError(f"Unhandled model_type for task 1: {model_type}")

            target_cpu = target.squeeze(0).cpu()      # [C_out, H, W]
            predicted_cpu = predicted.squeeze(0).cpu()  # [C_out, H, W]

            # Pull SDF visualisation from the conditioning that was actually
            # passed to the model (so history models show their *current*
            # frame, not the oldest one in the window).
            sdf_viz = _extract_latest_sdf_for_viz(input_data, conditioning, model_type)

            if not skip_denormalize:
                for j, fn in enumerate(target_names):
                    target_cpu[j] = dataset._denormalize_field(target_cpu[j], fn)
                    predicted_cpu[j] = dataset._denormalize_field(predicted_cpu[j], fn)
                sdf_viz = dataset._denormalize_field(sdf_viz, 'sdf')

            results.append((sdf_viz, target_cpu[temp_idx], predicted_cpu[temp_idx]))

    return results


def run_bootstrap_trajectory_task1(model, dataset, start_idx, num_frames, device,
                                   num_steps: int, solver: str,
                                   skip_denormalize: bool = False):
    """AR-bootstrap rollout returning per-frame (sdf, gt_temp, pred_temp).

    Mirrors :func:`scripts.physics_metrics_task123.run_ar_bootstrap_inference_batch`
    but runs as a single rollout starting at ``start_idx`` so the resulting
    frames form a contiguous time series suitable for a GIF.
    """
    target_names = list(model.task_cfg.get('target_names', ['temperature']))
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else 0

    sample = dataset[start_idx]
    if getattr(dataset, 'return_wall_temp', False):
        cond_hist, cond_seq, target_seq, _ = sample
    else:
        cond_hist, cond_seq, target_seq = sample

    cond_hist = cond_hist.unsqueeze(0).to(device)
    cond_seq = cond_seq.unsqueeze(0).to(device)
    target_seq = target_seq.unsqueeze(0).to(device)

    cond_hist_e = cond_hist[:, :, conditioning_channels]
    cond_seq_e = cond_seq[:, :, conditioning_channels]
    target_seq_e = target_seq[:, :, target_channels]

    B, _, C_cond, H, W = cond_hist_e.shape
    L = cond_seq_e.shape[1]
    C_out = target_seq_e.shape[2]

    prev = model.bootstrap_initial_state(cond_hist_e, cond_seq_e[:, 0])

    results = []
    frames_to_process = min(num_frames, L)

    pbar = tqdm(total=num_frames, desc="  flow_matching_ar_bootstrap")

    with torch.no_grad():
        for l in range(frames_to_process):
            cond = cond_seq_e[:, l]
            tgt = target_seq_e[:, l]
            avail = (torch.zeros(B, 1, H, W, device=device) if l == 0
                     else torch.ones(B, 1, H, W, device=device))
            pred = model.sample(
                condition=cond,
                prev_output=prev,
                shape=(B, C_out, H, W),
                device=device,
                availability_mask=avail,
                num_integration_steps=num_steps,
                solver=solver,
            )
            tgt_cpu = tgt.squeeze(0).cpu()
            pred_cpu = pred.squeeze(0).cpu()
            sdf_cpu = cond.squeeze(0).cpu()[0]  # only SDF is in conditioning for task 1

            if not skip_denormalize:
                for j, fn in enumerate(target_names):
                    tgt_cpu[j] = dataset._denormalize_field(tgt_cpu[j], fn)
                    pred_cpu[j] = dataset._denormalize_field(pred_cpu[j], fn)
                sdf_cpu = dataset._denormalize_field(sdf_cpu, 'sdf')

            results.append((sdf_cpu, tgt_cpu[temp_idx], pred_cpu[temp_idx]))
            prev = pred
            pbar.update(1)

        # Continue rolling out beyond the segment by walking the trajectory
        # one frame at a time (mirrors generate_model_comparison_gifs.run_bootstrap_trajectory).
        if num_frames > L:
            effective_start = getattr(dataset, 'effective_start_time', 0)
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
                pred = model.sample(
                    condition=cond_t,
                    prev_output=prev,
                    shape=(B, C_out, H, W),
                    device=device,
                    availability_mask=avail,
                    num_integration_steps=num_steps,
                    solver=solver,
                )
                tgt_cpu = tgt_t.squeeze(0).cpu()
                pred_cpu = pred.squeeze(0).cpu()
                sdf_cpu = cond_t.squeeze(0).cpu()[0]
                if not skip_denormalize:
                    for j, fn in enumerate(target_names):
                        tgt_cpu[j] = dataset._denormalize_field(tgt_cpu[j], fn)
                        pred_cpu[j] = dataset._denormalize_field(pred_cpu[j], fn)
                    sdf_cpu = dataset._denormalize_field(sdf_cpu, 'sdf')
                results.append((sdf_cpu, tgt_cpu[temp_idx], pred_cpu[temp_idx]))
                prev = pred
                pbar.update(1)

    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Per-model loader/runner
# ---------------------------------------------------------------------------

def run_task1_model_inference(model_name: str, model_type: str,
                              checkpoint_path: str, args, task_cfg,
                              norm_stats, device):
    """Load a Task-1 model + dataset, run inference, return list of frames.

    Returns:
        List of (sdf, gt_temp, pred_temp) torch tensors or None on failure.
    """
    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None

    optim_cfg = DictConfig({'name': 'adamw', 'lr': 0.001})
    scheduler_cfg = DictConfig({'name': 'cosine'})

    model_cfg = build_model_cfg(model_type, task_cfg, args,
                                checkpoint_path=checkpoint_path)
    model = load_model_from_checkpoint(
        checkpoint_path, model_cfg, optim_cfg, scheduler_cfg, task_cfg,
        model_type=model_type,
        normalization_stats=norm_stats,
        norm_mode=args.norm_mode,
    )

    is_ar_bootstrap = (model_type in ('flow_matching_ar_bootstrap', 'edm_ar_bootstrap'))
    is_history = (model_type == 'flow_matching_history')

    history_length = getattr(model, 'history_length', 10)
    history_stride = getattr(model, 'history_stride', 1)
    rollout_length = getattr(model, 'rollout_length', 5)
    history_window = getattr(model, 'history_window', 10) if is_history else 10

    dataset = load_dataset(
        args.data_file,
        output_fields=['temperature', 'velx', 'vely'],
        start_time=args.start_time,
        return_wall_temp=False,
        is_autoregressive=False,
        is_ar_bootstrap=is_ar_bootstrap,
        is_history_model=is_history,
        history_window=history_window,
        history_length=history_length,
        history_stride=history_stride,
        rollout_length=rollout_length,
        downsample_factor=args.downsample_factor,
        normalization_stats=norm_stats,
        norm_mode=args.norm_mode,
    )

    if is_ar_bootstrap:
        sample_indices = parse_sample_range(args.samples, len(dataset))
        if not sample_indices:
            print("  Empty sample range, skipping")
            del model
            torch.cuda.empty_cache()
            return None
        print(f"  Dataset size: {len(dataset)}, frames: {len(sample_indices)} "
              f"(rollout from segment {sample_indices[0]})")
        results = run_bootstrap_trajectory_task1(
            model, dataset, sample_indices[0], len(sample_indices),
            device, args.num_inference_steps, args.solver,
            skip_denormalize=args.plot_normalized,
        )
    else:
        sample_indices = parse_sample_range(args.samples, len(dataset))
        print(f"  Dataset size: {len(dataset)}, frames: {len(sample_indices)}")
        # Suppress inner per-sample prints (model load is already verbose)
        results = run_standard_trajectory_task1(
            model, dataset, sample_indices, device,
            args.num_inference_steps, args.solver, model_type,
            skip_denormalize=args.plot_normalized,
        )

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _temp_to_cmap_space(t: np.ndarray, bulk_temp: float, heater_temp: float,
                       plot_normalized: bool) -> np.ndarray:
    if plot_normalized:
        return np.clip((t + 1.0) / 2.0, 0, 1)
    return np.clip((t - bulk_temp) / (heater_temp - bulk_temp), 0, 1)


def _plot_sdf(ax, sdf_np: np.ndarray,
              sdf_vmin: float, sdf_vmax: float,
              n_contour_levels: int = 10) -> None:
    """SDF plot styled to match :mod:`scripts.generate_task2_figures`.

    - ``RdYlBu`` colormap with ``TwoSlopeNorm`` anchored at 0
    - Dotted white iso-contours spanning [sdf_vmin, sdf_vmax]
    - A bold black contour at the zero level (liquid/vapor interface)
    """
    # TwoSlopeNorm requires vmin < vcenter < vmax. Guard against degenerate
    # ranges where one side of the SDF lacks variation (rare but possible).
    eps = 1e-6
    vmin = min(sdf_vmin, -eps)
    vmax = max(sdf_vmax, eps)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    ax.imshow(sdf_np, cmap='RdYlBu', norm=norm, origin='lower')
    contour_levels = np.linspace(vmin, vmax, n_contour_levels)
    ax.contour(sdf_np, levels=contour_levels, colors='white',
               linewidths=0.6, linestyles='dotted', alpha=0.5)
    ax.contour(sdf_np, levels=[0.0], colors='black', alpha=0.8, linewidths=1.2)
    ax.axis('off')


def render_per_model_frame(sdf, gt_temp, pred_temp, frame_idx,
                           bulk_temp, heater_temp, cmap_vmin,
                           save_path, dpi=150,
                           model_display_name='Predicted',
                           plot_normalized=False):
    """Single per-model frame: [SDF GT] [Temp GT] [Pred T]."""
    cmap_t = temp_cmap()
    sdf_np = sdf.numpy() if hasattr(sdf, 'numpy') else np.asarray(sdf)
    gt_np = gt_temp.numpy() if hasattr(gt_temp, 'numpy') else np.asarray(gt_temp)
    pr_np = pred_temp.numpy() if hasattr(pred_temp, 'numpy') else np.asarray(pred_temp)

    gt_cm = _temp_to_cmap_space(gt_np, bulk_temp, heater_temp, plot_normalized)
    pr_cm = _temp_to_cmap_space(pr_np, bulk_temp, heater_temp, plot_normalized)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    _plot_sdf(axes[0], sdf_np, float(sdf_np.min()), float(sdf_np.max()))
    axes[0].set_title('SDF (GT)', fontsize=11)

    im1 = axes[1].imshow(gt_cm, cmap=cmap_t, origin='lower', vmin=cmap_vmin, vmax=1.0)
    axes[1].set_title('Temperature (GT)', fontsize=11)
    axes[1].axis('off')
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    tick_pos = [cmap_vmin, 0.25, 0.5, 0.75, 1.0]
    cb1.set_ticks(tick_pos)
    if plot_normalized:
        cb1.set_ticklabels([f'{t * 2.0 - 1.0:.2f}' for t in tick_pos])
    else:
        cb1.set_ticklabels([f'{t * (heater_temp - bulk_temp) + bulk_temp:.0f}°C' for t in tick_pos])

    im2 = axes[2].imshow(pr_cm, cmap=cmap_t, origin='lower', vmin=cmap_vmin, vmax=1.0)
    axes[2].set_title(f'{model_display_name}  |  Frame {frame_idx}', fontsize=11)
    axes[2].axis('off')
    cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_ticks(tick_pos)
    if plot_normalized:
        cb2.set_ticklabels([f'{t * 2.0 - 1.0:.2f}' for t in tick_pos])
    else:
        cb2.set_ticklabels([f'{t * (heater_temp - bulk_temp) + bulk_temp:.0f}°C' for t in tick_pos])

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def render_combined_frame_task1(frame_idx, sdf, gt_temp, model_preds,
                                model_display_names,
                                bulk_temp, heater_temp, cmap_vmin,
                                sdf_vmin_global, sdf_vmax_global,
                                save_path, dpi=150, plot_normalized=False):
    """Combined-row frame: [SDF GT] [Temp GT] [model 1] [model 2] ... [cbar].

    Args:
        sdf: (H, W) numpy or tensor — ground-truth SDF for this frame.
        gt_temp: (H, W) ground-truth temperature.
        model_preds: list of (H, W) per-model predictions.
        model_display_names: list of strings, same order.
        sdf_vmin_global, sdf_vmax_global: global min/max for the SDF colormap
            across all frames (keeps the TwoSlopeNorm endpoints stable).
    """
    n_models = len(model_preds)
    n_main_cols = 2 + n_models  # SDF, GT, then models

    cmap_t = temp_cmap()
    sdf_np = sdf.numpy() if hasattr(sdf, 'numpy') else np.asarray(sdf)
    gt_np = gt_temp.numpy() if hasattr(gt_temp, 'numpy') else np.asarray(gt_temp)

    gt_cm = _temp_to_cmap_space(gt_np, bulk_temp, heater_temp, plot_normalized)
    pred_cms = []
    for mp in model_preds:
        mp_np = mp.numpy() if hasattr(mp, 'numpy') else np.asarray(mp)
        pred_cms.append(_temp_to_cmap_space(mp_np, bulk_temp, heater_temp, plot_normalized))

    sdf_vmin = sdf_vmin_global if sdf_vmin_global is not None else float(sdf_np.min())
    sdf_vmax = sdf_vmax_global if sdf_vmax_global is not None else float(sdf_np.max())

    col_w = 3.0
    cbar_w = 0.4
    row_h = 3.2
    fig_w = col_w * n_main_cols + cbar_w
    fig_h = row_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    # one extra column at the right for the shared temperature colorbar
    gs = gridspec.GridSpec(1, n_main_cols + 1,
                           width_ratios=[1] * n_main_cols + [0.05],
                           wspace=0.08)

    # SDF GT — same RdYlBu + zero-level contour style as generate_task2_figures.py
    ax_sdf = fig.add_subplot(gs[0, 0])
    _plot_sdf(ax_sdf, sdf_np, sdf_vmin, sdf_vmax)
    ax_sdf.set_title('SDF (GT)', fontsize=10)

    # Temperature GT
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_gt.imshow(gt_cm, cmap=cmap_t, origin='lower', vmin=cmap_vmin, vmax=1.0)
    ax_gt.set_title('Temperature (GT)', fontsize=10)
    ax_gt.axis('off')

    # Model predictions
    im_t = None
    for j, (pcm, name) in enumerate(zip(pred_cms, model_display_names)):
        ax = fig.add_subplot(gs[0, j + 2])
        im_t = ax.imshow(pcm, cmap=cmap_t, origin='lower', vmin=cmap_vmin, vmax=1.0)
        ax.set_title(f'{name}  |  Frame {frame_idx}', fontsize=10)
        ax.axis('off')

    # Shared temperature colorbar
    cax_t = fig.add_subplot(gs[0, n_main_cols])
    if im_t is not None:
        cbar_t = fig.colorbar(im_t, cax=cax_t)
    else:
        # No models — colorbar from the GT itself
        cbar_t = fig.colorbar(
            ax_gt.images[0] if ax_gt.images else plt.cm.ScalarMappable(cmap=cmap_t),
            cax=cax_t,
        )
    tick_pos = [p for p in [cmap_vmin, 0.25, 0.5, 0.75, 1.0] if p >= cmap_vmin]
    cbar_t.set_ticks(tick_pos)
    if plot_normalized:
        cbar_t.set_ticklabels([f'{t * 2.0 - 1.0:.2f}' for t in tick_pos])
    else:
        cbar_t.set_ticklabels([f'{t * (heater_temp - bulk_temp) + bulk_temp:.0f}°C'
                               for t in tick_pos])

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


def generate_per_model_gif(results, model_label, model_display_name,
                           output_dir, bulk_temp, heater_temp,
                           cmap_vmin, dpi, gif_duration, plot_normalized):
    """Render per-model frames and assemble a temperature comparison GIF
    showing [SDF GT, Temp GT, Pred T] over time."""
    frames_dir = os.path.join(output_dir, model_label, 'frames')
    paths = []
    for i, (sdf, gt_temp, pred_temp) in enumerate(results):
        p = os.path.join(frames_dir, f'frame_{i:04d}.png')
        render_per_model_frame(
            sdf, gt_temp, pred_temp, i,
            bulk_temp, heater_temp, cmap_vmin, p, dpi,
            model_display_name=model_display_name,
            plot_normalized=plot_normalized,
        )
        paths.append(p)
    if paths:
        gif_p = os.path.join(output_dir, f'{model_label}_temperature.gif')
        create_gif(paths, gif_p, gif_duration)


# ---------------------------------------------------------------------------
# Short-key resolver (mirrors inference_metrics_task1.py)
# ---------------------------------------------------------------------------

SHORT_KEYS = {
    'fm': 'flow_matching',
    'historyfm': 'flow_matching_history',
    'fmhist': 'flow_matching_history',
    'flow_matching_history': 'flow_matching_history',
    'unet': 'unet',
    'ddpm': 'bubble_ddpm',
    'vesde': 've_sde',
    've_sde': 've_sde',
    'ffno': 'ffno',
    'dpde': 'diffusionpde',
    'diffusionpde': 'diffusionpde',
    'hbarfm': 'flow_matching_ar_bootstrap',
    'fmar': 'flow_matching_ar_bootstrap',
}


def resolve_models(args_models) -> "OrderedDict[str, dict]":
    if args_models is None:
        return MODEL_REGISTRY
    selected = []
    for key in args_models:
        mt = SHORT_KEYS.get(key.lower(), key)
        display = MODEL_NAMES.get(mt)
        if display is not None and display in MODEL_REGISTRY:
            selected.append(display)
        else:
            print(f"WARNING: Unknown model key '{key}', skipping. "
                  f"Valid keys: {list(SHORT_KEYS.keys()) + list(MODEL_NAMES.keys())}")
    return OrderedDict((k, MODEL_REGISTRY[k]) for k in selected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate temperature comparison GIFs for Task 1 (temperature_from_sdf)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-file', type=str,
                        default='/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5',
                        help='Path to HDF5 validation data file')
    parser.add_argument('--output-dir', type=str,
                        default='./ICML/CamReady/Task1_gifs',
                        help='Output directory for GIFs')
    parser.add_argument('--samples', type=str, default='0-50',
                        help='Sample range (e.g., "100-140" or "0,5,10")')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='ODE / denoising integration steps')
    parser.add_argument('--solver', type=str, default='rk4',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver')
    parser.add_argument('--start-time', type=int, default=800,
                        help='Starting timestep in HDF5 file')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Spatial downsample factor (4 = 128x128)')
    parser.add_argument('--normalization-stats', type=str,
                        default='/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json',
                        help='Path to normalization_stats.json')
    parser.add_argument('--norm-mode', type=str, default='all',
                        choices=['none', 'all', 'temperature_only'],
                        help='Normalization mode (must match training)')
    parser.add_argument('--bulk-temp', type=float, default=None,
                        help='Bulk temperature for colormap (auto-detected from norm stats if None)')
    parser.add_argument('--heater-temp', type=float, default=None,
                        help='Heater temperature for colormap (auto-detected from norm stats if None)')
    parser.add_argument('--cmap-vmin', type=float, default=0.012,
                        help='Temperature colormap vmin (0-1)')
    parser.add_argument('--gif-duration', type=int, default=100,
                        help='Duration per frame in ms')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for rendered frames')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of models to run (short keys or model types). '
                             f'Valid: {list(SHORT_KEYS.keys()) + list(MODEL_NAMES.keys())}')
    parser.add_argument('--seed', type=int, default=32,
                        help='Random seed for reproducibility')
    parser.add_argument('--combined-gif', action='store_true', default=True,
                        help='Generate a combined-row GIF with SDF | GT | all models')
    parser.add_argument('--no-combined-gif', dest='combined_gif',
                        action='store_false',
                        help='Disable combined-row GIF generation')
    parser.add_argument('--per-model-gifs', action='store_true', default=True,
                        help='Generate per-model GIFs (SDF | GT | Pred)')
    parser.add_argument('--no-per-model-gifs', dest='per_model_gifs',
                        action='store_false',
                        help='Disable per-model GIF generation')
    parser.add_argument('--plot-normalized', action='store_true', default=False,
                        help='Plot data in normalized space (skip denormalization). '
                             'Colorbars show normalized values instead of physical units.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models_to_run = resolve_models(args.models)

    print('=' * 80)
    print('  Multi-Model GIF Generation — Task 1 (temperature_from_sdf)')
    print('=' * 80)
    print(f'  Data file:       {args.data_file}')
    print(f'  Start time:      {args.start_time}')
    print(f'  Samples:         {args.samples}')
    print(f'  Downsample:      {args.downsample_factor}x')
    print(f'  Inference steps: {args.num_inference_steps}')
    print(f'  Solver:          {args.solver}')
    print(f'  Seed:            {args.seed}')
    print(f'  Models ({len(models_to_run)}):     {list(models_to_run.keys())}')
    print(f'  Output dir:      {args.output_dir}')
    print('=' * 80)

    task_cfg = load_task_config('temperature_from_sdf')

    # Use the first available checkpoint's directory as the fallback for
    # locating normalization stats (matches the metrics script behaviour).
    first_ckpt = next(iter(models_to_run.values()))['checkpoint']
    norm_stats = load_normalization_stats(first_ckpt, args.normalization_stats)

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
        print('Plotting in NORMALIZED space (skipping denormalization)')
        print('  Temperature colorbar: normalized [-1, 1]')
    else:
        print(f'Temperature colormap range: [{bulk_temp:.1f}, {heater_temp:.1f}] C')

    os.makedirs(args.output_dir, exist_ok=True)

    # Run every model, saving per-model GIFs as we go.
    all_results: "OrderedDict[str, list]" = OrderedDict()
    model_labels: "OrderedDict[str, str]" = OrderedDict()

    for model_name, info in models_to_run.items():
        model_type = info['model_type']
        checkpoint = info['checkpoint']

        print(f'\n{"=" * 60}')
        print(f'Model: {model_name}  ({model_type})')
        print(f'Checkpoint: {checkpoint}')
        print(f'{"=" * 60}')

        try:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)

            # Match the metrics script: suppress noisy per-model dataset/model
            # output to keep the high-level progress readable.
            results = run_task1_model_inference(
                model_name, model_type, checkpoint, args, task_cfg,
                norm_stats, device,
            )
        except Exception as exc:
            print(f'  ERROR: {exc}')
            import traceback
            traceback.print_exc()
            results = None

        if results is None or not results:
            print('  Skipping (no frames generated)')
            continue

        print(f'  Generated {len(results)} frames.')

        # Sanitize a filesystem-safe label
        label = model_name.lower().replace(' ', '_').replace('-', '_')
        for ch in ('(', ')', '/', '.'):
            label = label.replace(ch, '_')
        while '__' in label:
            label = label.replace('__', '_')
        label = label.strip('_') or model_type
        model_labels[model_name] = label

        if args.per_model_gifs:
            generate_per_model_gif(
                results, label, model_name,
                args.output_dir, bulk_temp, heater_temp,
                args.cmap_vmin, args.dpi, args.gif_duration,
                plot_normalized=args.plot_normalized,
            )

        if args.combined_gif:
            all_results[model_name] = results

    # ------------------------------------------------------------------
    # Combined comparison GIF: [SDF GT] [Temp GT] [model 1] [model 2] ...
    # ------------------------------------------------------------------
    if args.combined_gif and len(all_results) >= 1:
        print(f'\nRendering combined GIF for {len(all_results)} models...')
        first_name = next(iter(all_results.keys()))
        first_results = all_results[first_name]
        n_frames = min(len(r) for r in all_results.values())
        display_names = list(all_results.keys())

        # Pick a single global SDF [vmin, vmax] so the TwoSlopeNorm endpoints
        # stay stable across frames (matches the styling in generate_task2_figures.py).
        sdf_vmin_global = 0.0
        sdf_vmax_global = 0.0
        for i in range(n_frames):
            sdf_i = first_results[i][0]
            sdf_np = sdf_i.numpy() if hasattr(sdf_i, 'numpy') else np.asarray(sdf_i)
            sdf_vmin_global = min(sdf_vmin_global, float(sdf_np.min()))
            sdf_vmax_global = max(sdf_vmax_global, float(sdf_np.max()))
        # Guard against degenerate ranges where one side is empty.
        if sdf_vmin_global >= 0:
            sdf_vmin_global = -1e-6
        if sdf_vmax_global <= 0:
            sdf_vmax_global = 1e-6

        combined_dir = os.path.join(args.output_dir, 'combined', 'frames')
        combined_paths = []

        for i in range(n_frames):
            sdf, gt_temp, _ = first_results[i]
            preds = []
            for name in display_names:
                _, _, pred = all_results[name][i]
                preds.append(pred)

            p = os.path.join(combined_dir, f'combined_{i:04d}.png')
            render_combined_frame_task1(
                i, sdf, gt_temp, preds, display_names,
                bulk_temp, heater_temp, args.cmap_vmin,
                sdf_vmin_global, sdf_vmax_global,
                p, args.dpi,
                plot_normalized=args.plot_normalized,
            )
            combined_paths.append(p)

        gif_p = os.path.join(args.output_dir, 'combined_all_models_task1.gif')
        create_gif(combined_paths, gif_p, args.gif_duration)

    print(f'\nAll GIFs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
