#!/usr/bin/env python3
"""
Rollout Metrics for AR Bootstrap Flow Matching.

Runs autoregressive rollout inference for a configurable number of steps,
computes per-timestep physics metrics, and produces:
  1. A summary table (average metrics over each step window)
  2. Per-metric rollout plots with uncertainty bands from multiple seeds

Supported tasks (auto-detected from the checkpoint's ``task_cfg.target_names``,
override with ``--task``):

  task1: temperature_from_sdf  (target = temperature)
    Uses the inference task-1 metrics except Temp. Max Rel L2 (redundant with
    per-frame Relative L2 in rollout): Relative L2, Max Error, IRMSE, BRMSE,
    HF Energy Ratio, Wall Heat Flux Rel. Error (%).

  task2: velocity_from_interface  (targets = velx, vely, temperature)
    Uses the inference task-2 metrics except Vel./Temp. Max Rel L2 (same reason):
    6 velocity + 6 temperature metrics (see inference scripts for definitions).

  Optional ``--fourier`` adds the Fourier spectral band metrics from the
  corresponding inference script.

Since the model is stochastic (flow matching), we run multiple random seeds
and show the range (min–max band) and mean across seeds.

Usage:
    # Task 1 rollout with default checkpoint
    python scripts/rollout_metrics.py --task task1 --steps 1,100,200,300

    # Task 2 rollout
    python scripts/rollout_metrics.py --task task2 --steps 1,100,200,300

    # Quick test
    python scripts/rollout_metrics.py --steps 1,5,10 --num-seeds 3

    # Custom checkpoint
    python scripts/rollout_metrics.py --steps 1,50,100 --checkpoint /path/to/ckpt
"""

import sys
import os
import json
import re
import argparse
import random
import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.physics_metrics_task123 import (
    load_dataset,
    load_ground_truth_data,
)
from scripts.inference_metrics_task1 import (
    compute_task1_metrics,
    _get_metric_names as _get_metric_names_task1,
)
from scripts.inference_metrics_task2 import (
    compute_task2_metrics,
    _get_metric_names as _get_metric_names_task2,
)
from bubblefusion.models.flow_matching_ar_bootstrap import (
    ConditionalFlowMatchingARBootstrapLightning,
)
from bubblefusion.data.bubbleml import compute_normalization_stats

# ----------------------------------------------------------------------------
# Default checkpoints per task
# ----------------------------------------------------------------------------

DEFAULT_CHECKPOINT_TASK1 = (
    # "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/"
    # "flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_"
    # "temperature_from_sdf_pb_subcooled_singlestep_none_ds4_51527954/"
    # "checkpoints/epoch=23-step=019920.ckpt"
    "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/"
    "flow_matching_ar_bootstrap_32_256_2_False_hist64_roll5_attn_d128_L2_p8_"
    "temperature_from_sdf_pb_subcooled_singlestep_none_ds4_52646121/"
    "checkpoints/last.ckpt"
)

DEFAULT_CHECKPOINT_TASK2 = (
    # "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/"
    # "flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_"
    # "velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/"
    # "checkpoints/epoch=07-step=013280.ckpt"
    "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/"
    "flow_matching_ar_bootstrap_32_256_2_False_hist64_roll5_attn_d128_L2_p8_"
    "velocity_from_interface_pb_subcooled_singlestep_none_ds4_52453282/"
    "checkpoints/last.ckpt"
)

# Backwards-compatibility alias (older invocations imported this name)
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_TASK1

# ----------------------------------------------------------------------------
# Per-task metric name lists (delegated to inference metric scripts)
# ----------------------------------------------------------------------------

# Max Rel L2 collapses to Relative L2 when metrics are evaluated one frame at
# a time; keep it in inference scripts but omit from rollout plots/tables.
_ROLLOUT_EXCLUDED_METRICS = frozenset({
    'Temp. Max Rel L2',
    'Vel. Max Rel L2',
})


def get_metric_names(task: str, include_fourier: bool = False):
    if task == 'task1':
        names = _get_metric_names_task1(include_fourier)
    elif task in ('task2', 'task3'):
        names = _get_metric_names_task2(include_fourier)
    else:
        raise ValueError(f"Unknown task: {task}")
    return [n for n in names if n not in _ROLLOUT_EXCLUDED_METRICS]


# ============================================================================
# Per-frame metric helpers
# ============================================================================

def load_massflux(data_file, start_time, downsample_factor, num_frames):
    """Load raw massflux from HDF5, aligned with load_ground_truth_data slicing."""
    with h5py.File(data_file, 'r') as f:
        if 'massflux' not in f:
            return None
        mf_raw = f['massflux'][start_time:start_time + num_frames]
        if downsample_factor > 1:
            from scipy.ndimage import zoom
            scale = 1.0 / downsample_factor
            return zoom(mf_raw, (1, scale, scale), order=1)
        return mf_raw


def compute_per_frame_metrics_task1(
    gt_temp, pred_temp, sdf, heater_temp, downsample_factor=4,
    massflux=None, include_fourier=False,
):
    """Per-timestep Task 1 metrics using inference_metrics_task1 definitions."""
    T = gt_temp.shape[0]
    metric_names = get_metric_names('task1', include_fourier)
    result = {name: np.empty(T) for name in metric_names}

    for t in range(T):
        mf_slice = massflux[t:t + 1] if massflux is not None else None
        metrics_t = compute_task1_metrics(
            gt_temp[t:t + 1], pred_temp[t:t + 1],
            sdf[t:t + 1], heater_temp,
            massflux=mf_slice,
            downsample_factor=downsample_factor,
            include_fourier=include_fourier,
        )
        for name in metric_names:
            result[name][t] = metrics_t[name]

    return result


def compute_per_frame_metrics_task2(
    gt_velx, gt_vely, gt_temp,
    pred_velx, pred_vely, pred_temp,
    sdf, heater_temp, downsample_factor=4,
    massflux=None, include_fourier=False,
):
    """Per-timestep Task 2 metrics using inference_metrics_task2 definitions."""
    T = gt_temp.shape[0]
    metric_names = get_metric_names('task2', include_fourier)
    result = {name: np.empty(T) for name in metric_names}

    for t in range(T):
        mf_slice = massflux[t:t + 1] if massflux is not None else None
        metrics_t = compute_task2_metrics(
            gt_velx[t:t + 1], gt_vely[t:t + 1], gt_temp[t:t + 1],
            pred_velx[t:t + 1], pred_vely[t:t + 1], pred_temp[t:t + 1],
            sdf[t:t + 1], heater_temp,
            massflux=mf_slice,
            downsample_factor=downsample_factor,
            include_fourier=include_fourier,
        )
        for name in metric_names:
            result[name][t] = metrics_t[name]

    return result


# ============================================================================
# Single-seed rollout inference (returns per-frame GT + pred dicts)
# ============================================================================

def run_rollout_single_seed(
    model, dataset, device, num_frames, num_inference_steps, solver,
):
    """Run AR bootstrap rollout for *num_frames* frames.

    Returns:
        gt_fields, pred_fields  -- dicts mapping each name in
        ``model.task_cfg.target_names`` to a (num_frames, H, W) numpy array.
    """
    model = model.to(device)
    model.eval()

    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels

    rollout_length = dataset.rollout_length

    gt_lists = {name: [] for name in target_names}
    pred_lists = {name: [] for name in target_names}

    frames_generated = 0
    segment_stride = rollout_length

    with torch.no_grad():
        seg_i = 0
        while frames_generated < num_frames:
            segment_idx = seg_i * segment_stride
            if segment_idx >= len(dataset):
                break

            sample_data = dataset[segment_idx]
            if dataset.return_wall_temp:
                cond_hist, cond_seq, target_seq, _ = sample_data
            else:
                cond_hist, cond_seq, target_seq = sample_data

            cond_hist = cond_hist.unsqueeze(0).to(device)
            cond_seq = cond_seq.unsqueeze(0).to(device)
            target_seq = target_seq.unsqueeze(0).to(device)

            cond_hist_ext = cond_hist[:, :, conditioning_channels, :, :]
            cond_seq_ext = cond_seq[:, :, conditioning_channels, :, :]
            target_seq_ext = target_seq[:, :, target_channels, :, :]

            B, _, C_cond, H, W = cond_hist_ext.shape
            L = cond_seq_ext.shape[1]
            C_out = target_seq_ext.shape[2]

            current_cond_0 = cond_seq_ext[:, 0]
            prev_output = model.bootstrap_initial_state(cond_hist_ext, current_cond_0)

            for l in range(L):
                if frames_generated >= num_frames:
                    break

                current_cond = cond_seq_ext[:, l]
                target_l = target_seq_ext[:, l]

                availability_mask = (
                    torch.zeros(B, 1, H, W, device=device)
                    if l == 0
                    else torch.ones(B, 1, H, W, device=device)
                )

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

                for j, field_name in enumerate(target_names):
                    target_cpu[j] = dataset._denormalize_field(target_cpu[j], field_name)
                    predicted_cpu[j] = dataset._denormalize_field(predicted_cpu[j], field_name)
                    gt_lists[field_name].append(target_cpu[j].numpy())
                    pred_lists[field_name].append(predicted_cpu[j].numpy())

                prev_output = predicted
                frames_generated += 1

            seg_i += 1

    gt_fields = {name: np.stack(arrs[:num_frames]) for name, arrs in gt_lists.items()}
    pred_fields = {name: np.stack(arrs[:num_frames]) for name, arrs in pred_lists.items()}
    return gt_fields, pred_fields


# ============================================================================
# Normalization stats loader (mirrors history_length_ablation.py)
# ============================================================================

def load_normalization_stats(checkpoint_path, explicit_path=None, data_file=None, start_time=100):
    if explicit_path and os.path.exists(explicit_path):
        with open(explicit_path, 'r') as f:
            return json.load(f)

    ckpt_dir = os.path.dirname(checkpoint_path)
    if 'checkpoints' in ckpt_dir:
        ckpt_dir = os.path.dirname(ckpt_dir)
    stats_file = os.path.join(ckpt_dir, 'normalization_stats.json')
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)

    if data_file:
        return compute_normalization_stats(filenames=[data_file], start_time=start_time, verbose=False)

    raise FileNotFoundError("Cannot find or compute normalization stats")


# ============================================================================
# Summary table
# ============================================================================

def build_summary_table(all_per_frame, steps_list, metric_names):
    """Build a summary table from per-seed per-frame metrics.

    Args:
        all_per_frame: list of dicts (one per seed) mapping metric -> (T,) array
        steps_list: list of step counts, e.g. [1, 100, 200, 300]
        metric_names: ordered list of metric names

    Returns:
        pandas DataFrame with metric names as rows and step counts as columns.
    """
    rows = []
    for metric in metric_names:
        row = {"Metric": metric}
        for n_steps in steps_list:
            vals = []
            for seed_metrics in all_per_frame:
                arr = seed_metrics[metric]
                n = min(n_steps, len(arr))
                vals.append(np.nanmean(arr[:n]))
            mean_val = np.nanmean(vals)
            std_val = np.nanstd(vals)
            if len(vals) > 1:
                row[f"{n_steps} steps"] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[f"{n_steps} steps"] = f"{mean_val:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Rollout plots
# ============================================================================

def _short_metric_label(metric: str) -> str:
    """Compact y-axis label: drop parenthetical qualifiers like ``(excl. interface)``."""
    return re.sub(r'\s*\([^)]*\)', '', metric).strip()


def _split_velocity_temperature_metrics(metric_names):
    """Split task-2 metric names into velocity (left) and temperature (right) groups."""
    vel_metrics = []
    temp_metrics = []
    for name in metric_names:
        if name.startswith('Vel.') or name.startswith('Vorticity'):
            vel_metrics.append(name)
        elif name.startswith('Temp.') or name.startswith('Wall Heat Flux'):
            temp_metrics.append(name)
    return vel_metrics, temp_metrics


def _plot_rollout_metric(
    ax, metric, all_per_frame, max_T, timesteps, steps_list, num_seeds,
    color, axis_label_fs, legend_fs, tick_fs, show_xlabel=False,
    show_legend=True,
):
    """Draw mean curve + seed range band for one metric on *ax*."""
    stacked = np.stack([m[metric][:max_T] for m in all_per_frame], axis=0)
    mean_curve = np.nanmean(stacked, axis=0)
    min_curve = np.nanmin(stacked, axis=0)
    max_curve = np.nanmax(stacked, axis=0)

    ax.plot(timesteps, mean_curve, color=color, linewidth=1.5, label='Mean')
    ax.fill_between(timesteps, min_curve, max_curve, color=color, alpha=0.2,
                    label=f'Range ({num_seeds} seeds)')

    for s in steps_list:
        if s <= max_T:
            ax.axvline(x=s, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_ylabel(_short_metric_label(metric), fontsize=axis_label_fs)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    if show_legend:
        ax.legend(fontsize=legend_fs, loc='upper right')
    ax.grid(True, alpha=0.3)
    if show_xlabel:
        ax.set_xlabel('Rollout Timestep', fontsize=axis_label_fs)


def _set_bottom_xaxis(ax, axis_label_fs, tick_fs):
    """Show rollout timestep ticks and label on the bottom subplot of a column."""
    ax.set_xlabel('Rollout Timestep', fontsize=axis_label_fs)
    ax.tick_params(axis='x', which='major', labelbottom=True, labelsize=tick_fs)


def generate_rollout_plots(all_per_frame, output_dir, steps_list, metric_names, task='task1'):
    """Generate one plot per metric showing per-timestep values with band."""
    title_fs = 22
    axis_label_fs = 18
    legend_fs = 15
    tick_fs = 16

    num_seeds = len(all_per_frame)
    max_T = max(len(list(m.values())[0]) for m in all_per_frame)
    timesteps = np.arange(1, max_T + 1)

    colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed']
    vel_colors = ['#2563eb', '#1d4ed8', '#3b82f6', '#60a5fa', '#93c5fd',
                  '#1e40af', '#172554', '#0ea5e9']
    temp_colors = ['#dc2626', '#b91c1c', '#ef4444', '#f87171', '#fca5a5',
                   '#991b1b', '#7f1d1d']

    vel_metrics, temp_metrics = _split_velocity_temperature_metrics(metric_names)
    use_two_columns = task in ('task2', 'task3') and vel_metrics and temp_metrics

    if use_two_columns:
        n_rows = max(len(vel_metrics), len(temp_metrics))
        # wspace controls the gap between columns (fraction of avg axes width).
        # tight_layout(w_pad=...) does not reliably change this for 2-column grids.
        column_wspace = 0.19
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(17, 3.5 * n_rows), sharex=True,
            gridspec_kw={'wspace': column_wspace},
        )
        if n_rows == 1:
            axes = axes.reshape(1, 2)

        axes[0, 0].set_title('Velocity', fontsize=title_fs, fontweight='bold')
        axes[0, 1].set_title('Temperature', fontsize=title_fs, fontweight='bold')

        for row, metric in enumerate(vel_metrics):
            _plot_rollout_metric(
                axes[row, 0], metric, all_per_frame, max_T, timesteps, steps_list,
                num_seeds, vel_colors[row % len(vel_colors)],
                axis_label_fs, legend_fs, tick_fs,
                show_legend=(row == 0),
            )

        for row, metric in enumerate(temp_metrics):
            _plot_rollout_metric(
                axes[row, 1], metric, all_per_frame, max_T, timesteps, steps_list,
                num_seeds, temp_colors[row % len(temp_colors)],
                axis_label_fs, legend_fs, tick_fs,
                show_legend=(row == 0),
            )

        for row in range(len(vel_metrics), n_rows):
            axes[row, 0].set_visible(False)
        for row in range(len(temp_metrics), n_rows):
            axes[row, 1].set_visible(False)

        # Bottom row may be hidden when column lengths differ; label last used row.
        _set_bottom_xaxis(axes[len(vel_metrics) - 1, 0], axis_label_fs, tick_fs)
        _set_bottom_xaxis(axes[len(temp_metrics) - 1, 1], axis_label_fs, tick_fs)
    else:
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)),
                                 sharex=True)
        if len(metric_names) == 1:
            axes = [axes]

        for idx, metric in enumerate(metric_names):
            _plot_rollout_metric(
                axes[idx], metric, all_per_frame, max_T, timesteps, steps_list,
                num_seeds, colors[idx % len(colors)],
                axis_label_fs, legend_fs, tick_fs,
            )
            axes[idx].set_title(metric, fontsize=title_fs, fontweight='bold')

        _set_bottom_xaxis(axes[-1], axis_label_fs, tick_fs)

    if use_two_columns:
        # Avoid tight_layout here: hidden axes + tight_layout ignore wspace.
        fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.05,
                            wspace=column_wspace, hspace=0.28)
    else:
        plt.tight_layout()

    plot_path = os.path.join(output_dir, 'rollout_metrics.png')
    save_kwargs = {'dpi': 300}
    if not use_two_columns:
        save_kwargs['bbox_inches'] = 'tight'
    fig.savefig(plot_path, **save_kwargs)
    plt.close(fig)
    print(f"\nRollout plot saved to: {plot_path}")

    # Individual metric plots
    for idx, metric in enumerate(metric_names):
        fig_single, ax = plt.subplots(figsize=(10, 4))
        _plot_rollout_metric(
            ax, metric, all_per_frame, max_T, timesteps, steps_list,
            num_seeds, colors[idx % len(colors)],
            axis_label_fs, legend_fs, tick_fs, show_xlabel=True,
        )
        ax.set_title(f'Rollout: {metric}', fontsize=title_fs, fontweight='bold')
        plt.tight_layout()

        safe_name = metric.lower().replace(' ', '_').replace('(', '').replace(')', '')
        safe_name = safe_name.replace(',', '').replace('.', '').replace('%', 'pct')
        fig_single.savefig(os.path.join(output_dir, f'rollout_{safe_name}.png'),
                           dpi=300, bbox_inches='tight')
        plt.close(fig_single)


# ============================================================================
# Main
# ============================================================================

def _detect_task_from_target_names(target_names):
    """Auto-detect task from target_names: task1 if temperature-only."""
    names = set(target_names)
    if names == {'temperature'}:
        return 'task1'
    if 'velx' in names and 'vely' in names:
        return 'task2'
    return 'task2'


def main():
    parser = argparse.ArgumentParser(
        description='Rollout metrics for AR Bootstrap Flow Matching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--task', type=str, default='task1',
                        choices=['task1', 'task2', 'task3', 'auto'],
                        help='Which task to run: task1=temperature_from_sdf '
                             '(temperature-only metrics), task2/task3=velocity_from_interface '
                             '(velocity+temperature metrics). "auto" infers from the checkpoint.')
    parser.add_argument('--steps', type=str, default='1,100,200,300',
                        help='Comma-separated list of rollout step counts for summary table')
    parser.add_argument('--num-seeds', type=int, default=10,
                        help='Number of random seeds for stochastic evaluation')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base seed (seeds will be base_seed, base_seed+1, ...)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: per-task default)')
    parser.add_argument('--output-dir', type=str, default='./ICML/CamReady/Table7_Rollout_Task2',
                        help='Directory to save results (default: ./ICML/rollout_metrics/<task>/rk4)')
    parser.add_argument('--data-file', type=str,
                        default='/share/crsp/lab/amowli/share/BubbleML_2/'
                                'PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5',
                        help='Path to HDF5 data file')
    parser.add_argument('--start-time', type=int, default=900,
                        help='Starting timestep in HDF5 file')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Spatial downsampling factor')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='ODE integration steps per frame')
    parser.add_argument('--solver', type=str, default='heun',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver')
    parser.add_argument('--normalization-stats', type=str,
                        default='/share/crsp/lab/amowli/xianwz2/diffusion/'
                                'bubblefusion/normalization_stats.json',
                        help='Path to normalization_stats.json')
    parser.add_argument('--norm-mode', type=str, default='all',
                        choices=['none', 'all', 'temperature_only'],
                        help='Normalization mode (must match training)')
    parser.add_argument('--row-indices', type=str, default='0,8,16,24,32',
                        help='Comma-separated row indices for temperature analysis')
    parser.add_argument('--fourier', action='store_true', default=False,
                        help='Include Fourier spectral band metrics from the inference scripts')
    args = parser.parse_args()

    # Resolve defaults that depend on --task
    if args.checkpoint is None:
        args.checkpoint = (DEFAULT_CHECKPOINT_TASK1 if args.task == 'task1'
                           else DEFAULT_CHECKPOINT_TASK2)
    if args.output_dir is None:
        task_label = 'task1' if args.task == 'task1' else 'task2'
        args.output_dir = f'./ICML/rollout_metrics/{task_label}/{args.solver}'

    steps_list = sorted(int(s.strip()) for s in args.steps.split(','))
    max_steps = max(steps_list)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Rollout Metrics — AR Bootstrap Flow Matching")
    print("=" * 70)
    print(f"  Task                    : {args.task}")
    print(f"  Steps for summary table : {steps_list}")
    print(f"  Max rollout length      : {max_steps}")
    print(f"  Number of seeds         : {args.num_seeds}")
    print(f"  Checkpoint              : {args.checkpoint}")
    print(f"  Output dir              : {args.output_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    norm_stats = load_normalization_stats(
        args.checkpoint, args.normalization_stats, args.data_file, args.start_time,
    )

    model = ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
        args.checkpoint, normalization_stats=norm_stats, strict=True,
    )
    print(f"Model loaded — history_length={model.history_length}, "
          f"rollout_length={model.rollout_length}, "
          f"history_stride={model.history_stride}")

    # Resolve task (auto-detection if requested)
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    print(f"  Model target_names      : {target_names}")
    if args.task == 'auto':
        args.task = _detect_task_from_target_names(target_names)
        print(f"  Auto-detected task      : {args.task}")

    if args.task == 'task1' and target_names != ['temperature']:
        print(f"  ⚠️  Task is task1 but target_names={target_names}; expected ['temperature'].")
    if args.task in ('task2', 'task3') and not ({'velx', 'vely'} <= set(target_names)):
        print(f"  ⚠️  Task is {args.task} but target_names={target_names}; "
              f"expected to include 'velx' and 'vely'.")

    metric_names = get_metric_names(args.task, include_fourier=args.fourier)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load dataset & ground truth
    # ------------------------------------------------------------------
    dataset = load_dataset(
        args.data_file,
        output_fields=['temperature', 'velx', 'vely'],
        start_time=args.start_time,
        normalize_temperature=False,
        return_wall_temp=False,
        noise_cfg=None,
        use_clean_inputs=False,
        is_temporal=False,
        is_autoregressive=False,
        is_ar_bootstrap=True,
        history_length=model.history_length,
        history_stride=model.history_stride,
        temporal_stride=1,
        rollout_length=model.rollout_length,
        downsample_factor=args.downsample_factor,
        normalization_stats=norm_stats,
        norm_mode=args.norm_mode,
    )

    sdf_gt_full, _, _, _, _, _, heater_temp = load_ground_truth_data(
        args.data_file, args.start_time, args.downsample_factor,
    )

    sdf_slice = sdf_gt_full[:max_steps]
    massflux_slice = load_massflux(
        args.data_file, args.start_time, args.downsample_factor, max_steps,
    )
    if massflux_slice is not None:
        print(f"  Loaded raw mass flux: shape={massflux_slice.shape}")
    else:
        print("  Warning: no 'massflux' field in source HDF5 — "
              "falling back to SDF zero-crossing interface mask")

    # ------------------------------------------------------------------
    # Multi-seed rollout
    # ------------------------------------------------------------------
    all_per_frame = []

    for seed_idx in range(args.num_seeds):
        seed = args.base_seed + seed_idx
        print(f"\n{'='*50}")
        print(f"  Seed {seed_idx + 1}/{args.num_seeds}  (seed={seed})")
        print(f"{'='*50}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        gt_fields, pred_fields = run_rollout_single_seed(
            model, dataset, device, max_steps,
            args.num_inference_steps, args.solver,
        )

        gt_temp = gt_fields['temperature']
        pred_temp = pred_fields['temperature']
        actual_frames = gt_temp.shape[0]
        sdf_for_metrics = sdf_slice[:actual_frames]
        mf_for_metrics = (massflux_slice[:actual_frames]
                          if massflux_slice is not None else None)

        print(f"  Generated {actual_frames} frames, computing per-frame metrics...")

        if args.task == 'task1':
            per_frame = compute_per_frame_metrics_task1(
                gt_temp, pred_temp,
                sdf_for_metrics, heater_temp,
                downsample_factor=args.downsample_factor,
                massflux=mf_for_metrics,
                include_fourier=args.fourier,
            )
        else:
            per_frame = compute_per_frame_metrics_task2(
                gt_fields['velx'], gt_fields['vely'], gt_temp,
                pred_fields['velx'], pred_fields['vely'], pred_temp,
                sdf_for_metrics, heater_temp,
                downsample_factor=args.downsample_factor,
                massflux=mf_for_metrics,
                include_fourier=args.fourier,
            )
        all_per_frame.append(per_frame)

        # Quick summary for this seed
        for metric in metric_names:
            arr = per_frame[metric]
            print(f"    {metric}: mean={np.nanmean(arr):.4f}, "
                  f"last={arr[-1]:.4f}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  ROLLOUT METRICS — SUMMARY TABLE")
    print("=" * 90)

    df = build_summary_table(all_per_frame, steps_list, metric_names)
    print(df.to_string(index=False))

    csv_path = os.path.join(args.output_dir, 'rollout_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    # Also save raw per-frame data for each seed
    raw_path = os.path.join(args.output_dir, 'rollout_per_frame.npz')
    save_dict = {}
    for seed_idx, per_frame in enumerate(all_per_frame):
        for metric, arr in per_frame.items():
            safe_key = metric.replace(' ', '_').replace('(', '').replace(')', '')
            safe_key = safe_key.replace(',', '').replace('.', '').replace('%', 'pct')
            save_dict[f'seed{seed_idx}_{safe_key}'] = arr
    np.savez_compressed(raw_path, **save_dict)
    print(f"Raw per-frame data saved to: {raw_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating rollout plots...")
    generate_rollout_plots(all_per_frame, args.output_dir, steps_list, metric_names,
                           task=args.task)

    print(f"\nRollout metrics complete. Results in: {args.output_dir}")


if __name__ == '__main__':
    main()
