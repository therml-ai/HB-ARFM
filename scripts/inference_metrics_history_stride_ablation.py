#!/usr/bin/env python3
"""
History-Stride Ablation for Task-2 flow matching models.

Two model architectures share the same per-stride ablation harness here,
selectable via ``--model-type`` (see ``MODEL_REGISTRIES``):

  * ``flow_matching_history`` (default)
        Non-autoregressive baseline. Fixed history window W=10 trained at
        different strides S; conditioning covers raw timesteps
        ``[t - (W-1)*S, ..., t - S, t]`` — a span of ``(W-1)*S + 1`` raw
        frames at constant channel count.
  * ``flow_matching_ar_bootstrap``
        Autoregressive bootstrap rollout (H=10, rollout L=5, attention
        history encoder). Each dataset access yields a full L-frame
        rollout, which the harness flattens into the same metric pipeline.

With history window/length 10 the temporal span grows as

    S = 1  ->   10 raw frames spanned
    S = 2  ->   19 raw frames spanned
    S = 3  ->   28 raw frames spanned
    S = 4  ->   37 raw frames spanned
    S = 5  ->   46 raw frames spanned
    ... (history-only: up to S = 10 -> 91 raw frames)

The same metric suite as scripts/inference_metrics_task2.py is used (15 base
metrics, optional 6 Fourier band metrics).

Seed ablation (``--num-seeds N`` with N > 1) runs each stride with N noise
seeds for the stochastic flow-matching sampler while keeping the snapshot
indices identical across seeds. This isolates pure model stochasticity and
the script then writes both the mean (the usual CSV) and the per-seed std
to disk along with std-trend / coefficient-of-variation plots. Expected
behaviour: larger history strides give the model more context and therefore
should reduce the seed-std of every metric.

Usage:
    # Flow Matching History (default)
    python scripts/inference_metrics_history_stride_ablation.py
    python scripts/inference_metrics_history_stride_ablation.py --num-samples 50
    python scripts/inference_metrics_history_stride_ablation.py --fourier
    python scripts/inference_metrics_history_stride_ablation.py --strides 1 3 5
    python scripts/inference_metrics_history_stride_ablation.py --num-seeds 30
    python scripts/inference_metrics_history_stride_ablation.py \\
        --strides 1 2 3 4 5 6 7 8 9 10 --num-seeds 30 --num-samples 50

    # Flow Matching AR Bootstrap (same harness, different model family)
    python scripts/inference_metrics_history_stride_ablation.py \\
        --model-type flow_matching_ar_bootstrap
    python scripts/inference_metrics_history_stride_ablation.py \\
        --model-type flow_matching_ar_bootstrap --strides 1 2 3 4 5 \\
        --num-seeds 30 --num-samples 50
"""

import sys
import os
import json
import random
import argparse
import contextlib
import io
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.physics_metrics_task123 import (
    load_task_config,
    load_model_from_checkpoint,
    load_dataset,
    load_ground_truth_data,
)
from scripts.inference_metrics_task2 import (
    compute_task2_metrics,
    _radial_energy_spectrum,
    plot_energy_spectra,
    save_inference_hdf5,
    load_normalization_stats,
    build_model_cfg,
    _get_metric_names,
)
from bubblefusion.models.flow_matching_ar_bootstrap import (
    ConditionalFlowMatchingARBootstrapLightning,
)

# matplotlib backend is set to 'Agg' inside scripts.inference_metrics_task2
# (imported above), so importing pyplot here is safe in headless environments.
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ============================================================================
# Plotting helpers
# ============================================================================

def _parse_stride_from_column(col: str) -> int:
    """Extract the integer stride from a column name like 'S=3'.  Returns 0
    if the column cannot be parsed (the column is then sorted to the front,
    which is harmless because such columns shouldn't appear in practice).
    """
    try:
        return int(str(col).split('=')[-1])
    except (ValueError, IndexError):
        return 0


def _classify_metric(name: str) -> str:
    """Bucket a metric name into 'velocity' / 'thermal' / 'other'.

    Velocity bucket: all ``Vel.*`` metrics and ``Vorticity ...``.
    Thermal bucket:  all ``Temp.*`` metrics, plus heat-flux / HF-energy.
    Anything else falls into the ``other`` bucket (which the split plots
    skip; the combined plots still include it).
    """
    n = name.lower()
    if n.startswith("vel.") or n.startswith("vorticity"):
        return "velocity"
    if n.startswith("temp.") or "wall heat flux" in n or "hf energy" in n:
        return "thermal"
    return "other"


def _split_metric_groups(metric_names) -> dict:
    """Return {'velocity': [...], 'thermal': [...], 'other': [...]} preserving
    the relative ordering of ``metric_names`` within each bucket.
    """
    groups: dict = {"velocity": [], "thermal": [], "other": []}
    for m in metric_names:
        groups[_classify_metric(m)].append(m)
    return groups


def plot_metric_trends(df: pd.DataFrame, output_dir: str,
                        filename: str = 'history_stride_ablation_trend.png',
                        var_label: str = 'S',
                        var_long_label: str = 'stride',
                        title_prefix: str = 'History-Stride Ablation') -> str:
    """Plot how every metric varies across ablation values on one axes.

    Each metric is normalised by its value at the *first* column so every
    curve starts at 1.0. This makes trends directly comparable across
    metrics whose absolute scales differ by orders of magnitude (e.g.
    ``Temp. Max Error`` ~30 vs. ``Vel. Amplitude Ratio`` ~1).

    Args:
        df: DataFrame indexed by metric name with columns of the form
            ``'<var_label>=<value>'`` (e.g. ``'S=3'`` or ``'H=10'``).
        output_dir: Directory in which to save the figure.
        filename: Output PNG name.
        var_label: Short symbol used in column names and the ylabel formula
            (``'S'`` for stride / ``'H'`` for history length).
        var_long_label: Long descriptor used in the x-axis label
            (combined as ``f'History {var_long_label} {var_label}'``).
        title_prefix: Plot-title prefix (e.g. ``'History-Stride Ablation'``).

    Returns:
        Absolute path of the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort columns by numeric value so the x-axis is monotonic regardless
    # of input ordering.
    sorted_cols = sorted(df.columns, key=_parse_stride_from_column)
    df_sorted = df[sorted_cols]
    stride_vals = [_parse_stride_from_column(c) for c in sorted_cols]

    # Normalise to the first column; replace zeros in the denominator with
    # NaN so we don't blow up on degenerate metrics (the corresponding lines
    # simply won't be drawn).
    base = df_sorted.iloc[:, 0].replace(0.0, np.nan)
    norm = df_sorted.div(base, axis=0)

    n_metrics = len(norm.index)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_metrics, 1)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X',
               '<', '>', 'p', 'h', 'H', '8', 'd', '+', 'x']

    fig, ax = plt.subplots(figsize=(11, 7))

    for i, metric in enumerate(norm.index):
        ax.plot(stride_vals, norm.loc[metric].values,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=7, linewidth=1.6,
                label=metric)

    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7,
                label='_baseline')
    ax.set_xlabel(f'History {var_long_label} {var_label}', fontsize=12)
    ax.set_ylabel(f'Metric value relative to {var_label}={stride_vals[0]}',
                  fontsize=12)
    ax.set_title(f'{title_prefix}: Metric Trend Across {var_long_label.capitalize()}s',
                 fontsize=13)
    ax.set_xticks(stride_vals)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False, ncol=1)

    fig.tight_layout()
    fname = os.path.join(output_dir, filename)
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fname


def plot_metric_std(df_std: pd.DataFrame, output_dir: str,
                     df_mean: Optional[pd.DataFrame] = None,
                     filename_norm: str = 'history_stride_ablation_std_norm.png',
                     filename_raw: str = 'history_stride_ablation_std_raw.png',
                     filename_cv: str = 'history_stride_ablation_cv.png',
                     var_label: str = 'S',
                     var_long_label: str = 'stride',
                     title_prefix: str = 'History-Stride Ablation') -> dict:
    """Plot the cross-seed std of each metric as a function of the ablation variable.

    The hypothesis under test is that larger conditioning windows reduce
    model stochasticity, so the seed-std of each metric should decrease
    along the swept axis. Up to three figures are written:

    1. ``filename_norm``: std normalised by its value at the smallest
       ablation value (every curve starts at 1.0) — best for visualising
       relative trends.
    2. ``filename_raw``: raw std values on a log y-axis — preserves absolute
       magnitudes which differ by orders of magnitude across metrics.
    3. ``filename_cv``: coefficient of variation (std / |mean|) — produced
       only when ``df_mean`` is supplied; this is the noise-to-signal ratio.

    Args:
        df_std: DataFrame indexed by metric name with columns of the form
            ``'<var_label>=<value>'`` (e.g. ``'S=3'`` or ``'H=10'``).
        output_dir: Directory in which to save the figure(s).
        df_mean: Optional DataFrame with the same shape as ``df_std`` holding
            the per-value means; when provided the CV plot is also produced.
        filename_norm/raw/cv: Output PNG names for the three figures.
        var_label / var_long_label / title_prefix: Axis-/title-text labels
            (see :func:`plot_metric_trends`).

    Returns:
        Dict mapping each generated plot's short key ('norm', 'raw', 'cv')
        to its absolute path.
    """
    os.makedirs(output_dir, exist_ok=True)

    sorted_cols = sorted(df_std.columns, key=_parse_stride_from_column)
    df_sorted = df_std[sorted_cols]
    stride_vals = [_parse_stride_from_column(c) for c in sorted_cols]

    n_metrics = len(df_sorted.index)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_metrics, 1)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X',
               '<', '>', 'p', 'h', 'H', '8', 'd', '+', 'x']

    xlabel = f'History {var_long_label} {var_label}'
    pluralised = f'{var_long_label.capitalize()}s'
    out_paths: dict = {}

    # --- (1) Normalised std relative to first column ----------------------
    base = df_sorted.iloc[:, 0].replace(0.0, np.nan)
    norm = df_sorted.div(base, axis=0)

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, metric in enumerate(norm.index):
        ax.plot(stride_vals, norm.loc[metric].values,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=7, linewidth=1.6,
                label=metric)
    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7,
               label='_baseline')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f'Seed-std relative to {var_label}={stride_vals[0]}', fontsize=12)
    ax.set_title(f'{title_prefix}: Seed-Std Trend Across {pluralised}', fontsize=13)
    ax.set_xticks(stride_vals)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    p = os.path.join(output_dir, filename_norm)
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    out_paths['norm'] = p

    # --- (2) Raw std on log y-axis ----------------------------------------
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, metric in enumerate(df_sorted.index):
        vals = df_sorted.loc[metric].values.astype(float)
        # Replace zeros so log scale doesn't fail
        vals = np.where(vals <= 0.0, np.nan, vals)
        ax.plot(stride_vals, vals,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=7, linewidth=1.6,
                label=metric)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Seed-std (raw, log scale)', fontsize=12)
    ax.set_title(f'{title_prefix}: Raw Seed-Std Per Metric', fontsize=13)
    ax.set_xticks(stride_vals)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    p = os.path.join(output_dir, filename_raw)
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    out_paths['raw'] = p

    # --- (3) Coefficient of variation (std / |mean|) ----------------------
    if df_mean is not None:
        df_mean_sorted = df_mean[sorted_cols].reindex(df_sorted.index)
        denom = df_mean_sorted.abs().replace(0.0, np.nan)
        cv = df_sorted.div(denom)

        fig, ax = plt.subplots(figsize=(11, 7))
        for i, metric in enumerate(cv.index):
            ax.plot(stride_vals, cv.loc[metric].values,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=7, linewidth=1.6,
                    label=metric)
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Coefficient of variation  std / |mean|  (log scale)',
                      fontsize=12)
        ax.set_title(f'{title_prefix}: Noise-to-Signal Across {pluralised}',
                     fontsize=13)
        ax.set_xticks(stride_vals)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, frameon=False, ncol=1)
        fig.tight_layout()
        p = os.path.join(output_dir, filename_cv)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        out_paths['cv'] = p

    return out_paths


def _plot_norm_lines(df: pd.DataFrame, *, ax, title: str, ylabel: str,
                      xlabel: str = 'History stride S') -> None:
    """Helper: draw one normalised line per metric on ``ax``.

    The first column is treated as the baseline (each row is divided by its
    own value at the smallest column), so every curve starts at 1.0.
    """
    stride_vals = [_parse_stride_from_column(c) for c in df.columns]
    base = df.iloc[:, 0].replace(0.0, np.nan)
    norm = df.div(base, axis=0)

    n_metrics = len(norm.index)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_metrics, 1)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X',
               '<', '>', 'p', 'h', 'H', '8', 'd', '+', 'x']
    for i, metric in enumerate(norm.index):
        ax.plot(stride_vals, norm.loc[metric].values,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=8, linewidth=1.8,
                label=metric)
    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8,
               alpha=0.7, label='_baseline')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(stride_vals)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False, ncol=1)


def plot_metric_trends_split(
    df: pd.DataFrame,
    output_dir: str,
    filename_velocity: str = 'history_stride_ablation_trend_velocity.png',
    filename_thermal: str = 'history_stride_ablation_trend_thermal.png',
    var_label: str = 'S',
    var_long_label: str = 'stride',
    title_prefix: str = 'History-Stride Ablation',
) -> dict:
    """Mean-metric trend plots split into velocity vs thermal groups.

    Each plot mirrors :func:`plot_metric_trends` (one line per metric,
    normalised to the smallest column) but only contains the metrics from
    its bucket. Useful when the combined plot becomes too crowded.
    """
    os.makedirs(output_dir, exist_ok=True)

    sorted_cols = sorted(df.columns, key=_parse_stride_from_column)
    df_sorted = df[sorted_cols]
    groups = _split_metric_groups(list(df_sorted.index))

    xlabel = f'History {var_long_label} {var_label}'

    out_paths: dict = {}
    for key, fname, title in (
        ("velocity", filename_velocity,
         f"{title_prefix}: Metric Trend (Velocity)"),
        ("thermal", filename_thermal,
         f"{title_prefix}: Metric Trend (Thermal)"),
    ):
        metrics = groups[key]
        if not metrics:
            continue
        fig, ax = plt.subplots(figsize=(11, 7))
        ref_value = _parse_stride_from_column(df_sorted.columns[0])
        _plot_norm_lines(
            df_sorted.loc[metrics], ax=ax, title=title,
            ylabel=f'Metric value relative to {var_label}={ref_value}',
            xlabel=xlabel,
        )
        fig.tight_layout()
        p = os.path.join(output_dir, fname)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        out_paths[key] = p
    return out_paths


def plot_metric_std_split(
    df_std: pd.DataFrame,
    output_dir: str,
    filename_velocity: str = 'history_stride_ablation_std_norm_velocity.png',
    filename_thermal: str = 'history_stride_ablation_std_norm_thermal.png',
    var_label: str = 'S',
    var_long_label: str = 'stride',
    title_prefix: str = 'History-Stride Ablation',
) -> dict:
    """Normalised seed-std trend plots split into velocity vs thermal.

    Each plot shows the cross-seed std of every metric in its bucket,
    normalised by the value at the smallest ablation value.
    """
    os.makedirs(output_dir, exist_ok=True)

    sorted_cols = sorted(df_std.columns, key=_parse_stride_from_column)
    df_sorted = df_std[sorted_cols]
    groups = _split_metric_groups(list(df_sorted.index))

    xlabel = f'History {var_long_label} {var_label}'

    out_paths: dict = {}
    for key, fname, title in (
        ("velocity", filename_velocity,
         f"{title_prefix}: Seed-Std Trend (Velocity)"),
        ("thermal", filename_thermal,
         f"{title_prefix}: Seed-Std Trend (Thermal)"),
    ):
        metrics = groups[key]
        if not metrics:
            continue
        fig, ax = plt.subplots(figsize=(11, 7))
        ref_value = _parse_stride_from_column(df_sorted.columns[0])
        _plot_norm_lines(
            df_sorted.loc[metrics], ax=ax, title=title,
            ylabel=f'Seed-std relative to {var_label}={ref_value}',
            xlabel=xlabel,
        )
        fig.tight_layout()
        p = os.path.join(output_dir, fname)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        out_paths[key] = p
    return out_paths


def plot_mean_std_band(
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
    metrics: list,
    output_dir: str,
    filename: str = 'history_stride_ablation_mean_std_band.png',
    band_sigmas: float = 1.0,
    n_seeds: Optional[int] = None,
    var_label: str = 'S',
    var_long_label: str = 'stride',
    title_prefix: str = 'History-Stride Ablation',
    var_values: Optional[tuple] = None,
    title_model: Optional[str] = None,
    var_axis_label: Optional[str] = None,
) -> Optional[str]:
    """Plot mean trend + std-band (mean ± band_sigmas·std) across strides.

    The first metric is plotted on the left y-axis, the second on the right
    y-axis (twin-y), so two metrics on very different scales (e.g. ``Temp.
    Relative L2`` ~0.03 vs. ``Vel. Relative L2`` ~0.7) can share one figure
    without one dominating the other visually. The std band uses the same
    seed-std stored in ``df_std`` and is drawn as a filled region at half
    opacity. Returns the saved path, or ``None`` if no requested metric is
    available.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Restrict to metrics that actually exist in both tables and preserve
    # the user-requested ordering (so left-axis is always metrics[0]).
    avail = [m for m in metrics
             if m in df_mean.index and m in df_std.index]
    missing = [m for m in metrics if m not in avail]
    if missing:
        print(f"  [band plot] WARNING: metric(s) not in tables, skipping: "
              f"{missing}")
    if not avail:
        return None

    sorted_cols = sorted(df_mean.columns, key=_parse_stride_from_column)
    if var_values is not None:
        keep = set(var_values)
        sorted_cols = [c for c in sorted_cols
                       if _parse_stride_from_column(c) in keep]
    if not sorted_cols:
        print(f"  [band plot] WARNING: no columns match var_values={var_values}")
        return None
    stride_vals = [_parse_stride_from_column(c) for c in sorted_cols]

    # Publication-sized text (+8 pt over the original 12/13/10 defaults).
    fs_title, fs_label, fs_legend, fs_tick = 21, 20, 18, 18

    palette = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    fig, ax_left = plt.subplots(figsize=(10, 6.5))
    axes = [ax_left]
    if len(avail) >= 2:
        axes.append(ax_left.twinx())

    handles, labels = [], []
    for i, metric in enumerate(avail[: len(axes)]):
        ax = axes[i]
        color = palette[i % len(palette)]
        mean_vals = df_mean.loc[metric, sorted_cols].to_numpy(dtype=float)
        std_vals = df_std.loc[metric, sorted_cols].to_numpy(dtype=float)
        lower = mean_vals - band_sigmas * std_vals
        upper = mean_vals + band_sigmas * std_vals

        (line,) = ax.plot(stride_vals, mean_vals,
                          color=color, marker='o', linewidth=2.0,
                          markersize=7, label=f'{metric} (mean)')
        ax.fill_between(stride_vals, lower, upper,
                        color=color, alpha=0.20,
                        label=f'{metric} (±{band_sigmas:g}σ)')

        ax.tick_params(axis='y', labelcolor=color, labelsize=fs_tick)
        if metric.startswith('Temp.'):
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        handles.append(line)
        labels.append(metric)

    ax_left.set_xlabel(
        f'History {var_long_label} {var_axis_label or var_label}',
        fontsize=fs_label,
    )
    ax_left.set_xticks(stride_vals)
    ax_left.tick_params(axis='x', labelsize=fs_tick)
    ax_left.grid(True, alpha=0.3)
    subtitle = f'{title_prefix}: Mean ± Std Band Across Seeds'
    title = f'{title_model}\n{subtitle}' if title_model else subtitle
    ax_left.set_title(title, fontsize=fs_title)
    ax_left.legend(handles, labels, loc='upper right', ncol=1,
                   frameon=True, framealpha=0.92, fontsize=fs_legend)
    fig.tight_layout()

    p = os.path.join(output_dir, filename)
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return p


# ============================================================================
# Self-contained inference for flow_matching_history
# ----------------------------------------------------------------------------
# We do NOT call scripts.physics_metrics_task123.run_inference_batch here
# because that function does `extract_channels(input_batch, conditioning_channels)`,
# which collapses the BulkFlowHistory's pre-flattened [W*C_cond, H, W] window
# back to a single frame of C_cond channels — incompatible with the history
# UNet's expected in_channels = W*C_cond + C_out. Doing inference inline keeps
# this script independent of edits to physics_metrics_task123.py.
# ============================================================================

@torch.no_grad()
def run_history_inference_single(
    model,
    dataset,
    sample_idx: int,
    device: str,
    num_inference_steps: int,
    solver: str,
):
    """Single-sample inference for a flow_matching_history model.

    Returns: (gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp)
        each as a 2D float32 numpy array of shape (H, W) -- one frame.
    """
    sample = dataset[sample_idx]
    if dataset.return_wall_temp:
        input_data, output_data, _ = sample
    else:
        input_data, output_data = sample

    conditioning = input_data.unsqueeze(0).to(device)  # [1, W*C_cond, H, W]

    target_channels = list(model.target_channels)        # e.g. [1, 2, 0]
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))

    target_full = output_data.unsqueeze(0).to(device)    # [1, C_out_raw, H, W]
    target = target_full[:, target_channels, :, :]       # [1, len(target_channels), H, W]

    H, W = target.shape[2], target.shape[3]
    predicted = model.flow_matching.sample(
        condition=conditioning,
        shape=(1, len(target_channels), H, W),
        device=device,
        num_integration_steps=num_inference_steps,
        solver=solver,
    )

    target = target.squeeze(0).cpu()
    predicted = predicted.squeeze(0).cpu()
    for j, field_name in enumerate(target_names):
        target[j] = dataset._denormalize_field(target[j], field_name)
        predicted[j] = dataset._denormalize_field(predicted[j], field_name)

    velx_idx = target_names.index('velx') if 'velx' in target_names else None
    vely_idx = target_names.index('vely') if 'vely' in target_names else None
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else None

    def _get(arr_tensor, idx):
        return arr_tensor[idx].numpy() if idx is not None else None

    return (
        _get(target,    velx_idx),
        _get(target,    vely_idx),
        _get(target,    temp_idx),
        _get(predicted, velx_idx),
        _get(predicted, vely_idx),
        _get(predicted, temp_idx),
    )


@torch.no_grad()
def run_ar_bootstrap_inference_segment(
    model,
    dataset,
    segment_idx: int,
    device: str,
    num_inference_steps: int,
    solver: str,
):
    """Single-segment inference for a ``flow_matching_ar_bootstrap`` model.

    Bootstraps the initial state from the conditioning history, then
    autoregressively rolls out ``L = dataset.rollout_length`` frames feeding
    every prediction back as ``prev_output``. Frames are denormalised before
    being returned. This is a self-contained mirror of the AR-bootstrap
    rollout in :func:`scripts.physics_metrics_task123.run_ar_bootstrap_inference_batch`,
    rewritten for a single segment to fit cleanly into the per-sample
    metric-aggregation loop used elsewhere in this script.

    Returns: (gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp),
        each a float32 numpy array of shape (L, H, W) — one frame per
        rollout step. Channels that aren't requested by the task return
        ``None`` (mirroring :func:`run_history_inference_single`).
    """
    sample = dataset[segment_idx]
    if dataset.return_wall_temp:
        cond_hist, cond_seq, target_seq, _ = sample
    else:
        cond_hist, cond_seq, target_seq = sample

    cond_hist = cond_hist.unsqueeze(0).to(device)
    cond_seq = cond_seq.unsqueeze(0).to(device)
    target_seq = target_seq.unsqueeze(0).to(device)

    conditioning_channels = list(model.conditioning_channels)
    target_channels = list(model.target_channels)
    target_names = list(model.task_cfg.get('target_names',
                                           ['velx', 'vely', 'temperature']))

    cond_hist_x = cond_hist[:, :, conditioning_channels, :, :]
    cond_seq_x = cond_seq[:, :, conditioning_channels, :, :]
    target_seq_x = target_seq[:, :, target_channels, :, :]

    B, _, _, H, W = cond_hist_x.shape
    _, L, _, _, _ = cond_seq_x.shape
    C_out = target_seq_x.shape[2]

    # Bootstrap from history; t=0 is bootstrapped (availability=0), t>=1 is
    # autoregressive (availability=1) — same convention as the batch helper.
    prev_output = model.bootstrap_initial_state(cond_hist_x, cond_seq_x[:, 0])

    gt_frames, pred_frames = [], []
    for l in range(L):
        current_cond = cond_seq_x[:, l]
        target_l = target_seq_x[:, l]
        availability_mask = (torch.zeros if l == 0 else torch.ones)(
            B, 1, H, W, device=device
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

        gt_frames.append(target_cpu)
        pred_frames.append(predicted_cpu)
        prev_output = predicted

    velx_idx = target_names.index('velx') if 'velx' in target_names else None
    vely_idx = target_names.index('vely') if 'vely' in target_names else None
    temp_idx = target_names.index('temperature') if 'temperature' in target_names else None

    def _stack(frames, idx):
        if idx is None:
            return None
        return np.stack([f[idx].numpy() for f in frames], axis=0)  # [L, H, W]

    return (
        _stack(gt_frames, velx_idx),
        _stack(gt_frames, vely_idx),
        _stack(gt_frames, temp_idx),
        _stack(pred_frames, velx_idx),
        _stack(pred_frames, vely_idx),
        _stack(pred_frames, temp_idx),
    )


def run_inference_sample(
    model_type: str,
    model,
    dataset,
    idx: int,
    device: str,
    num_inference_steps: int,
    solver: str,
):
    """Dispatch a single inference call by model type.

    Always returns arrays of shape ``(T, H, W)``:
      * ``T = 1`` for ``flow_matching_history`` (a single frame),
      * ``T = dataset.rollout_length`` for ``flow_matching_ar_bootstrap``.

    The uniform 3D output lets the per-seed accumulation loop concatenate
    samples along axis 0 without any model-specific branching.
    """
    if model_type == 'flow_matching_history':
        gvx, gvy, gt_, pvx, pvy, pt = run_history_inference_single(
            model, dataset, idx, device, num_inference_steps, solver,
        )
        def _wrap(a):
            return None if a is None else a[None, ...]
        return _wrap(gvx), _wrap(gvy), _wrap(gt_), _wrap(pvx), _wrap(pvy), _wrap(pt)
    if model_type == 'flow_matching_ar_bootstrap':
        return run_ar_bootstrap_inference_segment(
            model, dataset, idx, device, num_inference_steps, solver,
        )
    raise ValueError(f"Unknown model_type: {model_type!r} "
                     f"(supported: {MODEL_TYPES})")


def _frames_per_sample(model_type: str, dataset) -> int:
    """Return how many raw frames a single dataset access produces.

    Needed so the SDF / interface-velocity slices line up with the GT/pred
    arrays produced by the inference dispatcher. ``flow_matching_history``
    yields one frame per access; ``flow_matching_ar_bootstrap`` yields
    ``dataset.rollout_length`` consecutive frames per access.
    """
    if model_type == 'flow_matching_history':
        return 1
    if model_type == 'flow_matching_ar_bootstrap':
        return int(getattr(dataset, 'rollout_length', 1))
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _load_model_for_type(
    model_type: str,
    checkpoint_path: str,
    task_cfg,
    optim_cfg,
    scheduler_cfg,
    norm_stats,
    args,
):
    """Load a checkpoint into a Lightning module appropriate for its type.

    For ``flow_matching_history`` we go through the standard
    ``load_model_from_checkpoint`` / ``build_model_cfg`` path shared with the
    Task-2 inference scripts. For ``flow_matching_ar_bootstrap`` we use
    Lightning's direct ``load_from_checkpoint`` constructor — that path
    pulls all hyperparameters straight from the saved ckpt, which is the
    same loader pattern used by ``scripts/history_length_ablation.py``.
    """
    if model_type == 'flow_matching_history':
        model_cfg = build_model_cfg(model_type, task_cfg, args,
                                    checkpoint_path=checkpoint_path)
        return load_model_from_checkpoint(
            checkpoint_path, model_cfg, optim_cfg, scheduler_cfg, task_cfg,
            model_type=model_type,
            normalization_stats=norm_stats,
            norm_mode=args.norm_mode,
        )
    if model_type == 'flow_matching_ar_bootstrap':
        return ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
            checkpoint_path,
            normalization_stats=norm_stats,
            strict=True,
        )
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _load_dataset_for_type(model_type: str, model, args, norm_stats):
    """Load the validation dataset shape that matches ``model_type``.

    Both shapes share start-time / downsample / normalization plumbing, but
    differ in which constructor flag (``is_history_model`` vs
    ``is_ar_bootstrap``) is set and which dimensional hyperparameters are
    needed.
    """
    if model_type == 'flow_matching_history':
        history_window = int(getattr(model, 'history_window', 10))
        history_stride = int(getattr(model, 'history_stride', 1))
        return load_dataset(
            args.data_file,
            output_fields=['temperature', 'velx', 'vely'],
            start_time=args.start_time,
            return_wall_temp=False,
            is_autoregressive=False,
            is_ar_bootstrap=False,
            is_history_model=True,
            history_window=history_window,
            history_stride=history_stride,
            downsample_factor=args.downsample_factor,
            normalization_stats=norm_stats,
            norm_mode=args.norm_mode,
        )
    if model_type == 'flow_matching_ar_bootstrap':
        history_length = int(getattr(model, 'history_length', 10))
        history_stride = int(getattr(model, 'history_stride', 1))
        rollout_length = int(getattr(model, 'rollout_length', 5))
        return load_dataset(
            args.data_file,
            output_fields=['temperature', 'velx', 'vely'],
            start_time=args.start_time,
            return_wall_temp=False,
            noise_cfg=None,
            use_clean_inputs=False,
            is_temporal=False,
            is_autoregressive=False,
            is_ar_bootstrap=True,
            history_length=history_length,
            history_stride=history_stride,
            temporal_stride=1,
            rollout_length=rollout_length,
            downsample_factor=args.downsample_factor,
            normalization_stats=norm_stats,
            norm_mode=args.norm_mode,
        )
    raise ValueError(f"Unknown model_type: {model_type!r}")


# ============================================================================
# Model Registries — one entry per history_stride, per model architecture
# ============================================================================

LOG_ROOT = "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs"

# Each entry maps a stride to either:
#   - a full path to a .ckpt file, or
#   - a directory (relative to LOG_ROOT or absolute) whose "checkpoints/"
#     subfolder will be searched for last.ckpt / the highest-epoch checkpoint.

# Flow Matching History (single-step, history-window conditioning, W=10):
STRIDE_REGISTRY = OrderedDict([
    (1,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52007481/checkpoints/epoch=23-step=039936.ckpt"),
    (2,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52007489/checkpoints/epoch=23-step=039936.ckpt"),
    (3,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52007500/checkpoints/epoch=23-step=039936.ckpt"),
    (4,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52007519/checkpoints/epoch=23-step=039936.ckpt"),
    (5,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52007524/checkpoints/epoch=23-step=039936.ckpt"),
    (6,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52452934/checkpoints/epoch=23-step=039936.ckpt"),
    (7,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52452939/checkpoints/epoch=23-step=039936.ckpt"),
    (8,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52452956/checkpoints/epoch=23-step=039936.ckpt"),
    (9,  f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52452968/checkpoints/epoch=23-step=039936.ckpt"),
    (10, f"{LOG_ROOT}/flow_matching_history_default_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52452969/checkpoints/epoch=23-step=039936.ckpt"),
])

# Flow Matching AR Bootstrap (autoregressive rollout, hist10_roll5 attention):
# stride is the history_stride; H=10 and rollout_length=5 are fixed for this
# registry. Identical raw-frame span semantics to the history models so
# results are directly comparable when paired by stride.
AR_BOOTSTRAP_STRIDE_REGISTRY = OrderedDict([
    (1, f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265190/checkpoints/last.ckpt"),
    (2, f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265193/checkpoints/last.ckpt"),
    (3, f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265198/checkpoints/last.ckpt"),
    (4, f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265203/checkpoints/last.ckpt"),
    (5, f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265204/checkpoints/last.ckpt"),
])

# Flow Matching AR Bootstrap — history-LENGTH ablation (H varies, S=1):
# Same architecture as the stride registry above (attn_d128_L2_p8, roll=5)
# but the conditioning history length itself is what changes. Stride is
# fixed at 1 so the conditioning window covers exactly H consecutive raw
# frames. Reuses the existing hist3/5/7/9/10 checkpoints (also referenced
# from scripts/history_length_ablation.py) plus hist1/4/16/64 for a wider
# log-spaced sweep.
AR_BOOTSTRAP_HISTORY_LENGTH_REGISTRY = OrderedDict([
    (1,  f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist1_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52452989/checkpoints/epoch=09-step=008300.ckpt"),
    (3,  f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist3_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251761/checkpoints/epoch=07-step=013280.ckpt"),
    (4,  f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist4_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52452992/checkpoints/epoch=09-step=008300.ckpt"),
    (5,  f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist5_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251759/checkpoints/epoch=07-step=013280.ckpt"),
    (7,  f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist7_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251762/checkpoints/epoch=07-step=013280.ckpt"),
    (9,  f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist9_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251758/checkpoints/epoch=07-step=013280.ckpt"),
    (10, f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/checkpoints/epoch=07-step=013280.ckpt"),
    (16, f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist16_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52453062/checkpoints/epoch=07-step=013280.ckpt"),
    (64, f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist64_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52453282/checkpoints/epoch=09-step=016600.ckpt"),
])


@dataclass(frozen=True)
class AblationConfig:
    """All metadata needed to run, label, and save one ablation experiment.

    Each entry in :data:`ABLATIONS` bundles the model architecture (which
    determines how the checkpoint is loaded + how inference is run) with
    the variable being swept (stride vs. history-length) and the labels /
    file names / output subdirectory used for that sweep. This is what
    lets a single harness produce ``S=1, S=2, ...`` columns + plots for
    one experiment and ``H=1, H=3, ..., H=64`` columns + plots for
    another without any branching in the main loop.

    Attributes:
        name: CLI key (also the ``--model-type`` value).
        model_type: Underlying model family for the dispatcher
            (``flow_matching_history`` or ``flow_matching_ar_bootstrap``).
        var_label: Short symbol for table columns / legends — typically
            ``"S"`` (stride) or ``"H"`` (history length).
        var_long_label: Long form used in axis labels and prose, e.g.
            ``"stride"`` or ``"length"`` (paired with ``"History "`` prefix).
        var_attr: Attribute on the loaded model used to validate that the
            checkpoint matches its registry key (``"history_stride"`` or
            ``"history_length"``).
        default_output_subdir: Full default output sub-path under
            ``./ICML/`` when ``--output-dir`` is not set. Differs per
            ablation so concurrent runs of different ablations don't
            stomp on each other's files.
        file_prefix: Base prefix for every CSV / PNG / HDF5 written by
            this ablation, e.g. ``"history_stride_ablation"`` /
            ``"history_length_ablation"``.
        title_prefix: Title prefix used in plots (e.g.
            ``"History-Stride Ablation"`` / ``"History-Length Ablation"``).
        registry: Ordered mapping from variable value (int) to checkpoint
            path (string).
        band_var_values: Optional subset of registry keys plotted in the
            mean±std-band figure (``None`` = all evaluated values).
        default_band_sigmas: Default band width in seed-std units when
            ``--band-sigmas`` is not set on the CLI.
        band_title_model: Optional short model name for line 1 of the
            mean±std-band figure title (line 2 is the ablation subtitle).
        var_axis_label: Optional plot-axis symbol override (e.g. ``"w"`` on
            the x-axis while CSV columns still use ``var_label``).
    """
    name: str
    model_type: str
    var_label: str
    var_long_label: str
    var_attr: str
    default_output_subdir: str
    file_prefix: str
    title_prefix: str
    registry: OrderedDict = field(repr=False)
    band_var_values: Optional[tuple] = None
    default_band_sigmas: float = 1.0
    band_title_model: Optional[str] = None
    var_axis_label: Optional[str] = None


# Dispatch table: ``--model-type`` selects an entry here. The string is kept
# as ``--model-type`` rather than e.g. ``--ablation`` to avoid breaking any
# existing CLI invocations; the choices are descriptive enough.
ABLATIONS = OrderedDict([
    ("flow_matching_history", AblationConfig(
        name="flow_matching_history",
        model_type="flow_matching_history",
        var_label="S",
        var_long_label="stride",
        var_attr="history_stride",
        # Backward-compat default: keep the legacy folder, no model subdir.
        default_output_subdir="metrics_history_stride_ablation",
        file_prefix="history_stride_ablation",
        title_prefix="History-Stride Ablation",
        registry=STRIDE_REGISTRY,
        default_band_sigmas=3.0,
        band_title_model="HistoryFM",
    )),
    ("flow_matching_ar_bootstrap", AblationConfig(
        name="flow_matching_ar_bootstrap",
        model_type="flow_matching_ar_bootstrap",
        var_label="S",
        var_long_label="stride",
        var_attr="history_stride",
        # AR-bootstrap stride ablation lives under its own model subdir so
        # it can't collide with the history-model stride ablation results.
        default_output_subdir="metrics_history_stride_ablation/flow_matching_ar_bootstrap",
        file_prefix="history_stride_ablation",
        title_prefix="History-Stride Ablation",
        registry=AR_BOOTSTRAP_STRIDE_REGISTRY,
        default_band_sigmas=3.0,
    )),
    ("flow_matching_ar_bootstrap_history_length", AblationConfig(
        name="flow_matching_ar_bootstrap_history_length",
        model_type="flow_matching_ar_bootstrap",
        var_label="H",
        var_long_label="length",
        var_attr="history_length",
        # History-LENGTH ablation gets its own top-level folder.
        default_output_subdir="metrics_history_length_ablation",
        file_prefix="history_length_ablation",
        title_prefix="History-Length Ablation",
        registry=AR_BOOTSTRAP_HISTORY_LENGTH_REGISTRY,
        band_var_values=(1, 4, 16, 64),
        default_band_sigmas=3.0,
        band_title_model="HB-ARFM",
        var_axis_label="w",
    )),
])
ABLATION_NAMES = list(ABLATIONS.keys())

# Legacy aliases kept so external code that imports these names still works.
MODEL_REGISTRIES = OrderedDict(
    (name, cfg.registry) for name, cfg in ABLATIONS.items()
)
MODEL_TYPES = ABLATION_NAMES


def resolve_checkpoint(path_or_dir: str) -> Optional[str]:
    """Resolve a checkpoint specification to an actual .ckpt file path.

    - If ``path_or_dir`` ends with .ckpt and exists, return it as-is.
    - Otherwise, look inside ``path_or_dir/checkpoints/`` (or ``path_or_dir``
      itself if it ends with ``checkpoints``) for ``last.ckpt`` first, then
      fall back to the highest ``epoch=*.ckpt``.

    Returns None if nothing usable is found.
    """
    if path_or_dir is None:
        return None
    p = path_or_dir.rstrip('/')
    if p.endswith('.ckpt') and os.path.exists(p):
        return p
    # Allow user to pass either the run dir or the run dir's checkpoints subdir.
    if os.path.basename(p) == 'checkpoints':
        ckpt_dir = p
    else:
        ckpt_dir = os.path.join(p, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return None

    last = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.exists(last):
        return last

    epoch_ckpts = [f for f in os.listdir(ckpt_dir)
                   if f.startswith('epoch=') and f.endswith('.ckpt')]
    if not epoch_ckpts:
        return None

    def _epoch_num(fname):
        try:
            return int(fname.split('epoch=')[1].split('-')[0])
        except Exception:
            return -1

    epoch_ckpts.sort(key=_epoch_num)
    return os.path.join(ckpt_dir, epoch_ckpts[-1])


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="History-stride ablation for Task-2 flow matching models "
                    "(flow_matching_history, flow_matching_ar_bootstrap)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model-type', type=str, default='flow_matching_history',
                        choices=MODEL_TYPES,
                        help='Which model family to ablate over. The corresponding '
                             'registry (one checkpoint per stride) is selected from '
                             'MODEL_REGISTRIES. Pass "flow_matching_ar_bootstrap" to '
                             'run the same stride ablation against the AR-bootstrap '
                             'checkpoints (W=10, rollout=5).')
    parser.add_argument('--data-file', type=str,
                        default='/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5',
                        help='Path to HDF5 validation data file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for CSV/plots. If omitted, defaults '
                             'to "./ICML/metrics_history_stride_ablation/random/<N>" '
                             'for flow_matching_history (backward-compat) and to '
                             '"./ICML/metrics_history_stride_ablation/<model_type>/'
                             'random/<N>" for any other model type so concurrent '
                             'runs don\'t overwrite each other.')
    parser.add_argument('--start-time', type=int, default=800,
                        help='Starting timestep in HDF5 file')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Spatial downsample factor (4 = 128x128)')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='ODE integration steps')
    parser.add_argument('--solver', type=str, default='heun',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver for flow matching')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Starting frame index for evaluation')
    parser.add_argument('--frame-end', type=int, default=5,
                        help='Ending frame index (exclusive)')
    parser.add_argument('--normalization-stats', type=str,
                        default='/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json',
                        help='Path to shared normalization_stats.json')
    parser.add_argument('--seed', type=int, default=32,
                        help='Random seed for reproducibility. With --num-seeds=1 '
                             'this controls both index sampling and model noise; '
                             'with --num-seeds>1 it controls index sampling only '
                             '(model-noise seeds are derived from --seed and '
                             '--seed-step).')
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='If >1, run inference with this many random seeds per '
                             'stride to measure the noise floor of the stochastic '
                             'flow-matching sampler. For every seed the SAME '
                             '--num-samples random snapshots are evaluated (sample '
                             'indices stay fixed; only the torch noise differs), so '
                             'the resulting std isolates model stochasticity. The '
                             'mean across seeds is written to the usual CSV; the std '
                             'and the raw per-seed values are written to extra CSVs '
                             'and a std-trend plot.')
    parser.add_argument('--seed-step', type=int, default=1,
                        help='Increment between consecutive noise seeds, i.e. '
                             'seeds_used = [seed, seed+step, seed+2*step, ...].')
    parser.add_argument('--norm-mode', type=str, default='all',
                        choices=['none', 'all', 'temperature_only'],
                        help='Normalization mode (must match training)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Random sampling mode: evaluate on N randomly chosen timesteps. '
                             'Overrides --frame-start/--frame-end.')
    parser.add_argument('--strides', type=int, nargs='+', default=None,
                        help='Subset of history strides to evaluate. Defaults to '
                             'every stride registered for the chosen --model-type '
                             '(use --help after picking --model-type for the exact '
                             'list).')
    parser.add_argument('--fourier', action='store_true', default=False,
                        help='Include 6 Fourier spectral band metrics (low/mid/high for vel & temp)')
    parser.add_argument('--plot-spectra', action='store_true', default=False,
                        help='Generate energy spectrum E(k) plots comparing all strides')
    parser.add_argument('--save-hdf5', action='store_true', default=False,
                        help='Save all inference results (GT + predictions + SDF) to HDF5')
    parser.add_argument('--band-metrics', type=str, nargs='+',
                        default=['Temp. Relative L2', 'Vel. Relative L2'],
                        help='Metric names (case-sensitive, exactly as written in the '
                             'CSV) to plot in the mean + std-band figure. The first '
                             'metric is drawn on the left y-axis, the second on a twin '
                             'right y-axis, so two metrics on very different scales can '
                             'share one plot. Only used when --num-seeds > 1.')
    parser.add_argument('--band-var-values', type=int, nargs='+', default=None,
                        help='Subset of ablation variable values (H or S) for the '
                             'mean±std-band plot only. Defaults to the active '
                             'AblationConfig (history-length ablation: 1 4 16 64).')
    parser.add_argument('--band-sigmas', type=float, default=None,
                        help='Width of the shaded std band in units of seed-std '
                             '(e.g. 1.0 = mean±1σ, 3.0 = mean±3σ). Defaults to '
                             'the active AblationConfig (history-length: 3.0).')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Look up the full ablation config (registry + labels + paths). The CLI
    # exposes this as ``--model-type``, but internally we work through the
    # AblationConfig — ``model_type`` is the *architecture* (used by the
    # load/inference dispatchers) and may differ from ``args.model_type``
    # (which is the ablation key, e.g. ``flow_matching_ar_bootstrap_history_length``).
    if args.model_type not in ABLATIONS:
        raise ValueError(f"Unknown --model-type {args.model_type!r}. "
                         f"Available: {ABLATION_NAMES}")
    ablation = ABLATIONS[args.model_type]
    model_type = ablation.model_type
    active_registry = ablation.registry

    # Resolve the default output directory from the active AblationConfig.
    # Each ablation declares its own ``default_output_subdir`` so concurrent
    # ablations land in different folders.
    if args.output_dir is None:
        n_for_dir = args.num_samples if args.num_samples is not None else 'seq'
        args.output_dir = (f'./ICML/{ablation.default_output_subdir}/'
                           f'random/{n_for_dir}')

    # Ablation-variable subset selection — defaults to every value in the
    # active registry. Unknown values are warned-about but not fatal so the
    # same CLI invocation can be reused across model types / ablations with
    # different available values.
    var_label = ablation.var_label
    var_long_label = ablation.var_long_label
    title_prefix = ablation.title_prefix
    file_prefix = ablation.file_prefix
    if args.strides is not None:
        strides_to_run = OrderedDict(
            (s, active_registry[s]) for s in args.strides if s in active_registry
        )
        for s in args.strides:
            if s not in active_registry:
                print(f"WARNING: {ablation.var_attr}={s} not registered for "
                      f"--model-type={args.model_type}, skipping. "
                      f"Valid: {list(active_registry.keys())}")
    else:
        strides_to_run = active_registry

    random_mode = args.num_samples is not None

    # Build the list of noise seeds used for the stochastic flow-matching
    # sampler. Sample indices are always drawn with `args.seed` (so each
    # ablation point evaluates on the same snapshots across seeds); only
    # the torch RNG state varies, isolating model stochasticity.
    if args.num_seeds < 1:
        raise ValueError(f"--num-seeds must be >= 1, got {args.num_seeds}")
    seeds_to_use = [args.seed + i * args.seed_step for i in range(args.num_seeds)]
    multi_seed = len(seeds_to_use) > 1

    pretty_model_type = (
        "Flow Matching History" if model_type == 'flow_matching_history'
        else "Flow Matching AR Bootstrap" if model_type == 'flow_matching_ar_bootstrap'
        else model_type
    )
    print("=" * 80)
    print(f"  {title_prefix} — {pretty_model_type} (Task 2)")
    print("=" * 80)
    print(f"  Ablation:          {ablation.name}")
    print(f"  Model type:        {model_type}")
    print(f"  Sweep variable:    {ablation.var_attr} ({var_label})")
    print(f"  Data file:         {args.data_file}")
    print(f"  Start time:        {args.start_time}")
    if random_mode:
        sample_unit = ('segments' if model_type == 'flow_matching_ar_bootstrap'
                       else 'timesteps')
        print(f"  Sampling mode:     RANDOM ({args.num_samples} {sample_unit})")
    else:
        print(f"  Frame range:       [{args.frame_start}, {args.frame_end})")
    print(f"  Downsample:        {args.downsample_factor}x")
    print(f"  Inference steps:   {args.num_inference_steps}")
    print(f"  Solver:            {args.solver}")
    print(f"  Seed:              {args.seed}")
    if multi_seed:
        print(f"  Seed ablation:     {args.num_seeds} seeds "
              f"(step={args.seed_step}) -> {seeds_to_use}")
    print(f"  {var_long_label.capitalize()}s ({len(strides_to_run)}): "
          f"      {list(strides_to_run.keys())}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Save HDF5:         {args.save_hdf5}")
    print("=" * 80)

    # Load task config once
    task_cfg = load_task_config('velocity_from_interface')

    # Load ground truth SDF, interface velocity, and heater temperature once
    (sdf_gt_full, _, _, _,
     velx_intf_full, vely_intf_full, heater_temp) = load_ground_truth_data(
        args.data_file, args.start_time, args.downsample_factor
    )

    # Load raw mass flux scalar field directly from source HDF5
    massflux_full = None
    with h5py.File(args.data_file, 'r') as _f:
        if 'massflux' in _f:
            _mf_raw = _f['massflux'][args.start_time:]
            if args.downsample_factor > 1:
                from scipy.ndimage import zoom
                scale = 1.0 / args.downsample_factor
                massflux_full = zoom(_mf_raw, (1, scale, scale), order=1)
            else:
                massflux_full = _mf_raw
            print(f"  Loaded raw mass flux: shape={massflux_full.shape}")
        else:
            print(f"  Warning: no 'massflux' field in source HDF5")

    optim_cfg = DictConfig({'name': 'adamw', 'lr': 0.001})
    scheduler_cfg = DictConfig({'name': 'cosine'})

    all_results = OrderedDict()        # mean across seeds (or single value)
    all_results_std = OrderedDict()    # std across seeds (multi-seed mode)
    all_results_per_seed = OrderedDict()  # raw per-seed metric dicts
    inference_cache = OrderedDict()

    spectra_data = {}
    gt_spectra = None

    for run_idx, (var_value, raw_ckpt) in enumerate(strides_to_run.items()):
        # ``var_value`` is the registry key — interpretation depends on the
        # active ablation: stride for the ``_stride_`` registries, history
        # length for the ``_history_length`` registry.
        model_name = f"{var_label}={var_value}"

        checkpoint_path = resolve_checkpoint(raw_ckpt)

        print(f"\n{'='*80}")
        print(f"  [{run_idx + 1}/{len(strides_to_run)}] "
              f"{ablation.var_attr}={var_value}  ({model_type})")
        print(f"  Provided:   {raw_ckpt}")
        print(f"  Resolved:   {checkpoint_path}")
        print(f"{'='*80}")

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"  SKIPPING — checkpoint not found")
            nan_dict = OrderedDict(
                (k, np.nan) for k in _get_metric_names(args.fourier)
            )
            all_results[model_name] = nan_dict
            if multi_seed:
                all_results_std[model_name] = OrderedDict(nan_dict)
                all_results_per_seed[model_name] = []
            continue

        try:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

            norm_stats = load_normalization_stats(checkpoint_path, args.normalization_stats)

            model = _load_model_for_type(
                model_type, checkpoint_path, task_cfg,
                optim_cfg, scheduler_cfg, norm_stats, args,
            )

            # Report what we actually got — shape attributes differ between
            # model families, so log whichever ones are present.
            history_window = int(getattr(model, 'history_window',
                                          getattr(model, 'history_length', 10)))
            history_stride = int(getattr(model, 'history_stride', 1))
            history_length = int(getattr(model, 'history_length', history_window))
            rollout_length = int(getattr(model, 'rollout_length', 1))
            # Validate the swept dimension matches the registry key.
            checkpoint_var = int(getattr(model, ablation.var_attr, -1))
            if checkpoint_var != var_value:
                print(f"  WARNING: checkpoint reports {ablation.var_attr}="
                      f"{checkpoint_var} but registry expected {var_value}; "
                      f"using checkpoint value.")
            if model_type == 'flow_matching_ar_bootstrap':
                print(f"  Loaded model: H={history_length}, S={history_stride}, "
                      f"L={rollout_length}, "
                      f"span={(history_length - 1) * history_stride + 1} raw frames")
            else:
                print(f"  Loaded model: W={history_window}, S={history_stride}, "
                      f"span={(history_window - 1) * history_stride + 1} raw frames")

            dataset = _load_dataset_for_type(model_type, model, args, norm_stats)

            total_available = len(dataset)
            T_per_sample = _frames_per_sample(model_type, dataset)

            # ----------------------------------------------------------------
            # Step 1: choose the snapshots ONCE for this stride. Sample
            # indices are drawn with `args.seed`, so they are identical
            # across noise seeds (and across strides whenever
            # `total_available` is unchanged) — this isolates model
            # stochasticity from data-sampling variance.
            #
            # For ``flow_matching_ar_bootstrap`` each dataset access yields
            # ``T_per_sample = rollout_length`` consecutive frames, so the
            # SDF/interface slices grab that many frames per sampled index
            # to stay aligned with the GT/pred stacks below.
            # ----------------------------------------------------------------
            if random_mode:
                n_samples = min(args.num_samples, total_available)
                random.seed(args.seed)
                sample_indices = sorted(
                    random.sample(range(total_available), n_samples)
                )
                sdf_slice = np.concatenate(
                    [sdf_gt_full[i:i + T_per_sample] for i in sample_indices]
                )
                vxi_slice = np.concatenate(
                    [velx_intf_full[i:i + T_per_sample] for i in sample_indices]
                )
                vyi_slice = np.concatenate(
                    [vely_intf_full[i:i + T_per_sample] for i in sample_indices]
                )
                if massflux_full is not None:
                    mf_slice = np.concatenate(
                        [massflux_full[i:i + T_per_sample] for i in sample_indices]
                    )
                else:
                    mf_slice = None
                sample_unit = ('segments' if model_type == 'flow_matching_ar_bootstrap'
                               else 'timesteps')
                total_frames = n_samples * T_per_sample
                print(f"\n  Random sampling: {n_samples} {sample_unit} "
                      f"from {total_available} available "
                      f"({total_frames} frames at T_per_sample={T_per_sample})")
            else:
                frame_start = args.frame_start
                frame_end_clamped = min(args.frame_end, total_available)
                sample_indices = list(range(frame_start, frame_end_clamped))
                # Use the same per-sample slicing as the random branch — for
                # T_per_sample == 1 this is equivalent to a single contiguous
                # slice; for T_per_sample > 1 it gathers the right rollout
                # frames for each segment (segments may overlap, in which
                # case the same SDF frame appears multiple times, matching
                # the GT/pred stacks produced by the rollouts).
                sdf_slice = np.concatenate(
                    [sdf_gt_full[i:i + T_per_sample] for i in sample_indices]
                )
                vxi_slice = np.concatenate(
                    [velx_intf_full[i:i + T_per_sample] for i in sample_indices]
                )
                vyi_slice = np.concatenate(
                    [vely_intf_full[i:i + T_per_sample] for i in sample_indices]
                )
                if massflux_full is not None:
                    mf_slice = np.concatenate(
                        [massflux_full[i:i + T_per_sample] for i in sample_indices]
                    )
                else:
                    mf_slice = None

            # ----------------------------------------------------------------
            # Step 2: per-seed inference loop. Only torch RNG state is varied
            # between seeds; numpy/random remain pegged to `args.seed` so the
            # dataset access is fully deterministic.
            # ----------------------------------------------------------------
            per_seed_metric_dicts = []
            first_seed_arrays = None  # cached for HDF5 / spectra below
            for seed_i, noise_seed in enumerate(seeds_to_use):
                torch.manual_seed(noise_seed)
                torch.cuda.manual_seed_all(noise_seed)

                gt_vx_list, gt_vy_list, gt_t_list = [], [], []
                pr_vx_list, pr_vy_list, pr_t_list = [], [], []

                desc = f"  {model_name}"
                if multi_seed:
                    desc += f" seed[{seed_i + 1}/{len(seeds_to_use)}]={noise_seed}"

                for idx in tqdm(sample_indices, desc=desc):
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        # The dispatcher returns 3D arrays of shape (T, H, W)
                        # with T = 1 for single-step models and
                        # T = rollout_length for AR-bootstrap models.
                        gvx, gvy, gt_, pvx, pvy, pt = run_inference_sample(
                            model_type, model, dataset, idx, device,
                            args.num_inference_steps, args.solver,
                        )
                    gt_vx_list.append(gvx)
                    gt_vy_list.append(gvy)
                    gt_t_list.append(gt_)
                    pr_vx_list.append(pvx)
                    pr_vy_list.append(pvy)
                    pr_t_list.append(pt)

                gt_velx = np.concatenate(gt_vx_list)
                gt_vely = np.concatenate(gt_vy_list)
                gt_temp = np.concatenate(gt_t_list)
                pred_velx = np.concatenate(pr_vx_list)
                pred_vely = np.concatenate(pr_vy_list)
                pred_temp = np.concatenate(pr_t_list)

                metrics_this_seed = compute_task2_metrics(
                    gt_velx, gt_vely, gt_temp,
                    pred_velx, pred_vely, pred_temp,
                    sdf_slice, heater_temp,
                    massflux=mf_slice,
                    downsample_factor=args.downsample_factor,
                    include_fourier=args.fourier,
                )
                per_seed_metric_dicts.append(metrics_this_seed)

                if seed_i == 0:
                    first_seed_arrays = (gt_velx, gt_vely, gt_temp,
                                         pred_velx, pred_vely, pred_temp)

                if multi_seed:
                    # Free large arrays before next seed (only keep first
                    # seed's outputs for HDF5 / spectra).
                    if seed_i > 0:
                        del gt_velx, gt_vely, gt_temp
                        del pred_velx, pred_vely, pred_temp

            # ----------------------------------------------------------------
            # Step 3: aggregate per-seed metrics into mean (always) and std
            # (multi-seed only). NaN-safe to tolerate occasional bad runs.
            # ----------------------------------------------------------------
            metric_keys = list(per_seed_metric_dicts[0].keys())
            if multi_seed:
                mean_dict = OrderedDict()
                std_dict = OrderedDict()
                for k in metric_keys:
                    vals = np.array([m[k] for m in per_seed_metric_dicts],
                                    dtype=float)
                    if np.sum(~np.isnan(vals)) > 1:
                        mean_dict[k] = float(np.nanmean(vals))
                        std_dict[k] = float(np.nanstd(vals, ddof=1))
                    elif np.sum(~np.isnan(vals)) == 1:
                        mean_dict[k] = float(np.nanmean(vals))
                        std_dict[k] = 0.0
                    else:
                        mean_dict[k] = np.nan
                        std_dict[k] = np.nan
                all_results[model_name] = mean_dict
                all_results_std[model_name] = std_dict
                all_results_per_seed[model_name] = per_seed_metric_dicts
            else:
                all_results[model_name] = per_seed_metric_dicts[0]

            # Restore arrays referenced below from the first seed.
            gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp = first_seed_arrays

            if args.save_hdf5:
                cache_entry = {
                    'model_type': model_type,
                    'checkpoint': checkpoint_path,
                    'gt_velx': gt_velx,
                    'gt_vely': gt_vely,
                    'gt_temp': gt_temp,
                    'pred_velx': pred_velx,
                    'pred_vely': pred_vely,
                    'pred_temp': pred_temp,
                    'sdf': sdf_slice,
                    'velx_interface': vxi_slice,
                    'vely_interface': vyi_slice,
                }
                if mf_slice is not None:
                    cache_entry['massflux'] = mf_slice
                inference_cache[model_name] = cache_entry

            if args.plot_spectra:
                pred_vel_spec = (_radial_energy_spectrum(pred_velx)
                                + _radial_energy_spectrum(pred_vely))
                pred_temp_spec = _radial_energy_spectrum(pred_temp)
                spectra_data[model_name] = {
                    "temp": pred_temp_spec,
                    "vel": pred_vel_spec,
                }
                if gt_spectra is None:
                    gt_vel_spec = (_radial_energy_spectrum(gt_velx)
                                  + _radial_energy_spectrum(gt_vely))
                    gt_temp_spec = _radial_energy_spectrum(gt_temp)
                    gt_spectra = {"temp": gt_temp_spec, "vel": gt_vel_spec}

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            nan_dict = OrderedDict(
                (k, np.nan) for k in _get_metric_names(args.fourier)
            )
            all_results[model_name] = nan_dict
            if multi_seed:
                all_results_std[model_name] = OrderedDict(nan_dict)
                all_results_per_seed[model_name] = []

    # ------------------------------------------------------------------
    # Save HDF5 (optional)
    # ------------------------------------------------------------------
    if args.save_hdf5 and inference_cache:
        # Filename pattern: ``<stride|length>_inference_results.hdf5``
        # — derived from the ablation prefix so AR-bootstrap history-length
        # runs don't overwrite the history-stride HDF5.
        hdf5_basename = ablation.file_prefix.replace('_ablation',
                                                      '_inference_results')
        hdf5_path = os.path.join(args.output_dir, f'{hdf5_basename}.hdf5')
        print(f"\n{'='*80}")
        print("  Saving inference results to HDF5")
        print(f"{'='*80}")
        save_inference_hdf5(hdf5_path, inference_cache, heater_temp, args)
        del inference_cache

    # ------------------------------------------------------------------
    # Energy spectrum plots (optional)
    # ------------------------------------------------------------------
    if args.plot_spectra and gt_spectra is not None and spectra_data:
        print(f"\n{'='*80}")
        print("  Generating energy spectrum plots")
        print(f"{'='*80}")
        spectra_dir = os.path.join(args.output_dir, 'fourier')
        plot_energy_spectra(spectra_data, gt_spectra, spectra_dir)

    # ------------------------------------------------------------------
    # Comparison CSV (rows = metrics, columns = ablation values). In
    # multi-seed mode this holds the mean across seeds; the std and the raw
    # per-seed values are written to additional CSVs and plots below.
    # All filenames are derived from ``ablation.file_prefix`` so the
    # history-length ablation lands under different names than the stride
    # ablation and the two can sit in the same folder without colliding.
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  Assembling comparison table")
    print(f"{'='*80}")

    df = pd.DataFrame(all_results)
    df.index.name = 'Metric'

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'{file_prefix}.csv')
    df.to_csv(csv_path)

    csv_path_T = os.path.join(args.output_dir, f'{file_prefix}_transposed.csv')
    df.T.to_csv(csv_path_T)

    print("\n" + df.to_string())
    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {csv_path_T}")

    df_std = None
    if multi_seed and all_results_std:
        df_std = pd.DataFrame(all_results_std)
        df_std.index.name = 'Metric'

        std_csv_path = os.path.join(args.output_dir,
                                    f'{file_prefix}_std.csv')
        df_std.to_csv(std_csv_path)
        std_csv_path_T = os.path.join(args.output_dir,
                                      f'{file_prefix}_std_transposed.csv')
        df_std.T.to_csv(std_csv_path_T)
        print(f"\n  Std across {len(seeds_to_use)} seeds:")
        print(df_std.to_string())
        print(f"  Saved: {std_csv_path}")
        print(f"  Saved: {std_csv_path_T}")

        # Long-format per-seed CSV (one row per (ablation-value, seed,
        # metric)) — this is the rawest possible record and lets downstream
        # notebooks recompute any statistic of interest. The first column is
        # named after the active ablation's ``var_attr`` (``history_stride``
        # vs. ``history_length``) so the file is self-describing.
        long_rows = []
        for model_name, seed_dicts in all_results_per_seed.items():
            for seed_i, m in enumerate(seed_dicts):
                noise_seed = seeds_to_use[seed_i] if seed_i < len(seeds_to_use) else seed_i
                for metric_name, value in m.items():
                    long_rows.append({
                        ablation.var_attr: model_name,
                        'seed_index': seed_i,
                        'seed': noise_seed,
                        'metric': metric_name,
                        'value': value,
                    })
        if long_rows:
            df_long = pd.DataFrame(long_rows)
            per_seed_csv = os.path.join(args.output_dir,
                                        f'{file_prefix}_per_seed.csv')
            df_long.to_csv(per_seed_csv, index=False)
            print(f"  Saved: {per_seed_csv}")

        # Combined mean+std summary table. Two representations:
        #   * A numerical CSV with a 2-level column header
        #     ``(<var>, statistic)`` where statistic in {'mean','std'}
        #     — convenient for further analysis with pandas.
        #   * A "mean ± std" formatted CSV — convenient as a paper-ready
        #     table that can be pasted into LaTeX/Markdown directly.
        sorted_stride_cols = sorted(df.columns, key=_parse_stride_from_column)
        df_combined = pd.concat(
            {s: pd.DataFrame({'mean': df[s], 'std': df_std[s]})
             for s in sorted_stride_cols},
            axis=1,
        )
        df_combined.index.name = 'Metric'
        combined_csv = os.path.join(args.output_dir,
                                    f'{file_prefix}_mean_std.csv')
        df_combined.to_csv(combined_csv)
        print(f"  Saved: {combined_csv}")

        df_fmt = pd.DataFrame(index=df.index, columns=sorted_stride_cols,
                              dtype=object)
        for stride_col in sorted_stride_cols:
            for metric in df.index:
                m = df.at[metric, stride_col]
                s = df_std.at[metric, stride_col]
                if np.isnan(m):
                    df_fmt.at[metric, stride_col] = 'nan'
                elif np.isnan(s) or s == 0.0:
                    df_fmt.at[metric, stride_col] = f'{m:.4g}'
                else:
                    df_fmt.at[metric, stride_col] = f'{m:.4g} ± {s:.2g}'
        df_fmt.index.name = 'Metric'
        fmt_csv = os.path.join(args.output_dir,
                               f'{file_prefix}_mean_std_formatted.csv')
        df_fmt.to_csv(fmt_csv)
        print(f"  Saved: {fmt_csv}")

    # ------------------------------------------------------------------
    # Bundle the ablation-specific axis/title labels so we don't have to
    # spell them out at every plot call below.
    # ------------------------------------------------------------------
    plot_labels = dict(
        var_label=var_label,
        var_long_label=var_long_label,
        title_prefix=title_prefix,
        title_model=ablation.band_title_model,
        var_axis_label=ablation.var_axis_label,
    )

    # ------------------------------------------------------------------
    # Metric-trend plot: one figure with one line per metric, normalised
    # so all 15+ curves share a comparable y-scale. We also save two
    # split versions (velocity / thermal) which are less crowded.
    # ------------------------------------------------------------------
    if df.shape[1] >= 2:
        trend_path = plot_metric_trends(
            df, args.output_dir,
            filename=f'{file_prefix}_trend.png',
            **plot_labels,
        )
        print(f"  Saved metric trend plot: {trend_path}")
        split_paths = plot_metric_trends_split(
            df, args.output_dir,
            filename_velocity=f'{file_prefix}_trend_velocity.png',
            filename_thermal=f'{file_prefix}_trend_thermal.png',
            **plot_labels,
        )
        for key, p in split_paths.items():
            print(f"  Saved metric trend plot ({key}): {p}")
    else:
        print(f"  Skipping trend plot (need at least 2 "
              f"{var_long_label} configurations).")

    # ------------------------------------------------------------------
    # Std-trend plots (multi-seed mode only): combined plus split.
    # ------------------------------------------------------------------
    if multi_seed and df_std is not None and df_std.shape[1] >= 2:
        std_plot_paths = plot_metric_std(
            df_std, args.output_dir, df_mean=df,
            filename_norm=f'{file_prefix}_std_norm.png',
            filename_raw=f'{file_prefix}_std_raw.png',
            filename_cv=f'{file_prefix}_cv.png',
            **plot_labels,
        )
        for key, p in std_plot_paths.items():
            print(f"  Saved std-trend plot ({key}): {p}")
        split_std_paths = plot_metric_std_split(
            df_std, args.output_dir,
            filename_velocity=f'{file_prefix}_std_norm_velocity.png',
            filename_thermal=f'{file_prefix}_std_norm_thermal.png',
            **plot_labels,
        )
        for key, p in split_std_paths.items():
            print(f"  Saved std-trend plot (norm/{key}): {p}")

        # Mean + std-band trend for the user-selected metrics
        # (defaults: Temp. Relative L2 and Vel. Relative L2 on twin-y).
        band_var_values = (tuple(args.band_var_values)
                           if args.band_var_values is not None
                           else ablation.band_var_values)
        band_sigmas = (args.band_sigmas if args.band_sigmas is not None
                       else ablation.default_band_sigmas)
        band_path = plot_mean_std_band(
            df_mean=df,
            df_std=df_std,
            metrics=args.band_metrics,
            output_dir=args.output_dir,
            filename=f'{file_prefix}_mean_std_band.png',
            band_sigmas=band_sigmas,
            n_seeds=len(seeds_to_use),
            var_values=band_var_values,
            **plot_labels,
        )
        if band_path is not None:
            print(f"  Saved mean+std-band plot: {band_path}")
    elif multi_seed:
        print(f"  Skipping std-trend plot (need at least 2 "
              f"{var_long_label} configurations).")

    n_ok = len([r for r in all_results.values()
                if not all(np.isnan(v) for v in r.values())])
    print(f"\n  Done! Evaluated {n_ok} / {len(strides_to_run)} "
          f"{ablation.var_attr} configurations successfully.")
    if multi_seed:
        print(f"  Seed ablation: {len(seeds_to_use)} seeds per "
              f"{var_long_label} ({seeds_to_use[0]} .. {seeds_to_use[-1]}).")


if __name__ == "__main__":
    main()
