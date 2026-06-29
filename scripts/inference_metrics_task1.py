#!/usr/bin/env python3
"""
Multi-Model Inference Metrics for Task 1 (temperature_from_sdf).

Evaluates temperature-only models on the same validation data and produces a
single CSV comparison table where rows = physics metrics, columns = models.

Region definitions
------------------
  interface : cells with |massflux| > 0   (raw scalar from the solver — the
              physically correct phase-change cells; loaded directly from the
              source HDF5 for metric purposes even though it is not a model
              input on task-1).  Falls back to the SDF zero-crossing only
              when the dataset has no massflux field.
  bulk      : sdf < 0 (liquid) AND not interface AND row index >=
              `near_wall_rows_full / downsample_factor` (excludes heater wall).

Domain conventions
------------------
  x in [-8, 8] mm, y in [0, 16] mm.  Heater span x in [-5.25, 5.25] mm.
  Grid spacing dx, dy are derived from the array shape so the metrics are
  correct at any downsample factor.

Default metrics (7):

  Temperature (7):
    1.  Temp. Relative L2        -- ||T_pred - T_gt||_2 / ||T_gt||_2
    2.  Temp. Max Rel L2         -- worst single-frame relative L2
    3.  Temp. Max Error          -- L-inf of temperature error
    4.  Temp. IRMSE              -- RMSE at interface cells (massflux-based;
                                    matches task-2.  Falls back to SDF
                                    zero-crossing only if the source HDF5
                                    has no massflux field.)
    5.  Temp. BRMSE              -- RMSE in bulk liquid cells
    6.  Temp. HF Energy Ratio    -- ideal = 1.0
          Formula (per frame t):
            F(t)       = FFT2D( field(t) )
            P(kx,ky)   = |F(kx,ky)|^2                     (power spectrum)
            k_r        = sqrt(kx^2 + ky^2)                (radial wavenumber)
            HF_frac(t) = sum P where k_r >= 12  /  sum P  (HF energy fraction)
          Then:
            HF Energy Ratio = mean_t[ HF_frac_pred(t) ] / mean_t[ HF_frac_gt(t) ]
          Quantifies whether the model preserves fine-scale thermal structure.
    7.  Wall Heat Flux Rel. Error (%)
                                    -- 100 * ||hflux_pred - hflux_gt||_2 /
                                              ||hflux_gt||_2 over time series
                                    (proper relative L2 norm; heater span
                                     x in [-5.25, 5.25])

  Optional Fourier spectral error (--fourier, 3 extra):
    8.  Temp. Fourier Low/Mid/High

Usage:
    python scripts/inference_metrics_task1.py
    python scripts/inference_metrics_task1.py --num-samples 50
    python scripts/inference_metrics_task1.py --fourier
    python scripts/inference_metrics_task1.py --models flow_matching unet diffusionpde
"""

import sys
import os
import json
import math
import random
import argparse
from collections import OrderedDict
from typing import Dict, Tuple

import contextlib
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.physics_metrics_task123 import (
    load_task_config,
    load_model_from_checkpoint,
    load_dataset,
    load_ground_truth_data,
    run_inference_batch,
    run_ar_bootstrap_inference_batch,
    extract_channels,
)

# ============================================================================
# Metric helper functions (self-contained)
#
# Physical domain conventions (PoolBoiling-Subcooled-FC72-2D):
#   x in [-8, 8] mm, y in [0, 16] mm   (square 16x16 domain in mm units)
#   heater span: x in [-5.25, 5.25]    (matches your inference-time configuration)
#   liquid : sdf < 0,    vapor : sdf >= 0
#   interface cells : |massflux| > 0  (raw scalar from the solver)
# ============================================================================

# Domain constants (mm) — single source of truth so dx, heater range and
# any future divergence/vorticity grid spacings are all consistent.  Kept in
# lock-step with scripts/inference_metrics_task2.py.
DOMAIN_X_MIN = -8.0
DOMAIN_X_MAX = 8.0
DOMAIN_Y_MIN = 0.0
DOMAIN_Y_MAX = 16.0
HEATER_X_MIN = -5.25
HEATER_X_MAX = 5.25


def _safe_relative_l2(pred: np.ndarray, true: np.ndarray,
                       eps: float = 1e-12) -> float:
    """Relative L2 error: ||pred - true||_2 / max(||true||_2, eps)."""
    num = np.sqrt(np.sum((pred - true) ** 2, dtype=np.float64))
    den = np.sqrt(np.sum(true ** 2, dtype=np.float64))
    return float(num / max(den, eps))


def _grid_spacings(H: int, W: int) -> Tuple[float, float]:
    """Cell sizes (dy, dx) inferred from the physical domain and array shape.

    For the standard PB-Subcooled 16x16 mm domain at downsample factor 4
    (H=W=128) this gives dy=dx=0.125 mm, matching the previous hardcoded
    `downsample_factor/32`.  Computing from the domain makes the helpers
    correct at any downsample factor or grid shape.
    """
    dx = (DOMAIN_X_MAX - DOMAIN_X_MIN) / W
    dy = (DOMAIN_Y_MAX - DOMAIN_Y_MIN) / H
    return dy, dx


def _interface_mask_from_massflux(massflux: np.ndarray) -> np.ndarray:
    """Boolean mask of interface cells: |massflux| > 0.

    This is the physically correct definition — the solver writes a non-zero
    massflux exactly at cells undergoing phase change — and is more reliable
    than thresholding velocity magnitudes or scanning for SDF sign changes.
    """
    return np.abs(massflux) > 0.0


def _interface_mask_zero_crossing(sdf: np.ndarray) -> np.ndarray:
    """SDF zero-crossing fallback used when massflux is unavailable.

    Returns the 1-cell ring of cells whose immediate neighbour has the
    opposite SDF sign.  Less precise than the massflux-based mask but a
    reasonable proxy.  Task-1 typically uses this fallback because massflux
    is not part of the model inputs.
    """
    if sdf.ndim == 2:
        sx = (sdf[:, :-1] * sdf[:, 1:]) < 0
        sy = (sdf[:-1, :] * sdf[1:, :]) < 0
        mx = np.zeros_like(sdf, dtype=bool)
        mx[:, :-1] |= sx; mx[:, 1:] |= sx
        my = np.zeros_like(sdf, dtype=bool)
        my[:-1, :] |= sy; my[1:, :] |= sy
        return mx | my
    sx = (sdf[:, :, :-1] * sdf[:, :, 1:]) < 0
    sy = (sdf[:, :-1, :] * sdf[:, 1:, :]) < 0
    mx = np.zeros_like(sdf, dtype=bool)
    mx[:, :, :-1] |= sx; mx[:, :, 1:] |= sx
    my = np.zeros_like(sdf, dtype=bool)
    my[:, :-1, :] |= sy; my[:, 1:, :] |= sy
    return mx | my


def _heatflux(dfun: np.ndarray, temp: np.ndarray, heater_temp: float,
              lc: float = 0.73e-3, thcl: float = 6.25e-2) -> np.ndarray:
    """Wall heat flux time series (T,) for FC-72.

    The heater spans x in [-5.25, 5.25] mm.  Grid spacing is derived from the
    physical domain and the array shape (so the helper is correct at any
    downsample factor).
    """
    T_frames, H, W = dfun.shape
    dy, dx = _grid_spacings(H, W)
    x_centers = DOMAIN_X_MIN + (np.arange(W) + 0.5) * dx
    y_centers = DOMAIN_Y_MIN + (np.arange(H) + 0.5) * dy
    x_grid, _ = np.meshgrid(x_centers, y_centers)
    heater_mask = (x_grid >= HEATER_X_MIN) & (x_grid <= HEATER_X_MAX)
    heater_mask_3d = np.broadcast_to(heater_mask, (T_frames, H, W))
    liquid_mask = dfun < 0
    temp_fields = (heater_mask_3d & liquid_mask).astype(float) * (heater_temp - temp)
    hflux_fields = thcl * (temp_fields / (dx * 0.5 * lc))
    return hflux_fields[:, 0, :].mean(axis=1)


def _bulk_liquid_mask(sdf: np.ndarray, interface_mask: np.ndarray,
                      downsample_factor: int = 1,
                      near_wall_rows_full: int = 16) -> np.ndarray:
    """Bulk liquid: sdf < 0, not at interface, not within `near_wall_rows_full`
    of the heater wall (rows scaled by the downsample factor).

    The heater-wall exclusion avoids flattering BRMSE by including cells that
    are pinned by the Dirichlet boundary condition.
    """
    T, H, W = sdf.shape
    scaled_wall = near_wall_rows_full // downsample_factor
    y_coords = np.arange(H)
    y_grid = np.broadcast_to(y_coords[None, :, None], (T, H, W))
    return (sdf < 0) & (~interface_mask) & (y_grid >= scaled_wall)


def _fourier_error(pred: np.ndarray, gt: np.ndarray,
                   Lx: float = 16.0, Ly: float = 16.0) -> Tuple[float, float, float]:
    """Spectral error in Fourier space, split into low/mid/high frequency bands.

    Pool boiling domain: x in [-8, 8], y in [0, 16]  =>  Lx = Ly = 16.
    """
    ILOW = 4
    IHIGH = 12

    nb, nx, ny = pred.shape

    pred_F = np.fft.fftn(pred, axes=[1, 2])
    gt_F = np.fft.fftn(gt, axes=[1, 2])

    err_F_sq = np.abs(pred_F - gt_F) ** 2

    n_bins = min(nx // 2, ny // 2)
    err_radial = np.zeros((nb, n_bins))

    for i in range(nx // 2):
        for j in range(ny // 2):
            k = int(math.floor(math.sqrt(i * i + j * j)))
            if k > n_bins - 1:
                continue
            err_radial[:, k] += err_F_sq[:, i, j]

    err_spectrum = np.sqrt(np.mean(err_radial, axis=0)) / (nx * ny) * Lx * Ly

    IHIGH_clamped = min(IHIGH, len(err_spectrum))
    low_err = float(np.mean(err_spectrum[:ILOW])) if ILOW <= len(err_spectrum) else np.nan
    mid_err = float(np.mean(err_spectrum[ILOW:IHIGH_clamped])) if IHIGH_clamped > ILOW else np.nan
    high_err = float(np.mean(err_spectrum[IHIGH_clamped:])) if IHIGH_clamped < len(err_spectrum) else np.nan

    return low_err, mid_err, high_err


def _radial_energy_spectrum(field: np.ndarray) -> np.ndarray:
    """Radial energy spectrum E(k) averaged over frames.

    Args:
        field: (T, H, W) array.
    Returns:
        1D array of length min(H//2, W//2), the mean radial energy per bin.
    """
    nb, nx, ny = field.shape
    F = np.fft.fftn(field, axes=[1, 2])
    power = np.abs(F) ** 2
    n_bins = min(nx // 2, ny // 2)
    spectrum = np.zeros((nb, n_bins))
    for i in range(nx // 2):
        for j in range(ny // 2):
            k = int(math.floor(math.sqrt(i * i + j * j)))
            if k > n_bins - 1:
                continue
            spectrum[:, k] += power[:, i, j]
    return np.mean(spectrum, axis=0)


def _hf_energy_ratio_temperature(
    gt_temp: np.ndarray, pred_temp: np.ndarray,
    k_threshold: int = 12,
) -> float:
    """HF Energy Ratio for temperature.  Ideal = 1.0."""
    eps = 1e-30
    hf_frac_gt, hf_frac_pred = [], []
    nb, nx, ny = gt_temp.shape
    for t in range(nb):
        gt_F = np.fft.fftn(gt_temp[t])
        pred_F = np.fft.fftn(pred_temp[t])
        gt_pow = np.abs(gt_F) ** 2
        pred_pow = np.abs(pred_F) ** 2

        kx = np.arange(nx)
        ky = np.arange(ny)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        kr = np.sqrt(KX.astype(float)**2 + KY.astype(float)**2)
        hf_mask = kr >= k_threshold

        gt_total = gt_pow.sum()
        pred_total = pred_pow.sum()
        if gt_total < eps or pred_total < eps:
            continue
        hf_frac_gt.append(float(gt_pow[hf_mask].sum() / gt_total))
        hf_frac_pred.append(float(pred_pow[hf_mask].sum() / pred_total))

    if not hf_frac_gt:
        return np.nan
    return float(np.mean(hf_frac_pred) / (np.mean(hf_frac_gt) + eps))


# ============================================================================
# Core metrics function — temperature-only
# ============================================================================

METRIC_NAMES_BASE = [
    "Temp. Relative L2",
    "Temp. Max Rel L2",
    "Temp. Max Error",
    "Temp. IRMSE",
    "Temp. BRMSE",
    "Temp. HF Energy Ratio",
    "Wall Heat Flux Rel. Error (%)",
]

METRIC_NAMES_FOURIER = [
    "Temp. Fourier Low",
    "Temp. Fourier Mid",
    "Temp. Fourier High",
]


def _get_metric_names(include_fourier: bool = False):
    names = list(METRIC_NAMES_BASE)
    if include_fourier:
        names.extend(METRIC_NAMES_FOURIER)
    return names


def compute_task1_metrics(
    gt_temp: np.ndarray, pred_temp: np.ndarray,
    sdf: np.ndarray,
    heater_temp: float,
    massflux: np.ndarray = None,
    downsample_factor: int = 1,
    include_fourier: bool = False,
) -> OrderedDict:
    """Compute Task-1 (temperature-only) metrics.

    Kept in lock-step with :func:`scripts.inference_metrics_task2.compute_task2_metrics`
    for the temperature side: same heater span ([-5.25, 5.25]), same grid
    spacings (inferred from the physical domain), same wall-heat-flux relative
    L2 definition, and same massflux-based interface mask when provided.

    Args:
        gt_temp:   Ground truth temperature (T, H, W).
        pred_temp: Predicted temperature (T, H, W).
        sdf:       Signed distance function (T, H, W).
        heater_temp:        Heater temperature in Celsius.
        massflux: Raw scalar mass flux (T, H, W).  When provided, the interface
            mask is defined as `|massflux| > 0` (the physically correct
            phase-change cells).  When None, falls back to the SDF zero-
            crossing mask — this is the usual case for task-1.
        downsample_factor:  Spatial downsampling factor (used for the
            near-wall row count in the bulk-liquid mask).
        include_fourier:    If True, include 3 Fourier band metrics.

    Returns:
        OrderedDict keyed by metric names.
    """
    num_frames = gt_temp.shape[0]
    eps = 1e-12

    if massflux is not None:
        temp_interface_mask = _interface_mask_from_massflux(massflux)
    else:
        temp_interface_mask = _interface_mask_zero_crossing(sdf)
    temp_bulk_mask = _bulk_liquid_mask(sdf, temp_interface_mask, downsample_factor)

    temp_err = pred_temp - gt_temp

    temp_rel_l2 = float(np.sqrt(np.sum(temp_err**2)) /
                        (np.sqrt(np.sum(gt_temp**2)) + eps))

    temp_max_rel_l2 = 0.0
    for t in range(num_frames):
        denom = np.sqrt(np.sum(gt_temp[t]**2)) + eps
        frame_rel = float(np.sqrt(np.sum(temp_err[t]**2)) / denom)
        temp_max_rel_l2 = max(temp_max_rel_l2, frame_rel)

    temp_max_err = float(np.max(np.abs(temp_err)))

    ti_vals = temp_err[temp_interface_mask]
    temp_irmse = float(np.sqrt(np.mean(ti_vals**2))) if len(ti_vals) > 0 else np.nan

    tb_vals = temp_err[temp_bulk_mask]
    temp_brmse = float(np.sqrt(np.mean(tb_vals**2))) if len(tb_vals) > 0 else np.nan

    temp_hf_ratio = _hf_energy_ratio_temperature(gt_temp, pred_temp)

    # Wall heat flux relative L2 error (reported as a percentage so the
    # metric name and downstream CSV consumers stay backward compatible).
    gt_hflux = _heatflux(sdf, gt_temp, heater_temp)
    pred_hflux = _heatflux(sdf, pred_temp, heater_temp)
    hf_rel = 100.0 * _safe_relative_l2(pred_hflux, gt_hflux, eps=eps)

    result = OrderedDict([
        ("Temp. Relative L2",              temp_rel_l2),
        ("Temp. Max Rel L2",               temp_max_rel_l2),
        ("Temp. Max Error",                temp_max_err),
        ("Temp. IRMSE",                    temp_irmse),
        ("Temp. BRMSE",                    temp_brmse),
        ("Temp. HF Energy Ratio",          temp_hf_ratio),
        ("Wall Heat Flux Rel. Error (%)",  hf_rel),
    ])

    if include_fourier:
        temp_fourier_low, temp_fourier_mid, temp_fourier_high = _fourier_error(pred_temp, gt_temp)
        result["Temp. Fourier Low"] = temp_fourier_low
        result["Temp. Fourier Mid"] = temp_fourier_mid
        result["Temp. Fourier High"] = temp_fourier_high

    return result


# ============================================================================
# Energy Spectrum Plotting (temperature only)
# ============================================================================

MODEL_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]

ILOW_BAND = 4
IHIGH_BAND = 12
SPECTRUM_K_AXIS_MAX = 1000


def plot_energy_spectra(
    spectra_data: Dict[str, np.ndarray],
    gt_spectrum: np.ndarray,
    output_dir: str,
) -> None:
    """Plot radial energy spectrum E(k) for GT and all models (temperature)."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    k_vals = np.arange(1, len(gt_spectrum) + 1)
    ax.loglog(k_vals, gt_spectrum, color='black', linewidth=2.5, label='Ground Truth')

    for idx, (model_name, spec) in enumerate(spectra_data.items()):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.loglog(k_vals, spec, color=color, linewidth=1.5, label=model_name)

    if ILOW_BAND < len(gt_spectrum):
        ax.axvline(x=ILOW_BAND, color='grey', linestyle='--', linewidth=0.8)
    if IHIGH_BAND < len(gt_spectrum):
        ax.axvline(x=IHIGH_BAND, color='grey', linestyle='--', linewidth=0.8)

    ax.set_xlabel('Wavenumber $k$', fontsize=13)
    ax.set_ylabel('$E(k)$', fontsize=13)
    ax.set_title('Temperature Energy Spectrum', fontsize=14)
    ax.set_xlim(1, SPECTRUM_K_AXIS_MAX)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, which='both', alpha=0.3)

    fname = os.path.join(output_dir, 'energy_spectrum_temp.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"  Saved energy spectrum plot: {fname}")


# ============================================================================
# Model Registry (display name -> {model_type, checkpoint})
# ============================================================================

LOG_ROOT = "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs"

# Display names per user spec
MODEL_NAMES = {
    'flow_matching': 'Flow Matching',
    'flow_matching_history': 'HistoryFM',
    'ffno': 'FFNO',
    'unet': 'U-Net',
    'bubble_ddpm': 'DDPM',
    've_sde': 'VE-SDE',
    'diffusionpde': 'DiffusionPDE',
    'flow_matching_ar_bootstrap': 'HB-ARFM',
}

# Checkpoint paths per user spec (model_type -> checkpoint)
CHECKPOINTS = {
    'flow_matching': f"{LOG_ROOT}/flow_matching_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47841592/checkpoints/epoch=32-step=054912.ckpt",
    'flow_matching_history': f"{LOG_ROOT}/flow_matching_history_default_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_52743668/checkpoints/last.ckpt",
    'unet':          f"{LOG_ROOT}/unet_32_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47842727/checkpoints/last.ckpt",
    'bubble_ddpm':   f"{LOG_ROOT}/bubble_ddpm_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849371/checkpoints/last.ckpt",
    've_sde':        f"{LOG_ROOT}/ve_sde_32_256_2_False_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849739/checkpoints/last.ckpt",
    'ffno':          f"{LOG_ROOT}/ffno_m12_w64_l4_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_47849886/checkpoints/last.ckpt",
    'diffusionpde':  f"{LOG_ROOT}/diffusionpde_ch32_b2_s50_zobs1.0_zpde0.5_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_51917285/checkpoints/last.ckpt",
    # 'flow_matching_ar_bootstrap': f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_51527954/checkpoints/epoch=23-step=019920.ckpt",
    # 'flow_matching_ar_bootstrap': f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_51527954/checkpoints/epoch=08-step=007470.ckpt",
    # 'flow_matching_ar_bootstrap': f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_51527954/checkpoints/last.ckpt",
    'flow_matching_ar_bootstrap': f"{LOG_ROOT}/flow_matching_ar_bootstrap_32_256_2_False_hist64_roll5_attn_d128_L2_p8_temperature_from_sdf_pb_subcooled_singlestep_none_ds4_52646121/checkpoints/last.ckpt",
}

# Deterministic ordering for the comparison table
_MODEL_ORDER = [
    'flow_matching_ar_bootstrap',
    'flow_matching',
    'flow_matching_history',
    've_sde',
    'diffusionpde',
    'bubble_ddpm',
    'unet',
    'ffno',
]

MODEL_REGISTRY = OrderedDict([
    (MODEL_NAMES[mt], {"model_type": mt, "checkpoint": CHECKPOINTS[mt]})
    for mt in _MODEL_ORDER
])

# ============================================================================
# Model config builder — reads architecture from checkpoint hyper_parameters,
# only overrides inference-time settings (solver, num_inference_steps, etc.)
# ============================================================================

INFERENCE_OVERRIDES = {
    'flow_matching':              lambda a: {'num_integration_steps': a.num_inference_steps,
                                             'inference': {'solver': a.solver, 'guidance_scale': 1.0}},
    'flow_matching_history':      lambda a: {'num_integration_steps': a.num_inference_steps,
                                             'inference': {'solver': a.solver}},
    'flow_matching_ar_bootstrap': lambda a: {'num_integration_steps': a.num_inference_steps,
                                             'inference': {'solver': a.solver}},
    'edm':                        lambda a: {'num_sampling_steps': a.num_inference_steps,
                                             'solver': a.solver},
    'edm_ar_bootstrap':           lambda a: {'inference': {'solver': a.solver}},
    've_sde':                     lambda a: {'num_sampling_steps': a.num_inference_steps,
                                             'num_inference_steps': a.num_inference_steps},
    'diffusionpde':               lambda a: {'num_sampling_steps': a.num_inference_steps,
                                             'solver': a.solver},
    'bubble_ddpm':                lambda a: {},
    'unet':                       lambda a: {},
    'ffno':                       lambda a: {},
}


def _deep_update(base, overrides: dict):
    """Recursively merge overrides into a DictConfig or dict."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base.get(k), (dict, DictConfig)):
            _deep_update(base[k], v)
        else:
            base[k] = v


def build_model_cfg(model_type: str, task_cfg: DictConfig, args,
                    checkpoint_path: str = None) -> DictConfig:
    """Build model_cfg by reading the checkpoint's saved hyper_parameters
    and only overriding inference-time settings.

    Falls back to hardcoded defaults if the checkpoint lacks hyper_parameters.
    """
    saved_cfg = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        hp = ckpt.get('hyper_parameters', {})
        raw = hp.get('model_cfg', None)
        if raw is not None:
            saved_cfg = OmegaConf.create(OmegaConf.to_container(raw, resolve=True)
                                         if isinstance(raw, DictConfig) else raw)
        del ckpt

    if saved_cfg is not None:
        overrides = INFERENCE_OVERRIDES.get(model_type, lambda a: {})(args)
        _deep_update(saved_cfg, overrides)
        print(f"  [model_cfg] Loaded from checkpoint hyper_parameters")
        return saved_cfg

    print(f"  [model_cfg] WARNING: checkpoint has no saved model_cfg, using hardcoded fallback")
    return _fallback_model_cfg(model_type, task_cfg, args)


def _fallback_model_cfg(model_type: str, task_cfg: DictConfig, args) -> DictConfig:
    """Hardcoded fallback configs for checkpoints that lack saved hyper_parameters."""
    num_cond = len(task_cfg.conditioning_channels)
    num_target = len(task_cfg.target_channels)

    if model_type == 'flow_matching':
        return DictConfig({
            'name': 'flow_matching',
            'in_channels': num_target + num_cond,
            'out_channels': num_target,
            'base_channels': 32,
            'time_embed_dim': 256,
            'num_res_blocks': 2,
            'attention_type': 'none',
            'adaptive_scale': False,
            'skip_scale': False,
            'dropout': 0.1,
            'num_integration_steps': args.num_inference_steps,
            'temp_min': 55.0, 'temp_max': 120.0,
            'inference': {'solver': args.solver, 'guidance_scale': 1.0},
        })
    elif model_type == 'flow_matching_history':
        return DictConfig({
            'name': 'flow_matching_history',
            'history_window': 10,
            'history_stride': 1,
            'base_channels': 32,
            'time_embed_dim': 256,
            'num_res_blocks': 2,
            'dropout': 0.1,
            'attention_type': 'none',
            'num_integration_steps': args.num_inference_steps,
            'temp_min': 55.0, 'temp_max': 120.0,
            'inference': {'solver': args.solver},
        })
    elif model_type in ('flow_matching_ar_bootstrap', 'edm_ar_bootstrap'):
        return DictConfig({
            'name': model_type,
            'in_channels': 10,
            'out_channels': num_target,
            'base_channels': 32,
            'time_embed_dim': 256,
            'num_res_blocks': 2,
            'use_attention': False,
            'dropout': 0.1,
            'num_integration_steps': args.num_inference_steps,
            'temp_min': 55.0, 'temp_max': 120.0,
            'history_length': 10,
            'rollout_length': 5,
            'use_availability_mask': True,
            'history_encoder_type': 'attention',
            'history_encoder_hidden': 32,
            'attention_encoder_embed_dim': 128,
            'attention_encoder_num_heads': 8,
            'attention_encoder_depth': 2,
            'attention_encoder_patch_size': 8,
            'attention_encoder_mlp_ratio': 4.0,
            'attention_encoder_dropout': 0.0,
            'attention_encoder_output_head': 'linear',
            'attention_encoder_max_history_length': 50,
            'bootstrap_loss_weight': 1.0,
            'ar_loss_weight': 1.0,
            'bootstrap_state_loss_weight': 0.5,
            'inference': {'solver': args.solver},
        })
    elif model_type == 've_sde':
        return DictConfig({
            'name': 've_sde',
            'in_channels': num_target + num_cond,
            'out_channels': num_target,
            'base_channels': 32,
            'sigma_embed_dim': 256,
            'num_res_blocks': 2,
            'use_attention': False,
            'dropout': 0.1,
            'sigma_min': 0.01,
            'sigma_max': 1.0,
            'num_sampling_steps': args.num_inference_steps,
            'sampling_method': 'pc',
            'snr': 0.16,
            'conditioning_strategy': 'none',
            'temp_min': 55.0, 'temp_max': 120.0,
            'num_inference_steps': args.num_inference_steps,
        })
    elif model_type == 'edm':
        return DictConfig({
            'name': 'edm',
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
            'temp_min': 55.0, 'temp_max': 120.0,
        })
    elif model_type == 'diffusionpde':
        return DictConfig({
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
            'temp_min': 55.0, 'temp_max': 120.0,
        })
    elif model_type == 'bubble_ddpm':
        return DictConfig({
            'name': 'bubble_ddpm',
            'in_channels': num_target + num_cond,
            'out_channels': num_target,
            'base_channels': 32,
            'time_embed_dim': 256,
            'num_res_blocks': 2,
            'use_attention': False,
            'dropout': 0.1,
            'num_timesteps': 1000,
            'beta_start': 1e-4,
            'beta_end': 2e-2,
            'num_inference_steps': 1000,
            'temp_min': 55.0, 'temp_max': 120.0,
        })
    elif model_type == 'unet':
        return DictConfig({
            'name': 'unet',
            'init_features': 32,
            'temp_min': 55.0, 'temp_max': 120.0,
        })
    elif model_type == 'ffno':
        return DictConfig({
            'name': 'ffno',
            'modes': 12,
            'width': 64,
            'n_layers': 4,
            'dropout': 0.0,
            'use_fork': False,
            'fourier_mode': 'full',
            'temp_min': 55.0, 'temp_max': 120.0,
        })
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# Normalization stats loader
# ============================================================================

def load_normalization_stats(checkpoint_path: str, explicit_path: str = None) -> dict:
    """Load normalization stats: explicit path > checkpoint dir > error."""
    if explicit_path and os.path.exists(explicit_path):
        with open(explicit_path, 'r') as f:
            return json.load(f)

    ckpt_dir = os.path.dirname(checkpoint_path)
    if "checkpoints" in ckpt_dir:
        ckpt_dir = os.path.dirname(ckpt_dir)
    stats_file = os.path.join(ckpt_dir, "normalization_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)

    raise FileNotFoundError(
        f"normalization_stats.json not found for {checkpoint_path}. "
        f"Provide --normalization-stats explicitly."
    )


# ============================================================================
# HDF5 export — self-contained inference results for downstream metric work
# ============================================================================

def save_inference_hdf5(
    filepath: str,
    inference_cache: Dict[str, dict],
    heater_temp: float,
    args,
) -> None:
    """Save all inference results to a single HDF5 file (temperature-only).

    File layout
    -----------
    /metadata                       (attributes)
    /<model_name>/
        ground_truth/
            temperature   (N, H, W) float32
        predictions/
            temperature   (N, H, W) float32
        sdf               (N, H, W) float32
        massflux          (N, H, W) float32   — raw scalar mass flux (if available)
        attrs: model_type, checkpoint, num_frames
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with h5py.File(filepath, 'w') as f:
        meta = f.create_group('metadata')
        meta.attrs['data_file'] = args.data_file
        meta.attrs['start_time'] = args.start_time
        meta.attrs['downsample_factor'] = args.downsample_factor
        meta.attrs['num_inference_steps'] = args.num_inference_steps
        meta.attrs['solver'] = args.solver
        meta.attrs['seed'] = args.seed
        meta.attrs['norm_mode'] = args.norm_mode
        meta.attrs['heater_temp'] = heater_temp
        if args.num_samples is not None:
            meta.attrs['num_samples'] = args.num_samples

        for model_name, data in inference_cache.items():
            safe_name = model_name.replace('/', '_')
            grp = f.create_group(safe_name)
            grp.attrs['model_name'] = model_name
            grp.attrs['model_type'] = data['model_type']
            grp.attrs['checkpoint'] = data['checkpoint']
            grp.attrs['num_frames'] = data['gt_temp'].shape[0]

            gt = grp.create_group('ground_truth')
            gt.create_dataset('temperature', data=data['gt_temp'].astype(np.float32),
                              compression='gzip', compression_opts=4)

            pred = grp.create_group('predictions')
            pred.create_dataset('temperature', data=data['pred_temp'].astype(np.float32),
                                compression='gzip', compression_opts=4)

            grp.create_dataset('sdf', data=data['sdf'].astype(np.float32),
                               compression='gzip', compression_opts=4)
            if 'massflux' in data:
                grp.create_dataset('massflux', data=data['massflux'].astype(np.float32),
                                   compression='gzip', compression_opts=4)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved inference HDF5: {filepath}  ({size_mb:.1f} MB)")
    print(f"  Models included: {list(inference_cache.keys())}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model inference metrics comparison for Task 1 (temperature_from_sdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-file', type=str,
                        default='/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5',
                        help='Path to HDF5 validation data file')
    parser.add_argument('--output-dir', type=str,
                        default='./ICML/CamReady/Table3_metrics_Task1/epoch23',
                        help='Output directory for CSV')
    parser.add_argument('--start-time', type=int, default=800,
                        help='Starting timestep in HDF5 file')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Spatial downsample factor (4 = 128x128)')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='ODE / denoising integration steps')
    parser.add_argument('--solver', type=str, default='rk4',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver for flow matching / EDM models')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Starting frame index for evaluation')
    parser.add_argument('--frame-end', type=int, default=5,
                        help='Ending frame index (exclusive)')
    parser.add_argument('--normalization-stats', type=str,
                        default='/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json',
                        help='Path to shared normalization_stats.json')
    parser.add_argument('--seed', type=int, default=32,
                        help='Random seed for reproducibility')
    parser.add_argument('--norm-mode', type=str, default='all',
                        choices=['none', 'all', 'temperature_only'],
                        help='Normalization mode (must match training)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Random sampling mode: evaluate on N randomly chosen timesteps. '
                             'Overrides --frame-start/--frame-end.')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of model_type keys to evaluate (default: all). '
                             'Valid: flow_matching, flow_matching_history, unet, bubble_ddpm, '
                             've_sde, ffno, diffusionpde, flow_matching_ar_bootstrap. '
                             'Short keys: fm, historyfm, unet, ddpm, vesde, ffno, dpde, hbarfm.')
    parser.add_argument('--fourier', action='store_true', default=False,
                        help='Include 3 Fourier spectral band metrics (low/mid/high for temperature)')
    parser.add_argument('--plot-spectra', action='store_true', default=False,
                        help='Generate energy spectrum E(k) plots comparing all models')
    parser.add_argument('--save-hdf5', action='store_true', default=False,
                        help='Save all inference results (GT + predictions + SDF) to a single '
                             'HDF5 file so others can compute metrics without re-running inference')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model subset selection — accept both short keys and model_type keys
    SHORT_KEYS = {
        'fm': 'flow_matching',
        'historyfm': 'flow_matching_history',
        'fmhist': 'flow_matching_history',
        'flow_matching_history': 'flow_matching_history',
        'unet': 'unet',
        'ddpm': 'bubble_ddpm',
        'vesde': 'vesde',
        've_sde': 've_sde',
        'ffno': 'ffno',
        'dpde': 'diffusionpde',
        'diffusionpde': 'diffusionpde',
        'hbarfm': 'flow_matching_ar_bootstrap',
        'fmar': 'flow_matching_ar_bootstrap',
    }
    if args.models is not None:
        selected = []
        for key in args.models:
            mt = SHORT_KEYS.get(key.lower(), key)
            if mt == 'vesde':
                mt = 've_sde'
            display = MODEL_NAMES.get(mt)
            if display is not None and display in MODEL_REGISTRY:
                selected.append(display)
            else:
                print(f"WARNING: Unknown model key '{key}', skipping. "
                      f"Valid keys: {list(SHORT_KEYS.keys()) + list(MODEL_NAMES.keys())}")
        models_to_run = OrderedDict((k, MODEL_REGISTRY[k]) for k in selected)
    else:
        models_to_run = MODEL_REGISTRY

    random_mode = args.num_samples is not None

    print("=" * 80)
    print("  Multi-Model Inference Metrics — Task 1 (temperature_from_sdf)")
    print("=" * 80)
    print(f"  Data file:         {args.data_file}")
    print(f"  Start time:        {args.start_time}")
    if random_mode:
        print(f"  Sampling mode:     RANDOM ({args.num_samples} timesteps)")
    else:
        print(f"  Frame range:       [{args.frame_start}, {args.frame_end})")
    print(f"  Downsample:        {args.downsample_factor}x")
    print(f"  Inference steps:   {args.num_inference_steps}")
    print(f"  Solver:            {args.solver}")
    print(f"  Seed:              {args.seed}")
    print(f"  Models ({len(models_to_run)}):       {list(models_to_run.keys())}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Save HDF5:         {args.save_hdf5}")
    print("=" * 80)

    task_cfg = load_task_config('temperature_from_sdf')

    # Only need SDF and heater temperature for task-1 metrics
    (sdf_gt_full, _, _, _, _, _, heater_temp) = load_ground_truth_data(
        args.data_file, args.start_time, args.downsample_factor
    )

    # Load raw mass flux scalar field directly from source HDF5 so the
    # interface mask used for Temp. IRMSE / BRMSE matches the physically
    # correct definition used in inference_metrics_task2.py (|massflux| > 0).
    # Task-1 models do not see massflux as input, but the metric helpers do.
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
            print(f"  Warning: no 'massflux' field in source HDF5 — "
                  f"falling back to SDF zero-crossing interface mask")

    optim_cfg = DictConfig({'name': 'adamw', 'lr': 0.001})
    scheduler_cfg = DictConfig({'name': 'cosine'})

    all_results = OrderedDict()
    inference_cache = OrderedDict()

    spectra_data = {}
    gt_spectrum = None

    for model_name, model_info in models_to_run.items():
        model_type = model_info['model_type']
        checkpoint_path = model_info['checkpoint']

        print(f"\n{'='*80}")
        print(f"  [{list(models_to_run.keys()).index(model_name)+1}/{len(models_to_run)}] "
              f"{model_name}  ({model_type})")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"{'='*80}")

        if not os.path.exists(checkpoint_path):
            print(f"  SKIPPING — checkpoint not found: {checkpoint_path}")
            all_results[model_name] = OrderedDict((k, np.nan) for k in _get_metric_names(args.fourier))
            continue

        try:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

            norm_stats = load_normalization_stats(checkpoint_path, args.normalization_stats)

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

            total_available = len(dataset)

            if random_mode and is_ar_bootstrap:
                num_segments = -(-args.num_samples // rollout_length)
                num_segments = min(num_segments, total_available)

                random.seed(args.seed)
                random_starts = sorted(random.sample(range(total_available), num_segments))
                total_frames = num_segments * rollout_length

                print(f"\n  Random sampling: {num_segments} rollouts x {rollout_length} frames "
                      f"= {total_frames} total frames at random positions")

                gt_t_list, pr_t_list, sdf_list, mf_list = [], [], [], []

                for idx in tqdm(random_starts, desc=f"  {model_name}"):
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        _, _, gt_t, _, _, pr_t = \
                            run_ar_bootstrap_inference_batch(
                                model, dataset, device, args.num_inference_steps,
                                max_samples=1, start_idx=idx, solver=args.solver)

                    gt_t_list.append(gt_t)
                    pr_t_list.append(pr_t)
                    sdf_list.append(sdf_gt_full[idx:idx + rollout_length])
                    if massflux_full is not None:
                        mf_list.append(massflux_full[idx:idx + rollout_length])

                gt_temp = np.concatenate(gt_t_list)
                pred_temp = np.concatenate(pr_t_list)
                sdf_slice = np.concatenate(sdf_list)
                mf_slice = np.concatenate(mf_list) if mf_list else None

            elif random_mode:
                n_samples = min(args.num_samples, total_available)
                random.seed(args.seed)
                random_indices = sorted(random.sample(range(total_available), n_samples))

                print(f"\n  Random sampling: {n_samples} timesteps from {total_available} available")

                gt_t_list, pr_t_list, sdf_list, mf_list = [], [], [], []

                for idx in tqdm(random_indices, desc=f"  {model_name}"):
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        _, _, gt_t, _, _, pr_t = \
                            run_inference_batch(
                                model, dataset, device, args.num_inference_steps,
                                max_samples=1, model_type=model_type,
                                start_idx=idx, solver=args.solver)

                    gt_t_list.append(gt_t)
                    pr_t_list.append(pr_t)
                    sdf_list.append(sdf_gt_full[idx:idx + 1])
                    if massflux_full is not None:
                        mf_list.append(massflux_full[idx:idx + 1])

                gt_temp = np.concatenate(gt_t_list)
                pred_temp = np.concatenate(pr_t_list)
                sdf_slice = np.concatenate(sdf_list)
                mf_slice = np.concatenate(mf_list) if mf_list else None

            elif is_ar_bootstrap:
                frame_start = args.frame_start
                frame_end = args.frame_end
                desired_frames = frame_end - frame_start
                num_segments = (desired_frames + rollout_length - 1) // rollout_length
                num_segments = min(num_segments, total_available - frame_start)
                actual_frames = num_segments * rollout_length
                sdf_slice = sdf_gt_full[frame_start:frame_start + actual_frames]
                mf_slice = (massflux_full[frame_start:frame_start + actual_frames]
                            if massflux_full is not None else None)

                _, _, gt_temp, _, _, pred_temp = \
                    run_ar_bootstrap_inference_batch(
                        model, dataset, device, args.num_inference_steps,
                        max_samples=num_segments,
                        start_idx=frame_start,
                        solver=args.solver,
                    )
                n_gen = gt_temp.shape[0] if gt_temp is not None else 0
                sdf_slice = sdf_slice[:n_gen]
                if mf_slice is not None:
                    mf_slice = mf_slice[:n_gen]

            else:
                frame_start = args.frame_start
                frame_end = args.frame_end
                frame_end_clamped = min(frame_end, total_available)
                num_frames = frame_end_clamped - frame_start
                sdf_slice = sdf_gt_full[frame_start:frame_end_clamped]
                mf_slice = (massflux_full[frame_start:frame_end_clamped]
                            if massflux_full is not None else None)

                _, _, gt_temp, _, _, pred_temp = \
                    run_inference_batch(
                        model, dataset, device, args.num_inference_steps,
                        max_samples=num_frames,
                        model_type=model_type,
                        start_idx=frame_start,
                        solver=args.solver,
                    )

            all_results[model_name] = compute_task1_metrics(
                gt_temp, pred_temp,
                sdf_slice, heater_temp,
                massflux=mf_slice,
                downsample_factor=args.downsample_factor,
                include_fourier=args.fourier,
            )

            if args.save_hdf5:
                cache_entry = {
                    'model_type': model_type,
                    'checkpoint': checkpoint_path,
                    'gt_temp': gt_temp,
                    'pred_temp': pred_temp,
                    'sdf': sdf_slice,
                }
                if mf_slice is not None:
                    cache_entry['massflux'] = mf_slice
                inference_cache[model_name] = cache_entry

            if args.plot_spectra:
                pred_temp_spec = _radial_energy_spectrum(pred_temp)
                spectra_data[model_name] = pred_temp_spec
                if gt_spectrum is None:
                    gt_spectrum = _radial_energy_spectrum(gt_temp)

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = OrderedDict((k, np.nan) for k in _get_metric_names(args.fourier))

    # ========================================================================
    # Save inference results HDF5 (optional)
    # ========================================================================
    if args.save_hdf5 and inference_cache:
        hdf5_path = os.path.join(args.output_dir, 'task1_inference_results.hdf5')
        print(f"\n{'='*80}")
        print("  Saving inference results to HDF5")
        print(f"{'='*80}")
        save_inference_hdf5(hdf5_path, inference_cache, heater_temp, args)
        del inference_cache

    # ========================================================================
    # Energy spectrum plots (optional)
    # ========================================================================
    if args.plot_spectra and gt_spectrum is not None and spectra_data:
        print(f"\n{'='*80}")
        print("  Generating energy spectrum plots")
        print(f"{'='*80}")
        spectra_dir = os.path.join(args.output_dir, 'fourier')
        plot_energy_spectra(spectra_data, gt_spectrum, spectra_dir)

    # ========================================================================
    # Assemble comparison CSV (rows = metrics, columns = models)
    # ========================================================================
    print(f"\n{'='*80}")
    print("  Assembling comparison table")
    print(f"{'='*80}")

    df = pd.DataFrame(all_results)
    df.index.name = 'Metric'

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'task1_model_comparison.csv')
    df.to_csv(csv_path)

    csv_path_T = os.path.join(args.output_dir, 'task1_model_comparison_transposed.csv')
    df.T.to_csv(csv_path_T)

    print("\n" + df.to_string())
    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {csv_path_T}")
    print(f"\n  Done! Evaluated {len([r for r in all_results.values() if not all(np.isnan(v) for v in r.values())])} / {len(models_to_run)} models successfully.")


if __name__ == "__main__":
    main()
