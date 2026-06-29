#!/usr/bin/env python3
"""
Measurement-Space Consistency Metrics (Horizon-Based).

Evaluates how measurement-space consistency degrades over long autoregressive
rollouts. For each of N starting points, runs a full AR rollout and computes
metrics at discrete horizon checkpoints (e.g. 1, 100, 200, 300 steps).

Metrics: H(x_pred) vs y_obs, where:
  - y_obs  = interface velocity used as model conditioning (ground truth)
  - x_pred = predicted bulk velocity field
  - H(.)   = observation operator that extracts interface velocity from bulk

The observation operator applies:
  v_interface = v_bulk * interface_region + (massflux / rho_gas) * normal

Usage:
    python scripts/measurement_space_consistency.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --data-file /path/to/Twall_96.hdf5 \
        --horizons 1 100 200 300 \
        --num-starts 10

    # Multiple validation files:
    python scripts/measurement_space_consistency.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --data-file /path/to/Twall_96.hdf5 /path/to/Twall_108.hdf5 \
        --horizons 1 50 100 200 300 --num-starts 5
"""

import sys
import os
import csv
import json
import argparse
from collections import defaultdict

import torch
import numpy as np
import h5py as h5
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bubblefusion.models.flow_matching_ar_bootstrap import (
    ConditionalFlowMatchingARBootstrapLightning,
)
from bubblefusion.data.bubbleml import (
    BulkFlowARBootstrap,
    NormalizationHelper,
)


# ============================================================================
# Robust model loading
#
# The Lightning module's constructor builds its architecture from the
# ``model_cfg`` dict passed at init time.  ``save_hyperparameters`` records that
# dict in the checkpoint, but the model code has since been refactored (e.g.
# ``AttentionHistoryEncoder`` switched from ``img_h``/``img_w`` to ``img_size``,
# new optional features like push-forward and AR temporal decay), so relying on
# Lightning's implicit hparams reconstruction with ``strict=True`` is brittle.
#
# Instead, we follow the same pattern as ``scripts/inference_metrics_task2.py``:
# read the saved ``model_cfg`` from the checkpoint, override only inference-time
# settings, and pass it back to ``load_from_checkpoint`` explicitly together
# with ``task_cfg``, with ``strict=False``.
# ============================================================================

def _load_cfg_from_checkpoint(checkpoint_path: str):
    """Pull ``model_cfg`` and ``task_cfg`` out of the checkpoint hparams."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hp = ckpt.get('hyper_parameters', {}) or {}

    def _to_dict_cfg(raw):
        if raw is None:
            return None
        if isinstance(raw, DictConfig):
            return OmegaConf.create(OmegaConf.to_container(raw, resolve=True))
        return OmegaConf.create(raw)

    model_cfg = _to_dict_cfg(hp.get('model_cfg', None))
    task_cfg = _to_dict_cfg(hp.get('task_cfg', None))
    del ckpt
    return model_cfg, task_cfg


def _default_task_cfg() -> DictConfig:
    """Fallback task config for velocity_from_interface (matches the YAML)."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'bubblefusion', 'config', 'task_cfg', 'velocity_from_interface.yaml',
    )
    if os.path.exists(cfg_path):
        return OmegaConf.load(cfg_path)
    return DictConfig({
        'name': 'velocity_from_interface',
        'conditioning_channels': [0, 1, 2],
        'conditioning_names': ['sdf', 'velx_interface', 'vely_interface'],
        'target_channels': [1, 2, 0],
        'target_names': ['velx', 'vely', 'temperature'],
    })


def load_ar_bootstrap_model(
    checkpoint_path: str,
    norm_stats: dict,
    solver: str,
    num_inference_steps: int,
):
    """Build model_cfg from the checkpoint and load the Lightning module.

    Overrides only inference-time settings (solver, num_integration_steps); all
    architectural choices come straight from the saved hyper_parameters so the
    state_dict shapes match.
    """
    model_cfg, task_cfg = _load_cfg_from_checkpoint(checkpoint_path)

    if model_cfg is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no saved model_cfg in "
            "hyper_parameters; cannot reconstruct the model architecture."
        )

    inference_cfg = OmegaConf.to_container(model_cfg.get('inference', {}) or {}, resolve=True)
    inference_cfg['solver'] = solver
    model_cfg['inference'] = inference_cfg
    model_cfg['num_integration_steps'] = num_inference_steps

    if task_cfg is None:
        task_cfg = _default_task_cfg()

    optim_cfg = DictConfig({'name': 'adamw', 'lr': 1e-3})
    scheduler_cfg = DictConfig({'name': 'cosine'})

    model = ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
        checkpoint_path,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        task_cfg=task_cfg,
        normalization_stats=norm_stats,
        strict=False,
    )
    return model


# ============================================================================
# Observation operator & helpers (unchanged)
# ============================================================================

def extract_interface_velocity_from_bulk(
    sdf: torch.Tensor,
    mass_flux: torch.Tensor,
    velx_bulk: torch.Tensor,
    vely_bulk: torch.Tensor,
    rho_gas: float,
    dy: float,
    dx: float,
) -> tuple:
    """
    Apply the observation operator H(.) to extract interface velocity from bulk velocity.

    This is the same formula used in the dataset to compute the conditioning inputs
    from the simulation data. By applying it to predicted bulk velocities, we can
    check measurement-space consistency.

    Args:
        sdf: Signed distance function [H, W]
        mass_flux: Mass flux field [H, W]
        velx_bulk: Predicted bulk x-velocity [H, W]
        vely_bulk: Predicted bulk y-velocity [H, W]
        rho_gas: Gas phase density
        dy, dx: Grid spacings

    Returns:
        velx_interface, vely_interface: Extracted interface velocity [H, W]
        interface_mask: Boolean mask of interface pixels [H, W]
    """
    interface_region = (mass_flux != 0).float()

    norm_y, norm_x = torch.gradient(sdf, spacing=[dy, dx])
    norm_y = norm_y * interface_region
    norm_x = norm_x * interface_region
    mag = torch.norm(torch.stack([norm_x, norm_y]), dim=0) + 1e-8
    norm_y = norm_y / mag
    norm_x = norm_x / mag

    velx_interface = velx_bulk * interface_region + (mass_flux / rho_gas) * norm_x
    vely_interface = vely_bulk * interface_region + (mass_flux / rho_gas) * norm_y

    return velx_interface, vely_interface, interface_region.bool()


def _downsample_2d(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample a 2D tensor [H, W] by the given factor using bilinear interpolation."""
    if factor == 1:
        return tensor
    t = tensor.unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, scale_factor=1.0 / factor, mode='bilinear', align_corners=False
    )
    return t.squeeze(0).squeeze(0)


def _upsample_2d(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Upsample a 2D tensor [H, W] to target size using bilinear interpolation."""
    if tensor.shape[0] == target_h and tensor.shape[1] == target_w:
        return tensor
    t = tensor.unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, size=(target_h, target_w), mode='bilinear', align_corners=False
    )
    return t.squeeze(0).squeeze(0)


def load_raw_fields_fullres(h5_file, timestep):
    """Load raw (un-normalized) fields at native resolution from HDF5."""
    sdf = torch.tensor(h5_file["dfun"][timestep]).float()
    velx = torch.tensor(h5_file["velx"][timestep]).float()
    vely = torch.tensor(h5_file["vely"][timestep]).float()
    massflux = torch.tensor(h5_file["massflux"][timestep]).float()
    return sdf, velx, vely, massflux


def apply_H_at_fullres_then_downsample(
    sdf_full: torch.Tensor,
    massflux_full: torch.Tensor,
    velx_bulk: torch.Tensor,
    vely_bulk: torch.Tensor,
    rho_gas: float,
    dy_full: float,
    dx_full: float,
    downsample_factor: int,
) -> tuple:
    """
    Apply observation operator H at full resolution, then downsample.

    This matches the dataset's pipeline: compute interface velocity at 512x512,
    then downsample to 128x128. For predicted bulk velocity (which is at 128x128),
    it is first upsampled to 512x512 before applying H.

    Returns downsampled interface velocities and mask.
    """
    full_h, full_w = sdf_full.shape

    if velx_bulk.shape[0] != full_h or velx_bulk.shape[1] != full_w:
        velx_full = _upsample_2d(velx_bulk, full_h, full_w)
        vely_full = _upsample_2d(vely_bulk, full_h, full_w)
    else:
        velx_full = velx_bulk
        vely_full = vely_bulk

    velx_int_full, vely_int_full, mask_full = extract_interface_velocity_from_bulk(
        sdf_full, massflux_full, velx_full, vely_full,
        rho_gas, dy_full, dx_full,
    )

    velx_int_ds = _downsample_2d(velx_int_full, downsample_factor)
    vely_int_ds = _downsample_2d(vely_int_full, downsample_factor)
    mask_ds = _downsample_2d(mask_full.float(), downsample_factor) > 0.5

    return velx_int_ds, vely_int_ds, mask_ds


def compute_metrics_on_interface(
    pred_velx_int: np.ndarray,
    pred_vely_int: np.ndarray,
    obs_velx_int: np.ndarray,
    obs_vely_int: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Compute measurement-space consistency metrics on interface pixels.

    Returns dict with:
        - rmse_velx, rmse_vely, rmse_combined: RMSE
        - mae_velx, mae_vely, mae_combined: MAE on interface pixels
        - cosine_similarity: Mean cosine similarity of velocity vectors at interface
        - relative_magnitude_error: | |v_pred| - |v_obs| | / |v_obs|
        - num_interface_pixels: Count of interface pixels
    """
    if mask.sum() == 0:
        return {k: float('nan') for k in [
            'rmse_velx', 'rmse_vely', 'rmse_combined',
            'mae_velx', 'mae_vely', 'mae_combined',
            'cosine_similarity', 'relative_magnitude_error',
            'num_interface_pixels',
        ]}

    px = pred_velx_int[mask]
    py = pred_vely_int[mask]
    ox = obs_velx_int[mask]
    oy = obs_vely_int[mask]

    mse_x = float(np.mean((px - ox) ** 2))
    mse_y = float(np.mean((py - oy) ** 2))
    mae_x = float(np.mean(np.abs(px - ox)))
    mae_y = float(np.mean(np.abs(py - oy)))

    mse_combined = (mse_x + mse_y) / 2
    mae_combined = (mae_x + mae_y) / 2

    dot = px * ox + py * oy
    mag_pred = np.sqrt(px ** 2 + py ** 2) + 1e-10
    mag_obs = np.sqrt(ox ** 2 + oy ** 2) + 1e-10
    cos_sim = dot / (mag_pred * mag_obs)
    cos_sim_mean = float(np.mean(cos_sim))

    rel_mag_err = float(np.mean(np.abs(mag_pred - mag_obs) / mag_obs))

    return {
        'rmse_velx': float(np.sqrt(mse_x)),
        'rmse_vely': float(np.sqrt(mse_y)),
        'rmse_combined': float(np.sqrt(mse_combined)),
        'mae_velx': mae_x,
        'mae_vely': mae_y,
        'mae_combined': mae_combined,
        'cosine_similarity': cos_sim_mean,
        'relative_magnitude_error': rel_mag_err,
        'num_interface_pixels': int(mask.sum()),
    }


# ============================================================================
# Dataset-to-HDF5 index mapping
# ============================================================================

def resolve_sample_to_hdf5(dataset, sample_idx):
    """Map a dataset sample index to (file_idx, segment_start_timestep)."""
    effective_start = dataset.effective_start_time
    rollout_len = dataset.rollout_length
    samples_per_traj = [
        max(0, tl - effective_start - rollout_len + 1)
        for tl in dataset.traj_lens
    ]
    cumsum = np.cumsum(samples_per_traj)
    file_idx = int(np.searchsorted(cumsum, sample_idx, side="right"))
    local_idx = sample_idx - (int(cumsum[file_idx - 1]) if file_idx > 0 else 0)
    segment_start = local_idx + effective_start
    return file_idx, segment_start


# ============================================================================
# Horizon-based evaluation
# ============================================================================

def run_horizon_evaluation(
    model,
    dataset,
    h5_files,
    fluid_params_list,
    norm_helper,
    device,
    sample_indices,
    horizons,
    num_steps,
    solver,
    downsample_factor,
):
    """
    Run long AR rollouts from selected starting points and compute
    measurement-space consistency at each horizon checkpoint.

    For each starting point:
    1. Bootstrap initial state from conditioning history
    2. Autoregress up to max(horizons) steps
    3. At each horizon checkpoint, apply H(x_pred) and compare to y_obs

    Returns:
        horizon_metrics: dict of horizon -> list of metric dicts (one per start)
        raw_rows: list of dicts for per-sample CSV export
    """
    conditioning_channels = model.conditioning_channels
    target_channels = model.target_channels
    target_names = list(model.task_cfg.get('target_names', ['velx', 'vely', 'temperature']))
    conditioning_names = list(model.task_cfg.get(
        'conditioning_names', ['sdf', 'velx_interface', 'vely_interface']
    ))

    velx_target_idx = target_names.index('velx')
    vely_target_idx = target_names.index('vely')

    # Position of the interface velocity channels inside the *raw* dataset
    # conditioning tensor (before the model's channel selection).  We use the
    # raw tensor below to read y_obs and denormalize it.
    velx_int_cond_idx = conditioning_names.index('velx_interface')
    vely_int_cond_idx = conditioning_names.index('vely_interface')

    sorted_horizons = sorted(horizons)
    max_horizon = sorted_horizons[-1]
    horizon_set = set(sorted_horizons)

    horizon_metrics = {h: [] for h in sorted_horizons}
    raw_rows = []

    for start_i, sample_idx in enumerate(sample_indices):
        sample = dataset[sample_idx]
        cond_hist, cond_seq, target_seq = sample[:3]
        cond_hist = cond_hist.unsqueeze(0).to(device)
        cond_seq = cond_seq.unsqueeze(0).to(device)

        cond_hist_e = cond_hist[:, :, conditioning_channels]
        cond_seq_e = cond_seq[:, :, conditioning_channels]
        target_seq_e = target_seq.unsqueeze(0).to(device)[:, :, target_channels]

        B, _, C_cond, H, W = cond_hist_e.shape
        L = cond_seq_e.shape[1]
        C_out = target_seq_e.shape[2]

        prev = model.bootstrap_initial_state(cond_hist_e, cond_seq_e[:, 0])

        file_idx, segment_start = resolve_sample_to_hdf5(dataset, sample_idx)
        h5_file = h5_files[file_idx]
        params = fluid_params_list[file_idx]
        rho_gas = params["rhogas"]
        sdf_shape = h5_file["dfun"].shape
        dy = (params["y_max"] - params["y_min"]) / sdf_shape[1]
        dx = (params["x_max"] - params["x_min"]) / sdf_shape[2]

        steps_to_run = min(max_horizon, L)

        for step in range(steps_to_run):
            cond = cond_seq_e[:, step]
            avail = (torch.zeros(B, 1, H, W, device=device) if step == 0
                     else torch.ones(B, 1, H, W, device=device))

            with torch.no_grad():
                pred = model.sample(
                    condition=cond, prev_output=prev,
                    shape=(B, C_out, H, W), device=device,
                    availability_mask=avail,
                    num_integration_steps=num_steps, solver=solver,
                )

            horizon_step = step + 1  # 1-indexed horizon

            if horizon_step in horizon_set:
                pred_velx_norm = pred[0, velx_target_idx].cpu()
                pred_vely_norm = pred[0, vely_target_idx].cpu()
                pred_velx_phys = norm_helper.denormalize_velocity(pred_velx_norm)
                pred_vely_phys = norm_helper.denormalize_velocity(pred_vely_norm)

                actual_timestep = segment_start + step
                sdf_full, _, _, massflux_full = load_raw_fields_fullres(
                    h5_file, actual_timestep
                )

                pred_velx_int, pred_vely_int, interface_mask = apply_H_at_fullres_then_downsample(
                    sdf_full, massflux_full, pred_velx_phys, pred_vely_phys,
                    rho_gas, dy, dx, downsample_factor,
                )

                obs_velx_int_norm = cond_seq[0, step, velx_int_cond_idx].cpu()
                obs_vely_int_norm = cond_seq[0, step, vely_int_cond_idx].cpu()
                obs_velx_int_phys = norm_helper.denormalize_velocity(obs_velx_int_norm)
                obs_vely_int_phys = norm_helper.denormalize_velocity(obs_vely_int_norm)

                mask_np = interface_mask.numpy()
                metrics = compute_metrics_on_interface(
                    pred_velx_int.numpy(), pred_vely_int.numpy(),
                    obs_velx_int_phys.numpy(), obs_vely_int_phys.numpy(),
                    mask_np,
                )

                horizon_metrics[horizon_step].append(metrics)

                row = {
                    'start_idx': start_i,
                    'sample_idx': sample_idx,
                    'horizon': horizon_step,
                    'file_idx': file_idx,
                    'timestep': actual_timestep,
                }
                row.update(metrics)
                raw_rows.append(row)

            prev = pred

        print(f"  [{start_i + 1}/{len(sample_indices)}] "
              f"Start idx={sample_idx}, file={file_idx}, "
              f"t0={segment_start}, rolled {steps_to_run} steps")

    return horizon_metrics, raw_rows


def run_gt_sanity_check(
    dataset,
    h5_files,
    fluid_params_list,
    norm_helper,
    sample_indices,
    horizons,
    downsample_factor,
    conditioning_names=('sdf', 'velx_interface', 'vely_interface'),
):
    """
    Sanity check: apply H to GT bulk velocity and compare to the observation.
    Should give near-zero error, validating that the observation operator is
    consistent with how the dataset was constructed.
    """
    cond_names = list(conditioning_names)
    velx_int_cond_idx = cond_names.index('velx_interface')
    vely_int_cond_idx = cond_names.index('vely_interface')

    sorted_horizons = sorted(horizons)
    max_horizon = sorted_horizons[-1]
    horizon_set = set(sorted_horizons)

    gt_horizon_metrics = {h: [] for h in sorted_horizons}

    for start_i, sample_idx in enumerate(sample_indices):
        sample = dataset[sample_idx]
        cond_hist, cond_seq, target_seq = sample[:3]

        file_idx, segment_start = resolve_sample_to_hdf5(dataset, sample_idx)
        h5_file = h5_files[file_idx]
        params = fluid_params_list[file_idx]
        rho_gas = params["rhogas"]
        sdf_shape = h5_file["dfun"].shape
        dy = (params["y_max"] - params["y_min"]) / sdf_shape[1]
        dx = (params["x_max"] - params["x_min"]) / sdf_shape[2]

        L = cond_seq.shape[0]
        steps_to_check = min(max_horizon, L)

        for step in range(steps_to_check):
            horizon_step = step + 1
            if horizon_step not in horizon_set:
                continue

            actual_timestep = segment_start + step
            sdf_full, velx_full, vely_full, massflux_full = load_raw_fields_fullres(
                h5_file, actual_timestep
            )

            gt_velx_int, gt_vely_int, interface_mask = apply_H_at_fullres_then_downsample(
                sdf_full, massflux_full, velx_full, vely_full,
                rho_gas, dy, dx, downsample_factor,
            )

            obs_velx_norm = cond_seq[step, velx_int_cond_idx]
            obs_vely_norm = cond_seq[step, vely_int_cond_idx]
            obs_velx_phys = norm_helper.denormalize_velocity(obs_velx_norm)
            obs_vely_phys = norm_helper.denormalize_velocity(obs_vely_norm)

            mask_np = interface_mask.numpy()
            metrics = compute_metrics_on_interface(
                gt_velx_int.numpy(), gt_vely_int.numpy(),
                obs_velx_phys.numpy(), obs_vely_phys.numpy(),
                mask_np,
            )
            gt_horizon_metrics[horizon_step].append(metrics)

    return gt_horizon_metrics


# ============================================================================
# Output: table printing and CSV saving
# ============================================================================

METRIC_KEYS = [
    'rmse_combined', 'rmse_velx', 'rmse_vely',
    'mae_combined', 'mae_velx', 'mae_vely',
    'cosine_similarity', 'relative_magnitude_error',
]


def _aggregate_horizon(metric_dicts):
    """Aggregate a list of per-start metric dicts into mean/std per metric."""
    agg = {}
    for key in METRIC_KEYS + ['num_interface_pixels']:
        vals = [m[key] for m in metric_dicts if not np.isnan(m[key])]
        if vals:
            agg[key] = {'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'count': len(vals)}
        else:
            agg[key] = {'mean': float('nan'), 'std': float('nan'), 'count': 0}
    return agg


def _fmt_mean_std(mean: float, std: float, ndigits: int = 3) -> str:
    """Format mean and std as 'mean±std' with fixed decimal places."""
    if np.isnan(mean):
        return 'nan'
    s = 0.0 if np.isnan(std) else std
    return f'{mean:.{ndigits}f}±{s:.{ndigits}f}'


def print_horizon_table(horizon_metrics, title="MEASUREMENT-SPACE CONSISTENCY"):
    """Print a clean horizon table to stdout."""
    sorted_horizons = sorted(horizon_metrics.keys())

    print("\n" + "=" * 100)
    print(f"  {title}")
    print(f"  H(x_pred) vs y_obs  |  observation operator = interface velocity extraction")
    print("=" * 100)

    print(f"\n  {'Horizon':>8}  {'RMSE':>18}  {'MAE':>18}  "
          f"{'Cos Sim':>18}  {'Rel Mag Err':>18}  {'N_pix':>18}  {'N':>4}")
    print(f"  {'-' * 110}")

    for h in sorted_horizons:
        metric_dicts = horizon_metrics[h]
        if not metric_dicts:
            print(f"  {h:>8}  {'(no data)':>18}")
            continue
        agg = _aggregate_horizon(metric_dicts)
        rmse = agg['rmse_combined']
        mae = agg['mae_combined']
        cos = agg['cosine_similarity']
        rel = agg['relative_magnitude_error']
        npx = agg['num_interface_pixels']
        print(f"  {h:>8}  "
              f"{_fmt_mean_std(rmse['mean'], rmse['std']):>18}  "
              f"{_fmt_mean_std(mae['mean'], mae['std']):>18}  "
              f"{_fmt_mean_std(cos['mean'], cos['std']):>18}  "
              f"{_fmt_mean_std(rel['mean'], rel['std']):>18}  "
              f"{_fmt_mean_std(npx['mean'], npx['std']):>18}  "
              f"{rmse['count']:>4}")

    # Also print per-component breakdown
    print(f"\n  Per-component RMSE breakdown:")
    print(f"  {'Horizon':>8}  {'RMSE_velx':>18}  {'RMSE_vely':>18}")
    print(f"  {'-' * 48}")
    for h in sorted_horizons:
        if not horizon_metrics[h]:
            continue
        agg = _aggregate_horizon(horizon_metrics[h])
        vx = agg['rmse_velx']
        vy = agg['rmse_vely']
        print(f"  {h:>8}  {_fmt_mean_std(vx['mean'], vx['std']):>18}  "
              f"{_fmt_mean_std(vy['mean'], vy['std']):>18}")


def save_horizon_csv(output_dir, horizon_metrics, raw_rows, gt_horizon_metrics, args):
    """Save horizon table, per-sample details, GT sanity check, and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    sorted_horizons = sorted(horizon_metrics.keys())

    # --- horizon_table.csv ---
    table_path = os.path.join(output_dir, 'horizon_table.csv')
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        cols = ['horizon'] + list(METRIC_KEYS) + ['num_interface_pixels', 'count']
        writer.writerow(cols)

        for h in sorted_horizons:
            if not horizon_metrics[h]:
                continue
            agg = _aggregate_horizon(horizon_metrics[h])
            row = [h]
            for key in METRIC_KEYS:
                row.append(_fmt_mean_std(agg[key]['mean'], agg[key]['std']))
            row.append(_fmt_mean_std(
                agg['num_interface_pixels']['mean'],
                agg['num_interface_pixels']['std'],
            ))
            row.append(agg['rmse_combined']['count'])
            writer.writerow(row)
    print(f"  Saved: {table_path}")

    # --- per_sample.csv ---
    if raw_rows:
        per_sample_path = os.path.join(output_dir, 'per_sample.csv')
        fieldnames = list(raw_rows[0].keys())
        with open(per_sample_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_rows:
                writer.writerow({k: (f'{v:.8f}' if isinstance(v, float) else v)
                                 for k, v in row.items()})
        print(f"  Saved: {per_sample_path}")

    # --- gt_sanity_check.csv ---
    if gt_horizon_metrics:
        gt_path = os.path.join(output_dir, 'gt_sanity_check.csv')
        with open(gt_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'horizon', 'rmse_combined', 'mae_combined',
                'cosine_similarity', 'relative_magnitude_error', 'count',
            ])
            for h in sorted_horizons:
                if not gt_horizon_metrics.get(h):
                    continue
                agg = _aggregate_horizon(gt_horizon_metrics[h])
                writer.writerow([
                    h,
                    _fmt_mean_std(agg['rmse_combined']['mean'], agg['rmse_combined']['std']),
                    _fmt_mean_std(agg['mae_combined']['mean'], agg['mae_combined']['std']),
                    _fmt_mean_std(
                        agg['cosine_similarity']['mean'],
                        agg['cosine_similarity']['std'],
                    ),
                    _fmt_mean_std(
                        agg['relative_magnitude_error']['mean'],
                        agg['relative_magnitude_error']['std'],
                    ),
                    agg['rmse_combined']['count'],
                ])
        print(f"  Saved: {gt_path}")

    # --- metrics.json ---
    json_path = os.path.join(output_dir, 'metrics.json')
    output = {
        'args': vars(args),
        'horizon_table': {
            str(h): _aggregate_horizon(horizon_metrics[h])
            for h in sorted_horizons if horizon_metrics[h]
        },
        'gt_sanity_check': {
            str(h): _aggregate_horizon(gt_horizon_metrics[h])
            for h in sorted_horizons if gt_horizon_metrics.get(h)
        } if gt_horizon_metrics else {},
    }
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {json_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Measurement-space consistency at long rollout horizons: H(x_pred) vs y_obs'
    )
    parser.add_argument('--checkpoint', type=str,
                        # default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/checkpoints/epoch=07-step=013280.ckpt",
                        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist64_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_52453282/checkpoints/epoch=09-step=016600.ckpt",
                        help='Path to model checkpoint')
    parser.add_argument(
        '--data-file', type=str, nargs='+',
        default=['/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5'],
        help='One or more HDF5 data files (must be a list: BulkFlowARBootstrap expects List[str])',
    )
    parser.add_argument('--output-dir', type=str, default='./ICML/CamReady/Table1_MeasurementConsistency',
                        help='Directory to save all output files (CSV, JSON)')
    parser.add_argument('--normalization-stats', type=str,
                        default="/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json",
                        help='Path to normalization_stats.json (auto-detected if None)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[5, 50, 100, 200, 300],
                        help='Rollout horizons (in AR steps) at which to evaluate')
    parser.add_argument('--num-starts', type=int, default=5,
                        help='Number of starting points for statistical averaging')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='ODE integration steps per AR step')
    parser.add_argument('--solver', type=str, default='rk4',
                        choices=['euler', 'heun', 'midpoint', 'rk4'])
    parser.add_argument('--start-time', type=int, default=500,
                        help='Dataset start time (skip initial transients)')
    parser.add_argument('--downsample-factor', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    sorted_horizons = sorted(args.horizons)
    max_horizon = sorted_horizons[-1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data files: {args.data_file}")
    print(f"Horizons: {sorted_horizons}")
    print(f"Num starts: {args.num_starts}")
    print(f"Output dir: {args.output_dir}")

    # ---- Load normalization stats ----
    if args.normalization_stats and os.path.exists(args.normalization_stats):
        with open(args.normalization_stats, 'r') as f:
            norm_stats = json.load(f)
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        if 'checkpoints' in ckpt_dir:
            ckpt_dir = os.path.dirname(ckpt_dir)
        stats_file = os.path.join(ckpt_dir, 'normalization_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                norm_stats = json.load(f)
            print(f"Loaded normalization stats from: {stats_file}")
        else:
            from bubblefusion.data.bubbleml import compute_normalization_stats
            norm_stats = compute_normalization_stats(
                args.data_file, args.start_time, verbose=True
            )

    norm_helper = NormalizationHelper(norm_stats, norm_mode='all')

    # ---- Load model ----
    # Use the checkpoint's saved model_cfg/task_cfg so the architecture matches
    # the state_dict.  Inference-only settings (solver / num_integration_steps)
    # are overridden from CLI args.
    print(f"\nLoading model...")
    model = load_ar_bootstrap_model(
        checkpoint_path=args.checkpoint,
        norm_stats=norm_stats,
        solver=args.solver,
        num_inference_steps=args.num_inference_steps,
    )
    model.eval()
    model = model.to(device)

    hist_len = getattr(model, 'history_length', 10)
    hist_stride = getattr(model, 'history_stride', 1)
    print(f"  History length: {hist_len}, stride: {hist_stride}")
    print(f"  Using rollout_length={max_horizon} for dataset (max horizon)")

    conditioning_names = list(model.task_cfg.get(
        'conditioning_names', ['sdf', 'velx_interface', 'vely_interface']
    ))

    # ---- Load dataset with rollout_length = max_horizon ----
    print(f"\nLoading dataset...")
    dataset = BulkFlowARBootstrap(
        filenames=args.data_file,
        output_fields=['temperature', 'velx', 'vely'],
        start_time=args.start_time,
        normalization_stats=norm_stats,
        history_length=hist_len,
        history_stride=hist_stride,
        rollout_length=max_horizon,
        downsample_factor=args.downsample_factor,
        norm_mode='all',
    )
    total_samples = len(dataset)
    print(f"  Dataset size: {total_samples}")

    if total_samples == 0:
        print("ERROR: No samples available. Check start_time and horizon length "
              "against data file length.")
        return

    # ---- Select non-overlapping starting points ----
    stride = max_horizon
    available_indices = list(range(0, total_samples, stride))
    num_starts = min(args.num_starts, len(available_indices))
    if num_starts < args.num_starts:
        print(f"  Warning: Only {len(available_indices)} non-overlapping starts available "
              f"(requested {args.num_starts}). Using {num_starts}.")
    rng = np.random.RandomState(args.seed)
    sample_indices = sorted(rng.choice(available_indices, size=num_starts, replace=False))
    print(f"  Selected {num_starts} starting points: {sample_indices}")

    # ---- Open raw HDF5 files ----
    h5_files = [h5.File(f, 'r') for f in args.data_file]
    fluid_params_list = []
    for data_path in args.data_file:
        params_path = data_path.replace('.hdf5', '.json')
        with open(params_path, 'r') as f:
            fluid_params_list.append(json.load(f))

    # ---- Run horizon evaluation ----
    print(f"\nRunning horizon evaluation...")
    print(f"  Horizons: {sorted_horizons}")
    print(f"  Starting points: {num_starts}")
    print(f"  Solver: {args.solver}, steps: {args.num_inference_steps}")
    print(f"  Max AR steps per trajectory: {max_horizon}\n")

    horizon_metrics, raw_rows = run_horizon_evaluation(
        model=model,
        dataset=dataset,
        h5_files=h5_files,
        fluid_params_list=fluid_params_list,
        norm_helper=norm_helper,
        device=device,
        sample_indices=sample_indices,
        horizons=sorted_horizons,
        num_steps=args.num_inference_steps,
        solver=args.solver,
        downsample_factor=args.downsample_factor,
    )

    print_horizon_table(horizon_metrics)

    # ---- GT Sanity check ----
    print("\n" + "=" * 100)
    print("  SANITY CHECK: GT bulk velocity -> H(x_GT) vs y_obs")
    print("  (Should show near-zero error if observation operator is consistent)")
    print("=" * 100)

    h5_files_gt = [h5.File(f, 'r') for f in args.data_file]
    gt_horizon_metrics = run_gt_sanity_check(
        dataset=dataset,
        h5_files=h5_files_gt,
        fluid_params_list=fluid_params_list,
        norm_helper=norm_helper,
        sample_indices=sample_indices,
        horizons=sorted_horizons,
        downsample_factor=args.downsample_factor,
        conditioning_names=conditioning_names,
    )
    for f_gt in h5_files_gt:
        f_gt.close()

    print_horizon_table(gt_horizon_metrics, title="GT SANITY CHECK")

    # ---- Close HDF5 and save ----
    for f in h5_files:
        f.close()

    print(f"\nSaving results to: {args.output_dir}/")
    save_horizon_csv(args.output_dir, horizon_metrics, raw_rows,
                     gt_horizon_metrics, args)

    print(f"\nDone. All results in: {args.output_dir}/")


if __name__ == '__main__':
    main()
