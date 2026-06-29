#!/usr/bin/env python3
"""
History Length / Stride Ablation for AR Bootstrap Flow Matching (Task 2).

Supports two ablation modes:
  --mode history_length : vary history length (H=3,5,7,9,...) with fixed stride
  --mode stride         : vary history stride (S=1,2,3,4,5) with fixed history length

Loops over the selected checkpoint group, runs inference + physics metrics,
collects results into a comparison CSV table, and generates plots.

Reuses functions from physics_metrics_task123.py to avoid code duplication.

Usage:
    python scripts/history_length_ablation.py --mode history_length --seed 42
    python scripts/history_length_ablation.py --mode stride --seed 42
    python scripts/history_length_ablation.py --mode stride --output-dir ./ICML/stride_ablation
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.physics_metrics_task123 import (
    load_task_config,
    load_dataset,
    load_ground_truth_data,
    run_ar_bootstrap_inference_batch,
    compute_all_physics_metrics,
)
from bubblefusion.models.flow_matching_ar_bootstrap import ConditionalFlowMatchingARBootstrapLightning
from bubblefusion.data.bubbleml import compute_normalization_stats

# ============================================================================
# Checkpoint registries
# ============================================================================

# --- History length ablation (varying H, fixed stride=1) ---
HISTORY_LENGTH_CHECKPOINTS = {
    # Temporal Mixer
    # 5:  "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist5_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230575/checkpoints/epoch=21-step=018260.ckpt",
    # 10: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230563/checkpoints/epoch=21-step=018260.ckpt",
    # 15: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist15_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50249983/checkpoints/epoch=21-step=018260.ckpt",
    # 20: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist20_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230602/checkpoints/epoch=21-step=018260.ckpt",
    # 30: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist30_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230609/checkpoints/epoch=21-step=018260.ckpt",
    # 40: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist40_roll5_tmix_spatial_tweights_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50230611/checkpoints/last.ckpt",

    # Attention
    # 2:  "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist2_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251763/checkpoints/epoch=07-step=013280.ckpt",
    3: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist3_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251761/checkpoints/epoch=07-step=013280.ckpt",
    5: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist5_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251759/checkpoints/epoch=07-step=013280.ckpt",
    7: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist7_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251762/checkpoints/epoch=07-step=013280.ckpt",
    9: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist9_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50251758/checkpoints/epoch=07-step=013280.ckpt",
    10: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/checkpoints/epoch=07-step=013280.ckpt",
}

# --- Stride ablation (fixed H=10, varying stride) ---
STRIDE_CHECKPOINTS = {
    1: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265190/checkpoints/last.ckpt",
    2: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265193/checkpoints/last.ckpt",
    3: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265198/checkpoints/last.ckpt",
    4: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265203/checkpoints/last.ckpt",
    5: "/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_velocity_from_interface_pb_subcooled_singlestep_none_ds4_50265204/checkpoints/last.ckpt",
}

# Metrics to extract from the full metrics dict
METRIC_KEYS = [
    ("Interface Temp Error (°C)",      lambda m: m['interface_temperature']['error_mean']),
    ("Wall Heat Flux Rel. Error (%)",   lambda m: m['wall_heat_flux']['relative_error']),
    ("Vel. Divergence (excl. intf.)",   lambda m: m['velocity_divergence_excl_interface']['pred_mean']),
    ("Vorticity RMSE (excl. intf.)",    lambda m: m['vorticity_excluding_interface']['error_rmse']),
    ("Temp MAE (bulk liquid, °C)",      lambda m: m['region_errors']['temperature']['bulk_liquid']['mae']),
    ("Velocity MAE (bulk liquid)",      lambda m: _avg_vel_mae(m)),
]


def _avg_vel_mae(metrics: dict) -> float:
    """Average of velx and vely bulk-liquid MAE."""
    velx_mae = metrics['region_errors']['velx']['bulk_liquid']['mae']
    vely_mae = metrics['region_errors']['vely']['bulk_liquid']['mae']
    return (velx_mae + vely_mae) / 2.0


def load_normalization_stats(checkpoint_path: str, explicit_path: str = None,
                             data_file: str = None, start_time: int = 100) -> dict:
    """Load normalization stats with priority: explicit > checkpoint dir > compute."""
    if explicit_path and os.path.exists(explicit_path):
        print(f"   Loading normalization stats from: {explicit_path}")
        with open(explicit_path, 'r') as f:
            return json.load(f)

    ckpt_dir = os.path.dirname(checkpoint_path)
    if 'checkpoints' in ckpt_dir:
        ckpt_dir = os.path.dirname(ckpt_dir)
    stats_file = os.path.join(ckpt_dir, 'normalization_stats.json')
    if os.path.exists(stats_file):
        print(f"   Loading normalization stats from training dir: {stats_file}")
        with open(stats_file, 'r') as f:
            return json.load(f)

    if data_file:
        print(f"   Computing normalization stats from data file (may not match training)...")
        return compute_normalization_stats(filenames=[data_file], start_time=start_time, verbose=False)

    raise FileNotFoundError("Cannot find or compute normalization stats")


# ============================================================================
# Plotting
# ============================================================================

def generate_plots(df: pd.DataFrame, output_dir: str, x_col: str,
                    x_label: str, title: str, filename: str):
    """Generate 2x3 subplot figure: each metric vs. the ablation variable."""
    x_values = df[x_col].values
    metric_cols = [c for c in df.columns if c != x_col]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(title, fontsize=16, y=0.98)

    for idx, col in enumerate(metric_cols):
        ax = axes[idx // 3, idx % 3]
        values = df[col].values
        ax.plot(x_values, values, 'o-', color='#2563eb', linewidth=2, markersize=8)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xticks(x_values)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to: {plot_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='History Length Ablation for AR Bootstrap (Task 2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mode', type=str, default='history_length',
                        choices=['history_length', 'stride'],
                        help='Ablation mode: history_length (vary H) or stride (vary stride, fixed H=10)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save CSV table and plots (auto-set per mode if omitted)')
    parser.add_argument('--data-file', type=str,
                        default='/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5',
                        help='Path to HDF5 data file')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Starting frame index for metrics (0-based)')
    parser.add_argument('--frame-end', type=int, default=10,
                        help='Ending frame index (exclusive)')
    parser.add_argument('--start-time', type=int, default=300,
                        help='Starting timestep in HDF5 file')
    parser.add_argument('--downsample-factor', type=int, default=4,
                        help='Spatial downsampling factor (1=full 512, 4=128)')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='Number of ODE integration steps per frame')
    parser.add_argument('--solver', type=str, default='heun',
                        choices=['euler', 'heun', 'midpoint', 'rk4'],
                        help='ODE solver')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--normalization-stats', type=str,
                        default='/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json',
                        help='Path to normalization_stats.json')
    parser.add_argument('--norm-mode', type=str, default='all',
                        choices=['none', 'all', 'temperature_only'],
                        help='Normalization mode (must match training)')
    parser.add_argument('--row-indices', type=str, default='0,8,16,24,32',
                        help='Comma-separated row indices for temperature analysis')
    args = parser.parse_args()

    if args.output_dir is None:
        if args.mode == 'stride':
            args.output_dir = './ICML/stride_ablation/attention/400'
        else:
            args.output_dir = './ICML/history_length_ablation/attention'
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {args.seed} (deterministic mode)")

    row_indices = [int(x.strip()) for x in args.row_indices.split(',')]

    # ----------------------------------------------------------------
    # Load shared resources (task config, ground truth SDF, heater temp)
    # ----------------------------------------------------------------
    task_cfg = load_task_config('velocity_from_interface')

    sdf_gt_full, _, _, _, _, _, heater_temp = load_ground_truth_data(
        args.data_file, args.start_time, args.downsample_factor
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Select checkpoint group and labels based on mode
    if args.mode == 'stride':
        checkpoints = STRIDE_CHECKPOINTS
        x_col = 'History Stride'
        x_label = 'History Stride'
        plot_title = 'Stride Ablation — AR Bootstrap (H=10, Task 2)'
        plot_filename = 'stride_ablation.png'
        csv_filename = 'stride_ablation.csv'
        var_label = 'S'
    else:
        checkpoints = HISTORY_LENGTH_CHECKPOINTS
        x_col = 'History Length'
        x_label = 'History Length'
        plot_title = 'History Length Ablation — AR Bootstrap (Task 2)'
        plot_filename = 'history_length_ablation.png'
        csv_filename = 'history_length_ablation.csv'
        var_label = 'H'

    sorted_keys = sorted(checkpoints.keys())
    total = len(sorted_keys)
    all_results = []

    print(f"\nAblation mode: {args.mode} ({x_label}={sorted_keys})")

    # ----------------------------------------------------------------
    # Main loop: iterate over ablation variable
    # ----------------------------------------------------------------
    for run_idx, key_val in enumerate(sorted_keys):
        ckpt_path = checkpoints[key_val]

        print("\n" + "=" * 70)
        print(f"[{run_idx + 1}/{total}] {x_label} = {key_val}")
        print(f"Checkpoint: {ckpt_path}")
        print("=" * 70)

        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found, skipping {var_label}={key_val}")
            continue

        # Load normalization stats
        norm_stats = load_normalization_stats(
            ckpt_path, args.normalization_stats, args.data_file, args.start_time
        )

        # Load model directly from checkpoint (uses saved hyperparameters)
        model = ConditionalFlowMatchingARBootstrapLightning.load_from_checkpoint(
            ckpt_path,
            normalization_stats=norm_stats,
            strict=True,
        )
        encoder_cls = type(model.flow_matching.history_encoder).__name__
        print(f"   History encoder: {encoder_cls}")
        print(f"   History length: {model.history_length} (from checkpoint)")
        print(f"   History stride: {model.history_stride} (from checkpoint)")
        print(f"   Rollout length: {model.rollout_length} (from checkpoint)")
        print(f"   Use availability mask: {model.use_availability_mask} (from checkpoint)")
        print(f"   Default solver: {model.default_solver}")

        # Load dataset with history_length, stride, and rollout_length from checkpoint
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

        # Determine segment count from frame range
        rollout_length = dataset.rollout_length
        desired_frames = args.frame_end - args.frame_start
        num_segments = (desired_frames + rollout_length - 1) // rollout_length
        num_segments = min(num_segments, len(dataset) - args.frame_start)
        actual_frames = num_segments * rollout_length

        # Slice ground truth SDF
        sdf_slice = sdf_gt_full[args.frame_start:args.frame_start + actual_frames]

        print(f"   Segments: {num_segments}, frames: {actual_frames}")

        # Run inference
        gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp = (
            run_ar_bootstrap_inference_batch(
                model, dataset, device, args.num_inference_steps,
                max_samples=num_segments,
                start_idx=args.frame_start,
                solver=args.solver,
            )
        )

        # Align SDF with actual generated frames
        gen_frames = gt_temp.shape[0] if gt_temp is not None else 0
        if sdf_slice.shape[0] > gen_frames:
            sdf_slice = sdf_slice[:gen_frames]
        elif sdf_slice.shape[0] < gen_frames:
            gt_velx = gt_velx[:sdf_slice.shape[0]]
            gt_vely = gt_vely[:sdf_slice.shape[0]]
            gt_temp = gt_temp[:sdf_slice.shape[0]]
            pred_velx = pred_velx[:sdf_slice.shape[0]]
            pred_vely = pred_vely[:sdf_slice.shape[0]]
            pred_temp = pred_temp[:sdf_slice.shape[0]]

        # Compute physics metrics
        metrics = compute_all_physics_metrics(
            gt_velx, gt_vely, gt_temp,
            pred_velx, pred_vely, pred_temp,
            sdf_slice, heater_temp, row_indices,
            downsample_factor=args.downsample_factor,
        )

        # Extract key metrics
        row = {x_col: key_val}
        print(f"\n   --- Key Metrics ({var_label}={key_val}) ---")
        for name, extractor in METRIC_KEYS:
            try:
                val = extractor(metrics)
                row[name] = val
                print(f"   {name}: {val:.6f}")
            except (KeyError, TypeError) as e:
                row[name] = np.nan
                print(f"   {name}: N/A ({e})")

        all_results.append(row)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Build comparison table
    # ----------------------------------------------------------------
    if not all_results:
        print("\nNo results collected — check checkpoint paths.")
        sys.exit(1)

    df = pd.DataFrame(all_results)
    df = df.sort_values(x_col).reset_index(drop=True)

    csv_path = os.path.join(args.output_dir, csv_filename)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print(f"{args.mode.upper().replace('_', ' ')} ABLATION — COMPARISON TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"\nCSV saved to: {csv_path}")

    # ----------------------------------------------------------------
    # Generate plots
    # ----------------------------------------------------------------
    generate_plots(df, args.output_dir, x_col, x_label, plot_title, plot_filename)

    print(f"\n{args.mode.replace('_', ' ').title()} ablation complete.")


if __name__ == '__main__':
    main()
