#!/usr/bin/env python3
"""
Noise Robustness Sweep: Evaluate model performance under varying input noise levels.

Loads a trained AR Bootstrap model once, then sweeps over multiple noise levels
(clean → very high), computing physics metrics at each level. Results are saved
to CSV and plotted as a multi-panel figure.

Addresses reviewer concern: "The paper provides insufficient study of robustness
under noisy or incomplete upstream observations."

Usage:
    python scripts/noise_robustness_sweep.py \
        --checkpoint /path/to/last.ckpt \
        --data-file /path/to/Twall_96.hdf5 \
        --output-dir ./noise_sweep_results

    # Quick test (fewer frames):
    python scripts/noise_robustness_sweep.py \
        --checkpoint /path/to/last.ckpt \
        --frame-end 5

    # Custom noise levels:
    python scripts/noise_robustness_sweep.py \
        --checkpoint /path/to/last.ckpt \
        --noise-levels 0,0:0.5,0.25:1,0.5:2,1:4,2:8,4
"""

import os
import sys
import json
import argparse
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bubblefusion.models.flow_matching_ar_bootstrap import (
    ConditionalFlowMatchingARBootstrapLightning,
)
from bubblefusion.data.bubbleml import compute_normalization_stats, BulkFlowARBootstrap

from scripts.physics_metrics_task123 import (
    load_task_config,
    load_model_from_checkpoint,
    load_dataset,
    load_ground_truth_data,
    run_ar_bootstrap_inference_batch,
    compute_velocity_divergence,
    compute_heatflux,
    compute_interface_mask_massflux,
)

# ============================================================================
# Default noise levels: (label, sdf_noise_std, vel_noise_std)
# ============================================================================
DEFAULT_NOISE_LEVELS = [
    ("Clean",     0.0, 0.0),
    ("Low",       0.5, 0.25),
    ("Medium",    1.0, 0.5),
    ("High",      2.0, 1.0),
    ("Very High", 4.0, 2.0),
]


def parse_noise_levels(spec: str) -> List[Tuple[str, float, float]]:
    """Parse user-supplied noise level specification.

    Format: ``sdf,vel:sdf,vel:...`` where the first pair is interpreted as
    "Clean" when both are 0.  Labels are auto-generated.
    """
    levels = []
    for i, pair_str in enumerate(spec.split(":")):
        parts = pair_str.strip().split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Each noise level must be 'sdf_std,vel_std', got '{pair_str}'"
            )
        sdf_std, vel_std = float(parts[0]), float(parts[1])
        if sdf_std == 0 and vel_std == 0:
            label = "Clean"
        else:
            label = f"σ_sdf={sdf_std},σ_vel={vel_std}"
        levels.append((label, sdf_std, vel_std))
    return levels


def build_noise_cfg(sdf_std: float, vel_std: float) -> Optional[Dict]:
    """Return a noise config dict for the dataset, or None when clean."""
    if sdf_std == 0 and vel_std == 0:
        return None
    return {
        "enabled": True,
        "noise_type": "gaussian",
        "sdf_noise_std": sdf_std,
        "vel_noise_std": vel_std,
    }


def compute_metrics_for_level(
    model,
    task_cfg,
    noise_cfg: Optional[Dict],
    data_file: str,
    normalization_stats: Dict,
    sdf_gt_full: np.ndarray,
    velx_interface_gt: np.ndarray,
    vely_interface_gt: np.ndarray,
    heater_temp: float,
    args,
) -> Dict[str, float]:
    """Run inference at one noise level and return the five target metrics."""

    dataset = load_dataset(
        data_file,
        output_fields=["temperature", "velx", "vely"],
        start_time=args.start_time,
        return_wall_temp=False,
        noise_cfg=noise_cfg,
        use_clean_inputs=False,
        is_temporal=False,
        is_autoregressive=False,
        is_ar_bootstrap=True,
        history_length=args.history_length,
        history_stride=args.history_stride,
        rollout_length=args.rollout_length,
        downsample_factor=args.downsample_factor,
        normalization_stats=normalization_stats,
        norm_mode=args.norm_mode,
    )

    rollout_length = dataset.rollout_length
    desired_num_frames = args.frame_end - args.frame_start
    num_segments = (desired_num_frames + rollout_length - 1) // rollout_length
    num_segments = min(num_segments, len(dataset) - args.frame_start)
    actual_num_frames = num_segments * rollout_length

    gt_velx, gt_vely, gt_temp, pred_velx, pred_vely, pred_temp = (
        run_ar_bootstrap_inference_batch(
            model,
            dataset,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_inference_steps=args.num_inference_steps,
            max_samples=num_segments,
            start_idx=args.frame_start,
            solver=args.solver,
        )
    )

    # --- Metric 1-3: RMSE for temp, velx, vely (overall) ---
    temp_rmse = float(np.sqrt(np.mean((gt_temp - pred_temp) ** 2)))
    velx_rmse = float(np.sqrt(np.mean((gt_velx - pred_velx) ** 2)))
    vely_rmse = float(np.sqrt(np.mean((gt_vely - pred_vely) ** 2)))

    # --- Metric 4: velocity divergence excluding interface ---
    sdf_slice = sdf_gt_full[args.frame_start : args.frame_start + actual_num_frames]
    velx_int_slice = velx_interface_gt[args.frame_start : args.frame_start + actual_num_frames]
    vely_int_slice = vely_interface_gt[args.frame_start : args.frame_start + actual_num_frames]

    pred_div = compute_velocity_divergence(
        pred_velx, pred_vely, downsample_factor=args.downsample_factor
    )
    interface_mask = compute_interface_mask_massflux(velx_int_slice, vely_int_slice)
    non_interface = ~interface_mask
    div_excl_interface = float(np.mean(np.abs(pred_div[non_interface])))

    # --- Metric 5: wall heat flux relative error ---
    gt_hflux = compute_heatflux(
        sdf_slice, gt_temp, heater_temp, downsample_factor=args.downsample_factor
    )
    pred_hflux = compute_heatflux(
        sdf_slice, pred_temp, heater_temp, downsample_factor=args.downsample_factor
    )
    gt_hflux_mean = np.mean(gt_hflux)
    if gt_hflux_mean != 0:
        wall_hf_rel_error = float(
            np.mean(np.abs(gt_hflux - pred_hflux)) / np.abs(gt_hflux_mean) * 100
        )
    else:
        wall_hf_rel_error = float("nan")

    return {
        "temp_rmse": temp_rmse,
        "velx_rmse": velx_rmse,
        "vely_rmse": vely_rmse,
        "div_excl_interface_pred_mean": div_excl_interface,
        "wall_heat_flux_rel_error": wall_hf_rel_error,
    }


def save_csv(
    results: List[Dict],
    output_path: str,
) -> None:
    """Write the sweep results to a CSV file."""
    fieldnames = [
        "noise_level",
        "sdf_noise_std",
        "vel_noise_std",
        "temp_rmse",
        "velx_rmse",
        "vely_rmse",
        "div_excl_interface_pred_mean",
        "wall_heat_flux_rel_error",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nCSV saved to: {output_path}")


def make_plots(
    results: List[Dict],
    output_path: str,
) -> None:
    """Generate a multi-panel plot of metrics vs noise level."""
    labels = [r["noise_level"] for r in results]
    x = np.arange(len(labels))

    metrics_info = [
        ("temp_rmse", "Temperature RMSE (°C)", "#2563EB"),
        ("velx_rmse", "Vel-x RMSE", "#DC2626"),
        ("vely_rmse", "Vel-y RMSE", "#16A34A"),
        ("div_excl_interface_pred_mean", "Divergence (excl. interface)", "#9333EA"),
        ("wall_heat_flux_rel_error", "Wall Heat Flux Rel. Error (%)", "#EA580C"),
    ]

    title_fs = 25
    axis_label_fs = 27
    tick_fs = 24

    fig, axes = plt.subplots(1, 5, figsize=(32, 8), constrained_layout=True)

    for ax, (key, title, color) in zip(axes, metrics_info):
        values = [float(r[key]) for r in results]
        ax.bar(x, values, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.plot(x, values, "o-", color="black", markersize=5, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=tick_fs)
        ax.set_title(title, fontsize=title_fs, fontweight="bold")
        ax.set_ylabel(title.split("(")[0].strip(), fontsize=axis_label_fs)
        ax.set_xlabel("Noise Level", fontsize=axis_label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Noise robustness sweep for AR Bootstrap model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/share/crsp/lab/amowli/xianwz2/logs/bubblefusion_logs/"
        "flow_matching_ar_bootstrap_32_256_2_False_hist10_roll5_attn_d128_L2_p8_"
        "velocity_from_interface_pb_subcooled_singlestep_none_ds4_50237283/"
        "checkpoints/last.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="/share/crsp/lab/amowli/share/BubbleML_2/"
        "PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5",
        help="Path to HDF5 data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ICML/noise_sweep_results_100",
        help="Output directory for CSV and plot",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default=None,
        help="Custom noise levels as 'sdf,vel:sdf,vel:...' (e.g. '0,0:1,0.5:2,1'). "
        "Overrides default 5-level sweep.",
    )

    parser.add_argument("--start-time", type=int, default=100)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=100)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--norm-mode", type=str, default="all")
    parser.add_argument("--solver", type=str, default="rk4",
                        choices=["euler", "heun", "midpoint", "rk4"])

    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--history-stride", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--history-encoder-type", type=str, default="attention",
                        choices=["conv3d", "temporal_mixer", "attention"])
    parser.add_argument("--history-encoder-hidden", type=int, default=32)
    parser.add_argument("--attention-encoder-embed-dim", type=int, default=128)
    parser.add_argument("--attention-encoder-num-heads", type=int, default=8)
    parser.add_argument("--attention-encoder-depth", type=int, default=2)
    parser.add_argument("--attention-encoder-patch-size", type=int, default=8)
    parser.add_argument("--attention-encoder-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attention-encoder-dropout", type=float, default=0.0)
    parser.add_argument("--attention-encoder-output-head", type=str, default="linear")
    parser.add_argument("--attention-encoder-max-history-length", type=int, default=50)

    args = parser.parse_args()

    # Noise levels
    if args.noise_levels is not None:
        noise_levels = parse_noise_levels(args.noise_levels)
    else:
        noise_levels = DEFAULT_NOISE_LEVELS

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  NOISE ROBUSTNESS SWEEP")
    print("=" * 70)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Data file:   {args.data_file}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Frames:      [{args.frame_start}:{args.frame_end}]")
    print(f"Downsample:  {args.downsample_factor}x")
    print(f"Noise levels ({len(noise_levels)}):")
    for label, sdf_s, vel_s in noise_levels:
        print(f"   {label:12s}  sdf_std={sdf_s:.2f}  vel_std={vel_s:.2f}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load task config
    # ------------------------------------------------------------------
    task_cfg = load_task_config("velocity_from_interface")

    # ------------------------------------------------------------------
    # Step 2: Build model config (matches training script defaults)
    # ------------------------------------------------------------------
    model_cfg = DictConfig({
        "name": "flow_matching_ar_bootstrap",
        "in_channels": 10,
        "out_channels": 3,
        "base_channels": 32,
        "time_embed_dim": 256,
        "num_res_blocks": 2,
        "use_attention": False,
        "dropout": 0.1,
        "num_integration_steps": args.num_inference_steps,
        "temp_min": 55.0,
        "temp_max": 120.0,
        "history_length": args.history_length,
        "rollout_length": args.rollout_length,
        "use_availability_mask": True,
        "history_encoder_type": args.history_encoder_type,
        "history_encoder_hidden": args.history_encoder_hidden,
        "attention_encoder_embed_dim": args.attention_encoder_embed_dim,
        "attention_encoder_num_heads": args.attention_encoder_num_heads,
        "attention_encoder_depth": args.attention_encoder_depth,
        "attention_encoder_patch_size": args.attention_encoder_patch_size,
        "attention_encoder_mlp_ratio": args.attention_encoder_mlp_ratio,
        "attention_encoder_dropout": args.attention_encoder_dropout,
        "attention_encoder_output_head": args.attention_encoder_output_head,
        "attention_encoder_max_history_length": args.attention_encoder_max_history_length,
        "bootstrap_loss_weight": 1.0,
        "ar_loss_weight": 1.0,
        "bootstrap_state_loss_weight": 0.5,
        "inference": {"solver": args.solver},
    })
    optim_cfg = DictConfig({"name": "adamw", "lr": 0.001})
    scheduler_cfg = DictConfig({"name": "cosine"})

    # ------------------------------------------------------------------
    # Step 3: Load normalization stats
    # ------------------------------------------------------------------
    normalization_stats = None
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if "checkpoints" in checkpoint_dir:
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    stats_file = os.path.join(checkpoint_dir, "normalization_stats.json")

    if os.path.exists(stats_file):
        print(f"\nLoading normalization stats from: {stats_file}")
        with open(stats_file, "r") as f:
            normalization_stats = json.load(f)
    else:
        print(f"\nnormalization_stats.json not found in checkpoint dir, computing from data...")
        normalization_stats = compute_normalization_stats(
            filenames=[args.data_file],
            start_time=args.start_time,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # Step 4: Load model (once)
    # ------------------------------------------------------------------
    model = load_model_from_checkpoint(
        args.checkpoint,
        model_cfg,
        optim_cfg,
        scheduler_cfg,
        task_cfg,
        model_type="flow_matching_ar_bootstrap",
        normalization_stats=normalization_stats,
        norm_mode=args.norm_mode,
    )

    if hasattr(model, "history_length") and model.history_length != args.history_length:
        print(f"Overriding --history-length {args.history_length} -> {model.history_length} (checkpoint)")
        args.history_length = model.history_length
    if hasattr(model, "history_stride") and model.history_stride != args.history_stride:
        print(f"Overriding --history-stride {args.history_stride} -> {model.history_stride} (checkpoint)")
        args.history_stride = model.history_stride
    if hasattr(model, "rollout_length") and model.rollout_length != args.rollout_length:
        print(f"Overriding --rollout-length {args.rollout_length} -> {model.rollout_length} (checkpoint)")
        args.rollout_length = model.rollout_length

    # ------------------------------------------------------------------
    # Step 5: Load ground truth data (once)
    # ------------------------------------------------------------------
    sdf_gt, _, _, _, velx_interface_gt, vely_interface_gt, heater_temp = (
        load_ground_truth_data(args.data_file, args.start_time, args.downsample_factor)
    )

    # ------------------------------------------------------------------
    # Step 6: Sweep over noise levels
    # ------------------------------------------------------------------
    all_results: List[Dict] = []

    for level_idx, (label, sdf_std, vel_std) in enumerate(noise_levels):
        print(f"\n{'='*70}")
        print(f"  [{level_idx+1}/{len(noise_levels)}] Noise level: {label}  "
              f"(sdf_std={sdf_std:.2f}, vel_std={vel_std:.2f})")
        print(f"{'='*70}")

        noise_cfg = build_noise_cfg(sdf_std, vel_std)

        if noise_cfg is not None:
            print(f"  Noise config: {noise_cfg}")
        else:
            print("  Clean inputs (no noise)")

        metrics = compute_metrics_for_level(
            model=model,
            task_cfg=task_cfg,
            noise_cfg=noise_cfg,
            data_file=args.data_file,
            normalization_stats=normalization_stats,
            sdf_gt_full=sdf_gt,
            velx_interface_gt=velx_interface_gt,
            vely_interface_gt=vely_interface_gt,
            heater_temp=heater_temp,
            args=args,
        )

        print(f"\n  Results for '{label}':")
        print(f"    temp_rmse                  = {metrics['temp_rmse']:.6f}")
        print(f"    velx_rmse                  = {metrics['velx_rmse']:.6f}")
        print(f"    vely_rmse                  = {metrics['vely_rmse']:.6f}")
        print(f"    div_excl_interface_pred_mean = {metrics['div_excl_interface_pred_mean']:.6f}")
        print(f"    wall_heat_flux_rel_error   = {metrics['wall_heat_flux_rel_error']:.2f}%")

        row = {
            "noise_level": label,
            "sdf_noise_std": sdf_std,
            "vel_noise_std": vel_std,
            **metrics,
        }
        all_results.append(row)

    # ------------------------------------------------------------------
    # Step 7: Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "noise_robustness_results.csv")
    save_csv(all_results, csv_path)

    # ------------------------------------------------------------------
    # Step 8: Generate plot
    # ------------------------------------------------------------------
    plot_path = os.path.join(args.output_dir, "noise_robustness_plot.png")
    make_plots(all_results, plot_path)

    print(f"\nDone! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
