#!/usr/bin/env python3
"""
Single-frame Task 1 comparison figure (temperature_from_sdf).

For one BulkFlow index (--frame), runs inference with every model in
inference_metrics_task1.MODEL_REGISTRY and saves a 2-row panel where
column 0 carries the inputs / GT and the remaining columns are split
into a top and bottom row of model temperature predictions:

  Row 0 (top):    SDF      | FFNO    | U-Net      | DDPM          | VE-SDE
  Row 1 (bottom): Temp GT  | HB-ARFM | HistoryFM  | Flow Matching | DiffusionPDE

No physics metrics are computed or drawn on the figure.

Usage:
    python scripts/generate_task1_figures.py --frame 100
    python scripts/generate_task1_figures.py --frame 100 --no-range-titles
    python scripts/generate_task1_figures.py --frame 100 --models fm historyfm
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
from collections import OrderedDict
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bubblefusion.data import BulkFlow
from scripts.generate_paper_figures import temp_cmap
from scripts.inference_metrics_task1 import (
    MODEL_REGISTRY,
    build_model_cfg,
    load_normalization_stats,
)
from scripts.physics_metrics_task123 import (
    load_dataset,
    load_model_from_checkpoint,
    load_task_config,
    run_ar_bootstrap_inference_batch,
    run_inference_batch,
)

DEFAULT_DATA_FILE = (
    "/share/crsp/lab/amowli/share/BubbleML_2/PoolBoiling-Subcooled-FC72-2D/Twall_96.hdf5"
)
DEFAULT_NORMALIZATION_STATS = (
    "/share/crsp/lab/amowli/xianwz2/diffusion/bubblefusion/normalization_stats.json"
)

# Short keys map to MODEL_REGISTRY display names (see inference_metrics_task1.MODEL_NAMES).
SHORT_KEYS = {
    "fm": "Flow Matching",
    "historyfm": "HistoryFM",
    "hbarfm": "HB-ARFM",
    "vesde": "VE-SDE",
    "diffpde": "DiffusionPDE",
    "dpde": "DiffusionPDE",
    "ddpm": "DDPM",
    "unet": "U-Net",
    "ffno": "FFNO",
}

# Fixed two-row panel layout (left-to-right within each row).
TOP_ROW_MODELS = ["FFNO", "U-Net", "DDPM", "VE-SDE"]
BOTTOM_ROW_MODELS = ["HB-ARFM", "HistoryFM", "Flow Matching", "DiffusionPDE"]


def _resolve_models_to_run(model_args: list[str] | None) -> OrderedDict:
    """Return the subset of MODEL_REGISTRY actually requested by the user.

    The two-row panel layout positions are fixed (TOP_ROW_MODELS /
    BOTTOM_ROW_MODELS), so a subset just leaves the corresponding cells blank.
    """
    if model_args is None:
        names = list(MODEL_REGISTRY.keys())
    else:
        names = []
        for key in model_args:
            full = SHORT_KEYS.get(key.lower(), key)
            if full in MODEL_REGISTRY:
                names.append(full)
            else:
                print(
                    f"WARNING: Unknown model key '{key}', skipping. "
                    f"Valid: {list(SHORT_KEYS.keys())}"
                )
    return OrderedDict((k, MODEL_REGISTRY[k]) for k in names)


def _inference_args_ns(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        num_inference_steps=args.num_inference_steps,
        solver=args.solver,
        downsample_factor=args.downsample_factor,
    )


def _run_single_model_frame(
    model_name: str,
    model_type: str,
    checkpoint_path: str,
    task_cfg: DictConfig,
    optim_cfg: DictConfig,
    scheduler_cfg: DictConfig,
    frame_idx: int,
    timestep: int,
    args: argparse.Namespace,
    device: str,
) -> np.ndarray | None:
    """Returns predicted temperature (H, W) in physical units, or None."""
    if not os.path.exists(checkpoint_path):
        print(f"  SKIP {model_name}: checkpoint missing\n {checkpoint_path}")
        return None

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    norm_stats = load_normalization_stats(checkpoint_path, args.normalization_stats)
    cfg_ns = _inference_args_ns(args)
    model_cfg = build_model_cfg(
        model_type, task_cfg, cfg_ns, checkpoint_path=checkpoint_path
    )

    try:
        model = load_model_from_checkpoint(
            checkpoint_path,
            model_cfg,
            optim_cfg,
            scheduler_cfg,
            task_cfg,
            model_type=model_type,
            normalization_stats=norm_stats,
            norm_mode=args.norm_mode,
        )
    except Exception as e:
        print(f"  SKIP {model_name}: failed to load model: {e}")
        return None

    is_ar_bootstrap = model_type in ("flow_matching_ar_bootstrap", "edm_ar_bootstrap")
    is_history = model_type == "flow_matching_history"

    history_length = getattr(model, "history_length", 10)
    history_stride = getattr(model, "history_stride", 1)
    rollout_length = getattr(model, "rollout_length", 5)
    history_window = getattr(model, "history_window", 10) if is_history else 10

    try:
        dataset = load_dataset(
            args.data_file,
            output_fields=["temperature", "velx", "vely"],
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
    except Exception as e:
        del model
        torch.cuda.empty_cache()
        print(f"  SKIP {model_name}: dataset error: {e}")
        return None

    try:
        if is_ar_bootstrap:
            eff = dataset.effective_start_time
            seg_idx = timestep - eff
            if seg_idx < 0:
                print(
                    f"  WARN {model_name}: seg_idx={seg_idx} < 0; clamping to 0 "
                    f"(timestep={timestep}, effective_start={eff})"
                )
                seg_idx = 0
            if seg_idx >= len(dataset):
                seg_idx = len(dataset) - 1
                print(
                    f"  WARN {model_name}: seg_idx clamped to {seg_idx} "
                    f"(len={len(dataset)})"
                )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                _, _, _, _, _, pred_temp = run_ar_bootstrap_inference_batch(
                    model,
                    dataset,
                    device,
                    args.num_inference_steps,
                    max_samples=1,
                    start_idx=seg_idx,
                    solver=args.solver,
                )
            pred_temp = pred_temp[0]
        else:
            idx = frame_idx
            if idx >= len(dataset):
                idx = len(dataset) - 1
                print(
                    f"  WARN {model_name}: frame_idx {frame_idx} -> {idx} "
                    f"(dataset len {len(dataset)})"
                )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                _, _, _, _, _, pred_temp = run_inference_batch(
                    model,
                    dataset,
                    device,
                    args.num_inference_steps,
                    max_samples=1,
                    model_type=model_type,
                    start_idx=idx,
                    solver=args.solver,
                )
            pred_temp = pred_temp[0]

        return pred_temp
    except Exception as e:
        print(f"  SKIP {model_name}: inference error: {e}")
        return None
    finally:
        del model
        torch.cuda.empty_cache()


def generate_task1_panel(args: argparse.Namespace) -> str:
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    models_to_run = _resolve_models_to_run(args.models)
    if not models_to_run:
        raise SystemExit("No models selected.")

    task_cfg = load_task_config("temperature_from_sdf")
    target_channels = list(task_cfg.target_channels)

    with open(args.normalization_stats, "r") as f:
        ref_norm = json.load(f)

    ref_ds = BulkFlow(
        filenames=[args.data_file],
        output_fields=["temperature", "velx", "vely"],
        start_time=args.start_time,
        normalization_stats=ref_norm,
        downsample_factor=args.downsample_factor,
        norm_mode=args.norm_mode,
    )

    frame_idx = args.frame
    if frame_idx >= len(ref_ds):
        frame_idx = len(ref_ds) - 1
        print(f"Frame index too large; using {frame_idx}")
    if frame_idx < 0:
        frame_idx = 0
    timestep = ref_ds.start_time + frame_idx

    input_data, target_data = ref_ds[frame_idx]
    gt_temp = ref_ds._denormalize_field(
        target_data[target_channels[0]].clone(), "temperature"
    ).numpy()
    sdf = ref_ds._denormalize_field(input_data[0].clone(), "sdf").numpy()

    temp_vmin_phys, temp_vmax_phys = float(gt_temp.min()), float(gt_temp.max())
    sdf_vmin, sdf_vmax = float(sdf.min()), float(sdf.max())

    temp_vmin = args.temp_vmin

    optim_cfg = DictConfig({"name": "adamw", "lr": 0.001})
    scheduler_cfg = DictConfig({"name": "cosine"})

    predictions: dict[str, np.ndarray | None] = {}
    for model_name, info in models_to_run.items():
        print(f"\n--- {model_name} ({info['model_type']}) ---")
        out = _run_single_model_frame(
            model_name,
            info["model_type"],
            info["checkpoint"],
            task_cfg,
            optim_cfg,
            scheduler_cfg,
            frame_idx,
            timestep,
            args,
            device,
        )
        predictions[model_name] = out

    n_model_cols = max(len(TOP_ROW_MODELS), len(BOTTOM_ROW_MODELS))
    width_ratios = [1, args.col_gap] + [1] * n_model_cols
    n_cols = len(width_ratios)

    fig = plt.figure(figsize=(args.fig_width, args.fig_height))
    gs = gridspec.GridSpec(
        2,
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.03,
        hspace=args.hspace,
    )

    label_fontsize = args.fontsize
    range_fontsize = args.range_fontsize
    last_temp_im = None

    def plot_temp(ax, data: np.ndarray, show_range: bool) -> None:
        nonlocal last_temp_im
        data_n = (data - temp_vmin_phys) / (temp_vmax_phys - temp_vmin_phys)
        data_n = np.clip(data_n, 0, 1)
        im = ax.imshow(data_n, cmap=temp_cmap(), origin="lower", vmin=temp_vmin, vmax=1.0)
        ax.axis("off")
        last_temp_im = im
        if show_range and args.range_titles:
            ax.set_title(
                f"[{data.min():.1f}, {data.max():.1f}]°C",
                fontsize=range_fontsize,
                pad=2,
                color="black",
            )

    def plot_sdf(ax, data: np.ndarray) -> None:
        norm = TwoSlopeNorm(vcenter=0, vmin=sdf_vmin, vmax=sdf_vmax)
        ax.imshow(data, cmap="RdYlBu", norm=norm, origin="lower")
        contour_levels = np.linspace(sdf_vmin, sdf_vmax, 10)
        ax.contour(
            data,
            levels=contour_levels,
            colors="white",
            linewidths=0.6,
            linestyles="dotted",
            alpha=0.5,
        )
        ax.contour(data, levels=[0], colors="black", alpha=0.8, linewidths=1.2)
        ax.axis("off")

    ax_sdf = fig.add_subplot(gs[0, 0])
    plot_sdf(ax_sdf, sdf)
    ax_sdf.text(
        0.5,
        args.label_offset,
        "SDF",
        transform=ax_sdf.transAxes,
        fontsize=label_fontsize,
        fontweight="bold",
        ha="center",
    )

    ax_gt = fig.add_subplot(gs[1, 0])
    plot_temp(ax_gt, gt_temp, show_range=True)
    ax_gt.text(
        0.5,
        args.label_offset,
        "Ground Truth",
        transform=ax_gt.transAxes,
        fontsize=label_fontsize,
        fontweight="bold",
        ha="center",
    )

    def _draw_model_row(row: int, names: list[str]) -> None:
        for i, model_name in enumerate(names):
            col = i + 2
            ax = fig.add_subplot(gs[row, col])
            pred = predictions.get(model_name)
            if pred is not None:
                plot_temp(ax, pred, show_range=True)
            else:
                ax.axis("off")
            ax.text(
                0.5,
                args.label_offset,
                model_name,
                transform=ax.transAxes,
                fontsize=label_fontsize,
                fontweight="bold",
                ha="center",
            )

    _draw_model_row(0, TOP_ROW_MODELS)
    _draw_model_row(1, BOTTOM_ROW_MODELS)

    cbar_left = 0.90 + args.cbar_pad
    if last_temp_im is not None:
        cax_t = fig.add_axes([cbar_left, 0.18, args.cbar_width, 0.7])
        cbar_t = fig.colorbar(last_temp_im, cax=cax_t)
        cbar_t.set_label("°C", fontsize=args.cbar_fontsize)
        tick_positions = [temp_vmin, 0.5, 1.0]
        tick_positions = [t for t in tick_positions if t >= temp_vmin]
        tick_labels = [
            f"{temp_vmin_phys + t * (temp_vmax_phys - temp_vmin_phys):.0f}"
            for t in tick_positions
        ]
        cbar_t.set_ticks(tick_positions)
        cbar_t.set_ticklabels(tick_labels)
        cbar_t.ax.tick_params(labelsize=args.cbar_tick_fontsize)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"task1_panel_frame{frame_idx}.png")
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 1 single-frame multi-model comparison figure (no metrics).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-file", type=str, default=DEFAULT_DATA_FILE)
    p.add_argument("--start-time", type=int, default=800)
    p.add_argument("--downsample-factor", type=int, default=4)
    p.add_argument("--frame", type=int, default=0, help="BulkFlow dataset index")
    p.add_argument("--output-dir", type=str, default="./ICML/CamReady/Figure8_Task1")
    p.add_argument("--normalization-stats", type=str, default=DEFAULT_NORMALIZATION_STATS)
    p.add_argument(
        "--norm-mode",
        type=str,
        default="all",
        choices=["none", "all", "temperature_only"],
    )
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument(
        "--solver",
        type=str,
        default="rk4",
        choices=["euler", "heun", "midpoint", "rk4"],
    )
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset: fm historyfm hbarfm vesde diffpde ddpm unet ffno. "
        "Models not specified leave their fixed-position cells blank.",
    )
    p.add_argument("--temp-vmin", type=float, default=0.012)
    p.add_argument("--hspace", type=float, default=0.15)
    p.add_argument(
        "--fontsize",
        type=int,
        default=18,
        help="Font size for panel labels below each subplot (SDF, GT, model names).",
    )
    p.add_argument(
        "--range-fontsize",
        type=int,
        default=20,
        dest="range_fontsize",
        help="Font size for [min,max] numeric titles above temperature panels.",
    )
    p.add_argument(
        "--cbar-fontsize",
        type=int,
        default=18,
        dest="cbar_fontsize",
        help="Font size for colorbar axis label (°C).",
    )
    p.add_argument(
        "--cbar-tick-fontsize",
        type=int,
        default=17,
        dest="cbar_tick_fontsize",
        help="Font size for colorbar tick labels.",
    )
    p.add_argument("--label-offset", type=float, default=-0.12)
    p.add_argument("--cbar-width", type=float, default=0.008)
    p.add_argument("--cbar-pad", type=float, default=0.01)
    p.add_argument("--col-gap", type=float, default=0.1, dest="col_gap")
    p.add_argument("--fig-width", type=float, default=16.0)
    p.add_argument("--fig-height", type=float, default=6.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--range-titles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show [min,max] above temperature panels (default: on). "
        "Pass --no-range-titles to hide them.",
    )
    args = p.parse_args()
    generate_task1_panel(args)


if __name__ == "__main__":
    main()
