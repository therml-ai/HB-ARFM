#!/usr/bin/env python3
"""
Script to extract Overall Rel L2 metrics from inference output and plot against sample ID.

Usage:
    # From log file:
    python scripts/plot_rel_l2_metrics.py --log-file inference.log
    
    # From stdin (piped):
    python inference_script.py 2>&1 | python scripts/plot_rel_l2_metrics.py
    
    # From log file with custom output:
    python scripts/plot_rel_l2_metrics.py --log-file inference.log --output plot.png
"""

import re
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Try to import scipy for better smoothing, fall back to simple moving average
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def parse_rel_l2_from_line(line):
    """
    Parse Overall Rel L2 value from a line of output.
    
    Expected format: "✓ Frame {frame_number}/{total}: Overall Rel L2 = {value}"
    
    Returns:
        tuple: (sample_id, rel_l2) or (None, None) if not found
    """
    # Pattern to match: "✓ Frame {number}/{number}: Overall Rel L2 = {float}"
    pattern = r'✓\s+Frame\s+(\d+)/\d+:\s+Overall\s+Rel\s+L2\s+=\s+([\d.]+)'
    match = re.search(pattern, line)
    
    if match:
        sample_id = int(match.group(1))
        rel_l2 = float(match.group(2))
        return sample_id, rel_l2
    
    return None, None


def extract_metrics_from_file(file_path):
    """
    Extract sample IDs and Rel L2 values from a log file.
    
    Args:
        file_path: Path to the log file
        
    Returns:
        tuple: (sample_ids, rel_l2_values) as numpy arrays
    """
    sample_ids = []
    rel_l2_values = []
    
    with open(file_path, 'r') as f:
        for line in f:
            sample_id, rel_l2 = parse_rel_l2_from_line(line)
            if sample_id is not None and rel_l2 is not None:
                sample_ids.append(sample_id)
                rel_l2_values.append(rel_l2)
    
    return np.array(sample_ids), np.array(rel_l2_values)


def extract_metrics_from_stdin():
    """
    Extract sample IDs and Rel L2 values from stdin.
    
    Returns:
        tuple: (sample_ids, rel_l2_values) as numpy arrays
    """
    sample_ids = []
    rel_l2_values = []
    
    for line in sys.stdin:
        sample_id, rel_l2 = parse_rel_l2_from_line(line)
        if sample_id is not None and rel_l2 is not None:
            sample_ids.append(sample_id)
            rel_l2_values.append(rel_l2)
    
    return np.array(sample_ids), np.array(rel_l2_values)


def smooth_data(x, y, window_size=None):
    """
    Smooth data using Savitzky-Golay filter or moving average.
    
    Args:
        x: Sample IDs (sorted)
        y: Rel L2 values
        window_size: Window size for smoothing (auto-determined if None)
        
    Returns:
        Smoothed y values
    """
    if len(y) < 3:
        return y
    
    # Auto-determine window size (should be odd and less than data length)
    if window_size is None:
        window_size = min(51, len(y) // 4 * 2 + 1)  # Approximately 1/4 of data, made odd
        if window_size < 3:
            window_size = 3
    
    # Ensure window_size is odd and valid
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, len(y))
    if window_size < 3:
        return y
    
    if HAS_SCIPY:
        # Use Savitzky-Golay filter for better smoothing
        try:
            smoothed = savgol_filter(y, window_size, 3)  # polynomial order 3
            return smoothed
        except:
            # Fall back to moving average if savgol fails
            pass
    
    # Simple moving average fallback
    smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='same')
    return smoothed


def plot_rel_l2_metrics(sample_ids, rel_l2_values, output_path=None, show_plot=True):
    """
    Plot Rel L2 values against sample IDs.
    
    Args:
        sample_ids: Array of sample IDs
        rel_l2_values: Array of Rel L2 values
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    if len(sample_ids) == 0:
        print("No data found to plot!")
        return
    
    # Sort by sample_id to ensure proper ordering
    sort_idx = np.argsort(sample_ids)
    sample_ids_sorted = sample_ids[sort_idx]
    rel_l2_sorted = rel_l2_values[sort_idx]
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute smoothed trend line
    smoothed_values = smooth_data(sample_ids_sorted, rel_l2_sorted)
    
    # Plot smoothed trend line in the background (behind everything)
    ax.plot(sample_ids_sorted, smoothed_values, '-', linewidth=10, 
            alpha=0.4, color='#7CFC00', label='Smoothed Trend', zorder=1)
    
    # Plot the data points on top
    ax.plot(sample_ids_sorted, rel_l2_sorted, 'o-', markersize=4, linewidth=1.5, 
            alpha=0.7, color='#2E86AB', label='Overall Rel L2', zorder=2)
    
    # Add statistics
    mean_val = np.mean(rel_l2_values)
    std_val = np.std(rel_l2_values)
    min_val = np.min(rel_l2_values)
    max_val = np.max(rel_l2_values)
    
    # Add mean line
    ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.4f}')
    
    # Add ±1 std lines
    ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, 
               alpha=0.7, label=f'Mean ± Std: {mean_val:.4f} ± {std_val:.4f}')
    ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Sample ID (Frame Number)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Rel L2', fontsize=12, fontweight='bold')
    ax.set_title(f'Overall Rel L2 vs Sample ID\n'
                 f'Mean: {mean_val:.4f} ± {std_val:.4f}, Range: [{min_val:.4f}, {max_val:.4f}]',
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract and plot Overall Rel L2 metrics from inference output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file containing inference output (if not provided, reads from stdin)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the plot (default: rel_l2_plot.png in current directory)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (only save if --output is provided)'
    )
    
    args = parser.parse_args()
    
    # Extract metrics
    if args.log_file:
        if not Path(args.log_file).exists():
            print(f"Error: Log file not found: {args.log_file}")
            sys.exit(1)
        print(f"Reading metrics from: {args.log_file}")
        sample_ids, rel_l2_values = extract_metrics_from_file(args.log_file)
    else:
        print("Reading metrics from stdin...")
        sample_ids, rel_l2_values = extract_metrics_from_stdin()
    
    # Check if we found any data
    if len(sample_ids) == 0:
        print("Error: No Rel L2 metrics found in the input!")
        print("Expected format: '✓ Frame {number}/{number}: Overall Rel L2 = {value}'")
        sys.exit(1)
    
    print(f"✓ Found {len(sample_ids)} data points")
    print(f"  Sample ID range: [{sample_ids.min()}, {sample_ids.max()}]")
    print(f"  Rel L2 range: [{rel_l2_values.min():.4f}, {rel_l2_values.max():.4f}]")
    print(f"  Mean Rel L2: {np.mean(rel_l2_values):.4f} ± {np.std(rel_l2_values):.4f}")
    
    # Determine output path
    output_path = args.output
    if output_path is None and not args.no_show:
        output_path = "rel_l2_plot.png"
    
    # Plot
    plot_rel_l2_metrics(
        sample_ids, 
        rel_l2_values, 
        output_path=output_path,
        show_plot=not args.no_show
    )


if __name__ == '__main__':
    main()
