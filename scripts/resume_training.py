#!/usr/bin/env python3
"""
Utility script to resume training from a specific checkpoint.
This script helps find and resume training from checkpoints in your log directory.
"""

import os
import glob
import argparse
from datetime import datetime

def find_checkpoints(log_dir):
    """Find all available checkpoints in the log directory."""
    checkpoint_patterns = [
        os.path.join(log_dir, "lightning_logs", "**", "checkpoints", "*.ckpt"),
        os.path.join(log_dir, "checkpoints", "*.ckpt"),
        os.path.join(log_dir, "hpc_ckpt_*.ckpt"),
        os.path.join(log_dir, "*.ckpt")
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        found = glob.glob(pattern, recursive=True)
        for ckpt in found:
            if os.path.isfile(ckpt):
                mtime = os.path.getmtime(ckpt)
                size = os.path.getsize(ckpt)
                checkpoints.append({
                    'path': ckpt,
                    'mtime': mtime,
                    'datetime': datetime.fromtimestamp(mtime),
                    'size_mb': size / (1024 * 1024)
                })
    
    # Sort by modification time (newest first)
    return sorted(checkpoints, key=lambda x: x['mtime'], reverse=True)

def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Path to the log directory containing checkpoints')
    parser.add_argument('--list_only', action='store_true',
                        help='Only list available checkpoints without resuming')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint path to resume from')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' does not exist")
        return 1
    
    checkpoints = find_checkpoints(args.log_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in '{args.log_dir}'")
        return 1
    
    print(f"Found {len(checkpoints)} checkpoint(s) in '{args.log_dir}':")
    print("-" * 80)
    for i, ckpt in enumerate(checkpoints):
        print(f"{i+1:2d}. {ckpt['path']}")
        print(f"     Modified: {ckpt['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Size: {ckpt['size_mb']:.1f} MB")
        print()
    
    if args.list_only:
        return 0
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.isfile(checkpoint_path):
            print(f"Error: Checkpoint file '{checkpoint_path}' does not exist")
            return 1
    else:
        # Use the latest checkpoint
        checkpoint_path = checkpoints[0]['path']
        print(f"Using latest checkpoint: {checkpoint_path}")
    
    # Generate the resume command
    print("-" * 80)
    print("To resume training, run:")
    print(f"python scripts/train.py checkpoint_path={checkpoint_path}")
    print()
    print("Or add it to your training script:")
    print(f"    checkpoint_path={checkpoint_path} \\")
    
    return 0

if __name__ == "__main__":
    exit(main())

