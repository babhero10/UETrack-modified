#!/usr/bin/env python3
"""
Evaluate UETrack on custom dataset and generate CSV predictions.

Usage:
    python tracking/evaluate_custom_loader.py --tracker_name uetrack --tracker_param uetrack_tiny \
        --split public_lb --csv_output submission.csv
"""

import os
import sys
import argparse
import csv
import json
import numpy as np
from pathlib import Path

# Add project path
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.environment import env_settings


def load_public_lb_sequences(manifest_dir):
    """Return the ordered list of public_lb sequence keys from manifest.json."""
    manifest_path = os.path.join(manifest_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        # Fall back to contestant_manifest.json
        manifest_path = os.path.join(manifest_dir, 'contestant_manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    if 'public_lb' not in manifest:
        raise ValueError("No 'public_lb' split found in manifest")
    return list(manifest['public_lb'].keys())


def generate_csv_submission(results_dir, output_file, sequence_ids):
    """
    Generate CSV submission from tracker results.

    Tracker saves results as:
        {results_dir}/{dataset_name}/{seq_name}.txt   (tab-separated floats, one bbox per line)

    Args:
        results_dir:  tracker.results_dir  (e.g. env.results_path/uetrack/uetrack_tiny)
        output_file:  path to write submission.csv
        sequence_ids: ordered list of ids from the manifest, e.g. ["dataset1/Car_video", ...]
    """
    results_dict = {}
    missing = []

    for seq_id in sequence_ids:
        # seq_id looks like "dataset1/Car_video"
        result_file = Path(results_dir) / f"{seq_id}.txt"
        if not result_file.exists():
            missing.append(seq_id)
            continue
        with open(result_file, 'r') as f:
            first_line = f.readline().strip()
        if not first_line:
            missing.append(seq_id)
            continue
        # Tracker saves tab-separated floats
        parts = first_line.replace(',', '\t').split()
        if len(parts) < 4:
            missing.append(seq_id)
            continue
        x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        results_dict[seq_id] = (x, y, w, h)

    if missing:
        print(f"Warning: {len(missing)} sequences have no results: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'w', 'h'])
        for seq_id in sequence_ids:
            if seq_id in results_dict:
                x, y, w, h = results_dict[seq_id]
                writer.writerow([seq_id, int(x), int(y), int(w), int(h)])

    print(f"CSV written to: {output_file}  ({len(results_dict)}/{len(sequence_ids)} sequences)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate tracker on custom dataset')
    parser.add_argument('--tracker_name', type=str, default='uetrack')
    parser.add_argument('--tracker_param', type=str, default='uetrack_tiny')
    parser.add_argument('--dataset_name', type=str, default='custom_manifest')
    parser.add_argument('--split', type=str, default='public_lb')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--csv_output', type=str, default='submission.csv')
    parser.add_argument('--skip_tracking', action='store_true',
                        help='Skip tracking; only regenerate CSV from existing results')
    args = parser.parse_args()

    env = env_settings()
    manifest_dir = env.custom_manifest_path

    # Resolve the same results_dir the tracker will use
    results_dir = os.path.join(env.results_path, args.tracker_name, args.tracker_param)

    if not args.skip_tracking:
        print(f"\nRunning tracker on {args.dataset_name} ({args.split} split)")
        dataset = get_dataset(args.dataset_name)
        if len(dataset) == 0:
            print(f"Error: no sequences found in {args.dataset_name}")
            return
        print(f"Found {len(dataset)} sequences")
        trackers = [Tracker(args.tracker_name, args.tracker_param, args.dataset_name)]
        run_dataset(dataset, trackers, debug=0, threads=args.threads, num_gpus=1)

    print(f"\nGenerating CSV from: {results_dir}")
    sequence_ids = load_public_lb_sequences(manifest_dir)
    print(f"Expecting {len(sequence_ids)} public_lb sequences")
    generate_csv_submission(results_dir, args.csv_output, sequence_ids)
    print("Done!")


if __name__ == '__main__':
    main()
