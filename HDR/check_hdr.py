#!/usr/bin/env python3
"""
General-purpose HDR Integrity Audit Tool.
Evaluates dynamic range, statistics, and numerical integrity of Radiance (.hdr) files.
"""

import cv2
import numpy as np


def audit_hdr(hdr_path):
    hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
    if hdr is None:
        print(f"Error: Could not read HDR file at {hdr_path}")
        return

    print(f"\n--- HDR Integrity Audit: {hdr_path} ---")
    print(f"Shape: {hdr.shape} | Dtype: {hdr.dtype}")

    # 1. Range Stats
    v_min, v_max = np.min(hdr), np.max(hdr)
    v_mean, v_med = np.mean(hdr), np.median(hdr)
    print("\n[1] Dynamic Range:")
    print(f"  Min/Max: {v_min:.6e} / {v_max:.6f}")
    print(f"  Mean/Med: {v_mean:.6f} / {v_med:.6f}")

    # Dynamic Range in Stops
    if v_min > 0:
        stops = np.log2(v_max / v_min)
        print(f"  Dynamic Range: {stops:.2f} stops (EV)")
    else:
        # Avoid log of zero, use a small epsilon
        valid_min = np.min(hdr[hdr > 1e-12]) if np.any(hdr > 1e-12) else 1e-12
        stops = np.log2(v_max / valid_min)
        print(f"  Dynamic Range: ~{stops:.2f} stops (estimated from non-zero min)")

    # 2. Distribution
    print("\n[2] Percentile Distribution:")
    for p in [1, 10, 50, 90, 99, 99.9]:
        print(f"  {p:4}%: {np.percentile(hdr, p):.6f}")

    # 3. Integrity Check
    nans = np.sum(np.isnan(hdr))
    infs = np.sum(np.isinf(hdr))
    zeros = np.sum(hdr == 0)
    print("\n[3] Numerical Integrity:")
    print(f"  NaN Count: {nans}")
    print(f"  Inf Count: {infs}")
    print(f"  Zero Count: {zeros} ({zeros/hdr.size*100:.2f}%)")

    # 4. Clipping Analysis
    # Many displays clip at 1.0 if not tone mapped.
    overlines = np.sum(hdr > 1.0)
    print(f"\n[4] Clipping (Values > 1.0):")
    print(f"  Count: {overlines} ({overlines/hdr.size*100:.2f}%)")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Audit HDR dynamic range and numerical integrity."
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="memorial",
        help="Input path or dataset name",
    )
    args = parser.parse_args()

    # Smart path resolution
    base_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.input)
    if not dataset_path.exists():
        potential_paths = [
            base_dir / "dataset" / args.input / f"{args.input}.hdr",
            base_dir / "output" / f"{args.input}_debevec1997.hdr",
            base_dir / "output" / f"{args.input}_mitsunaga1999.hdr",
        ]
        for p in potential_paths:
            if p.exists():
                dataset_path = p
                break

    audit_hdr(str(dataset_path))
