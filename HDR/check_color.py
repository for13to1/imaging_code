#!/usr/bin/env python3
"""
General-purpose HDR Chromaticity Audit Tool.
Evaluates color balance and channel statistics of Radiance (.hdr) files.
"""

import argparse
import cv2
import numpy as np


def audit_color(hdr_path):
    hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
    if hdr is None:
        print(f"Error: Could not read HDR file at {hdr_path}")
        return

    print(f"\n--- HDR Color Audit: {hdr_path} ---")
    print(f"Shape: {hdr.shape} | Dtype: {hdr.dtype}")

    # Channels (OpenCV default BGR)
    b, g, r = hdr[:, :, 0], hdr[:, :, 1], hdr[:, :, 2]

    # 1. Basic Stats
    print("\n[1] Channel Statistics (Linear):")
    for name, ch in [("Red", r), ("Green", g), ("Blue", b)]:
        print(
            f"  {name:5}: Mean={np.mean(ch):.6f}, Max={np.max(ch):.4f}, Min={np.min(ch):.4f}"
        )

    # 2. Chromaticity Balance (Relative to Green)
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    print("\n[2] Relative Scales (Normalized to Green):")
    print(f"  R/G: {mean_r/mean_g:.4f}")
    print(f"  G/G: 1.0000")
    print(f"  B/G: {mean_b/mean_g:.4f}")

    # 3. Global Chromaticity Coordinates (x, y)
    total = r + g + b + 1e-12
    x = np.mean(r / total)
    y = np.mean(g / total)
    print(f"\n[3] Average Chromaticity (x, y):")
    print(f"  x: {x:.4f} (Red component)")
    print(f"  y: {y:.4f} (Green component)")
    print(f"  (D65 White Point is approx x=0.3127, y=0.3290)")

    # 4. Saturation Check
    dark_mask = (r < 0.01) & (g < 0.01) & (b < 0.01)
    print(f"\n[4] Pixels near black (<0.01): {np.sum(dark_mask)/hdr.size*100*3:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit HDR chromaticity and color balance."
    )
    parser.add_argument("input", type=str, help="Path to the .hdr file")
    args = parser.parse_args()
    audit_color(args.input)
