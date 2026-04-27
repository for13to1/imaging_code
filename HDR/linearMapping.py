#!/usr/bin/env python3
"""
Linear Mapping Methods for HDR to LDR Conversion

Strictly linear scaling methods that map HDR values to display range
through affine transformations (scale + shift) followed by clipping.

Methods Implemented:
1. Direct Clipping / Truncation
2. Linear Min-Max Normalization
3. Exposure-Based Linear Scaling
4. Percentile-Based Linear Scaling
5. Mean/Std Linear Normalization

All methods assume input HDR image is in linear RGB color space with
floating-point values. The output is an 8-bit unsigned integer image
suitable for standard display devices.

Example:
    >>> from linearMapping import LinearMapping, load_hdr
    >>> hdr = load_hdr("memorial.hdr")
    >>> mapper = LinearMapping(gamma=2.2)
    >>> ldr = mapper.process(hdr, method="min_max_normalize")
"""

import cv2
import numpy as np
from pathlib import Path

__all__ = ["LinearMapping", "load_hdr"]


class LinearMapping:
    """
    Collection of strictly linear mapping methods for HDR display.

    All methods follow the same pipeline:
        HDR input -> linear affine mapping -> clip to [0, 1] -> gamma -> 8-bit output
    """

    def __init__(self, gamma: float = 2.2):
        """
        Args:
            gamma: Display gamma for post-mapping transfer function (default 2.2).
                   Must be a positive number.

        Raises:
            ValueError: If gamma is not positive.
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = gamma

    def _apply_gamma(self, img: np.ndarray) -> np.ndarray:
        """Apply gamma correction. Input must already be in [0, 1]."""
        return img ** (1.0 / self.gamma)

    def _to_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert [0, 1] float image to 8-bit unsigned integer."""
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    def direct_clip(self, img_hdr: np.ndarray) -> np.ndarray:
        """
        Method 1: Direct clipping to [0, 255] without any scaling.

        Simply truncates/clips HDR values to 8-bit range.
        Fastest but worst quality; highlights are harshly clipped.
        """
        img = np.clip(img_hdr, 0.0, 255.0) / 255.0
        return self._to_uint8(self._apply_gamma(img))

    def min_max_normalize(self, img_hdr: np.ndarray) -> np.ndarray:
        """
        Method 2: Linear min-max normalization to [0, 1].

        Affine mapping: darkest pixel -> 0, brightest pixel -> 1.
        """
        v_min = np.min(img_hdr)
        v_max = np.max(img_hdr)
        if v_max - v_min < 1e-9:
            return np.zeros_like(img_hdr, dtype=np.uint8)
        img_norm = (img_hdr - v_min) / (v_max - v_min)
        img = np.clip(img_norm, 0.0, 1.0)
        return self._to_uint8(self._apply_gamma(img))

    def exposure_scale(self, img_hdr: np.ndarray, exposure: float = 1.0) -> np.ndarray:
        """
        Method 3: Exposure-based linear scaling.

        Linearly scales HDR by an exposure factor, then clips to [0, 1].
        Mimics camera exposure adjustment.

        Args:
            exposure: Exposure multiplier (default 1.0).
                      >1.0 brightens, <1.0 darkens the image.

        Raises:
            ValueError: If exposure is not positive.
        """
        if exposure <= 0:
            raise ValueError(f"exposure must be positive, got {exposure}")
        img_scaled = img_hdr * exposure
        img = np.clip(img_scaled, 0.0, 1.0)
        return self._to_uint8(self._apply_gamma(img))

    def percentile_scale(
        self, img_hdr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0
    ) -> np.ndarray:
        """
        Method 4: Percentile-based linear scaling.

        Affine mapping using percentiles instead of absolute min/max
        to avoid outliers destroying contrast.

        Args:
            low_pct: Lower percentile mapped to 0 (default 1.0).
                     Must be in [0, 100).
            high_pct: Upper percentile mapped to 1 (default 99.0).
                      Must be in (0, 100] and greater than low_pct.

        Raises:
            ValueError: If percentiles are invalid.
        """
        if not (0.0 <= low_pct < 100.0):
            raise ValueError(f"low_pct must be in [0, 100), got {low_pct}")
        if not (0.0 < high_pct <= 100.0):
            raise ValueError(f"high_pct must be in (0, 100], got {high_pct}")
        if low_pct >= high_pct:
            raise ValueError(f"low_pct ({low_pct}) must be < high_pct ({high_pct})")

        v_low = np.percentile(img_hdr, low_pct)
        v_high = np.percentile(img_hdr, high_pct)
        if v_high - v_low < 1e-9:
            return np.zeros_like(img_hdr, dtype=np.uint8)
        img_norm = (img_hdr - v_low) / (v_high - v_low)
        img = np.clip(img_norm, 0.0, 1.0)
        return self._to_uint8(self._apply_gamma(img))

    def mean_std_normalize(
        self, img_hdr: np.ndarray, target_mean: float = 0.5, target_std: float = 0.2
    ) -> np.ndarray:
        """
        Method 5: Mean/std linear normalization.

        Affine mapping so that mean and std match target values,
        then clips to [0, 1].

        Args:
            target_mean: Target mean after mapping (default 0.5).
                         Must be in [0, 1].
            target_std: Target standard deviation (default 0.2).
                        Must be positive.

        Raises:
            ValueError: If target parameters are invalid.
        """
        if not (0.0 <= target_mean <= 1.0):
            raise ValueError(f"target_mean must be in [0, 1], got {target_mean}")
        if target_std <= 0:
            raise ValueError(f"target_std must be positive, got {target_std}")

        v_mean = np.mean(img_hdr)
        v_std = np.std(img_hdr)
        if v_std < 1e-9:
            return np.zeros_like(img_hdr, dtype=np.uint8)
        img_norm = (img_hdr - v_mean) / v_std
        img_norm = img_norm * target_std + target_mean
        img = np.clip(img_norm, 0.0, 1.0)
        return self._to_uint8(self._apply_gamma(img))

    def process(
        self,
        img_hdr: np.ndarray,
        method: str = "min_max_normalize",
        exposure: float = 1.0,
        low_pct: float = 1.0,
        high_pct: float = 99.0,
        target_mean: float = 0.5,
        target_std: float = 0.2,
    ) -> np.ndarray:
        """
        Unified entry point for all linear mapping methods.

        Args:
            img_hdr: Input HDR image in linear RGB, float32 or float64.
            method: Mapping method name. One of:
                - "direct_clip"
                - "min_max_normalize"
                - "exposure_scale"
                - "percentile_scale"
                - "mean_std_normalize"
            exposure: Exposure multiplier for "exposure_scale" method.
            low_pct: Lower percentile for "percentile_scale" method.
            high_pct: Upper percentile for "percentile_scale" method.
            target_mean: Target mean for "mean_std_normalize" method.
            target_std: Target std for "mean_std_normalize" method.

        Returns:
            8-bit LDR image (np.uint8).

        Raises:
            ValueError: If method name is unknown.
        """
        img_hdr = img_hdr.astype(np.float32)

        if method == "direct_clip":
            return self.direct_clip(img_hdr)
        elif method == "min_max_normalize":
            return self.min_max_normalize(img_hdr)
        elif method == "exposure_scale":
            return self.exposure_scale(img_hdr, exposure=exposure)
        elif method == "percentile_scale":
            return self.percentile_scale(img_hdr, low_pct=low_pct, high_pct=high_pct)
        elif method == "mean_std_normalize":
            return self.mean_std_normalize(
                img_hdr, target_mean=target_mean, target_std=target_std
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Choose from: direct_clip, min_max_normalize, exposure_scale, "
                f"percentile_scale, mean_std_normalize"
            )


def load_hdr(path: str) -> np.ndarray:
    """Load HDR image into linear RGB float32.

    Args:
        path: Path to the HDR image file.

    Returns:
        HDR image as float32 array in RGB order, shape (H, W, 3).

    Raises:
        FileNotFoundError: If the file cannot be read.
    """
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read HDR file: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Linear Mapping Methods for HDR to LDR Conversion"
    )
    parser.add_argument("input", type=str, nargs="?", default="memorial")
    parser.add_argument(
        "--method",
        type=str,
        default="min_max_normalize",
        choices=[
            "direct_clip",
            "min_max_normalize",
            "exposure_scale",
            "percentile_scale",
            "mean_std_normalize",
        ],
        help="Linear mapping method to apply",
    )
    parser.add_argument("--gamma", type=float, default=2.2, help="Display gamma")
    parser.add_argument(
        "--exposure", type=float, default=1.0, help="Exposure multiplier"
    )
    parser.add_argument("--low-pct", type=float, default=1.0, help="Lower percentile")
    parser.add_argument("--high-pct", type=float, default=99.0, help="Upper percentile")
    parser.add_argument("--target-mean", type=float, default=0.5, help="Target mean")
    parser.add_argument("--target-std", type=float, default=0.2, help="Target std")
    parser.add_argument("--output", type=str, help="Output PNG path")
    args = parser.parse_args()

    # Path resolution
    in_p = Path(args.input)
    if not in_p.exists():
        in_p = Path(__file__).parent / "dataset" / args.input / f"{args.input}.hdr"

    out_p = args.output or str(
        Path(__file__).parent / "output" / f"{in_p.stem}_linear_{args.method}.png"
    )
    Path(out_p).parent.mkdir(parents=True, exist_ok=True)

    try:
        hdr = load_hdr(str(in_p))
        mapper = LinearMapping(gamma=args.gamma)
        ldr = mapper.process(
            hdr,
            method=args.method,
            exposure=args.exposure,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            target_mean=args.target_mean,
            target_std=args.target_std,
        )
        cv2.imwrite(str(out_p), cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_p} | Method: {args.method} | Gamma: {args.gamma}")
    except Exception as e:
        print(f"Error: {e}")
