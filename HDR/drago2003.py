#!/usr/bin/env python3
"""
Adaptive Logarithmic Mapping For Displaying High Contrast Scenes (Drago 2003)

Based on:
Drago, F., Myszkowski, K., Annen, T., & Chiba, N. (2003).
Adaptive logarithmic mapping for displaying high contrast scenes.
Computer Graphics Forum, 22(3), 419-426.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


class Drago2003:
    """
    Adaptive Logarithmic Tone Mapping Operator (Drago et al. 2003).

    Implementation Fidelity Categories:
    1. [PAPER_STRICT]: Formulas and logic strictly following Sections 3, 4, and 5.
    2. [RATIONAL_OMISSION]:
       - Temporal smoothing for video (Section 5.1) is omitted for static processing.
       - Conditional tiling threshold (Section 5) is replaced by unconditional tiling.
       - Edge padding for 3x3 tiling is not specified in the paper.
       - Fixation center defaults to geometric center when not specified.
    """

    def __init__(
        self,
        bias: float = 0.85,
        exposure: float = 1.0,
        ld_max: float = 100.0,
        gamma: float = 2.2,
        use_center_weight: bool = False,
        center_area: float = 0.15,
        use_fast_mode: bool = False,
        pade_threshold: float = 0.1,
    ):
        """
        Args:
            bias: Bias parameter (Section 3.3, default 0.85).
            exposure: Global exposure factor (Section 3.1).
            ld_max: Maximum display luminance (Section 3.3, default 100 cd/m2).
            gamma: Display gamma for custom transfer function (Section 4, default 2.2).
            use_center_weight: Enable Gaussian center-weighted adaptation (Section 3.1).
            center_area: Area fraction for Gaussian kernel (Section 3.1, default 0.15).
            use_fast_mode: Use 3x3 tiling for bias and Pade approximation for log (Section 5).
            pade_threshold: Heuristic threshold for Pade approximation (default 0.1).
        """
        self.bias = bias
        self.exposure = exposure
        self.ld_max = ld_max
        self.gamma = gamma
        self.use_center_weight = use_center_weight
        self.center_area = center_area
        self.use_fast_mode = use_fast_mode
        self.pade_threshold = pade_threshold

    def _pade_log1p(self, x: np.ndarray) -> np.ndarray:
        """[PAPER_STRICT] Section 5: Pade[1,1] approximation of ln(x+1)."""
        # Pade[1,1] approximant: ln(1+x) approx x / (1 + x/2)
        return x / (1.0 + x * 0.5)

    def _compute_gamma_params(self) -> Tuple[float, float, float]:
        """[PAPER_STRICT] Section 4: Solve for 'start' and 'slope' of tangent BT.709 curve."""
        power = 0.9 / self.gamma
        if abs(power - 1.0) < 1e-5:
            return 0.0, 1.0, power

        # Tangent condition: slope * start = 1.099 * start^power - 0.099
        # And slope = 1.099 * power * start^(power-1)
        # Solving leads to: 1.099 * start^power * (1 - power) = 0.099
        start = np.power(0.099 / (1.099 * (1.0 - power)), 1.0 / power)
        slope = 1.099 * power * np.power(start, power - 1.0)
        return start, slope, power

    def apply_custom_gamma(self, img: np.ndarray) -> np.ndarray:
        """[PAPER_STRICT] Section 4: Adaptive ITU-R BT.709 transfer function."""
        start, slope, power = self._compute_gamma_params()
        out = np.zeros_like(img)
        mask = img <= start
        out[mask] = slope * img[mask]
        out[~mask] = 1.099 * np.power(np.maximum(img[~mask], 0.0), power) - 0.099
        return np.clip(out, 0.0, 1.0)

    def _log1p_fast(self, x: np.ndarray) -> np.ndarray:
        """[ENGINEERING_ADAPTATION] Internal log1p with optional Pade approximation (Section 5)."""
        if self.use_fast_mode:
            # [ENGINEERING_ADAPTATION] Heuristic threshold for Pade approximation (default 0.1).
            mask = x < self.pade_threshold
            res = np.zeros_like(x)
            res[mask] = self._pade_log1p(x[mask])
            res[~mask] = np.log(x[~mask] + 1.0)
            return res
        return np.log(x + 1.0)

    def process(
        self, img_hdr: np.ndarray, fixation: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        Adaptive Logarithmic Mapping (Section 3.3).

        Args:
            img_hdr: Linear HDR image (float32).
            fixation: (y, x) coordinates of viewing fixation for center-weighting.
                     Defaults to image center if None.
        """
        img_xyz = cv2.cvtColor(img_hdr, cv2.COLOR_RGB2XYZ)
        Y = img_xyz[:, :, 1]
        # [ENGINEERING_ADAPTATION] Small epsilon to prevent log(-inf) for zero luminance pixels.
        log_Y = np.log(np.maximum(Y, 1e-9))
        h, w = Y.shape

        # 1. World Adaptation (Section 3.1)
        if self.use_center_weight:
            # [PAPER_STRICT] Section 3.1 Gaussian kernel area (15% default).
            # Here alpha represents the standard deviation as a fraction of image size.
            # For a 2D Gaussian e^(-r^2/2sigma^2), the integral (total energy) is 2pi*sigma^2.
            # Setting 2pi*sigma^2 = center_area * H * W gives:
            # 2pi * (alpha*H)*(alpha*W) = center_area * H * W (assuming circular symmetry for alpha)
            # alpha = sqrt(center_area / 2pi)
            alpha = np.sqrt(self.center_area / (2.0 * np.pi))

            # [ENGINEERING_ADAPTATION] Fixed to geometric center if fixation is None.
            cy, cx = fixation if fixation is not None else (h // 2, w // 2)

            y_ax = np.arange(h) - cy
            x_ax = np.arange(w) - cx
            gy = np.exp(-(y_ax**2) / (2 * (h * alpha) ** 2))
            gx = np.exp(-(x_ax**2) / (2 * (w * alpha) ** 2))

            # [PAPER_STRICT] Section 3.1 Gaussian convolved logarithmic average.
            L_wa = np.exp(np.dot(gy, np.dot(log_Y, gx)) / (np.sum(gy) * np.sum(gx)))
        else:
            # [PAPER_STRICT] Section 3.1 Global logarithmic average.
            L_wa = np.exp(np.mean(log_Y))

        # 2. Brightness Compensation (Section 3.3)
        # [PAPER_STRICT] Scalefactor to maintain constant brightness across different bias values.
        L_wa_adj = L_wa / ((1.0 + self.bias - 0.85) ** 5.0)

        # 3. Scaling
        L_w = Y * (self.exposure / L_wa_adj)
        L_wmax = np.max(Y) * (self.exposure / L_wa_adj)

        # 4. Adaptive Logarithmic Mapping (Equation 4)
        # [PAPER_STRICT] bias_power is the exponent of the Perlin-Hoffert bias function.
        bias_power = np.log(self.bias) / np.log(0.5)

        if self.use_fast_mode:
            # [PAPER_STRICT] Section 5: Optimization using 3x3 pixel tiles.
            h3, w3 = (h + 2) // 3, (w + 2) // 3
            # [RATIONAL_OMISSION] Edge padding not specified in paper; using edge replicate.
            pad_h, pad_w = h3 * 3 - h, w3 * 3 - w
            L_w_padded = np.pad(L_w, ((0, pad_h), (0, pad_w)), mode="edge")

            # Compute average luminance per 3x3 tile
            L_w_tile = L_w_padded.reshape(h3, 3, w3, 3).mean(axis=(1, 3))

            # [ENGINEERING_ADAPTATION] Safe division for ratio computation
            ratio_tile = np.divide(
                L_w_tile, L_wmax, out=np.zeros_like(L_w_tile), where=L_wmax > 1e-9
            )
            base_tile = 2.0 + np.power(ratio_tile, bias_power) * 8.0

            # Upsample base back to full resolution
            base_full = base_tile.repeat(3, axis=0).repeat(3, axis=1)
            base = base_full[:h, :w]
        else:
            # [PAPER_STRICT] Standard per-pixel base calculation.
            # [ENGINEERING_ADAPTATION] Safe division for ratio computation
            ratio_w = np.divide(
                L_w, L_wmax, out=np.zeros_like(L_w), where=L_wmax > 1e-9
            )
            base = 2.0 + np.power(ratio_w, bias_power) * 8.0

        # [PAPER_STRICT] Equation (4) adaptive logarithmic mapping.
        L_dmax_term = (self.ld_max * 0.01) / np.log10(L_wmax + 1.0)
        L_d = L_dmax_term * (self._log1p_fast(L_w) / np.log(base))

        # 5. Color Restoration and Gamma (Section 4)
        # [PAPER_STRICT] Section 3.3: tone mapped Y replaces original Y in XYZ,
        # preserving chromaticity coordinates (x, y). Then convert back to RGB.
        # img_xyz has shape (H, W, 3) with channels [X, Y, Z]
        # [ENGINEERING_ADAPTATION] Safe division to handle zero luminance pixels.
        ratio = np.divide(L_d, Y, out=np.zeros_like(L_d), where=Y > 1e-9)
        img_xyz_new = img_xyz * ratio[..., np.newaxis]
        img_rgb = cv2.cvtColor(img_xyz_new.astype(np.float32), cv2.COLOR_XYZ2RGB)
        return (self.apply_custom_gamma(img_rgb) * 255.0).astype(np.uint8)


def load_hdr(path: str) -> np.ndarray:
    """Load HDR image into linear RGB float32."""
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Missing {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drago 2003 Reference Implementation")
    parser.add_argument("input", type=str, nargs="?", default="memorial")
    parser.add_argument("--bias", type=float, default=0.85)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--ldmax", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--pade", type=float, default=0.1, help="Pade approximation threshold"
    )
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    # Path resolution
    in_p = Path(args.input)
    if not in_p.exists():
        in_p = Path(__file__).parent / "dataset" / args.input / f"{args.input}.hdr"

    out_p = args.output or str(
        Path(__file__).parent / "output" / f"{in_p.stem}_drago2003.png"
    )
    Path(out_p).parent.mkdir(parents=True, exist_ok=True)

    try:
        hdr = load_hdr(str(in_p))
        tmo = Drago2003(
            bias=args.bias,
            exposure=args.exposure,
            ld_max=args.ldmax,
            gamma=args.gamma,
            use_center_weight=args.center,
            use_fast_mode=args.fast,
            pade_threshold=args.pade,
        )
        cv2.imwrite(out_p, cv2.cvtColor(tmo.process(hdr), cv2.COLOR_RGB2BGR))
        print(f"✅ Saved: {out_p}")
    except Exception:
        import traceback

        traceback.print_exc()
