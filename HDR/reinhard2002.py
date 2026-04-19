"""
Photographic Tone Reproduction for Digital Images (Reinhard 2002)

Based on:
Reinhard, E., Stark, M., Shirley, P., & Ferwerda, J. (2002).
Photographic tone reproduction for digital images.
ACM Transactions on Graphics (TOG), 21(3), 267-276.

[PAPER_STRICT] Implementation following the mathematical formulas in the paper.
"""

import numpy as np
import cv2
from pathlib import Path


class Reinhard2002:
    def __init__(
        self, key_value=0.18, l_white=None, phi=8.0, epsilon=0.05, use_local=True
    ):
        """
        Args:
            key_value (a): Geometric mean alignment target (default 0.18).
            l_white: Smallest luminance mapped to white (default max in scene).
            phi: Sharpening parameter for local operator (default 8.0).
            epsilon: Threshold for scale selection in local operator (default 0.05).
            use_local: If True, uses the local dodging & burning operator.
        """
        self.a = key_value
        self.l_white = l_white
        self.phi = phi
        self.epsilon = epsilon
        self.use_local = use_local

        # [PAPER_STRICT] Section 4: L = 0.27R + 0.67G + 0.06B
        self.Y_WEIGHTS = np.array([0.27, 0.67, 0.06])

    def get_luminance(self, img_rgb):
        return np.dot(img_rgb, self.Y_WEIGHTS)

    def process(self, img_rgb):
        img_rgb = img_rgb.astype(np.float64)
        lw = self.get_luminance(img_rgb)

        # 1. Calculate Log-Average Luminance (Eq 1)
        delta = 1e-6
        lw_log_avg = np.exp(np.mean(np.log(delta + lw)))

        # 2. Scale Luminance (Eq 2)
        l_scaled = (self.a / (lw_log_avg + 1e-8)) * lw

        if not self.use_local:
            # --- Global Operator (Eq 4) ---
            l_white = self.l_white if self.l_white is not None else l_scaled.max()
            ld = (l_scaled * (1.0 + l_scaled / (l_white**2 + 1e-8))) / (1.0 + l_scaled)
        else:
            # --- Local Operator (Dodging and Burning) ---
            # [PAPER_STRICT] Section 4: alpha1 = 1 / (2 * sqrt(2))
            alpha1 = 1.0 / (2.0 * np.sqrt(2.0))

            num_scales = 8
            scales = np.array([1.6**i for i in range(num_scales + 1)])

            # Precompute Gaussian blurred images V1
            v1_all = []
            for s in scales:
                # [PAPER_STRICT] Eq 5 vs standard Gaussian: sigma = (alpha * s) / sqrt(2)
                sigma = (alpha1 * s) / np.sqrt(2.0)
                v1 = cv2.GaussianBlur(l_scaled, (0, 0), max(sigma, 0.1))
                v1_all.append(v1)

            # Selection of optimal scale s_m (Eq 8)
            # Starting at the lowest scale, seek the FIRST scale where |V| < epsilon
            best_v1 = v1_all[-1].copy()  # Default to largest scale
            found = np.zeros(
                l_scaled.shape, dtype=bool
            )  # Track pixels that found their scale

            for i in range(num_scales):
                v1 = v1_all[i]
                v2 = v1_all[i + 1]
                s = scales[i]

                # Center-surround function (Eq 7)
                v = (v1 - v2) / ((2.0**self.phi * self.a / (s**2 + 1e-8)) + v1)

                # Find pixels where |V| < epsilon and not yet found
                mask = (np.abs(v) < self.epsilon) & (~found)
                best_v1[mask] = v1[mask]
                found[mask] = True

                if np.all(found):
                    break

            # Local operator (Eq 9)
            ld = l_scaled / (1.0 + best_v1)

        # CLIP and Color Restoration
        ld = np.clip(ld, 0, 1)
        scale_factor = ld / (l_scaled + 1e-8)
        img_out = img_rgb * scale_factor[:, :, np.newaxis]

        # Display Mapping (Gamma 2.2)
        img_out = np.clip(img_out, 0, 1)
        return (np.power(img_out, 1.0 / 2.2) * 255).astype(np.uint8)


def load_hdr(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Missing {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reinhard 2002 TMO - Strict Implementation"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="memorial",
        help="Input path or dataset name",
    )
    parser.add_argument("--key", type=float, default=0.18)
    parser.add_argument("--white", type=float, default=None)
    parser.add_argument("--global_only", action="store_true")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent

    # Smart path resolution
    dataset_path = Path(args.input)
    if not dataset_path.exists():
        potential_path = base_dir / "dataset" / args.input / f"{args.input}.hdr"
        if potential_path.exists():
            dataset_path = potential_path
        else:
            dataset_path = base_dir / "dataset" / args.input

    if args.output is None:
        out_dir = base_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "global" if args.global_only else "local"
        args.output = str(out_dir / f"{args.input}_reinhard2002_{suffix}.png")

    print(
        f"--- Processing {dataset_path.name} (Mode={'Global' if args.global_only else 'Local'}) ---"
    )

    try:
        img_hdr = load_hdr(str(dataset_path))
        tmo = Reinhard2002(
            key_value=args.key, l_white=args.white, use_local=not args.global_only
        )
        result = tmo.process(img_hdr)
        cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"✅ Result saved: {args.output}")
    except Exception as e:
        print(f"❌ Error: {e}")
