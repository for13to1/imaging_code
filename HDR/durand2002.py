"""
Fast Bilateral Filtering for the Display of High-Dynamic-Range Images (Durand 2002)

Based on:
Durand, F., & Dorsey, J. (2002).
Fast bilateral filtering for the display of high-dynamic-range images.
ACM transactions on graphics (TOG), 21(3), 257-266.
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter


class Durand2002:
    """
    ENGINEERING ASSUMPTIONS & DISCREPANCIES (Marker System):
    - [PAPER_STRICT]: Directly from the Durand 2002 paper (formula, constant, or step).
    - [ENGINEERING_ADAPTATION]: Omitted details, ambiguities, or modern stability/performance heuristics.

    1. Gaussian Definition: [PAPER_STRICT] Section 3.1 & Eq 2 define g(x) = exp(-x^2 / sigma^2).
       [ENGINEERING_ADAPTATION] Standard libraries (scipy) use exp(-x^2 / (2*sigma^2)).
       This implementation compensates by passing sigma / sqrt(2) to library functions.
    2. Color Space: [ENGINEERING_ADAPTATION] Assumes linear RGB input with Rec.709/sRGB primaries.
       [PAPER_STRICT] Section 6 mentions intensity calculations but doesn't define weights.
    3. Logarithm Base: [PAPER_STRICT] Durand's FAQ clarifies that 'log' in the paper refers to log10.
       This implementation strictly uses np.log10 for all domain transformations.
    4. Numerical Stability: [ENGINEERING_ADAPTATION] Added epsilon (1e-6) to log and division
       operations to prevent singularities in zero-valued pixels.
    5. Uncertainty Fixer (Section 5.3): [ENGINEERING_ADAPTATION] This section describes a heuristic
       to blend J and J_tilde based on log(k), but omits the actual mathematical mapping function.
       Due to this ambiguity (which forces flawed min-max normalizations) and the fact that the
       original authors completely omitted it in their official 2006 C++ implementation
       (Paris & Durand), this heuristic has been intentionally excluded to ensure robustness.
    6. Base Contrast: [ENGINEERING_ADAPTATION] The paper literally states "a base contrast of 5
       worked well". However, mapping to a linear contrast ratio of 5:1 causes severe washing out.
       The authors' later official C++ implementation for the Bilateral Grid (Paris & Durand, 2006;
       see https://people.csail.mit.edu/sparis/code/src/tone_mapping.tar.gz) explicitly states in its
       source code (tone_mapping.cpp): "meaningful values for the contrast are between 5.0 and 200.0.
       50.0 always gives satisfying results." Thus, 50.0 is used here.
    """

    def __init__(
        self,
        base_contrast: float = 50.0,
        sigma_r: float = 0.4,
        subsample_factor: int = 10,
    ):
        """
        Initialize the TMO with paper-referenced and performance parameters.

        Args:
            base_contrast: Target linear contrast ratio for the base layer (default 50.0 as per Paris & Durand 2006).
            sigma_r: Standard deviation in the intensity (log10) domain (default 0.4 as per Section 6).
            subsample_factor: Acceleration factor for spatial downsampling (default 10 as per Section 5.2).
        """
        self.base_contrast = base_contrast
        self.sigma_r = sigma_r
        self.subsample_factor = subsample_factor

        # [ENGINEERING_ADAPTATION] Standard Rec. 709 luminance weights.
        self.Y_WEIGHTS = np.array([0.2126729, 0.7151522, 0.0721750])

    def get_intensity(self, img_rgb: np.ndarray) -> np.ndarray:
        """Computes the luminance channel from linear RGB."""
        return np.dot(img_rgb, self.Y_WEIGHTS)

    def fast_bilateral_filter(self, L: np.ndarray, sigma_s: float) -> np.ndarray:
        """
        Piecewise-linear bilateral filtering with subsampling.
        (Strict implementation of Durand 2002, Sections 5.1 & 5.2)
        """
        h, w = L.shape

        # [ENGINEERING_ADAPTATION] Prevent excessive subsampling which causes blocky aliasing.
        # Ensure the standard deviation in the subsampled domain is at least ~1.5 pixels.
        max_subsample = max(1, int(sigma_s / (1.5 * np.sqrt(2))))
        actual_subsample = min(self.subsample_factor, max_subsample)

        # 1. Subsample the input (Nearest-neighbor as per Section 5.2)
        h_sub = max(h // actual_subsample, 2)
        w_sub = max(w // actual_subsample, 2)
        L_sub = cv2.resize(L, (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)

        # 2. Discretize intensity range (Section 5.1)
        min_L, max_L = L.min(), L.max()
        segment_step = self.sigma_r
        num_segments = int(np.ceil((max_L - min_L) / segment_step))
        num_segments = max(num_segments, 1)

        i_values = min_L + np.arange(num_segments + 1) * segment_step

        # 3. Piecewise linear processing on subsampled image
        sigma_s_sub = sigma_s / actual_subsample
        # [PAPER_STRICT] Compensation for exp(-x^2/sigma^2) vs library exp(-x^2/2sigma^2)
        lib_sigma_s = sigma_s_sub / np.sqrt(2)

        J_full_layers = []

        for i_j in i_values:
            # [PAPER_STRICT] Range weight g(I_p - i^j)
            G_j_sub = np.exp(-((L_sub - i_j) ** 2) / (self.sigma_r**2))
            H_j_sub = G_j_sub * L_sub

            # Blur with spatial Gaussian f
            G_j_blurred_sub = gaussian_filter(G_j_sub, sigma=lib_sigma_s)
            H_j_blurred_sub = gaussian_filter(H_j_sub, sigma=lib_sigma_s)

            # [PAPER_STRICT] Eq 6 normalization
            J_j_sub = H_j_blurred_sub / (G_j_blurred_sub + 1e-8)

            # Upsample J^j to full resolution (Bilinear)
            J_j_full = cv2.resize(J_j_sub, (w, h), interpolation=cv2.INTER_LINEAR)
            J_full_layers.append(J_j_full)

        # 4. Final interpolation at full resolution to preserve edges
        J_stack = np.array(J_full_layers)

        # [PAPER_STRICT] Linear interpolation between segments (Fig 10)
        u = (L - min_L) / segment_step
        idx_lower = np.clip(np.floor(u).astype(int), 0, num_segments - 1)
        u_rem = u - idx_lower

        # [ENGINEERING_ADAPTATION] Optimized broadcasting index to save memory (replaces np.indices)
        rows = np.arange(h)[:, None]
        cols = np.arange(w)
        J_lower = J_stack[idx_lower, rows, cols]
        J_upper = J_stack[idx_lower + 1, rows, cols]
        B = (1.0 - u_rem) * J_lower + u_rem * J_upper

        return B

    def process(self, img_rgb: np.ndarray) -> np.ndarray:
        """Execute the Durand 2002 tone mapping pipeline."""
        img_rgb = img_rgb.astype(np.float64)

        # 1. Intensity and Log Domain (Section 6)
        # [PAPER_STRICT] "We perform our calculations on the logs of pixel intensities"
        I = self.get_intensity(img_rgb)
        L = np.log10(I + 1e-6)

        # [PAPER_STRICT] Section 6: sigma_s constant to a value of 2% of the image size
        h, w = L.shape
        sigma_s = 0.02 * min(h, w)

        # 2. Fast Bilateral Filter
        B = self.fast_bilateral_filter(L, sigma_s)

        # 3. Detail Layer (Section 6)
        D = L - B

        # 4. Contrast Reduction of Base Layer (Section 6)
        # [PAPER_STRICT] Scale factor such that whole range is compressed to base contrast
        base_range = B.max() - B.min()
        target_range = np.log10(self.base_contrast)

        scale = target_range / base_range if base_range > 1e-6 else 1.0

        # [PAPER_STRICT] Align max to 0 (as per FAQ and Equation in Section 6)
        B_prime = (B - B.max()) * scale

        # 5. Recombination & Restoration
        O_log = B_prime + D
        O = 10.0**O_log

        # [PAPER_STRICT] "Recompose color after contrast reduction"
        color_scale = O / (I + 1e-6)
        img_out = img_rgb * color_scale[:, :, np.newaxis]

        # [ENGINEERING_ADAPTATION] Standard display mapping (Gamma 2.2)
        img_out = np.clip(img_out, 0.0, 1.0)
        return (np.power(img_out, 1.0 / 2.2) * 255).astype(np.uint8)


def load_hdr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Missing {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Durand 2002 Fast Bilateral TMO - Strict Reference Implementation"
    )
    parser.add_argument(
        "input", type=str, nargs="?", default="HDR/dataset/memorial/memorial.hdr"
    )

    group_alg = parser.add_argument_group("Algorithm Parameters")
    group_alg.add_argument(
        "--base_contrast",
        type=float,
        default=50.0,
        help="Target linear contrast ratio for base layer (default 50.0)",
    )
    group_alg.add_argument(
        "--sigma_r",
        type=float,
        default=0.4,
        help="Intensity domain sigma in log10 units (default 0.4)",
    )

    group_perf = parser.add_argument_group("Performance & Artifacts")
    group_perf.add_argument(
        "--subsample",
        type=int,
        default=10,
        help="Spatial downsampling factor (default 10)",
    )

    group_io = parser.add_argument_group("Input/Output")
    group_io.add_argument("--output", type=str, help="Output image path")

    args = parser.parse_args()

    if args.output is None:
        out_dir = Path("HDR/output")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"{Path(args.input).stem}_durand2002.png")

    print(f"--- Processing {args.input} with Fast Bilateral Filter ---")
    print(
        f"Parameters: Base Contrast={args.base_contrast}, sigma_r={args.sigma_r}, Subsample={args.subsample}"
    )

    img_hdr = load_hdr(args.input)
    tmo = Durand2002(
        base_contrast=args.base_contrast,
        sigma_r=args.sigma_r,
        subsample_factor=args.subsample,
    )
    result = tmo.process(img_hdr)
    cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"✅ Result saved: {args.output}")
