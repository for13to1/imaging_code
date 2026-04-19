"""
Larson/Ward 1997 HDR Tone Reproduction Implementation.

Based on:
Larson, G.W., Rushmeier, H., & Piatko, C. (1997).
A visibility matching tone reproduction operator for high dynamic range scenes.
IEEE Transactions on Visualization and Computer Graphics.
"""

from pathlib import Path
import traceback
import numpy as np
import cv2


class Ward1997:
    """
    ENGINEERING ASSUMPTIONS & DISCREPANCIES (Marker System):
    - [PAPER_STRICT]: Directly from the Larson 1997 paper (formula, constant, or step).
    - [ENGINEERING_ADAPTATION]: Omitted details, ambiguities, or modern stability/performance heuristics.

    1. Color Space: [ENGINEERING_ADAPTATION] This implementation assumes sRGB/D65.
       [PAPER_STRICT] Section 5.2 notes lack of RGB standards and uses CIE XYZ.
       Radiance (Larson's system) uses Equal Energy (EE) white point (1/3, 1/3).
    2. Calibration: [PAPER_STRICT] Assumes calibrated cd/m2. [ENGINEERING_ADAPTATION]
       Provides a 'scale' heuristic (default 60) for uncalibrated HDR radiance.
    3. Numerical Stability: [ENGINEERING_ADAPTATION] Added epsilons (1e-7, 1e-8)
       to prevent division by zero in scotopic/glare formulas, not discussed in paper.
    4. Acuity Filter: [ENGINEERING_ADAPTATION] The paper specifies a "Mip Map"
       but doesn't define the Gaussian sigma for blur. We use sigma = 1/(2R) (cycles/deg)
       as a standard engineering mapping to the MTF cutoff frequency.
    5. Logarithm Base: [PAPER_STRICT] Larson 1997 Section 4.1 defines 'log'
       as natural log (ln). This implementation strictly uses ln (np.log)
       for histogram binning, display mapping, and the ceiling formula (Eq 7c).
       [ENGINEERING_ADAPTATION] Table 1 data and Eq 15 explicitly use log10
       in the paper, so we perform local base conversion for those formulas.
    """

    def __init__(
        self,
        enable_glare=True,
        enable_acuity=True,
        enable_color=True,
        n_bins=100,
        ld_max=100.0,
        ld_min=1.0,
        scale=1.0,
        calibrated=True,
        tolerance=0.025,
    ):
        """
        Initialize the TMO with display and behavior parameters.

        Args:
            enable_glare: Include veiling glare in the adaptation map.
            enable_acuity: Apply spatially varying acuity loss blur.
            enable_color: Include mesopic/scotopic color sensitivity.
            n_bins: Number of histogram bins (default 100 as per paper Section 4.2).
            ld_max: Maximum display luminance (Ld_max, default 100 cd/m2).
            ld_min: Minimum display luminance (Ld_min, default 1 cd/m2).
            scale: Calibration factor to map input data to absolute cd/m2 (default 1.0 assumes calibrated input).
            calibrated: If True, input is assumed to be in absolute cd/m2 (paper assumption).
                       If False, scale factor will be auto-computed from median luminance.
            tolerance: Convergence criterion for the ceiling algorithm (Larson default: 0.025).
        """
        self.enable_glare = enable_glare
        self.enable_acuity = enable_acuity
        self.enable_color = enable_color
        self.n_bins = n_bins
        self.tolerance = tolerance
        # default scale 60 is an empirical heuristic for typical Radiance/HDR files
        # to map radiance (W/sr/m2) to absolute luminance (cd/m2).
        self.scale = scale
        self.calibrated = calibrated

        # Display parameters
        self.ld_max = ld_max
        self.ld_min = ld_min
        # [PAPER_STRICT] Section 4.1: log is natural log (ln)
        self.ln_ld_min = np.log(self.ld_min)
        self.ln_ld_max = np.log(self.ld_max)
        self.ln_ld_range = self.ln_ld_max - self.ln_ld_min

        # Physiological Constants (Larson 1997 Section 5.1 & 5.2)
        self.VFOVEAL = 0.913  # Foveal weight (Eq 8 and Eq 12)
        self.VGLARE = 0.087  # Peripheral/Veil weight (Eq 12)
        self.TOP_MESOPIC = 5.6  # cd/m2 (Section 5.2: "approximately 5.6 cd/m2")
        self.BOT_MESOPIC = 0.0056  # cd/m2 (Section 5.2: "approximately 0.0056 cd/m2")
        # [PAPER_STRICT] Section 4.2: lower threshold of human vision
        self.L_WORLD_MIN = 1e-4  # cd/m2

        # [ENGINEERING_ADAPTATION] Matrices follow modern sRGB/Rec.709 (D65) standard.
        # [PAPER_STRICT] Section 5.2 states there was no RGB standard at the time.
        # Note: Radiance uses Equal Energy (EE) matrices matching Larson's results.
        self.rgb2xyz = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        self.xyz2rgb = np.linalg.inv(self.rgb2xyz)

    def rgb_to_xyz(self, img_rgb: np.ndarray) -> np.ndarray:
        """Converts linear RGB (with sRGB/Rec.709 primaries) to CIE XYZ."""
        return np.dot(img_rgb, self.rgb2xyz.T)

    def xyz_to_rgb(self, img_xyz: np.ndarray) -> np.ndarray:
        """Converts CIE XYZ back to linear RGB (sRGB/Rec.709 primaries)."""
        return np.dot(img_xyz, self.xyz2rgb.T)

    def threshold_function_ferwerda(self, la: np.ndarray) -> np.ndarray:
        """
        TVI (Threshold vs Intensity) piecewise approximation from Larson/Ward 1997 Table 1.
        Returns the delta-luminance (JND - Just Noticeable Difference) threshold.
        """
        # [PAPER_STRICT] Table 1 data is explicitly given in log10(La).
        log10_la = np.log10(np.maximum(la, 1e-6))
        log10_dl = np.zeros_like(log10_la)

        # Five-segment fit to threshold data (from scotopic to photopic)
        m1 = log10_la < -3.94
        log10_dl[m1] = -2.86
        m2 = (log10_la >= -3.94) & (log10_la < -1.44)
        log10_dl[m2] = np.power(0.405 * log10_la[m2] + 1.6, 2.18) - 2.86
        m3 = (log10_la >= -1.44) & (log10_la < -0.0184)
        log10_dl[m3] = log10_la[m3] - 0.395
        m4 = (log10_la >= -0.0184) & (log10_la < 1.9)
        log10_dl[m4] = np.power(0.249 * log10_la[m4] + 0.65, 2.7) - 0.72
        m5 = log10_la >= 1.9
        log10_dl[m5] = log10_la[m5] - 1.255

        return np.power(10.0, log10_dl)

    def _prepare_adaptation_map(self, img_xyz, fov_deg):
        """Phase I: Preparation (Larson/Ward 1997 Section 6, Steps 1-4)."""
        rows, cols, _ = img_xyz.shape
        rad_fov = np.radians(fov_deg)

        # Step 1: Compute 1° foveal sample image (Eq 1)
        # [ENGINEERING_ADAPTATION] Fallback for wide angles or fisheye images.
        # [PAPER_STRICT] Eq 1 assumes rectilinear perspective: S = 2 * tan(theta/2) / 0.01745
        if fov_deg < 120.0:
            target_w = int(max(1, 2 * np.tan(np.radians(fov_deg) / 2) / 0.01745))
        else:
            target_w = int(max(1, fov_deg))  # Fallback logic not in paper.

        target_h = int(max(1, target_w * (rows / cols)))
        target_size = (int(target_w), int(target_h))

        # [PAPER_STRICT] Section 4.2: Box filter original image to foveal samples.
        foveal_xyz = cv2.boxFilter(
            img_xyz,
            -1,
            (max(1, cols // target_size[0]), max(1, rows // target_size[1])),
        )
        foveal_xyz = cv2.resize(
            foveal_xyz, target_size, interpolation=cv2.INTER_NEAREST
        )
        foveal_y = foveal_xyz[:, :, 1]

        if not self.enable_glare:
            return foveal_y, img_xyz, target_w

        # Step 2: Compute veil image (Eq 9 & 10)
        # Simulates intraocular scattering of bright peripheral sources.
        y_dir, x_dir = np.mgrid[0 : target_size[1], 0 : target_size[0]]

        ang_x = (x_dir / (target_size[0] - 1.0 + 1e-8) - 0.5) * rad_fov
        ang_y = (
            (y_dir / (target_size[1] - 1.0 + 1e-8) - 0.5)
            * rad_fov
            * (target_size[1] / target_size[0])
        )

        directions = np.stack(
            [
                np.sin(ang_x) * np.cos(ang_y),
                np.sin(ang_y),
                np.cos(ang_x) * np.cos(ang_y),
            ],
            axis=-1,
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

        xyz_flat = foveal_xyz.reshape(-1, 3)
        vecs = directions.reshape(-1, 3)
        veil_xyz_flat = np.zeros_like(xyz_flat)
        block_size = 1000

        for i in range(0, len(xyz_flat), block_size):
            end_i = min(i + block_size, len(xyz_flat))
            cos_a = np.clip(np.dot(vecs[i:end_i], vecs.T), -1, 1)

            with np.errstate(divide="ignore", invalid="ignore"):
                # [PAPER_STRICT] Eq 10 weighting: W = cos(theta) / (2 - 2 * cos(theta))
                # [ENGINEERING_ADAPTATION] Added 1e-7 epsilon for stability.
                weights = cos_a / (2.0 - 2.0 * cos_a + 1e-7)
                # [PAPER_STRICT] Eq 8 foveal half angle theta_f = 0.5 deg (0.00873 rad)
                weights[cos_a >= np.cos(0.00873)] = 0.0
                weights[np.isinf(weights)] = 0.0

            w_sum = np.sum(weights, axis=1, keepdims=True)
            veil_xyz_flat[i:end_i] = self.VGLARE * (
                np.dot(weights, xyz_flat) / (w_sum + 1e-8)
            )

        veil_foveal_xyz = veil_xyz_flat.reshape(target_size[1], target_size[0], 3)
        veil_full_xyz = cv2.resize(
            veil_foveal_xyz, (cols, rows), interpolation=cv2.INTER_LINEAR
        )

        # Step 3: Add veil to foveal adaptation image (Eq 12)
        la_foveal = self.VFOVEAL * foveal_y + veil_foveal_xyz[:, :, 1]

        # Step 4: Add veil to image (Eq 11)
        img_xyz_veiled = self.VFOVEAL * img_xyz + veil_full_xyz
        return la_foveal, img_xyz_veiled, target_size[0]

    def _apply_physiological_models(self, img_xyz, la_foveal, cdm2_scale, foveal_w):
        """Phase II: Sensitivities (Larson/Ward 1997 Section 6, Steps 5-6)."""
        rows, cols, _ = img_xyz.shape
        la_abs_f = la_foveal * cdm2_scale

        # Step 5: Blur image locally based on visual acuity function (Eq 15)
        # Higher adaptation = higher acuity = sharper image.
        if self.enable_acuity:
            # [PAPER_STRICT] Eq 15: visual acuity function (cycles/deg)
            # Formula explicitly uses log10.
            acuity_f = (
                17.25 * np.arctan(1.4 * np.log10(np.maximum(la_abs_f, 1e-6)) + 0.35)
                + 25.72
            )
            # [ENGINEERING_ADAPTATION] Mapping R to Gaussian sigma.
            # Heuristic: sigma (deg) = 1 / (2 * acuity).
            # (cols/foveal_w) converts degrees to pixel units based on center 1° sample width.
            sigma_full_px = cv2.resize(
                1.0 / (2.0 * np.maximum(acuity_f, 1.0)), (cols, rows)
            ) * (cols / foveal_w)

            max_sigma = float(np.max(sigma_full_px))
            if max_sigma < 0.5:
                img_xyz = img_xyz
            else:
                pyramid = [img_xyz.astype(np.float32)]
                levels = min(6, int(np.ceil(np.log2(max_sigma + 1))) + 1)
                for l in range(1, levels):
                    pyramid.append(cv2.GaussianBlur(pyramid[0], (0, 0), sigmaX=2.0**l))
                l_idx = np.clip(np.log2(np.maximum(sigma_full_px, 1e-3)), 0, levels - 1)
                l_lo = l_idx.astype(int)
                l_hi = np.minimum(l_lo + 1, levels - 1)
                alpha = (l_idx - l_lo.astype(float))[:, :, np.newaxis]
                img_xyz = np.zeros_like(img_xyz, dtype=np.float32)
                for l in range(levels):
                    # [ENGINEERING_ADAPTATION] Added [:,:,np.newaxis] to boolean masks to ensure correct broadcasting across 3 channels.
                    m_lo = (l_lo == l).astype(np.float32)[:, :, np.newaxis]
                    m_hi = (l_hi == l).astype(np.float32)[:, :, np.newaxis]
                    img_xyz += pyramid[l] * (m_lo * (1.0 - alpha) + m_hi * alpha)

        # Step 6: Apply color sensitivity function (Mesopic interpolation)
        # Ramps from grayscale (scotopic) to full color (photopic) between 0.0056 and 5.6 cd/m2.
        if self.enable_color:
            la_abs_full = cv2.resize(la_abs_f, (cols, rows))
            X, Y, Z = img_xyz[:, :, 0], img_xyz[:, :, 1], img_xyz[:, :, 2]
            # [PAPER_STRICT] Eq 13: Scotopic luminance approximation (Macbeth Chart fit)
            # [ENGINEERING_ADAPTATION] Added 1e-8 epsilon in denominator.
            y_scot = np.maximum(Y * (1.33 * (1 + (Y + Z) / (X + 1e-8)) - 1.68), 0)
            k = np.clip(
                (la_abs_full - self.BOT_MESOPIC)
                / (self.TOP_MESOPIC - self.BOT_MESOPIC),
                0,
                1,
            )
            # [ENGINEERING_ADAPTATION] Map scotopic component to neutral D65 white point.
            # Paper does not specify the white point for scotopic mapping.
            img_xyz = np.stack(
                [
                    k * X + (1 - k) * y_scot * 0.95045,
                    k * Y + (1 - k) * y_scot,
                    k * Z + (1 - k) * y_scot * 1.08892,
                ],
                axis=-1,
            )

        return img_xyz

    def _build_tone_mapping_function(self, la_foveal, cdm2_scale):
        """Phase III: Histogram Adjustment (Larson/Ward 1997 Section 6, Steps 7-8)."""
        # Step 7: Generate histogram of effective adaptation image
        all_samples = la_foveal.flatten() * cdm2_scale

        # [PAPER_STRICT] Section 4.2: "For the minimum value, we use either the
        # minimum 1° spot average or 10^-4 cd/m2, whichever is larger."
        world_min_val = np.min(all_samples)
        hist_min_val = max(world_min_val, self.L_WORLD_MIN)
        hist_max_val = np.max(all_samples)

        # [ENGINEERING_ADAPTATION] To maintain the total population T (Section 4.2.1)
        # while respecting the lower threshold of vision, we clamp sub-threshold
        # samples to hist_min_val so they are counted in the first histogram bin.
        clamped_samples = np.clip(all_samples, hist_min_val, hist_max_val)
        ln_samples = np.log(clamped_samples)

        ln_min = np.log(hist_min_val)
        ln_max = np.log(hist_max_val)
        ln_avg = np.mean(ln_samples)

        # [PAPER_STRICT] Section 4.4: Linear Fallback. If scene dynamic range (log)
        # fits within display, use a simple linear operator to preserve contrast.
        # [ENGINEERING_ADAPTATION] Using geometric mean alignment (ln_avg to ln_ld_mid)
        # as the specific linear scale factor, which is not defined in the paper.
        if (ln_max - ln_min) <= self.ln_ld_range:
            ln_ld_mid = (self.ln_ld_max + self.ln_ld_min) / 2.0
            mapping_offset = ln_ld_mid - ln_avg
            return (ln_min, ln_max), mapping_offset

        hist, bin_edges = np.histogram(
            ln_samples, bins=self.n_bins, range=(ln_min, ln_max)
        )
        widths = bin_edges[1:] - bin_edges[:-1]
        centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

        # Step 8: Adjust histogram to contrast sensitivity function (Eq 7c)
        f = hist.astype(np.float32)
        total_samples_original = np.sum(f)

        for _ in range(30):
            total_n = np.sum(f)
            p = f / total_n
            p_cum = np.cumsum(p)
            p_cum_mid = p_cum - 0.5 * p

            # [PAPER_STRICT] Eq 4: Bde = ln(Ldmin) + [ln(Ldmax) - ln(Ldmin)] * P(Bw)
            ld = np.exp(self.ln_ld_min + self.ln_ld_range * p_cum_mid)
            lw = np.exp(centers)

            delta_ld = self.threshold_function_ferwerda(ld)
            delta_lw = self.threshold_function_ferwerda(lw)

            # [PAPER_STRICT] Eq 7c: Directly using natural log widths/ranges.
            ceiling = (delta_ld / delta_lw) * (
                total_n * widths * lw / (self.ln_ld_range * ld)
            )

            over = f > (ceiling + 1e-4)
            trimmings = np.sum(f[over] - ceiling[over])
            # [PAPER_STRICT] Section 4.4: Tolerance criterion (default 2.5 percent).
            if trimmings <= self.tolerance * total_samples_original:
                break

            # [ENGINEERING_ADAPTATION] Hard limit of 30 iterations for stability.
            f[over] = ceiling[over]

        p_mapping = np.insert(np.cumsum(f / np.sum(f)), 0, 0.0)
        return bin_edges, p_mapping

    def process(self, img_xyz, fov_deg=140.0) -> np.ndarray:
        """Execute the full 10-step Larson/Ward tone reproduction pipeline."""
        img_xyz = img_xyz.astype(np.float32)

        # Preparation & Glare (Steps 1-4)
        la_foveal, img_xyz_veiled, fov_w = self._prepare_adaptation_map(
            img_xyz, fov_deg
        )

        if self.calibrated:
            actual_scale = self.scale
        else:
            # [ENGINEERING_ADAPTATION] Heuristic auto-calibration: align median
            # luminance to the provided 'scale' factor (default 60 cd/m2)
            # as the paper assumes pre-calibrated absolute luminance.
            la_median = np.median(la_foveal[la_foveal > 1e-6])
            actual_scale = self.scale / (la_median + 1e-8)

        # Physiological Modeling (Steps 5-6)
        img_xyz_processed = self._apply_physiological_models(
            img_xyz_veiled, la_foveal, actual_scale, fov_w
        )

        # Building Mapping (Steps 7-8)
        bin_edges, p_mapping_data = self._build_tone_mapping_function(
            la_foveal, actual_scale
        )
        p_mapping = p_mapping_data if isinstance(p_mapping_data, np.ndarray) else None

        # Step 9: Apply histogram adjustment to image
        py_abs = img_xyz_processed[:, :, 1] * actual_scale
        ln_lw = np.log(np.maximum(py_abs, 1e-7))

        if p_mapping is None:
            # [PAPER_STRICT] Section 4.4: Linear Fallback mapping (preserving contrast)
            mapping_offset = p_mapping_data
            ln_ld_full = ln_lw + mapping_offset
        else:
            p_mapping = p_mapping_data
            # Interpolation in natural log space
            ln_ld_vals = self.ln_ld_min + self.ln_ld_range * p_mapping
            ln_ld_full = np.interp(ln_lw, bin_edges, ln_ld_vals)

        ld_full = np.exp(ln_ld_full)

        img_xyz_mapped = (
            img_xyz_processed
            * (ld_full / (py_abs / actual_scale + 1e-8))[:, :, np.newaxis]
        )

        # [PAPER_STRICT] Step 10: Translate CIE results to display RGB values
        img_rgb = self.xyz_to_rgb(img_xyz_mapped)
        # [PAPER_STRICT] Section 6 Step 10: Subtract black level Ld_min before gamma.
        img_rgb = np.clip(
            (img_rgb - self.ld_min) / (self.ld_max - self.ld_min + 1e-8), 0, 1
        )

        # [ENGINEERING_ADAPTATION] Gamma 2.2 approximation for sRGB display.
        # Paper does not specify a gamma value.
        return (np.power(img_rgb, 1.0 / 2.2) * 255).astype(np.uint8)


def load_hdr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Missing {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Larson/Ward 1997 Reference Implementation"
    )
    parser.add_argument(
        "input", type=str, nargs="?", default="HDR/dataset/memorial/memorial.hdr"
    )
    group_env = parser.add_argument_group("Environment & Display")
    group_env.add_argument(
        "--fov", type=float, default=140.0, help="Field of view in degrees"
    )
    group_env.add_argument(
        "--scale",
        type=float,
        default=60.0,
        help="Absolute luminance scale factor (default 60 for typical HDR files)",
    )
    group_env.add_argument(
        "--calibrated",
        action="store_true",
        help="Input is already in absolute cd/m2 (skip auto-calibration)",
    )
    group_env.add_argument(
        "--ld-max", type=float, default=100.0, help="Max display luminance (Ld_max)"
    )
    group_env.add_argument(
        "--ld-min", type=float, default=1.0, help="Min display luminance (Ld_min)"
    )

    # Physiological Models
    group_phys = parser.add_argument_group("Physiological Models (Toggles)")
    group_phys.add_argument(
        "--glare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable intraocular scattering simulation",
    )
    group_phys.add_argument(
        "--acuity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable spatial acuity loss blur",
    )
    group_phys.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable mesopic color sensitivity transition",
    )

    # Advanced Parameters
    group_adv = parser.add_argument_group("Algorithm Parameters")
    group_adv.add_argument(
        "--bins", type=int, default=100, help="Number of histogram bins (default 100)"
    )
    group_adv.add_argument(
        "--tolerance",
        type=float,
        default=0.025,
        help="Ceiling truncation tolerance (default 0.025)",
    )

    args = parser.parse_args()

    # Auto-output path: HDR/output/<input>_ward1997.png
    output_dir = Path("HDR/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"{Path(args.input).stem}_ward1997.png")

    print(f"--- Processing {args.input} (FOV={args.fov}, Scale={args.scale}) ---")
    try:
        img_hdr = load_hdr(args.input)
        tmo = Ward1997(
            enable_glare=args.glare,
            enable_acuity=args.acuity,
            enable_color=args.color,
            n_bins=args.bins,
            ld_max=args.ld_max,
            ld_min=args.ld_min,
            scale=args.scale,
            calibrated=args.calibrated,
            tolerance=args.tolerance,
        )
        result = tmo.process(tmo.rgb_to_xyz(img_hdr), fov_deg=args.fov)
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"✅ Result saved: {output_path}")
    except Exception:
        traceback.print_exc()
