"""
Mitsunaga 1999 Radiometric Self-Calibration (RASCAL).

[PAPER_STRICT]: Algorithms explicitly defined in the Section 3-5 formulas.
[CAVE_SUPPLEMENT]: Engineering logic found in the author's CAVE-RASCAL C++ source.
"""

from pathlib import Path
from typing import List, Tuple
import struct
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear


class Mitsunaga1999:
    """[PAPER_STRICT] High-fidelity RASCAL implementation."""

    def __init__(
        self,
        samples=5000,
        max_order=10,
        noise_level=0.02,
        sat_level=0.98,
        i_max=1.0,
        convergence_thresh=1e-6,
        neutral_thresh=0.1,
    ):
        self.samples = samples
        self.max_order = max_order
        self.noise_level = noise_level
        self.sat_level = sat_level
        self.i_max = i_max
        self.convergence_thresh = convergence_thresh
        self.neutral_thresh = neutral_thresh
        self.channel_coeffs = [None] * 3
        self.channel_ratios = [None] * 3

    def _get_samples(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """[PAPER_STRICT] Section 6.1 & [CAVE_SUPPLEMENT]: Flat area selection."""
        # Use the middle exposure as the reference for stability
        ref_idx = len(images) // 2
        ref_img = images[ref_idx].astype(np.float64) / 255.0
        avg_lum = np.mean(ref_img, axis=2)
        h, w = avg_lum.shape

        # [PAPER_STRICT] Section 6.1: Spatial flatness check (Chi-squared test)
        # s=25 (5x5 neighborhood), rejection ratio psi=0.05 index: Chi2_24(0.05) ~ 36.415
        s = 25
        chi2_thresh = 36.415
        sigma_t = 0.01  # [PAPER_STRICT] Default temporal noise estimate (Sec 6.1)

        mean = cv2.boxFilter(avg_lum, -1, (5, 5))
        mean_sq = cv2.boxFilter(avg_lum**2, -1, (5, 5))
        var_s = np.maximum(mean_sq - mean**2, 0)

        flat_mask = (s * var_s / (sigma_t**2)) <= chi2_thresh

        # [PAPER_STRICT] Section 6.2: Vignetting/Boundary avoidance
        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
        flat_mask[:margin_h, :] = False
        flat_mask[-margin_h:, :] = False
        flat_mask[:, :margin_w] = False
        flat_mask[:, -margin_w:] = False

        valid_indices = []
        bins = np.linspace(0, 1, 101)
        samples_per_bin = self.samples // 100

        flat_lum, flat_mask_rav = avg_lum.ravel(), flat_mask.ravel()

        for i in range(100):
            bin_m = (flat_lum >= bins[i]) & (flat_lum < bins[i + 1]) & flat_mask_rav
            idx = np.where(bin_m)[0]
            if idx.size > 0:
                valid_indices.extend(
                    np.random.choice(idx, min(samples_per_bin, idx.size), replace=False)
                )

        if len(valid_indices) < self.samples:
            # Fallback to general mask if too few flat areas found
            mask = np.zeros_like(avg_lum, dtype=bool)
            mask[margin_h:-margin_h, margin_w:-margin_w] = True
            idx = np.where(mask.ravel())[0]
            valid_indices.extend(
                np.random.choice(
                    idx, min(self.samples - len(valid_indices), idx.size), replace=False
                )
            )

        coords = np.unravel_index(valid_indices, (h, w))
        print(
            f"    - Sampled {len(valid_indices)} points (Spatial Filter: {np.sum(flat_mask_rav)/flat_mask_rav.size:.1%} flat)"
        )
        return [img[coords].astype(np.float64) / 255.0 for img in images]

    def _solve_coefficients(
        self, m0: np.ndarray, m1: np.ndarray, R: np.ndarray, degree: int
    ) -> np.ndarray:
        """[PAPER_STRICT] Eq 7 & 8: Solve for c_1...c_{N-1} with f(0)=0 and f(1)=I_max constraints."""
        n_range = np.arange(1, degree + 1)
        X = (m0[:, None] ** n_range) - (R[:, None] * (m1[:, None] ** n_range))
        A = X[:, :-1] - X[:, -1:]
        B = -self.i_max * X[:, -1]
        res = lsq_linear(A, B, method="trf")
        coeffs = np.zeros(degree + 1)
        coeffs[1:-1] = res.x
        coeffs[-1] = self.i_max - np.sum(res.x)
        return coeffs

    def calibrate(self, images: List[np.ndarray], init_ratios: np.ndarray):
        """[PAPER_STRICT] Section 6.4: Joint channel calibration (shared ratios and degree)."""
        sampled = self._get_samples(images)
        p_count = sampled[0].shape[0]

        best_err = float("inf")
        best_coeffs, best_joint_ratios = None, None
        optimal_d = 0

        print(f"  > Starting Joint Calibration (Max Degree N={self.max_order})...")
        for d in range(1, self.max_order + 1):
            ratios = init_ratios.copy()
            prev_f = np.zeros((3, 101))
            current_coeffs = [None] * 3

            # Joint Iterative Solver
            for iter_idx in range(50):
                # 1. Update Coefficients for each channel using current shared ratios
                for c in range(3):
                    ch_samples = [s[:, c] for s in sampled]
                    m_q, m_q1 = np.concatenate(ch_samples[:-1]), np.concatenate(
                        ch_samples[1:]
                    )
                    # Keep mask for individual channel saturation/noise
                    mask = (m_q > self.noise_level) & (m_q1 < self.sat_level)
                    if not np.any(mask):
                        mask = np.ones_like(m_q, dtype=bool)

                    r_flat = np.repeat(ratios, p_count)
                    current_coeffs[c] = self._solve_coefficients(
                        m_q[mask], m_q1[mask], r_flat[mask], d
                    )

                # 2. Update Shared Ratios using all channels (Eq 10 & [CAVE_SUPPLEMENT])
                for q in range(len(sampled) - 1):
                    sia0, sia1 = 0.0, 0.0
                    for c in range(3):
                        poly_c = current_coeffs[c][::-1]
                        s0, s1 = sampled[q][:, c], sampled[q + 1][:, c]
                        v = (s0 > self.noise_level) & (s1 < self.sat_level)
                        if np.sum(v) > 10:
                            # [CAVE_SUPPLEMENT] Ratio-of-Sums update for stability
                            sia0 += np.sum(np.polyval(poly_c, s0[v]))
                            sia1 += np.sum(np.polyval(poly_c, s1[v]))
                    if sia1 > 1e-12:
                        ratios[q] = sia0 / sia1

                # 3. Check Convergence (all channels) - [CAVE_SUPPLEMENT]
                curr_f = np.array(
                    [
                        np.polyval(current_coeffs[c][::-1], np.linspace(0, 1, 101))
                        for c in range(3)
                    ]
                )
                if np.max(np.abs(curr_f - prev_f)) < self.convergence_thresh:
                    break
                prev_f = curr_f

            # [CAVE_SUPPLEMENT] Evaluation of Optimal Degree N via Total Monotonic Error
            total_err = 0.0
            monotonic = True
            for c in range(3):
                poly_c = current_coeffs[c][::-1]
                # [CAVE_SUPPLEMENT] mfCheckSimpleIncreasing equivalent
                if not np.all(
                    np.diff(np.polyval(poly_c, np.linspace(0, 1, 101))) >= -1e-7
                ):
                    monotonic = False
                    break
                ch_samples = [s[:, c] for s in sampled]
                m_q, m_q1 = np.concatenate(ch_samples[:-1]), np.concatenate(
                    ch_samples[1:]
                )
                r_flat = np.repeat(ratios, p_count)
                total_err += np.mean(
                    (np.polyval(poly_c, m_q) - r_flat * np.polyval(poly_c, m_q1)) ** 2
                )

            if monotonic and total_err < best_err:
                best_err = total_err
                best_coeffs = [c.copy() for c in current_coeffs]
                best_joint_ratios = ratios.copy()
                optimal_d = d

        if best_coeffs is None:
            raise RuntimeError("Calibration failed to find monotonic solution")

        self.channel_coeffs = best_coeffs
        self.channel_ratios = [best_joint_ratios] * 3  # Shared physical ratios
        print(f"    - Joint Optimal Order N={optimal_d}, Mean Error: {best_err:.6e}")
        return self.channel_coeffs, best_joint_ratios, optimal_d

    def combine(self, images: List[np.ndarray]) -> np.ndarray:
        """[PAPER_STRICT] Section 5 & 6.3: SNR-weighted synthesis."""
        h, w, channels = images[0].shape
        rad_map = np.zeros((h, w, channels), dtype=np.float64)
        for c in range(channels):
            coeffs = self.channel_coeffs[c]
            exps = np.ones(len(images))
            exps[1:] = 1.0 / self.channel_ratios[c]
            exps = np.cumprod(exps)
            # [PAPER_STRICT] Sec 6.3: Normalize scaled exposures so their mean is 1
            exps /= np.mean(exps)

            poly, deriv = coeffs[::-1], np.polyder(coeffs[::-1])
            acc_rad, acc_w = np.zeros((h, w)), np.zeros((h, w))
            for q, img in enumerate(images):
                m = img[:, :, c].astype(np.float64) / 255.0
                f_m = np.polyval(poly, m)
                # [PAPER_STRICT] Eq 14: Weighting function w(M) = f(M) / f'(M)
                w_m = np.maximum(f_m, 0) / (np.polyval(deriv, m) + 1e-8)
                # [CAVE_SUPPLEMENT] Apply noise/saturation clip to weighting
                w_m *= (m > self.noise_level) & (m < self.sat_level)
                acc_rad += w_m * (f_m / exps[q])
                acc_w += w_m
            channel_rad = acc_rad / (acc_w + 1e-12)
            max_v = np.max(channel_rad)
            if max_v > 1e-8:
                channel_rad /= max_v
            rad_map[:, :, c] = channel_rad
        return rad_map

    def balance_chromaticity(
        self, hdr: np.ndarray, images: List[np.ndarray]
    ) -> np.ndarray:
        """[PAPER_STRICT] Section 6.4 & [CAVE_SUPPLEMENT]: Chromaticity Alignment.
        Uses CAVE 'Best M' composite reference for maximum input SNR.
        """
        print("  > Balancing chromaticity (CAVE Best-M Composite)...")
        h, w, _ = hdr.shape
        best_m = np.zeros((h, w, 3), dtype=np.float64)
        best_dist = np.full((h, w), np.inf)

        # 1. [CAVE_SUPPLEMENT] Synthesize the Best-M reference image
        for img in images:
            ldr = img.astype(np.float64) / 255.0
            avg_ldr = np.mean(ldr, axis=2)
            dist = np.abs(avg_ldr - 0.5)

            # Mask valid pixels only
            valid = (
                (ldr[:, :, 0] > self.noise_level)
                & (ldr[:, :, 0] < self.sat_level)
                & (ldr[:, :, 1] > self.noise_level)
                & (ldr[:, :, 1] < self.sat_level)
                & (ldr[:, :, 2] > self.noise_level)
                & (ldr[:, :, 2] < self.sat_level)
            )

            update_mask = valid & (dist < best_dist)
            best_m[update_mask] = ldr[update_mask]
            best_dist[update_mask] = dist[update_mask]

        # 2. [PAPER_STRICT] Least-Squares Alignment (Eq 19-21)
        avg_ldr = np.mean(best_m, axis=2)
        neutral_mask = (
            (np.abs(best_m[:, :, 0] - avg_ldr) / (avg_ldr + 1e-5) < self.neutral_thresh)
            & (
                np.abs(best_m[:, :, 1] - avg_ldr) / (avg_ldr + 1e-5)
                < self.neutral_thresh
            )
            & (
                np.abs(best_m[:, :, 2] - avg_ldr) / (avg_ldr + 1e-5)
                < self.neutral_thresh
            )
        )

        mask = (avg_ldr > self.noise_level) & (avg_ldr < self.sat_level) & neutral_mask
        if np.sum(mask) < 100:
            mask = (avg_ldr > self.noise_level) & (avg_ldr < self.sat_level)

        ir, ig, ib = hdr[mask].T
        mr, mg, mb = best_m[mask].T

        a11 = np.sum(ig**2 * mr**2 + ig**2 * mb**2)
        a12 = -np.sum(ig * ib * mg * mb)
        a22 = np.sum(ib**2 * mr**2 + ib**2 * mg**2)
        b1 = np.sum(ir * ig * mr * mg)
        b2 = np.sum(ir * ib * mr * mb)

        denom = a11 * a22 - a12**2
        if abs(denom) < 1e-12:
            return hdr.astype(np.float32)

        k_g = (b1 * a22 - b2 * a12) / denom
        k_b = (b2 * a11 - b1 * a12) / denom
        scales = np.array([1.0, k_g, k_b])
        scales /= np.max(scales)
        print(
            f"    Chromaticity scales: R={scales[0]:.4f}, G={scales[1]:.4f}, B={scales[2]:.4f}"
        )

        # Apply chromaticity alignment first
        hdr *= scales

        # Then normalize global intensity to 0.18 middle gray target
        lum = np.dot(hdr, [0.2126, 0.7152, 0.0722])
        avg_lum = np.exp(np.mean(np.log(np.maximum(lum, 1e-6))))
        hdr *= 0.18 / (avg_lum + 1e-8)
        return hdr.astype(np.float32)


def save_rgbe(filename: str, radiance_map: np.ndarray):
    """Standard Radiance RGBE encoder."""
    radiance_map = np.asarray(radiance_map, dtype=np.float32)
    h, w = radiance_map.shape[:2]
    max_c = np.max(radiance_map, axis=-1, keepdims=True)
    exp = np.where(max_c > 1e-32, np.floor(np.log2(max_c) + 129), 0).astype(np.uint8)
    scale = np.power(2.0, exp.astype(np.float32) - 128)
    mantissa = np.where(
        exp > 0, np.clip(radiance_map * 256.0 / scale, 0, 255), 0
    ).astype(np.uint8)
    with open(filename, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=1.0\n\n")
        f.write(f"-Y {h} +X {w}\n".encode())
        for i in range(h):
            for j in range(w):
                f.write(
                    struct.pack(
                        "BBBB",
                        mantissa[i, j, 0],
                        mantissa[i, j, 1],
                        mantissa[i, j, 2],
                        exp[i, j, 0],
                    )
                )


def load_image_series(directory: str) -> Tuple[List[np.ndarray], np.ndarray]:
    dir_path = Path(directory)
    with open(dir_path / "image_list.txt", "r") as f:
        f.readline()
        n = int(f.readline().strip())
        f.readline()
        data = [f.readline().split() for _ in range(n)]
    imgs = [
        cv2.cvtColor(cv2.imread(str(dir_path / d[0])), cv2.COLOR_BGR2RGB) for d in data
    ]
    times = np.array([1.0 / float(d[1]) for d in data])
    idx = np.argsort(times)
    return [imgs[i] for i in idx], times[idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="memorial")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = root / "dataset" / args.dataset

    out_dir = root / "output"
    out_dir.mkdir(exist_ok=True)
    try:
        images, times = load_image_series(str(dataset_path))
        rascal = Mitsunaga1999()
        print("--- Phase I: Calibration ---")
        rascal.calibrate(images, times[:-1] / times[1:])
        print("--- Phase II: Synthesis ---")
        hdr = rascal.balance_chromaticity(rascal.combine(images), images)
        save_rgbe(str(out_dir / f"{args.dataset}_mitsunaga1999.hdr"), hdr)
        print(f"✅ Generated HDR: {out_dir / f'{args.dataset}_mitsunaga1999.hdr'}")
    except Exception:
        import traceback

        traceback.print_exc()
