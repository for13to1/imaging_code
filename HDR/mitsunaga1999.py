"""
Mitsunaga 1999 Radiometric Self-Calibration (RASCAL).

Based on:
Mitsunaga, T., & Nayar, S.K. (1999). Radiometric self calibration.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Reference source code consulted:
Columbia University CAVE-RASCAL (C++), Oct 31 1998.

This implementation strictly follows the paper equations where possible.
Deviations for stability are noted with [CAVE-RASCAL Reference] markers.
"""

from pathlib import Path
from typing import List, Tuple
import traceback
import struct
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear


class Mitsunaga1999:
    """
    [PAPER_STRICT] Implementation following Mitsunaga & Nayar 1999 paper.

    Key equations implemented:
    - Eq 3: Polynomial model I = f(M) = sum_{n=0}^N c_n M^n
    - Eq 7: Error function E = sum[sum(c_n M_q^n) - R * sum(c_n M_{q+1}^n)]^2
    - Eq 8: Constraint c_N = I_max - sum_{n=0}^{N-1} c_n (i.e., f(1) = I_max)
    - Eq 10: [Adjusted] Ratio update R = sum f(M_q) / sum f(M_{q+1}) (Ratio-of-Sums)
    - Eq 11: Convergence |f^k(M) - f^{k-1}(M)| < epsilon
    - Eq 14: SNR weight w = f(M) / f'(M)
    """

    def __init__(
        self,
        samples=5000,
        max_order=10,
        noise_level=0.02,
        sat_level=0.98,
        i_max=1.0,
        convergence_thresh=1e-6,
    ):
        self.samples = samples
        self.max_order = max_order
        self.noise_level = noise_level
        self.sat_level = sat_level
        self.i_max = i_max  # I_max for constraint f(1) = I_max
        self.convergence_thresh = convergence_thresh

    def _get_samples(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        [CAVE-RASCAL Reference] Uniform Intensity Sampling (Section 6.1/6.2 logic).
        Ensures samples are uniformly distributed across the intensity range 0.0-1.0.
        """
        rows, cols = images[0].shape[:2]
        h_m, w_m = int(rows * 0.1), int(cols * 0.1)

        avg_img = np.mean(images, axis=0)
        lum = np.mean(avg_img, axis=2)

        valid_indices = []
        bins = np.linspace(0, 1, 101)
        samples_per_bin = self.samples // 100

        for i in range(100):
            mask = (lum >= bins[i]) & (lum < bins[i + 1])
            mask[0:h_m, :] = False
            mask[-h_m:, :] = False
            mask[:, 0:w_m] = False
            mask[:, -w_m:] = False

            idx = np.where(mask.flatten())[0]
            if len(idx) > 0:
                s_idx = np.random.choice(
                    idx, min(samples_per_bin, len(idx)), replace=False
                )
                valid_indices.extend(s_idx)

        if len(valid_indices) < self.samples:
            remain = self.samples - len(valid_indices)
            all_v = np.where(np.ones((rows, cols), dtype=bool).flatten())[0]
            extra = np.random.choice(all_v, remain, replace=False)
            valid_indices.extend(extra)

        y, x = np.unravel_index(valid_indices, (rows, cols))
        return [img[y, x].astype(np.float64) / 255.0 for img in images]

    def _solve_coefficients(self, m_q, m_q1, R_flat, degree):
        """
        [PAPER_STRICT] Equation 8: Strict f(1) = 1 constraint.
        [STABILITY] Force f(0) = 0 (c_0 = 0) to avoid the degenerate f(M)=1 solution.
        """
        if degree < 1:
            return np.array([self.i_max])

        # We need to solve for c_1 ... c_N
        # c_0 is fixed to 0.
        # c_N is fixed to 1 - sum(c_1 ... c_{N-1})

        # Basis matrix for n from 1 to degree
        X = np.zeros((m_q.size, degree), dtype=np.float64)
        for n in range(1, degree + 1):
            X[:, n - 1] = (m_q**n) - R_flat * (m_q1**n)

        # If degree is 1, c_1 must be 1 (due to f(1)=1 and c_0=0)
        if degree == 1:
            return np.array([0.0, self.i_max])

        # Substitute c_N = 1 - sum_{n=1}^{N-1} c_n
        A_sub = X[:, :-1] - X[:, -1:]
        b_sub = -X[:, -1] * self.i_max

        res = lsq_linear(A_sub, b_sub, bounds=(-np.inf, np.inf), method="bvls")
        c_hidden = res.x  # These are c_1 ... c_{N-1}

        coeffs = np.zeros(degree + 1)
        coeffs[1:-1] = c_hidden
        coeffs[-1] = self.i_max - np.sum(c_hidden)
        # coeffs[0] remains 0.0

        return coeffs

    def _check_monotonicity(self, coeffs):
        """[PAPER_STRICT] Section 6.3: Monotonicity requirement."""
        poly = coeffs[::-1]
        x = np.linspace(0, 1, 101)
        y = np.polyval(poly, x)
        return np.all(np.diff(y) >= -1e-7)

    def _eval_poly(self, coeffs, x):
        return np.polyval(coeffs[::-1], x)

    def calibrate_channel(
        self,
        m_q: np.ndarray,
        m_q1: np.ndarray,
        pair_samples: List[Tuple[np.ndarray, np.ndarray]],
        init_ratios: np.ndarray,
        fixed_order: int = None,
        target_ratios: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Runs iterative self-calibration for a single channel (Section 4).

        Args:
            m_q: Sampled intensities from image q.
            m_q1: Sampled intensities from image q+1.
            pair_samples: List of intensity pairs (m_q, m_q1) for each exposure.
            init_ratios: Initial exposure ratios (usually from EXIF).
            fixed_order: If set, uses this fixed polynomial degree.
            target_ratios: If set, uses these fixed ratios instead of updating.

        Returns:
            Tuple of (best_coefficients, best_ratios, optimal_order).
        """
        num_pairs = len(pair_samples)
        points_per_pair = pair_samples[0][0].size
        best_coeffs, best_ratios, min_err = None, None, float("inf")

        # [PAPER_STRICT] Section 6.4: Same degree for all channels.
        orders = (
            [fixed_order] if fixed_order is not None else range(1, self.max_order + 1)
        )

        for d in orders:
            ratios = (
                target_ratios.copy()
                if target_ratios is not None
                else init_ratios.copy()
            )
            prev_coeffs = np.zeros(d + 1)

            for iteration in range(50):
                mask = (m_q > self.noise_level) & (m_q1 < self.sat_level)
                if not np.any(mask):
                    mask = np.ones_like(m_q, dtype=bool)

                r_flat = np.repeat(ratios, points_per_pair)
                coeffs = self._solve_coefficients(
                    m_q[mask], m_q1[mask], r_flat[mask], d
                )

                if target_ratios is not None:
                    break

                # [STABILITY] Staged Initialization used in CAVE reference.
                if iteration < 10:
                    continue

                # [CAVE-RASCAL] Ratio of Sums Update (Eq 10).
                for q in range(num_pairs):
                    s_q, s_q1 = pair_samples[q]
                    valid = (s_q > self.noise_level) & (s_q1 < self.sat_level)
                    if np.sum(valid) > 10:
                        f_q = self._eval_poly(coeffs, s_q[valid])
                        f_q1 = self._eval_poly(coeffs, s_q1[valid])
                        ratios[q] = np.sum(f_q) / (np.sum(f_q1) + 1e-10)

                if np.sum(np.abs(prev_coeffs)) > 0:
                    x_test = np.linspace(0.1, 0.9, 50)
                    delta = np.max(
                        np.abs(
                            self._eval_poly(coeffs, x_test)
                            - self._eval_poly(prev_coeffs, x_test)
                        )
                    )
                    if delta < self.convergence_thresh:
                        break
                prev_coeffs = coeffs.copy()

            if fixed_order is None and not self._check_monotonicity(coeffs):
                continue

            r_flat = np.repeat(ratios, points_per_pair)
            err = np.mean(
                (self._eval_poly(coeffs, m_q) - r_flat * self._eval_poly(coeffs, m_q1))
                ** 2
            )
            if err < min_err:
                min_err, best_coeffs, best_ratios = err, coeffs, ratios

        if best_coeffs is None:
            r_fallback = target_ratios if target_ratios is not None else init_ratios
            best_coeffs = self._solve_coefficients(
                m_q, m_q1, np.repeat(r_fallback, points_per_pair), 1
            )
            best_ratios = r_fallback

        return best_coeffs, best_ratios, best_coeffs.size - 1

    def calibrate(
        self, images: List[np.ndarray], init_ratios: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Orchestrates multi-channel calibration (Section 6.4).

        Independently recovers the CRF for Red, Green, and Blue channels while
        ensuring physical exposure ratios remain consistent across the sensor.

        Args:
            images: List of LDR images.
            init_ratios: Initial exposure ratios between images.

        Returns:
            Tuple of (green_coefficients, green_ratios, optimal_order).
        """
        sampled_images = self._get_samples(images)
        self.channel_coeffs = [None, None, None]
        self.channel_ratios = [None, None, None]

        c_order = [1, 0, 2]  # Green (Master Anchor), Red, Blue
        optimal_order = None

        for c in c_order:
            name = ["Red", "Green", "Blue"][c]
            print(f"  > Calibrating {name} channel...")

            pair_samples = []
            for q in range(len(sampled_images) - 1):
                pair_samples.append(
                    (sampled_images[q][:, c], sampled_images[q + 1][:, c])
                )

            current_m_q = np.concatenate([p[0] for p in pair_samples])
            current_m_q1 = np.concatenate([p[1] for p in pair_samples])

            # Step A: Independent Calibration.
            # Following Section 6.4, R, G, and B independently solve for both
            # the response function f(M) and the exposure ratios.
            # However, the paper explicitly states: "the transfer function f(M)
            # was assumed to be of the same degree for all channels."
            coeffs, ratios, order = self.calibrate_channel(
                current_m_q,
                current_m_q1,
                pair_samples,
                init_ratios,
                fixed_order=optimal_order,
            )

            self.channel_coeffs[c] = coeffs
            self.channel_ratios[c] = ratios

            if c == 1:
                optimal_order = order
                # If we just determined the optimal order for Green, we should
                # update c_order to ensure we don't need to re-scan for R/B.
                # Since Green is first in c_order=[1, 0, 2], optimal_order is
                # correctly populated for the remaining channels.

            print(f"    - {name} scale complete (Order N={order})")

        return self.channel_coeffs[1], self.channel_ratios[1], optimal_order

    def combine(self, images: List[np.ndarray]) -> np.ndarray:
        """Synthesizes HDR radiance map from the LDR stack (Section 5).

        Uses SNR-based weighting (Eq 14) to combine multiple exposures.

        Args:
            images: List of LDR images normalized to [0, 1].

        Returns:
            A linear radiance map [H, W, 3].
        """
        rows, cols, channels = images[0].shape
        radiance_map = np.zeros((rows, cols, channels), dtype=np.float64)

        for c in range(channels):
            coeffs = self.channel_coeffs[c]
            ratios = self.channel_ratios[c]

            num_im = len(images)
            exposure = np.zeros(num_im)
            sum_prod = 0.0
            for i in range(num_im):
                prod = 1.0
                for j in range(i, len(ratios)):
                    prod *= ratios[j]
                sum_prod += prod

            exposure[num_im - 1] = num_im / sum_prod
            for i in range(num_im - 1, 0, -1):
                exposure[i - 1] = exposure[i] * ratios[i - 1]

            exposure_norm = exposure / np.mean(exposure)
            c_prime = np.polyder(coeffs[::-1])

            channel_radiance = np.zeros((rows, cols), dtype=np.float64)
            weight_sum = np.zeros((rows, cols), dtype=np.float64)

            for q in range(num_im):
                m = images[q][:, :, c].astype(np.float64) / 255.0
                f_m = self._eval_poly(coeffs, m)
                f_p_m = np.polyval(c_prime, m)

                # [PAPER_STRICT] Eq 14: Weighting based on SNR (f / f').
                # Ensures pixels in high-slope regions of the CRF contribute more.
                w = np.abs(f_m / (f_p_m + 1e-12))
                mask = (m >= self.noise_level) & (m <= self.sat_level)
                w *= mask

                channel_radiance += w * (f_m / exposure_norm[q])
                weight_sum += w

            channel_radiance /= weight_sum + 1e-8

            max_val = np.max(channel_radiance)
            if max_val > 1e-8:
                channel_radiance /= max_val
            radiance_map[:, :, c] = channel_radiance

        return radiance_map

    def balance_chromaticity(self, hdr: np.ndarray, ldr_ref: np.ndarray) -> np.ndarray:
        """Balances HDR color using Global Least Squares Chromaticity Alignment (Section 6.4).

        This replaces the engineering 'Neutral Prior' with the mathematically
        rigorous global formulation from the original paper.

        Args:
            hdr: Unbalanced HDR radiance map.
            ldr_ref: Reference LDR image (usually the middle exposure).

        Returns:
            Chromaticity-balanced linear radiance map [H, W, 3].
        """
        rows, cols, _ = hdr.shape
        rad = hdr.reshape(-1, 3)
        ldr = ldr_ref.reshape(-1, 3).astype(np.float64) / 255.0

        # Masking: Use only valid, non-saturated, non-noisy pixels.
        # Note: We NO LONGER filter for neutral (gray) colors, following the global objective.
        valid_mask = (np.mean(ldr, axis=1) > self.noise_level) & (
            np.mean(ldr, axis=1) < self.sat_level
        )

        rad_v = rad[valid_mask]
        ldr_v = ldr[valid_mask]

        if len(rad_v) < 100:
            print("    [Warning] Too few valid pixels for global balancing.")
            return hdr.astype(np.float32)

        # [PAPER_STRICT] Section 6.4: Cross-channel alignment.
        # We solve for k_g, k_b assuming k_r = 1.0.
        # This matches the Cramer's Rule implementation in RRColorBalanceProc.cpp (Lines 175-221).
        ir, ig, ib = rad_v[:, 0], rad_v[:, 1], rad_v[:, 2]
        mr, mg, mb = ldr_v[:, 0], ldr_v[:, 1], ldr_v[:, 2]

        irg_mrg = np.sum(ir * ig * mr * mg)
        irb_mrb = np.sum(ir * ib * mr * mb)
        igg_mrr = np.sum(ig * ig * mr * mr)
        igg_mbb = np.sum(ig * ig * mb * mb)
        ibb_mrr = np.sum(ib * ib * mr * mr)
        ibb_mgg = np.sum(ib * ib * mg * mg)
        igb_mgb = np.sum(ig * ib * mg * mb)

        denom = (igg_mrr + igg_mbb) * (ibb_mrr + ibb_mgg) - igb_mgb**2
        if abs(denom) < 1e-12:
            print("    [Error] Singular matrix in global balancing.")
            return hdr.astype(np.float32)

        k_g = (irg_mrg * (ibb_mrr + ibb_mgg) + irb_mrb * igb_mgb) / denom
        k_b = (irb_mrb * (igg_mrr + igg_mbb) + irg_mrg * igb_mgb) / denom

        scales = np.array([1.0, k_g, k_b])
        scales /= np.max(scales)

        print(
            f"    Chromaticity scales: R={scales[0]:.4f}, G={scales[1]:.4f}, B={scales[2]:.4f}"
        )

        # [RADIANCE_NORMALIZATION] Anchor middle-gray for standard viewers.
        lum = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
        avg_lum = np.exp(np.mean(np.log(np.maximum(lum, 1e-6))))
        hdr *= 0.18 / (avg_lum + 1e-8)

        return (hdr * scales).astype(np.float32)

    def plot_crfs(self, save_path: str, title: str = "Recovered CRF (Mitsunaga 1999)"):
        """Plots the recovered response functions in log scale.

        Args:
            save_path: Path to save the resulting PNG plot.
            title: Title for the chart.
        """
        plt.figure(figsize=(8, 5))
        m = np.linspace(0, 1, 256)
        colors = ["r", "g", "b"]
        labels = ["Red Channel", "Green Channel", "Blue Channel"]

        for i in range(3):
            coeffs = self.channel_coeffs[i]
            f_m = np.polyval(coeffs[::-1], m)
            # Log Exposure ln(f(M)) consistent with Debevec 1997 visuals
            plt.plot(
                m * 255.0,
                np.log(f_m + 1e-8),
                color=colors[i],
                label=labels[i],
                linewidth=2,
            )

        plt.title(title)
        plt.xlabel("Pixel Value (M)")
        plt.ylabel("Log Exposure ln(f(M))")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.savefig(save_path, dpi=150)
        plt.close()


def save_rgbe(filename: str, radiance_map: np.ndarray) -> None:
    """Standard Radiance RGBE (.hdr) encoder for cross-viewer compatibility."""
    radiance_map = np.asarray(radiance_map, dtype=np.float32)
    rows, cols, _ = radiance_map.shape
    max_component = np.max(radiance_map, axis=-1, keepdims=True)
    shared_exp = np.where(
        max_component > 1e-32, np.floor(np.log2(max_component) + 129), 0
    ).astype(np.uint8)
    scale = np.power(2.0, shared_exp.astype(np.float32) - 128)
    mantissa = np.where(
        shared_exp > 0, np.clip(radiance_map * 256.0 / scale, 0, 255), 0
    ).astype(np.uint8)

    with open(filename, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=1.0\n\n")
        f.write(f"-Y {rows} +X {cols}\n".encode())
        for i in range(rows):
            for j in range(cols):
                f.write(
                    struct.pack(
                        "BBBB",
                        mantissa[i, j, 0],
                        mantissa[i, j, 1],
                        mantissa[i, j, 2],
                        shared_exp[i, j, 0],
                    )
                )


def load_image_series(directory: str) -> Tuple[List[np.ndarray], np.ndarray]:
    dir_path = Path(directory)
    list_path = dir_path / "image_list.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing image_list.txt in {directory}")

    images, times = [], []
    with open(list_path, "r") as f:
        f.readline()
        line = f.readline().strip()
        if not line:
            raise ValueError("Empty image count")
        n_images = int(line)
        f.readline()

        for _ in range(n_images):
            line = f.readline().strip()
            if not line:
                break
            parts = line.split()
            if len(parts) < 2:
                continue
            img_path = str(dir_path / parts[0])
            img = cv2.imread(img_path)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # parts[1] is 1/exposure_time (Section 2.2)
                times.append(1.0 / float(parts[1]))

    idx = np.argsort(times)
    images = [images[i] for i in idx]
    times = np.array(times)[idx]
    return images, times


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="memorial")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "dataset" / args.dataset
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Loading dataset: {args.dataset} ---")
    try:
        images, times = load_image_series(str(dataset_path))
        init_ratios = times[:-1] / times[1:]
        rascal = Mitsunaga1999()

        print("--- Phase I: Self-Calibration ---")
        coeffs, ratios, optimal_order = rascal.calibrate(images, init_ratios)
        print(f"Optimal Polynomial Order N: {optimal_order}")

        print("--- Phase II: Radiance Synthesis ---")
        hdr = rascal.combine(images)

        # Use middle image as color reference (Section 6.4 suggestion)
        hdr_balanced = rascal.balance_chromaticity(hdr, images[len(images) // 2])

        # Save results
        out_path = output_dir / f"{args.dataset}_mitsunaga1999.hdr"
        save_rgbe(str(out_path), hdr_balanced)
        print(f"✅ Generated HDR: {out_path}")

        plot_path = output_dir / f"{args.dataset}_mitsunaga1999_crf.png"
        rascal.plot_crfs(
            str(plot_path), title=f"Recovered CRF - {args.dataset} (Mitsunaga 1999)"
        )
        print(f"✅ Generated CRF: {plot_path}")

    except Exception:
        traceback.print_exc()
