#!/usr/bin/env python3
"""
Estimation-Theoretic Approach to Dynamic Range Enhancement Using Multiple Exposures (Robertson 2003)

Based on:
Robertson, M. A., Borman, S., & Stevenson, R. L. (2003).
Estimation-theoretic approach to dynamic range enhancement using multiple exposures.
Journal of Electronic Imaging, 12(2), 219-228.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline


class Robertson2003:
    """
    Robertson 2003 HDR Algorithm Implementation.

    Implementation Fidelity Categories:
    1. [PAPER_STRICT]: Iterative CRF estimation (Section 4) and weighted ML reconstruction (Section 3).
    2. [ENGINEERING_ADAPTATION]:
       - Original paper uses single-channel (green channel) input. This implementation
         extends to multi-channel (RGB) by applying the algorithm independently to each channel.
       - Convergence threshold and max iterations for CRF estimation.
       - Small epsilon for numerical stability.
    """

    def __init__(
        self,
        max_iters: int = 20,
        threshold: float = 1e-4,
        ldr_size: int = 256,
    ):
        self.max_iters = max_iters
        self.threshold = threshold
        self.ldr_size = ldr_size
        self.weights_fixed = self._get_fixed_weights()

    def _get_fixed_weights(self) -> np.ndarray:
        """
        [PAPER_STRICT] Section 4, Figure 4: Weighting function for CRF estimation.
        A bell-shaped curve that gives less weight to pixels near the extremes.
        """
        # Using a Gaussian-like weight function similar to Figure 4.
        # w(y) = exp(- (y - 127.5)^2 / (2 * sigma^2))
        # Here we use a common bell curve.
        y = np.arange(self.ldr_size)
        mid = (self.ldr_size - 1) / 2.0
        sigma = mid / 2.0  # Heuristic for the shape in Fig 4
        weights = np.exp(-((y - mid) ** 2) / (2 * sigma**2))
        return weights.astype(np.float32)

    def _get_valid_range(self, images: list[np.ndarray]) -> tuple[int, int]:
        """
        [PAPER_STRICT] Section 2 & 5: Determine zero and saturation points from histograms.
        "The zero and saturation points are taken as the peaks of these two histograms."
        """
        all_pixels = np.concatenate([img.ravel() for img in images])
        hist = np.bincount(all_pixels, minlength=self.ldr_size)

        # Zero point: peak in the lower 25% of the range
        z_min = np.argmax(hist[: self.ldr_size // 4])
        # Saturation point: peak in the upper 25% of the range
        z_max = (self.ldr_size // 4 * 3) + np.argmax(hist[self.ldr_size // 4 * 3 :])

        print(f"Detected valid range: [{z_min}, {z_max}]")
        return int(z_min), int(z_max)

    def _compute_certainty_weights(self, response: np.ndarray, valid_range: tuple[int, int]) -> np.ndarray:
        """
        [PAPER_STRICT] Section 3: Weights based on the derivative of the CRF on a log-exposure axis.
        """
        channels = response.shape[1]
        certainty = np.zeros_like(response)
        y = np.arange(self.ldr_size, dtype=np.float32)
        z_min, z_max = valid_range

        for ch in range(channels):
            log_exposure = np.log(np.maximum(response[:, ch], 1e-8))

            # Clean up for strict monotonicity
            unique_idx = np.unique(log_exposure, return_index=True)[1]
            if len(unique_idx) < 2:
                certainty[:, ch] = self.weights_fixed
                continue

            clean_log_exp = log_exposure[unique_idx]
            clean_y = y[unique_idx]

            # [PAPER_STRICT] Cubic spline fit: Pixel Value (y) as function of Log Exposure (log I)
            # bc_type=((1, 0.0), (1, 0.0)) ensures zero derivative at endpoints
            cs = CubicSpline(clean_log_exp, clean_y, bc_type=((1, 0.0), (1, 0.0)))

            derivative = cs(log_exposure, 1)
            derivative = np.maximum(derivative, 0.0)

            # [PAPER_STRICT] Mask out values outside valid range
            derivative[: z_min + 1] = 0.0
            derivative[z_max:] = 0.0

            max_val = np.max(derivative)
            if max_val > 1e-8:
                certainty[:, ch] = derivative / max_val
            else:
                certainty[:, ch] = self.weights_fixed

        return certainty.astype(np.float32)

    def calibrate(self, images: list[np.ndarray], times: np.ndarray) -> np.ndarray:
        """
        [PAPER_STRICT] Section 4: Estimate the camera response function (CRF).
        Returns response function I_m (ldr_size, channels).
        """
        n_images = len(images)
        h, w, channels = images[0].shape
        times = times.astype(np.float32)

        # [PAPER_STRICT] Identify valid range from histograms
        valid_range = self._get_valid_range(images)

        # Initialize I_m as linear function, I_128 = 1.0
        response = np.tile(np.arange(self.ldr_size, dtype=np.float32)[:, np.newaxis], (1, channels))
        response /= 128.0

        y = np.stack(images, axis=0)  # (N, H, W, C)

        # [PAPER_STRICT] The initial x^(0) is chosen according to Eq. (7), using the initial linear I^(0)
        num = np.zeros((h, w, channels), dtype=np.float32)
        den = np.zeros((h, w, channels), dtype=np.float32)
        for i in range(n_images):
            I_val = response[y[i], np.arange(channels)]
            w_val = self.weights_fixed[y[i]]
            num += w_val * times[i] * I_val
            den += w_val * (times[i] ** 2)
        x = num / (den + 1e-8)

        prev_obj = float("inf")

        for iteration in range(self.max_iters):
            # [PAPER_STRICT] Gauss-Seidel relaxation order:
            # 1. Minimize with respect to each I_m
            # 2. Scale restriction
            # 3. Minimize with respect to each x_j

            # Step 1: Estimate response I_m (Eq 11) using x_j from previous iteration
            new_response = np.zeros_like(response)
            card = np.zeros_like(response)

            for i in range(n_images):
                for ch in range(channels):
                    weighted_x = times[i] * x[:, :, ch]
                    new_response[:, ch] += np.bincount(
                        y[i, :, :, ch].ravel(),
                        weights=weighted_x.ravel(),
                        minlength=self.ldr_size,
                    )
                    card[:, ch] += np.bincount(y[i, :, :, ch].ravel(), minlength=self.ldr_size)

            mask = card > 0
            new_response[mask] /= card[mask]

            for ch in range(channels):
                missing = ~mask[:, ch]
                if np.any(missing):
                    known_idx = np.where(mask[:, ch])[0]
                    if len(known_idx) > 1:
                        new_response[missing, ch] = np.interp(
                            np.where(missing)[0], known_idx, new_response[known_idx, ch]
                        )

            # Step 2: Scale restriction: I_128 = 1.0
            for ch in range(channels):
                new_response[:, ch] /= new_response[128, ch] + 1e-8

            response = new_response

            # Step 3: Estimate irradiances x_j (Eq 12) using new I_m
            num = np.zeros((h, w, channels), dtype=np.float32)
            den = np.zeros((h, w, channels), dtype=np.float32)

            for i in range(n_images):
                I_val = response[y[i], np.arange(channels)]
                w_val = self.weights_fixed[y[i]]

                num += w_val * times[i] * I_val
                den += w_val * (times[i] ** 2)

            x = num / (den + 1e-8)

            # Check convergence
            curr_obj = 0
            for i in range(n_images):
                I_val = response[y[i], np.arange(channels)]
                w_val = self.weights_fixed[y[i]]
                curr_obj += np.sum(w_val * (I_val - times[i] * x) ** 2)

            print(f"Iteration {iteration}: Objective = {curr_obj:.4f}")

            if abs(prev_obj - curr_obj) < self.threshold:
                break

            prev_obj = curr_obj

        # Store valid range for reconstruction
        self.valid_range = valid_range
        return response

    def reconstruct(self, images: list[np.ndarray], times: np.ndarray, response: np.ndarray) -> np.ndarray:
        """
        [PAPER_STRICT] Section 3: Weighted average to form HDR radiance map.
        Uses certainty weights (derivative of CRF).
        """
        n_images = len(images)
        h, w, channels = images[0].shape
        times = times.astype(np.float32)

        # Use detected valid range or default
        v_range = getattr(self, "valid_range", (0, 255))
        certainty = self._compute_certainty_weights(response, v_range)

        num = np.zeros((h, w, channels), dtype=np.float32)
        den = np.zeros((h, w, channels), dtype=np.float32)

        y = np.stack(images, axis=0)

        for i in range(n_images):
            I_val = response[y[i], np.arange(channels)]
            w_val = certainty[y[i], np.arange(channels)]

            num += w_val * times[i] * I_val
            den += w_val * (times[i] ** 2)

        x = num / (den + 1e-8)
        return x


def load_image_series(directory: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Load exposure sequence and exposure times from image_list.txt."""
    dir_path = Path(directory)
    list_path = dir_path / "image_list.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"image_list.txt not found in {directory}")

    images = []
    times = []

    with open(list_path, encoding="utf-8") as f:
        # Skip header lines
        f.readline()  # # Number of Images
        n_images = int(f.readline().strip())
        f.readline()  # # Filename ...

        for _ in range(n_images):
            line = f.readline().strip()
            if not line:
                break
            parts = line.split()
            filename, inv_time = parts[0], parts[1]
            img_path = dir_path / filename

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # [Strict Dataset Format] value is 1/shutter_speed
            times.append(1.0 / float(inv_time))

    return images, np.array(times, dtype=np.float32)


def save_hdr(filename: str, radiance_map: np.ndarray) -> None:
    """Save radiance map in Radiance HDR (.hdr) format."""
    # Simple save using OpenCV
    cv2.imwrite(filename, cv2.cvtColor(radiance_map, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robertson 2003 HDR Reconstruction")
    parser.add_argument("input", type=str, nargs="?", default="memorial")
    parser.add_argument("--iters", type=int, default=20, help="Max iterations for CRF estimation")
    parser.add_argument("--output", type=str, help="Output HDR filename")

    args = parser.parse_args()

    # Path resolution
    in_p = Path(args.input)
    if not in_p.exists():
        in_p = Path(__file__).parent / "dataset" / args.input

    out_p = args.output or str(Path(__file__).parent / "output" / f"{in_p.name}_robertson2003.hdr")
    Path(out_p).parent.mkdir(parents=True, exist_ok=True)

    try:
        images, times = load_image_series(str(in_p))

        algo = Robertson2003(max_iters=args.iters)

        print("--- Step 1: Calibrating Response Function ---")
        response = algo.calibrate(images, times)

        print("--- Step 2: Reconstructing HDR Radiance Map ---")
        hdr = algo.reconstruct(images, times, response)

        save_hdr(out_p, hdr)
        print(f"✅ Saved: {out_p}")

    except Exception:
        import traceback

        traceback.print_exc()
