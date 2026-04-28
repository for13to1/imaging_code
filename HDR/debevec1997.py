#!/usr/bin/env python3
"""
Recovering High Dynamic Range Radiance Maps from Photographs (Debevec 1997)

Based on:
Debevec, P. E., & Malik, J. (1997).
Recovering high dynamic range radiance maps from photographs.
In Proceedings of the 24th annual conference on Computer graphics and interactive techniques (pp. 369-378).
"""

import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class Debevec1997:
    """
    Debevec 1997 HDR Algorithm Implementation.

    Implementation Fidelity Categories:
    1. [PAPER_STRICT]: CRF calibration via linear system (Section 2.1) and radiance reconstruction (Section 2.2).
    2. [ENGINEERING_ADAPTATION]:
       - Multi-channel (RGB) calibration handled independently.
       - Small epsilon to prevent log(0) and precision loss.
       - Prevention of division by zero and exponent overflow.
       - Forced monotonicity for discrete inversion.
       - Fallback handling for single-channel grayscale input/output.
       - Automated variance-based spatial sampling algorithm.

    Attributes:
        samples: Number of pixel locations to sample for calibration.
        lambda_smooth: Smoothness regularization weight for the linear system.
        ldr_size: Number of discrete pixel values (default 256 for 8-bit images).
        weights: Hat function weighting array w(z) of shape (ldr_size,).
    """

    def __init__(
        self,
        samples: int = 70,
        lambda_smooth: float = 100.0,
        ldr_size: int = 256,
    ):
        self.samples = samples
        self.lambda_smooth = lambda_smooth
        self.ldr_size = ldr_size
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """
        [PAPER_STRICT] Section 2, Equation 4: Weighting function w(z).
        "simple hat function" that gives higher weight to pixels near the center of the range.
        """
        z_min, z_max = 0, self.ldr_size - 1
        z_mid = (z_min + z_max) / 2.0
        return np.array(
            [z - z_min if z <= z_mid else z_max - z for z in range(self.ldr_size)],
            dtype=np.float32,
        )

    def _sample_pixels(self, images: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        [ENGINEERING_ADAPTATION] Section 2.1: Sample N locations for calibration.
        "pixels should be spatially well distributed... and sample the range of pixel values."
        "Furthermore, the pixels are best sampled from regions of the image with low intensity variance."

        Note: The paper mentions that pixel selection was performed by hand.
        This implementation provides an automated variance-based sampling algorithm
        as a practical replacement for manual selection.
        """
        rows, cols = images[0].shape[:2]

        # [ENGINEERING_ADAPTATION] Automated variance-based sampling algorithm to replace manual pixel selection.
        # Compute a low-variance mask across all images and channels
        # Use the middle exposure image as reference for variance computation
        mid_idx = len(images) // 2
        ref_img = images[mid_idx]
        if ref_img.ndim == 3:
            if ref_img.shape[2] == 3:
                gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = ref_img[:, :, 0].astype(np.float32)
        else:
            gray = ref_img.astype(np.float32)

        # Local variance using a 5x5 box filter
        mean_sq = cv2.blur(gray * gray, (5, 5))
        mean = cv2.blur(gray, (5, 5))
        variance = mean_sq - mean * mean

        # Variance threshold: keep pixels with variance below the median
        var_thresh = np.median(variance)
        low_var_mask = variance <= var_thresh

        # Build a candidate pool from a dense grid, filtered by low variance
        y_coords = np.linspace(5, rows - 6, int(np.sqrt(self.samples * 4)), dtype=int)
        x_coords = np.linspace(5, cols - 6, int(np.sqrt(self.samples * 4)), dtype=int)

        candidates = [(y, x) for y in y_coords for x in x_coords if low_var_mask[y, x]]

        if len(candidates) < self.samples:
            # Fallback: relax variance constraint
            candidates = [(y, x) for y in y_coords for x in x_coords]

        # Ensure brightness coverage: bin candidates by pixel value and sample evenly
        # Use the reference image's grayscale value for binning
        candidate_values = np.array([gray[y, x] for y, x in candidates], dtype=np.int32)
        n_bins = min(16, self.samples // 4)
        bin_edges = np.linspace(0, 255, n_bins + 1)

        selected = []
        samples_per_bin = self.samples // n_bins
        for b in range(n_bins):
            mask = (candidate_values >= bin_edges[b]) & (
                candidate_values < bin_edges[b + 1]
            )
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                n_pick = min(samples_per_bin, len(idxs))
                # Use deterministic sampling: pick evenly spaced indices for reproducibility
                if len(idxs) <= n_pick:
                    pick = idxs
                else:
                    step = len(idxs) / n_pick
                    pick = idxs[np.floor(np.arange(n_pick) * step).astype(int)]
                selected.extend([candidates[i] for i in pick])

        # If we still don't have enough, fill with deterministic candidates
        if len(selected) < self.samples:
            remaining = [c for c in candidates if c not in selected]
            n_needed = self.samples - len(selected)
            if len(remaining) > 0:
                # Deterministic fill: take evenly spaced from remaining
                step = len(remaining) / min(n_needed, len(remaining))
                extra_indices = np.floor(
                    np.arange(min(n_needed, len(remaining))) * step
                ).astype(int)
                selected.extend([remaining[i] for i in extra_indices])

        return selected[: self.samples]

    def calibrate(self, images: List[np.ndarray], times: np.ndarray) -> np.ndarray:
        """
        [PAPER_STRICT] Section 2.1: Solve the linear system g(Z_ij) = ln(E_i) + ln(t_j).

        Returns the inverse camera response function I(z) = f^{-1}(z) = exp(g(z)),
        which maps a pixel value z to the corresponding exposure X = E * Δt.
        This is NOT the camera response function f itself, but its inverse.
        """
        n_images = len(images)
        # Normalize all images to 3D (H, W, C) to support both grayscale and RGB input
        images = [img[:, :, np.newaxis] if img.ndim == 2 else img for img in images]
        channels = images[0].shape[2]
        points = self._sample_pixels(images)
        n_points = len(points)
        log_times = np.log(times)

        inverse_responses = []
        for ch in range(channels):
            # A matrix size: rows = (N*P + 1 + (ldr_size-2)), cols = (ldr_size + N)
            # where N = n_points (sampled pixel locations), P = n_images (exposures)
            n_eq = n_points * n_images + 1 + (self.ldr_size - 2)
            A = np.zeros((n_eq, self.ldr_size + n_points), dtype=np.float32)
            B = np.zeros(n_eq, dtype=np.float32)

            # 1. Data fitting equations: w(Z_ij) * (g(Z_ij) - ln(E_i)) = w(Z_ij) * ln(t_j)
            row = 0
            for i, (py, px) in enumerate(points):
                for j in range(n_images):
                    z = int(images[j][py, px, ch])
                    w = self.weights[z]
                    A[row, z] = w
                    A[row, self.ldr_size + i] = -w
                    B[row] = w * log_times[j]
                    row += 1

            # 2. Fix the scale: g(Z_mid) = 0
            # Paper: Z_mid = (Z_min + Z_max) / 2 = 127.5 for 8-bit images.
            # We round to nearest integer for matrix indexing.
            z_mid = int(round((self.ldr_size - 1) / 2.0))
            A[row, z_mid] = 1.0
            B[row] = 0.0
            row += 1

            # 3. Smoothing equations: lambda * sum (w(z) * g''(z))^2
            # To minimize this via least squares ||Ax - B||^2, the coefficient in A must be sqrt(lambda) * w(z)
            sqrt_lambda = np.sqrt(self.lambda_smooth)
            for z in range(self.ldr_size - 2):
                w = self.weights[z + 1]
                A[row, z] = sqrt_lambda * w
                A[row, z + 1] = -2 * sqrt_lambda * w
                A[row, z + 2] = sqrt_lambda * w
                row += 1

            # Solve via least squares (SVD)
            g = np.linalg.lstsq(A, B, rcond=None)[0][: self.ldr_size]
            inverse_responses.append(np.exp(g))

        return np.stack(inverse_responses, axis=1)

    def reconstruct(
        self, images: List[np.ndarray], times: np.ndarray, inverse_response: np.ndarray
    ) -> np.ndarray:
        """
        [PAPER_STRICT] Section 2.2, Equation 6: Form the weighted average radiance.
        ln(E_i) = sum(w(Z_ij) * (g(Z_ij) - ln(t_j))) / sum(w(Z_ij))

        Args:
            inverse_response: The inverse camera response function I(z) = f^{-1}(z),
                              mapping pixel values to exposure X = E * Δt.
                              This is the output of calibrate().
        """
        h, w = images[0].shape[:2]
        # Normalize all images to 3D (H, W, C) to support both grayscale and RGB input
        images = [img[:, :, np.newaxis] if img.ndim == 2 else img for img in images]
        channels = inverse_response.shape[1]

        # [ENGINEERING_ADAPTATION] Defensive: prevent log(0) or log(negative) in case of numerical instability
        log_inverse_response = np.log(np.maximum(inverse_response, 1e-8))
        log_times = np.log(times)

        num = np.zeros((h, w, channels), dtype=np.float32)
        den = np.zeros((h, w, channels), dtype=np.float32)

        for i, img in enumerate(images):
            for ch in range(channels):
                z = img[:, :, ch]
                w_val = self.weights[z]
                num[:, :, ch] += w_val * (log_inverse_response[z, ch] - log_times[i])
                den[:, :, ch] += w_val

        # Handle pixels that are saturated or zero in all exposures (den == 0).
        # Paper: weighting function ignores saturated values; these pixels have no valid data.
        # [ENGINEERING_ADAPTATION] Prevent division by zero and handle missing data with NaN.
        den_safe = np.where(den > 0, den, 1.0)
        # [ENGINEERING_ADAPTATION] Defensive: prevent exp overflow for extremely large values
        val = np.clip(num / den_safe, -80.0, 80.0)
        radiance = np.exp(val)
        radiance = np.where(den > 0, radiance, np.nan)

        if channels == 1:
            radiance = radiance.squeeze(axis=-1)

        return radiance

    def virtual_photograph(
        self,
        radiance_map: np.ndarray,
        inverse_response: np.ndarray,
        exposure_time: float,
    ) -> np.ndarray:
        """
        [PAPER_STRICT] Section 2.7: Map HDR radiance back to LDR via response function.
        Paper: Z = f(E * delta_t)

        Since we only store the inverse response f^{-1}(z) defined on discrete
        pixel values, the forward function f is recovered by direct inversion:
        f(X) = argmin_z |f^{-1}(z) - X|. This is the exact discrete inverse.

        Args:
            inverse_response: The inverse camera response function I(z) = f^{-1}(z),
                              mapping pixel values to exposure X = E * Δt.
        """
        is_grayscale = radiance_map.ndim == 2
        exposure_value = radiance_map * exposure_time
        if is_grayscale:
            exposure_value = exposure_value[:, :, np.newaxis]
            ldr_image = np.zeros(exposure_value.shape, dtype=np.uint8)
        else:
            ldr_image = np.zeros_like(radiance_map, dtype=np.uint8)

        for ch in range(inverse_response.shape[1]):
            # [ENGINEERING_ADAPTATION] Defensive: ensure strictly non-decreasing for searchsorted
            inv_resp_ch = np.maximum.accumulate(inverse_response[:, ch])
            # Discrete exact inversion: find the pixel value z whose inverse
            # response is closest to the target exposure.
            # searchsorted finds the first z where f^{-1}(z) >= exposure;
            # we then compare with z-1 to pick the nearest neighbor.
            idx = np.searchsorted(inv_resp_ch, exposure_value[:, :, ch])
            idx = np.clip(idx, 1, self.ldr_size - 1)
            # Compare distance to idx and idx-1, pick closer one.
            d_hi = np.abs(inv_resp_ch[idx] - exposure_value[:, :, ch])
            d_lo = np.abs(inv_resp_ch[idx - 1] - exposure_value[:, :, ch])
            nearest = np.where(d_lo < d_hi, idx - 1, idx)
            ldr_image[:, :, ch] = np.clip(nearest, 0, self.ldr_size - 1)

        return ldr_image.squeeze(axis=-1) if is_grayscale else ldr_image

    def balance_channels(
        self,
        radiance_map: np.ndarray,
        reference_color: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        [PAPER_STRICT] Section 2.6: Balance RGB channels.

        Default: The g(Z_mid)=0 constraint in calibration already ensures that
        (Z_mid, Z_mid, Z_mid) maps to equal radiance across R, G, B channels,
        meaning the pixel is achromatic. No extra processing is needed.

        With reference illuminant C: "the radiance values of the three channels
        should be scaled so that the pixel value (Z_mid, Z_mid, Z_mid) maps to
        a radiance with the same color ratios as C."
        """
        if (
            reference_color is None
            or radiance_map.ndim == 2
            or radiance_map.shape[-1] == 1
        ):
            # Default: already balanced by g(Z_mid) = 0 constraint
            return radiance_map

        # Scale channels so (Z_mid, Z_mid, Z_mid) maps to reference illuminant color ratios
        balanced = np.copy(radiance_map)
        ref = np.array(reference_color, dtype=np.float32)
        # Normalize reference color ratios (e.g., relative to green channel)
        # [ENGINEERING_ADAPTATION] Prevent division by zero with small epsilon
        ref /= ref[1] + 1e-8
        for ch in range(min(radiance_map.shape[2], 3)):
            balanced[:, :, ch] *= ref[ch]
        return balanced

    def absolute_calibrate(
        self,
        radiance_map: np.ndarray,
        known_radiance: float,
        pos: Tuple[int, int],
        channel: Optional[int] = None,
    ) -> np.ndarray:
        """
        [PAPER_STRICT] Section 2.5: Absolute Radiance Calibration.
        "determines the scale factor by photographing a light source of known radiance."

        Args:
            known_radiance: The known radiance value of the calibration light source.
            pos: (y, x) pixel coordinates of the calibration light source.
            channel: If None, uses the mean of all channels. If specified, uses only
                     that channel's value for computing the scale factor.
        """
        y, x = pos
        pixel_val = radiance_map[y, x]
        if radiance_map.ndim == 3 and channel is not None:
            current_val = pixel_val[channel]
        elif radiance_map.ndim == 3:
            current_val = np.mean(pixel_val)
        else:
            current_val = pixel_val
        # [ENGINEERING_ADAPTATION] Prevent division by zero with small epsilon
        scale_factor = known_radiance / (current_val + 1e-8)
        return radiance_map * scale_factor

    def merge_scans_with_responses(
        self,
        scan1: np.ndarray,
        scan2: np.ndarray,
        inverse_response1: np.ndarray,
        inverse_response2: np.ndarray,
    ) -> np.ndarray:
        """
        [PAPER_STRICT] Section 2.4: Using Multiple Digitizations.
        "The process is mathematically identical to the process used to combine the film exposures...
        only now the true transmittance of the negative replaces E_i."

        Args:
            inverse_response1, inverse_response2: Inverse response functions for each scan,
                mapping pixel values to exposure X (or transmittance for film scans).
        """
        scan1 = np.asarray(scan1, dtype=np.float32)
        scan2 = np.asarray(scan2, dtype=np.float32)
        is_grayscale = len(scan1.shape) == 2
        if is_grayscale:
            scan1 = scan1[:, :, np.newaxis]
            scan2 = scan2[:, :, np.newaxis]
        rows, cols, channels = scan1.shape
        # For film scans, these are transmittance values (T), not radiance (E).
        # For digital photos, these are exposure values X = E * Δt.
        exposure1 = np.zeros((rows, cols, channels), dtype=np.float32)
        exposure2 = np.zeros((rows, cols, channels), dtype=np.float32)
        for ch in range(channels):
            pixel_vals1 = scan1[:, :, ch].astype(np.int32)
            pixel_vals2 = scan2[:, :, ch].astype(np.int32)
            exposure1[:, :, ch] = inverse_response1[pixel_vals1, ch]
            exposure2[:, :, ch] = inverse_response2[pixel_vals2, ch]
        weight1 = np.zeros((rows, cols, channels), dtype=np.float32)
        weight2 = np.zeros((rows, cols, channels), dtype=np.float32)
        for ch in range(channels):
            weight1[:, :, ch] = self.weights[scan1[:, :, ch].astype(np.int32)]
            weight2[:, :, ch] = self.weights[scan2[:, :, ch].astype(np.int32)]

        total_weight = weight1 + weight2 + 1e-8
        # [PAPER_STRICT] Section 2.4: "The process is mathematically identical to the process used to combine the film exposures"
        # This means the weighted average MUST be performed in the logarithmic domain, just like Equation 6.
        # [ENGINEERING_ADAPTATION] Defensive: use np.maximum to avoid precision loss on tiny values instead of adding 1e-8
        merged_log_exposure = (
            np.log(np.maximum(exposure1, 1e-8)) * weight1
            + np.log(np.maximum(exposure2, 1e-8)) * weight2
        ) / total_weight
        merged_exposure = np.exp(merged_log_exposure)

        return merged_exposure.squeeze() if is_grayscale else merged_exposure


def save_rgbe(filename: str, radiance_map: np.ndarray):
    """Save in Radiance HDR (.hdr) format via OpenCV. OpenCV encodes to RGBE internally."""
    if radiance_map.ndim == 2:
        # [ENGINEERING_ADAPTATION] OpenCV's HDR encoder requires a 3-channel RGB/BGR array
        cv2.imwrite(filename, cv2.cvtColor(radiance_map, cv2.COLOR_GRAY2BGR))
    elif radiance_map.ndim == 3 and radiance_map.shape[2] == 1:
        gray = radiance_map.squeeze(axis=-1)
        cv2.imwrite(filename, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    else:
        cv2.imwrite(filename, cv2.cvtColor(radiance_map, cv2.COLOR_RGB2BGR))


def load_image_series(directory: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load image series and exposure times from a directory containing image_list.txt.

    Expected image_list.txt format (as used in the Debevec dataset):
      - Line 1: header comment
      - Line 2: number of images N
      - Line 3: header comment
      - Lines 4..4+N-1: "filename shutter_speed", where exposure_time = 1/shutter_speed

    Note: This function assumes that all images in the series are already
    geometrically aligned (registered). The paper (Section 3) mentions using
    normalized correlation for sub-pixel registration when necessary.
    Pre-alignment is the responsibility of the caller.
    """
    dir_path = Path(directory)
    list_path = dir_path / "image_list.txt"
    if not list_path.exists():
        raise FileNotFoundError(
            f"image_list.txt not found in {directory}\n"
            f"Expected format:\n"
            f"  <comment>\n"
            f"  <number_of_images>\n"
            f"  <comment>\n"
            f"  <filename> <shutter_speed>  (exposure_time = 1/shutter_speed)\n"
        )

    with open(list_path, "r") as f:
        f.readline()
        line = f.readline().strip()
        if not line:
            raise ValueError(
                f"Empty image list in {list_path}. "
                f"Expected second line to contain the number of images."
            )
        try:
            n = int(line)
        except ValueError as exc:
            raise ValueError(
                f"Expected integer image count on line 2 of {list_path}, got: '{line}'"
            ) from exc
        f.readline()
        images, times = [], []
        for idx in range(n):
            line = f.readline().strip()
            if not line:
                raise ValueError(
                    f"Expected {n} image entries in {list_path}, "
                    f"but line {idx + 4} is empty."
                )
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid format on line {idx + 4} of {list_path}: '{line}'\n"
                    f"Expected: '<filename> <shutter_speed>'"
                )
            img_path = dir_path / parts[0]
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(
                    f"Failed to load image: {img_path}\n"
                    f"Please ensure the image file exists and is a valid image format."
                )
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            try:
                shutter_speed = float(parts[1])
                if shutter_speed <= 0:
                    raise ValueError("shutter_speed must be positive")
                # [DATASET_FORMAT] exposure_time = 1 / shutter_speed
                times.append(1.0 / shutter_speed)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid shutter speed on line {idx + 4} of {list_path}: '{parts[1]}'\n"
                    f"Expected a positive numeric value."
                ) from exc
    return images, np.array(times, dtype=np.float32)


def save_inverse_response(
    filename: str, inverse_response: np.ndarray, metadata: Optional[dict] = None
) -> None:
    """Save the inverse camera response function I(z) = f^{-1}(z) to JSON."""
    data = {
        "inverse_response": inverse_response.tolist(),
        "channels": inverse_response.shape[1] if len(inverse_response.shape) > 1 else 1,
        "metadata": metadata or {},
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_inverse_response(filename: str) -> Tuple[np.ndarray, dict]:
    """Load the inverse camera response function I(z) = f^{-1}(z) from JSON."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    inverse_response = np.array(data["inverse_response"], dtype=np.float32)
    metadata = data.get("metadata", {})
    return inverse_response, metadata


def plot_inverse_response(
    inverse_response: np.ndarray,
    title: str = "Recovered Inverse Camera Response Function",
    log_scale: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot the inverse camera response function I(z) = f^{-1}(z)."""
    plt.figure(figsize=(8, 5))
    ldr_size = inverse_response.shape[0]
    z = np.arange(ldr_size)
    colors = ["r", "g", "b"]
    labels = ["Red Channel", "Green Channel", "Blue Channel"]
    channels = inverse_response.shape[1] if len(inverse_response.shape) > 1 else 1
    for i in range(channels):
        channel_resp = (
            inverse_response[:, i] if channels > 1 else inverse_response.flatten()
        )
        color = colors[i] if channels == 3 else "k"
        label = labels[i] if channels == 3 else "Luminance"
        if log_scale:
            # [ENGINEERING_ADAPTATION] Prevent log(0) with small epsilon for plotting
            plt.plot(
                z, np.log(channel_resp + 1e-8), color=color, label=label, linewidth=2
            )
            plt.ylabel(r"$\ln(E \cdot \Delta t)$")
        else:
            plt.plot(z, channel_resp, color=color, label=label, linewidth=2)
            plt.ylabel(r"$E \cdot \Delta t$")
    plt.title(title)
    plt.xlabel(r"Pixel Value $Z$")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.close()


def load_rgbe(filename: str) -> np.ndarray:
    """
    [ENGINEERING_ADAPTATION] Load Radiance HDR (.hdr) format via OpenCV.
    OpenCV handles RLE decompression and RGBE decoding natively.
    """
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load HDR image: {filename}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debevec 1997 HDR Implementation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="memorial",
        help="dataset directory name (default: memorial)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=70,
        help="number of pixels to sample for calibration (default: 70)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_smooth",
        type=float,
        default=100.0,
        help="smoothness weight (default: 100.0)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = base_dir / "dataset" / args.dataset
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.dataset}_debevec1997.hdr"
    plot_path = output_dir / f"{args.dataset}_debevec1997_crf.png"

    print(f"--- 1. Loading dataset: {args.dataset} ---")
    print(f"Path: {dataset_path}")

    try:
        images, times = load_image_series(str(dataset_path))
        print(
            f"Successfully loaded {len(images)} images, "
            f"exposure time range: {min(times):.4f}s ~ {max(times):.4f}s"
        )

        algo = Debevec1997(samples=args.samples, lambda_smooth=args.lambda_smooth)

        print(
            f"\n--- 2. Computing inverse camera response curve "
            f"(samples: {args.samples}, Lambda: {args.lambda_smooth}) ---"
        )
        inverse_response = algo.calibrate(images, times)

        print(f"\n--- 3. Generating and saving inverse response curve plot ---")
        plot_path_str = str(plot_path)
        plot_inverse_response(
            inverse_response,
            title=f"Recovered Inverse CRF - {args.dataset}",
            save_path=plot_path_str,
        )

        print("\n--- 4. Reconstructing Radiance Map ---")
        hdr = algo.reconstruct(images, times, inverse_response)

        save_rgbe(str(output_path), hdr)
        print(f"✅ Success! Result saved to: {output_path}")

    except Exception:
        import traceback

        traceback.print_exc()
