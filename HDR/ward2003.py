"""
Ward 2003 Fast, Robust Image Registration (MTB) Implementation.

Based on:
Ward, G. (2003). Fast, robust image registration for compositing high dynamic range
photographs from hand-held exposures. Journal of Graphics Tools, 8(2), 17-30.
"""

from pathlib import Path

import cv2
import numpy as np


class Ward2003:
    """
    ENGINEERING ASSUMPTIONS & DISCREPANCIES (Marker System):
    - [PAPER_STRICT]: Directly from the Ward 2003 paper (formula, constant, or step).
    - [ENGINEERING_ADAPTATION]: Omitted details, ambiguities, or modern stability/performance heuristics.

    1. Grayscale Formula: [PAPER_STRICT] Section 2 specifies
       grey = (54 * red + 183 * green + 19 * blue) / 256.
    2. Median Thresholding: [PAPER_STRICT] Section 2 defines MTB as 0 where
       pixels <= median and 1 where > median.
    3. Exclusion Bitmap: [PAPER_STRICT] Section 2.1 specifies excluding pixels
       within ±4 of the median value.
    4. Pyramid Depth: [PAPER_STRICT] Section 2.2 suggests shift_bits limit of 6
       (±64 pixels). [ENGINEERING_ADAPTATION] Auto-calculate depth based on resolution.
    5. Bitmap Shifting: [PAPER_STRICT] Section 2.2 notes borders must be cleared with 0s.
    """

    def __init__(
        self,
        exclusion_range: int = 4,
        shift_bits: int = 6,
        normalize_error: bool = False,
    ):
        """
        Initialize the MTB Aligner.

        Args:
            exclusion_range: Tolerance range around median for noise exclusion (default 4).
            shift_bits: Max shift allowed is 2^shift_bits (default 6 for +/- 64px).
            normalize_error: [ENGINEERING_ADAPTATION] If True, divides error by overlap area.
                             Default is False to strictly match the 2003 paper.
        """
        self.exclusion_range = exclusion_range
        self.shift_bits = shift_bits
        self.normalize_error = normalize_error

    def get_grayscale(self, img_rgb: np.ndarray) -> np.ndarray:
        """[PAPER_STRICT] Section 2 grayscale conversion formula."""
        # Note: img_rgb is assumed to be linear or sRGB as per paper context
        if img_rgb.ndim == 2:
            return img_rgb

        # [PAPER_STRICT] grey = (54 * red + 183 * green + 19 * blue) / 256
        # Using floating point for intermediate to avoid overflow before division
        gray = (54.0 * img_rgb[..., 0] + 183.0 * img_rgb[..., 1] + 19.0 * img_rgb[..., 2]) / 256.0
        return gray.astype(np.uint8)

    def get_percentile(self, gray: np.ndarray, percentile: float) -> int:
        """[PAPER_STRICT] Section 2: Determine percentile value from a histogram."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        target = gray.size * (percentile / 100.0)
        acc = 0
        for i, count in enumerate(hist):
            acc += count
            if acc >= target:
                return i
        return 255

    def compute_bitmaps(self, gray: np.ndarray, p: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
        """[PAPER_STRICT] Section 2 & 2.1: Compute MTB and Exclusion Bitmap with fixed percentile."""
        # [PAPER_STRICT] "It is crucial that the same percentile be used for both exposures being registered."
        val = self.get_percentile(gray, p)

        # [PAPER_STRICT] Section 2: Create MTB
        _, mtb = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)

        # [PAPER_STRICT] Section 2.1: Exclusion Bitmap: 0 if within ±exclusion_range of threshold, else 1.
        exc = cv2.inRange(
            gray,
            int(max(0, val - self.exclusion_range)),
            int(min(255, val + self.exclusion_range)),
        )
        exc = cv2.bitwise_not(exc)

        return mtb, exc

    def shift_bitmap(self, bitmap: np.ndarray, dx: int, dy: int, clear_border: bool = True) -> np.ndarray:
        """[PAPER_STRICT] Section 2.2 & 2.3: Shift bitmap with optional border clearing.

        Args:
            clear_border: [PAPER_STRICT] Section 2.3: Set to True for Exclusion Bitmaps (EB),
                          and False for Threshold Bitmaps (TB) to save redundant work.
        """
        rows, cols = bitmap.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        if clear_border:
            # [PAPER_STRICT] "shift 0s into the new image areas"
            return cv2.warpAffine(bitmap, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            # [PAPER_STRICT] Section 2.3: "no point in clearing the median bitmaps ... We can
            # therefore save time by not clearing the median bitmaps."
            # Using BORDER_REPLICATE as a "no-op" for the border.
            return cv2.warpAffine(bitmap, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    def get_exp_shift(self, img1: np.ndarray, img2: np.ndarray, shift_bits: int) -> tuple[int, int]:
        """[PAPER_STRICT] Section 2.2: Recursive pyramid search (GetExpShift)."""
        cur_shift = [0, 0]

        if shift_bits > 0:
            # Recursive step: Shrink images and find shift at lower resolution
            # [PAPER_STRICT] "filter it down by a factor of two in each dimension"
            sml_img1 = cv2.pyrDown(img1)
            sml_img2 = cv2.pyrDown(img2)
            prev_shift = self.get_exp_shift(sml_img1, sml_img2, shift_bits - 1)
            cur_shift[0] = prev_shift[0] * 2
            cur_shift[1] = prev_shift[1] * 2

        # [PAPER_STRICT] Section 2: "It is crucial that the same percentile be used for both exposures"
        # [PAPER_STRICT] "There may be certain exposure pairs that are either too light or too dark... we choose either the 17th or 83rd percentile... respectively."
        p = 50.0
        m1 = self.get_percentile(img1, 50.0)
        m2 = self.get_percentile(img2, 50.0)

        # Determine shared percentile based on extreme median values indicating too dark or too light
        if m1 < 10 or m2 < 10:
            # Too dark: raise threshold to 83rd to stay above the noise floor
            p = 83.0
        elif m1 > 245 or m2 > 245:
            # Too light: lower threshold to 17th to find structural details in the shadows
            p = 17.0

        tb1, eb1 = self.compute_bitmaps(img1, p)
        tb2, eb2 = self.compute_bitmaps(img2, p)

        # [PAPER_STRICT] Section 2.2: Initial error for comparison.
        # [ENGINEERING_STABILITY] Favor center shift (0,0) by checking it first.
        def _calc_raw_err(xs, ys):
            # [PAPER_STRICT] Section 2.3 Optimization: TB uses no-clear, EB uses clear
            stb2 = self.shift_bitmap(tb2, xs, ys, clear_border=False)
            seb2 = self.shift_bitmap(eb2, xs, ys, clear_border=True)
            combined_mask = cv2.bitwise_and(eb1, seb2)
            diff = cv2.bitwise_and(cv2.bitwise_xor(tb1, stb2), combined_mask)
            err = cv2.countNonZero(diff)
            if self.normalize_error:
                active = cv2.countNonZero(combined_mask)
                return err / active if active > 0 else 1.0
            return err

        best_shift = (cur_shift[0], cur_shift[1])
        min_err = _calc_raw_err(best_shift[0], best_shift[1])

        # [PAPER_STRICT] "minimum difference offset between them within a range of +/- 1 pixel"
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                xs, ys = cur_shift[0] + dx, cur_shift[1] + dy
                err = _calc_raw_err(xs, ys)

                # [PAPER_STRICT] "if (err < min_err) { min_err = err; ... }"
                if err < min_err:
                    min_err = err
                    best_shift = (xs, ys)

        return best_shift

    def align(self, images: list[np.ndarray]) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
        """
        [PAPER_STRICT] Section 2: "In practice, we register between adjacent exposures".
        Aligns a sequence of images to the reference (middle exposure).
        """
        if not images:
            return [], []

        num_imgs = len(images)
        ref_idx = num_imgs // 2

        print(f"Aligning sequence of {num_imgs} images using adjacent registration...")

        # [PAPER_STRICT] "we register between adjacent exposures... to minimize image content change"
        # We store cumulative shifts relative to the reference image
        final_shifts = [(0, 0)] * num_imgs

        # 1. Backward alignment from ref_idx-1 down to 0
        for i in range(ref_idx - 1, -1, -1):
            img_next = images[i + 1]  # The one closer to reference
            img_curr = images[i]

            gray_next = self.get_grayscale(img_next)
            gray_curr = self.get_grayscale(img_curr)

            s_bits = self.shift_bits
            if s_bits <= 0:
                s_bits = int(np.log2(min(gray_next.shape) / 16))

            # Find shift of current relative to next
            dx, dy = self.get_exp_shift(gray_next, gray_curr, s_bits)

            # Cumulative shift relative to reference
            # If next is shifted by (NX, NY), and curr matches next at (dx, dy)
            # Then curr is shifted by (NX + dx, NY + dy) relative to reference
            final_shifts[i] = (final_shifts[i + 1][0] + dx, final_shifts[i + 1][1] + dy)

        # 2. Forward alignment from ref_idx+1 to end
        for i in range(ref_idx + 1, num_imgs):
            img_prev = images[i - 1]  # The one closer to reference
            img_curr = images[i]

            gray_prev = self.get_grayscale(img_prev)
            gray_curr = self.get_grayscale(img_curr)

            s_bits = self.shift_bits
            if s_bits <= 0:
                s_bits = int(np.log2(min(gray_prev.shape) / 16))

            dx, dy = self.get_exp_shift(gray_prev, gray_curr, s_bits)

            final_shifts[i] = (final_shifts[i - 1][0] + dx, final_shifts[i - 1][1] + dy)

        # Apply shifts
        aligned_images = []
        for i, img in enumerate(images):
            dx, dy = final_shifts[i]
            rows, cols = img.shape[:2]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            aligned_images.append(aligned)
            print(f"  - Image {i}: Shift = ({dx}, {dy})")

        return aligned_images, final_shifts


def load_sequence(input_path: str) -> list[np.ndarray]:
    """Loads an image sequence from a directory or file list."""
    path = Path(input_path)
    if path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        img_paths = sorted([p for p in path.iterdir() if p.suffix.lower() in valid_exts])
    else:
        raise FileNotFoundError(f"Path {input_path} is not a directory.")

    images = []
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ward 2003 MTB Image Alignment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="memorial",
        help="Dataset name in dataset/ or full path",
    )
    parser.add_argument("--bits", type=int, default=6, help="Shift bits (max shift = 2^bits)")
    parser.add_argument("--tol", type=int, default=4, help="Exclusion tolerance")
    parser.add_argument("--output", type=str, help="Output directory for aligned images")

    args = parser.parse_args()

    # Path resolution
    base_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = base_dir / "dataset" / args.dataset

    try:
        images = load_sequence(str(dataset_path))
        if not images:
            print(f"No images found in {dataset_path}")
            exit(1)

        aligner = Ward2003(exclusion_range=args.tol, shift_bits=args.bits)
        aligned, shifts = aligner.align(images)

        if args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(aligned):
                out_name = out_dir / f"aligned_{i:02d}.png"
                cv2.imwrite(str(out_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"✅ Aligned images saved to {out_dir}")
        else:
            print("✅ Alignment completed (use --output to save results).")

    except Exception:
        import traceback

        traceback.print_exc()
