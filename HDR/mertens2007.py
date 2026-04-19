"""
Exposure Fusion (Mertens 2007)

Based on:
Mertens, T., Kautz, J., & Van Reeth, F. (2007).
Exposure fusion. In 15th Pacific Conference on Computer Graphics and Applications (PG'07) (pp. 382-390). IEEE.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional


class Mertens2007:
    """
    ENGINEERING ASSUMPTIONS & DISCREPANCIES (Marker System):
    - [PAPER_STRICT]: Directly from the Mertens 2007 paper (formula, constant, or step).
    - [ENGINEERING_ADAPTATION]: Omitted details, ambiguities, or modern stability/performance heuristics.

    1. Grayscale Conversion: [ENGINEERING_ADAPTATION] Uses Rec. 709 luminance weights
       for grayscale conversion, ensuring consistency with other HDR modules.
       [PAPER_STRICT] Section 3.1 merely mentions "grayscale version".
    2. Laplacian Kernel: [PAPER_STRICT] Section 3.1 specifies a "Laplacian filter".
       [ENGINEERING_ADAPTATION] Uses cv2.Laplacian with ksize=1 (3x3 aperture).
    3. Pyramid Alignment: [ENGINEERING_ADAPTATION] Uses explicit dstsize in pyrUp
       to ensure exact spatial alignment for non-power-of-two image dimensions.
    4. Numerical Stability: [ENGINEERING_ADAPTATION] Added epsilon (1e-12) during
       weight normalization to prevent division-by-zero in black regions.
    5. Memory Management: [ENGINEERING_ADAPTATION] Uses Two-Pass Streaming and
       accumulative pyramid blending to minimize peak memory usage O(Image).
    6. Weight Power Optimization: [ENGINEERING_ADAPTATION] Special-cases exponents
       0.0 and 1.0 to skip expensive power functions and uses epsilon for stability.
    """

    def __init__(
        self,
        w_c: float = 1.0,
        w_s: float = 1.0,
        w_e: float = 1.0,
        sigma_e: float = 0.2,
        levels: Optional[int] = None,
    ):
        """
        Initialize the Exposure Fusion parameters.

        Args:
            w_c: Contrast weight exponent (default 1.0).
            w_s: Saturation weight exponent (default 1.0).
            w_e: Well-exposedness weight exponent (default 1.0).
            sigma_e: Standard deviation for the Gaussian well-exposedness curve (default 0.2).
            levels: Number of pyramid levels. If None, auto-calculated.
        """
        self.w_c = w_c
        self.w_s = w_s
        self.w_e = w_e
        self.sigma_e = sigma_e
        self.levels = levels

        # [ENGINEERING_ADAPTATION] Standard Rec. 709 luminance weights.
        self.Y_WEIGHTS = np.array([0.2126729, 0.7151522, 0.0721750])

    def compute_weight_map(self, img: np.ndarray) -> np.ndarray:
        """
        Computes the raw weight map for a single image.
        (Implementation of Mertens 2007, Section 3.1)
        """
        # 1. Contrast (C) [PAPER_STRICT]
        gray = np.dot(img, self.Y_WEIGHTS).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
        C = np.abs(laplacian)

        # 2. Saturation (S) [PAPER_STRICT]
        S = np.std(img, axis=2)

        # 3. Well-exposedness (E) [PAPER_STRICT]
        # [ENGINEERING_ADAPTATION] Optimization: exp(a)*exp(b)*exp(c) = exp(a+b+c)
        sq_diff = (img - 0.5) ** 2
        E = np.exp(-np.sum(sq_diff, axis=2) / (2 * (self.sigma_e**2)))

        # 4. Combined Weight [PAPER_STRICT]
        # [ENGINEERING_ADAPTATION] Performance Optimization:
        # Skip power function for exponents 0.0 and 1.0.
        W = np.ones(img.shape[:2], dtype=np.float32)

        if self.w_c != 0:
            W *= C if self.w_c == 1.0 else (np.maximum(C, 1e-12) ** self.w_c)
        if self.w_s != 0:
            W *= S if self.w_s == 1.0 else (np.maximum(S, 1e-12) ** self.w_s)
        if self.w_e != 0:
            W *= E if self.w_e == 1.0 else (np.maximum(E, 1e-12) ** self.w_e)

        return W

    def build_gaussian_pyramid(self, img: np.ndarray, levels: int) -> List[np.ndarray]:
        """Builds a Gaussian pyramid."""
        pyr = [img.copy()]
        G = img
        for _ in range(levels - 1):
            G = cv2.pyrDown(G)
            pyr.append(G)
        return pyr

    def build_laplacian_pyramid(self, img: np.ndarray, levels: int) -> List[np.ndarray]:
        """Builds a Laplacian pyramid."""
        G_pyr = self.build_gaussian_pyramid(img, levels)
        L_pyr = []
        for i in range(levels - 1):
            h, w = G_pyr[i].shape[:2]
            GE = cv2.pyrUp(G_pyr[i + 1], dstsize=(w, h))
            L = G_pyr[i] - GE
            L_pyr.append(L)
        L_pyr.append(G_pyr[-1])
        return L_pyr

    def collapse_laplacian_pyramid(self, L_pyr: List[np.ndarray]) -> np.ndarray:
        """Collapses a Laplacian pyramid."""
        levels = len(L_pyr)
        R = L_pyr[-1]
        for i in range(levels - 2, -1, -1):
            h, w = L_pyr[i].shape[:2]
            R = cv2.pyrUp(R, dstsize=(w, h))
            R = R + L_pyr[i]
        return R

    def process(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Executes the exposure fusion pipeline using Two-Pass Streaming.
        (Implementation of Mertens 2007, Section 3.2)
        """
        N = len(images)
        if N < 2:
            raise ValueError("Exposure fusion requires at least 2 images.")

        h, w = images[0].shape[:2]
        if self.levels is None:
            self.levels = int(np.log2(min(h, w))) - 1

        # 1. First Pass: Compute global weight sum [ENGINEERING_ADAPTATION]
        # This allows us to normalize weights without storing all weight maps.
        print("First Pass: Analyzing exposure sequence...")
        W_sum = np.zeros((h, w), dtype=np.float32)
        for img in images:
            W_sum += self.compute_weight_map(img)
        W_sum += 1e-12  # Numerical stability

        # 2. Second Pass: Accumulative Blending [ENGINEERING_ADAPTATION]
        # Memory usage is now O(Image) instead of O(N * Image).
        print(f"Second Pass: Blending {N} images at {self.levels} levels...")
        R_pyr = None

        for k in range(N):
            # Re-calculate normalized weight map on the fly
            w_k = self.compute_weight_map(images[k]) / W_sum

            W_pyr_k = self.build_gaussian_pyramid(w_k, self.levels)
            I_pyr_k = self.build_laplacian_pyramid(images[k], self.levels)

            if R_pyr is None:
                R_pyr = [np.zeros_like(lp) for lp in I_pyr_k]

            for l in range(self.levels):
                w_map_level = W_pyr_k[l][..., np.newaxis]
                R_pyr[l] += w_map_level * I_pyr_k[l]

        # 3. Final Reconstruction [PAPER_STRICT]
        R = self.collapse_laplacian_pyramid(R_pyr)
        R = np.clip(R, 0.0, 1.0)
        return (R * 255.0).astype(np.uint8)


def load_exposure_sequence(input_path: str) -> List[np.ndarray]:
    """
    Loads and normalizes an image sequence.
    (Pre-processing step for Exposure Fusion)
    """
    path = Path(input_path)
    if path.is_file():
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            parent = path.parent
            img_paths = [
                parent / p if not Path(p).is_absolute() else Path(p) for p in lines
            ]
    elif path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        img_paths = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in valid_exts]
        )
    else:
        raise FileNotFoundError(f"Path {input_path} not found.")

    images = []
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0

        images.append(img.astype(np.float32))

    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mertens 2007 Exposure Fusion - High-Performance Implementation"
    )
    parser.add_argument(
        "--dataset", type=str, default="memorial", help="Dataset name or path"
    )
    parser.add_argument("--w_c", type=float, default=1.0, help="Contrast weight")
    parser.add_argument("--w_s", type=float, default=1.0, help="Saturation weight")
    parser.add_argument(
        "--w_e", type=float, default=1.0, help="Well-exposedness weight"
    )
    parser.add_argument("--output", type=str, help="Output path")

    args = parser.parse_args()

    # Smart path resolution
    base_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = base_dir / "dataset" / args.dataset

    if args.output is None:
        out_dir = base_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"{args.dataset}_mertens2007.png")

    try:
        images = load_exposure_sequence(str(dataset_path))
        print(f"--- Processing {dataset_path.name} with Exposure Fusion ---")
        print(f"Loaded {len(images)} images.")
        tmo = Mertens2007(w_c=args.w_c, w_s=args.w_s, w_e=args.w_e)
        result = tmo.process(images)
        cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"✅ Result saved: {args.output}")
    except Exception as e:
        import traceback

        traceback.print_exc()
