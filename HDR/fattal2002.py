#!/usr/bin/env python3
r"""
Gradient Domain High Dynamic Range Compression (Fattal 2002)

Based on:
Fattal, R., Lischinski, D., & Werman, M. (2002).
Gradient domain high dynamic range compression.
ACM Transactions on Graphics (TOG), 21(3), 249-256.
https://dl.acm.org/doi/10.1145/566654.566573

[PAPER_STRICT] Implementation following the mathematical formulas in the paper.

Algorithm Overview:
  1. Compute log-luminance $H = \log(L_{\mathrm{in}})$.
  2. Build Gaussian pyramid $H_0, \ldots, H_d$ (Section 4), min size 32.
  3. At each level $k$, compute central-difference gradients $\nabla H_k$ (Eq. in Sec. 4)
     with denominator $2^{k+1}$ to account for inter-level spacing.
  4. Compute per-level attenuation
     $\varphi_k = (\alpha / \|\nabla H_k\|) \cdot (\|\nabla H_k\| / \alpha)^{\beta}$ (Sec. 4).
  5. Accumulate attenuation top-down with linear upsampling:
     $\Phi = \prod_k \varphi_k$ (Sec. 4).
  6. Attenuate full-resolution gradients: $G = \nabla H_0 \cdot \Phi$.
  7. Solve Poisson equation $\nabla^2 I = \operatorname{div} G$ with Neumann BCs via DCT (Section 5).
  8. Reconstruct LDR: $L_{\mathrm{out}} = \exp(I)$, map color via
     $(C_{\mathrm{in}} / L_{\mathrm{in}})^s \cdot L_{\mathrm{out}}$ (Sec. 5).
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.fft import dctn, idctn


class Fattal2002:
    r"""
    Gradient Domain HDR Compression (Fattal et al. 2002).

    Implementation Fidelity Categories:
    - [PAPER_STRICT]: Directly follows the paper's Sections 3, 4, and 5.
    - [ENGINEERING_ADAPTATION]: Necessary decisions for numerical stability
      or ambiguous details not specified in the paper.
    - [RATIONAL_OMISSION]: Features clearly out of scope for static images.

    Key Implementation Notes:
    1. Poisson Solver: The paper recommends Full Multigrid ($O(n)$).
       [ENGINEERING_ADAPTATION] We use a DCT-based rapid solver ($O(n \log n)$)
       which the paper itself mentions as an alternative (Section 5), and which
       is simpler and numerically reliable in NumPy/SciPy.
    2. Gradient at pyramid level $k$: denominator is $2^{k+1}$ per the paper's
       central-difference formula (not 2, not 1), reflecting the coarser pixel
       spacing at level $k$.
    3. $\alpha$ default: paper states "0.1 times the average gradient magnitude" (Sec. 4).
    4. $\beta$ default: paper states "between 0.8 and 0.9" (Sec. 4). We default to 0.87.
    5. Color saturation $s$: paper states "values between 0.4 and 0.6" (Sec. 5).
    """

    # [PAPER_STRICT] Section 4: minimum coarsest pyramid dimension.
    MIN_PYRAMID_DIM = 32

    def __init__(
        self,
        alpha: float | None = None,
        beta: float = 0.87,
        saturation: float = 0.5,
        gamma: float = 2.2,
        white_percentile: float = 99.0,
    ):
        r"""
        Args:
            alpha: Gradient magnitude threshold $\alpha$. Gradients at this magnitude
                   are unchanged ($\varphi = 1$). Defaults to None, computed as
                   $0.1 \times$ mean gradient magnitude of the full-resolution
                   log-luminance (Sec. 4).
            beta: Compression exponent $\beta$. $\beta < 1$ attenuates large gradients;
                  $\beta = 1$ is a no-op. Paper range: 0.8–0.9. (Sec. 4, default 0.87).
            saturation: Color saturation exponent $s$ applied in color restoration step.
                        Paper range: 0.4–0.6. (Sec. 5, default 0.5).
            gamma: Display gamma for final LDR output (default 2.2).
            white_percentile: Percentile of the output luminance used as the white
                              point before display mapping (default 99.0).
                              [ENGINEERING_ADAPTATION] The paper states "shift and
                              scale the solution to fit display device limits"
                              (Section 5) without specifying the exact mapping.
                              Using a high percentile (rather than the absolute
                              maximum) prevents isolated bright highlights from
                              collapsing the mid-tone exposure.
        """
        self.alpha = alpha
        self.beta = beta
        self.saturation = saturation
        self.gamma = gamma
        self.white_percentile = white_percentile

        # [PAPER_STRICT] Section 4: luminance weights for RGB → Y (Rec. 709)
        # The paper does not specify weights; Rec. 709 is the standard choice.
        # [ENGINEERING_ADAPTATION] Using standard Rec. 709 weights.
        self.Y_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

    # ------------------------------------------------------------------
    # Step 1 – Gaussian Pyramid
    # ------------------------------------------------------------------

    def _build_gaussian_pyramid(self, H: np.ndarray) -> list[np.ndarray]:
        r"""
        [PAPER_STRICT] Section 4: Build Gaussian pyramid $H_0, \ldots, H_d$.
        $H_0$ is the full-resolution log-luminance; $H_d$ has width and height $\geq 32$.

        Uses cv2.pyrDown (Gaussian blur + $2\times$ downsampling), consistent with the
        paper's Gaussian pyramid definition.
        """
        pyramid = [H]
        current = H
        while True:
            h, w = current.shape
            next_h, next_w = (h + 1) // 2, (w + 1) // 2
            if next_h < self.MIN_PYRAMID_DIM or next_w < self.MIN_PYRAMID_DIM:
                break
            current = cv2.pyrDown(current.astype(np.float32)).astype(np.float64)
            pyramid.append(current)
        return pyramid  # pyramid[0] = finest, pyramid[-1] = coarsest

    # ------------------------------------------------------------------
    # Step 2 – Per-level gradients and attenuation
    # ------------------------------------------------------------------

    def _central_diff_gradients(self, Hk: np.ndarray, k: int):
        r"""
        [PAPER_STRICT] Section 4, Eq.:

        .. math::

            \nabla H_k = \left(
                \frac{H_k(x+1,y) - H_k(x-1,y)}{2^{k+1}},\;
                \frac{H_k(x,y+1) - H_k(x,y-1)}{2^{k+1}}
            \right)

        Boundary pixels are handled by replicating the edge value (Neumann-like),
        which is equivalent to a zero-derivative boundary.

        Returns:
            grad_x, grad_y: shape (H, W) each.
        """
        denom = 2.0 ** (k + 1)

        # Central differences with edge replication
        Hk_pad_x = np.pad(Hk, ((0, 0), (1, 1)), mode="edge")
        grad_x = (Hk_pad_x[:, 2:] - Hk_pad_x[:, :-2]) / denom

        Hk_pad_y = np.pad(Hk, ((1, 1), (0, 0)), mode="edge")
        grad_y = (Hk_pad_y[2:, :] - Hk_pad_y[:-2, :]) / denom

        return grad_x, grad_y

    def _compute_phi_k(
        self, grad_x: np.ndarray, grad_y: np.ndarray, alpha: float
    ) -> np.ndarray:
        r"""
        [PAPER_STRICT] Section 4:

        .. math::

            \varphi_k(x,y) = \frac{\alpha}{\|\nabla H_k(x,y)\|}
                \left(\frac{\|\nabla H_k(x,y)\|}{\alpha}\right)^{\beta}

        [ENGINEERING_ADAPTATION] To prevent numerical explosion in perfectly flat
        regions while maintaining the paper's "magnification" effect, we use a
        small epsilon and a rational cap.
        """
        mag = np.sqrt(grad_x**2 + grad_y**2)
        safe_mag = np.maximum(mag, 1e-9)
        phi = (alpha / safe_mag) * np.power(safe_mag / alpha, self.beta)

        # Limit magnification to 100x to avoid floating point overflow
        # across multiple levels of the pyramid.
        return np.minimum(phi, 100.0)

    # ------------------------------------------------------------------
    # Step 3 – Accumulate attenuation top-down
    # ------------------------------------------------------------------

    def _compute_Phi(self, pyramid: list[np.ndarray], alpha: float) -> np.ndarray:
        r"""
        [PAPER_STRICT] Section 4:

        .. math::

            \Phi_d = \varphi_d, \quad
            \Phi_k = L(\Phi_{k+1}) \cdot \varphi_k, \quad
            \Phi   = \Phi_0

        where $L$ denotes bilinear upsampling to the next finer level.
        Propagation is top-down (coarsest $\to$ finest), accumulating per-level
        attenuation maps by pointwise multiplication.
        """
        d = len(pyramid) - 1

        # Initialise at coarsest level
        gx_d, gy_d = self._central_diff_gradients(pyramid[d], d)
        Phi = self._compute_phi_k(gx_d, gy_d, alpha)

        # Propagate from level d-1 down to level 0
        for k in range(d - 1, -1, -1):
            target_h, target_w = pyramid[k].shape
            # $L(\Phi_{k+1})$: bilinear upsample to level $k$ size
            Phi_up = cv2.resize(
                Phi.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float64)

            gx_k, gy_k = self._central_diff_gradients(pyramid[k], k)
            phi_k = self._compute_phi_k(gx_k, gy_k, alpha)

            Phi = (
                Phi_up * phi_k
            )  # [PAPER_STRICT] pointwise product $\Phi_k = L(\Phi_{k+1}) \cdot \varphi_k$

        return Phi

    # ------------------------------------------------------------------
    # Step 4 – Attenuated gradient field G = ∇H_0 · Φ
    # ------------------------------------------------------------------

    def _compute_attenuated_gradients(self, H0: np.ndarray, Phi: np.ndarray):
        r"""
        [PAPER_STRICT] Section 3: $G(x,y) = \nabla H(x,y) \cdot \Phi(x,y)$.

        $\nabla H_0$ is computed with forward differences at the finest level ($k=0$),
        as required by the finite-difference discretisation in Section 5.

        Section 5 specifies:

        .. math::

            \nabla H(x,y) \approx
                \bigl(H(x+1,y) - H(x,y),\; H(x,y+1) - H(x,y)\bigr)
        """
        # Forward differences (Section 5)
        Gx = np.zeros_like(H0)
        Gy = np.zeros_like(H0)

        Gx[:, :-1] = H0[:, 1:] - H0[:, :-1]  # zero at right boundary (Neumann)
        Gy[:-1, :] = H0[1:, :] - H0[:-1, :]  # zero at bottom boundary (Neumann)

        Gx *= Phi
        Gy *= Phi
        return Gx, Gy

    # ------------------------------------------------------------------
    # Step 5 – Divergence of G
    # ------------------------------------------------------------------

    def _compute_divergence(self, Gx: np.ndarray, Gy: np.ndarray) -> np.ndarray:
        r"""
        [PAPER_STRICT] Section 5 backward differences:

        .. math::

            \operatorname{div}\,G \approx
                G_x(x,y) - G_x(x-1,y) + G_y(x,y) - G_y(x,y-1)

        This backward-difference divergence is consistent with the forward-
        difference gradient and with the 5-point Laplacian stencil.
        """
        div = np.zeros_like(Gx)

        # Backward difference in x: $G_x(x,y) - G_x(x-1,y)$
        div[:, 1:] += Gx[:, 1:] - Gx[:, :-1]
        div[:, 0] += Gx[:, 0]  # left boundary: $G_x(-1,y) = 0$

        # Backward difference in y: $G_y(x,y) - G_y(x,y-1)$
        div[1:, :] += Gy[1:, :] - Gy[:-1, :]
        div[0, :] += Gy[0, :]  # top boundary: $G_y(x,-1) = 0$

        return div

    # ------------------------------------------------------------------
    # Step 6 – Solve Poisson equation ∇²I = div G  (Neumann BCs)
    # ------------------------------------------------------------------

    def _solve_poisson_dct(self, rhs: np.ndarray) -> np.ndarray:
        r"""
        [ENGINEERING_ADAPTATION] DCT-based rapid Poisson solver ($O(n \log n)$).

        The paper recommends Full Multigrid (Section 5, $O(n)$), but also mentions
        a "rapid Poisson solver" via FFT as an alternative. The DCT-II / DCT-III
        pair diagonalises the 5-point Laplacian under Neumann BCs exactly, making
        it the canonical spectral solver for this problem.

        Neumann BCs correspond to DCT-II (the standard orthogonal DCT):

        .. math::

            \lambda(u,v) = 2\cos\!\left(\frac{\pi u}{W}\right)
                         + 2\cos\!\left(\frac{\pi v}{H}\right) - 4

        The DC component ($u=v=0$) has $\lambda=0$; we set $I(0,0) = 0$, which
        fixes the additive constant that is otherwise undetermined under Neumann
        BCs (the paper acknowledges this freedom in Section 5).
        """
        h, w = rhs.shape

        # Forward DCT-II on right-hand side
        rhs_dct = dctn(rhs, norm="ortho")

        # Build eigenvalue grid of the 5-point Laplacian under Neumann BCs
        u = np.arange(w)
        v = np.arange(h)
        lambda_x = 2.0 * np.cos(np.pi * u / w) - 2.0  # shape (W,)
        lambda_y = 2.0 * np.cos(np.pi * v / h) - 2.0  # shape (H,)
        Lambda = lambda_y[:, None] + lambda_x[None, :]  # shape (H, W)

        # Avoid division by zero at DC; solution defined up to a constant
        Lambda[0, 0] = 1.0

        I_dct = rhs_dct / Lambda
        I_dct[0, 0] = 0.0  # fix the additive constant

        # Inverse DCT-III (= inverse of DCT-II)
        I = idctn(I_dct, norm="ortho")
        return I

    # ------------------------------------------------------------------
    # Step 7 – Color restoration
    # ------------------------------------------------------------------

    def _restore_color(
        self,
        img_rgb: np.ndarray,
        L_in: np.ndarray,
        L_out: np.ndarray,
    ) -> np.ndarray:
        r"""
        [PAPER_STRICT] Section 5:

        .. math::

            C_{\mathrm{out}} = \left(\frac{C_{\mathrm{in}}}{L_{\mathrm{in}}}\right)^s
                               L_{\mathrm{out}}, \quad C \in \{R, G, B\}

        where $s$ is the saturation parameter ($0.4 \leq s \leq 0.6$).
        """
        eps = 1e-6
        ratio = (img_rgb / (L_in[:, :, np.newaxis] + eps)) ** self.saturation
        return ratio * L_out[:, :, np.newaxis]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Execute the Fattal 2002 tone mapping pipeline.

        Args:
            img_rgb: Linear HDR image ($H \times W \times 3$), float32 or float64, RGB order.

        Returns:
            LDR image (H × W × 3), uint8, gamma-corrected.
        """
        img = img_rgb.astype(np.float64)

        # --- Luminance ---
        L_in = np.dot(img, self.Y_WEIGHTS)
        eps = 1e-8
        L_safe = np.maximum(L_in, eps)

        # --- Log-luminance H = log(L) ---
        # [PAPER_STRICT] Sec. 3: "all computations are done on the logarithm
        # of the luminances" (natural log is standard; base doesn't affect the
        # final result since differences are what matter and the additive
        # constant is removed in the Poisson solve).
        H = np.log(L_safe)

        # --- Gaussian pyramid ---
        pyramid = self._build_gaussian_pyramid(H)
        num_levels = len(pyramid)

        # --- Alpha: $\alpha = 0.1 \times$ mean full-resolution gradient magnitude ---
        if self.alpha is None:
            gx0, gy0 = self._central_diff_gradients(pyramid[0], 0)
            mean_mag = np.mean(np.sqrt(gx0**2 + gy0**2))
            # [PAPER_STRICT] Sec. 4: $\alpha = 0.1 \times$ average gradient magnitude
            alpha = 0.1 * mean_mag if mean_mag > 1e-10 else 1e-4
        else:
            alpha = self.alpha

        # --- Multi-scale attenuation $\Phi$ ---
        Phi = self._compute_Phi(pyramid, alpha)

        # --- Attenuated gradient field G ---
        Gx, Gy = self._compute_attenuated_gradients(H, Phi)

        # --- Divergence of G ---
        div_G = self._compute_divergence(Gx, Gy)

        # --- Solve $\nabla^2 I = \operatorname{div} G$ ---
        I_log = self._solve_poisson_dct(div_G)

        # --- Reconstruct LDR luminance ---
        # [PAPER_STRICT] Section 5: "shift and scale the solution to fit display
        # device limits." The additive constant in $I$ is free (Neumann BCs).
        # [ENGINEERING_ADAPTATION] Linearly map I_log to [-I_range, 0] before exp
        # to utilize the full display dynamic range. Using percentiles avoids
        # outliers collapsing the range.
        p_low, p_high = np.percentile(I_log, [1.0, self.white_percentile])
        I_min, I_max = p_low, p_high
        I_range = I_max - I_min
        if I_range > 1e-8:
            I_log = (I_log - I_max) / I_range * np.log(20.0)  # Map to [-log(20), 0]
        else:
            I_log = I_log - I_max
        L_out = np.exp(I_log)  # L_out in [0.05, 1]

        # --- Color restoration ---
        img_out = self._restore_color(img, L_in, L_out)

        # --- Clip and apply display gamma ---
        # Clip to [0, 1]: pixels above the white point clip to white.
        img_out = np.clip(img_out, 0.0, 1.0)

        # [ENGINEERING_ADAPTATION] Standard display gamma encoding (default 2.2).
        img_out = np.power(img_out, 1.0 / self.gamma)
        return (img_out * 255.0).astype(np.uint8)


def load_hdr(path: str) -> np.ndarray:
    """Load an HDR image and return linear RGB float32."""
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fattal 2002 Gradient Domain HDR Compression — Strict Reference Implementation"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="memorial",
        help="Input HDR file path or dataset name (default: memorial)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Gradient magnitude threshold α. Default: 0.1 × mean grad magnitude (Sec. 4)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.87,
        help="Compression exponent β (paper range 0.8–0.9, default 0.87)",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=0.5,
        help="Color saturation exponent s (paper range 0.4–0.6, default 0.5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="Display gamma (default 2.2)",
    )
    parser.add_argument(
        "--white_percentile",
        type=float,
        default=99.0,
        help="Percentile of output luminance used as white point (default 99.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: output/<stem>_fattal2002.png)",
    )

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent

    # Smart path resolution (consistent with other HDR scripts)
    in_p = Path(args.input)
    if not in_p.exists():
        candidate = base_dir / "dataset" / args.input / f"{args.input}.hdr"
        if candidate.exists():
            in_p = candidate
        else:
            in_p = base_dir / "dataset" / args.input

    out_p = args.output or str(base_dir / "output" / f"{in_p.stem}_fattal2002.png")
    Path(out_p).parent.mkdir(parents=True, exist_ok=True)

    print(f"--- Fattal 2002 ---")
    print(f"Input : {in_p}")
    print(
        f"Params: alpha={args.alpha or 'auto'}, beta={args.beta}, "
        f"saturation={args.saturation}, gamma={args.gamma}, "
        f"white_percentile={args.white_percentile}"
    )

    try:
        hdr = load_hdr(str(in_p))
        tmo = Fattal2002(
            alpha=args.alpha,
            beta=args.beta,
            saturation=args.saturation,
            gamma=args.gamma,
            white_percentile=args.white_percentile,
        )
        result = tmo.process(hdr)
        cv2.imwrite(out_p, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved: {out_p}")
    except Exception:
        import traceback

        traceback.print_exc()
