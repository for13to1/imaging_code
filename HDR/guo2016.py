#!/usr/bin/env python3
"""
LIME: Low-Light Image Enhancement via Illumination Map Estimation (Guo 2016)

Based on:
Guo, X., Li, Y., & Ling, H. (2016).
LIME: Low-light image enhancement via illumination map estimation.
IEEE Transactions on Image Processing, 26(2), 982-993.
"""

from pathlib import Path

import cv2
import numpy as np


class Guo2016:
    """
    LIME: Low-Light Image Enhancement via Illumination Map Estimation.

    Implementation Fidelity Categories:
    1. [PAPER_STRICT]: Formulas and logic strictly following the paper.
    2. [ENGINEERING_ADAPTATION]: Omitted details, ambiguities, or modern stability/performance heuristics.
    3. [RATIONAL_OMISSION]: Features not implemented for simplicity.

    Engineering Adaptations:
    - Convergence: ‖∇T - G‖_F ≤ δ‖T̂‖_F (paper: "while not converged"). [ENGINEERING_ADAPTATION]
    - FFT preconditioner for CG in sped-up solver (paper: unspecified). [ENGINEERING_ADAPTATION]
    - BM3D sigma auto-normalized for [0, 1] float range. [ENGINEERING_ADAPTATION]
    - YUV via float32 to avoid uint8 roundtrip precision loss. [ENGINEERING_ADAPTATION]
    - filter2D border: BORDER_REFLECT_101 (not strict Gaussian convolution). [ENGINEERING_ADAPTATION]
    """

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.8,
        sigma: float = 2.0,
        strategy: int = 3,
        use_fast_solver: bool = True,
        eps: float = 1e-6,
        max_iter: int = 60,
        mu0: float = 1.0,
        rho: float = 2.0,
        delta: float = 1e-5,
        denoise_sigma: float | None = None,
    ):
        """
        Initialize LIME parameters.

        Args:
            alpha: Balance coefficient for structure-aware smoothing (default 0.15).
            gamma: Gamma correction parameter (default 0.8).
            sigma: Gaussian kernel standard deviation for Strategy III (default 2.0).
            strategy: Weighting strategy (1, 2, or 3, default 3).
            use_fast_solver: Use sped-up solver (CG) or exact solver (ALM-ADM).
            eps: Small constant to avoid division by zero.
            max_iter: Maximum iterations for exact solver (default 60).
            mu0: Initial penalty parameter for ALM (default 1.0).
            rho: Penalty growth rate for ALM (default 2.0).
            delta: Convergence threshold for solvers (default 1e-5).
            denoise_sigma: Noise standard deviation for BM3D post-processing.
                           If None, skip denoising. (default None)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.strategy = strategy
        self.use_fast_solver = use_fast_solver
        self.eps = eps
        self.max_iter = max_iter
        self.mu0 = mu0
        self.rho = rho
        self.delta = delta
        self.denoise_sigma = denoise_sigma

    def _compute_gradient(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute horizontal and vertical gradients using forward difference.
        [PAPER_STRICT] Section II-A: first order derivative filter.
        """
        # Horizontal gradient (right difference)
        grad_h = np.zeros_like(img)
        grad_h[:, :-1] = img[:, 1:] - img[:, :-1]
        grad_h[:, -1] = img[:, 0] - img[:, -1]  # Circular boundary

        # Vertical gradient (down difference)
        grad_v = np.zeros_like(img)
        grad_v[:-1, :] = img[1:, :] - img[:-1, :]
        grad_v[-1, :] = img[0, :] - img[-1, :]  # Circular boundary

        return grad_h, grad_v

    def _adjoint_gradient(self, grad_h: np.ndarray, grad_v: np.ndarray) -> np.ndarray:
        """
        D^T: backward difference with circular boundary.
        [PAPER_STRICT] Eq. (12) T sub-problem.
        """
        div = np.zeros_like(grad_h)

        # Horizontal part: D_h^T Y[i,j] = Y[i,j-1] - Y[i,j]
        div[:, 0] = grad_h[:, -1] - grad_h[:, 0]
        div[:, 1:] = grad_h[:, :-1] - grad_h[:, 1:]

        # Vertical part: D_v^T Y[i,j] = Y[i-1,j] - Y[i,j]
        div[0, :] += grad_v[-1, :] - grad_v[0, :]
        div[1:, :] += grad_v[:-1, :] - grad_v[1:, :]

        return div

    def _compute_weight_map(self, T_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute weight matrix W based on selected strategy.
        [PAPER_STRICT] Section II-D: Possible Weighting Strategies.
        """
        grad_h, grad_v = self._compute_gradient(T_hat)

        if self.strategy == 1:
            # Strategy I: Classic TV minimization [PAPER_STRICT] Eq. (20)
            W_h = np.ones_like(T_hat)
            W_v = np.ones_like(T_hat)
        elif self.strategy == 2:
            # Strategy II: Gradient-based weight [PAPER_STRICT] Eq. (21)
            W_h = 1.0 / (np.abs(grad_h) + self.eps)
            W_v = 1.0 / (np.abs(grad_v) + self.eps)
        elif self.strategy == 3:
            # Strategy III: RTV-inspired weight [PAPER_STRICT] Eq. (22)
            kernel_size = int(np.ceil(self.sigma * 3)) * 2 + 1
            W_h = self._compute_rtv_weight(grad_h, kernel_size)
            W_v = self._compute_rtv_weight(grad_v, kernel_size)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Must be 1, 2, or 3.")

        return W_h, W_v

    def _compute_rtv_weight(self, grad: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Compute RTV-inspired weight (Strategy III).
        [PAPER_STRICT] Eq. (22)
        """
        # Gaussian kernel
        sigma = self.sigma
        ax = np.arange(kernel_size) - kernel_size // 2
        xx, yy = np.meshgrid(ax, ax)
        G = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        G = G / G.sum()

        # Weighted sum of gradient
        grad_smooth = cv2.filter2D(grad.astype(np.float64), -1, G)

        # Weight computation
        W = cv2.filter2D(np.ones_like(grad, dtype=np.float64), -1, G) / (np.abs(grad_smooth) + self.eps)

        return W

    def _initial_estimate(self, L: np.ndarray) -> np.ndarray:
        """
        Initial illumination map estimation.
        [PAPER_STRICT] Eq. (2): T_hat(x) = max_{c in {R,G,B}} L^c(x)
        """
        return np.max(L, axis=2)

    def _exact_solver(self, T_hat: np.ndarray, W_h: np.ndarray, W_v: np.ndarray) -> np.ndarray:
        """
        Exact solver using ALM-ADM method.
        [PAPER_STRICT] Algorithm 1 and Section II-B.
        """
        h, w = T_hat.shape
        # [ENGINEERING_ADAPTATION] Paper uses T^{(0)}=0; overwritten by Eq. (13) in first iter anyway.
        G_h = np.zeros_like(T_hat)
        G_v = np.zeros_like(T_hat)
        Z_h = np.zeros_like(T_hat)
        Z_v = np.zeros_like(T_hat)
        mu = self.mu0

        # FFT of gradient operators: D_h(u,v)=e^{-j2πv/W}-1, D_v(u,v)=e^{-j2πu/H}-1
        vy = np.arange(h).reshape(-1, 1) / h
        vx = np.arange(w).reshape(1, -1) / w
        D_h_fft = np.exp(-2j * np.pi * vx) - 1.0
        D_v_fft = np.exp(-2j * np.pi * vy) - 1.0

        denom = 2.0 + mu * (np.abs(D_h_fft) ** 2 + np.abs(D_v_fft) ** 2)

        for _t in range(self.max_iter):
            # T sub-problem (Eq. 13)
            # Compute D^T * (G - Z/mu)
            rhs_h = G_h - Z_h / mu
            rhs_v = G_v - Z_v / mu
            DT_rhs = self._adjoint_gradient(rhs_h, rhs_v)

            numerator = 2.0 * T_hat + mu * DT_rhs
            T_new = np.real(np.fft.ifft2(np.fft.fft2(numerator) / denom))

            # G sub-problem (Eq. 15) - Shrinkage operation
            grad_T_h, grad_T_v = self._compute_gradient(T_new)
            S_h = grad_T_h + Z_h / mu
            S_v = grad_T_v + Z_v / mu

            threshold_h = self.alpha * W_h / mu
            threshold_v = self.alpha * W_v / mu

            G_h = np.sign(S_h) * np.maximum(np.abs(S_h) - threshold_h, 0)
            G_v = np.sign(S_v) * np.maximum(np.abs(S_v) - threshold_v, 0)

            # Z and mu update (Eq. 16)
            residual_h = grad_T_h - G_h
            residual_v = grad_T_v - G_v

            Z_h = Z_h + mu * residual_h
            Z_v = Z_v + mu * residual_v
            mu = mu * self.rho

            # Update denominator (Critical for ALM-ADM correctness in next iteration)
            denom = 2.0 + mu * (np.abs(D_h_fft) ** 2 + np.abs(D_v_fft) ** 2)

            # Convergence check
            residual_norm = np.sqrt(np.sum(residual_h**2 + residual_v**2))
            T_hat_norm = np.sqrt(np.sum(T_hat**2))

            if residual_norm <= self.delta * T_hat_norm:
                break

        return T_new

    def _fast_solver(self, T_hat: np.ndarray, W_h: np.ndarray, W_v: np.ndarray) -> np.ndarray:
        """
        Sped-up solver: solve (I + D_h^T Diag(w̃_h) D_h + D_v^T Diag(w̃_v) D_v) t = t̂.
        [PAPER_STRICT] Eq. (19), solved via conjugate gradient with FFT preconditioner.
        """
        from scipy.sparse.linalg import LinearOperator, cg

        h, w = T_hat.shape
        N = h * w

        grad_h, grad_v = self._compute_gradient(T_hat)

        # [PAPER_STRICT] Eq. (19): W_tilde_d = W_d / (|grad_d T_hat| + eps)
        W_tilde_h = W_h / (np.abs(grad_h) + self.eps)
        W_tilde_v = W_v / (np.abs(grad_v) + self.eps)

        # FFT preconditioner: treat weights as spatially constant
        vy = np.arange(h).reshape(-1, 1) / h
        vx = np.arange(w).reshape(1, -1) / w
        D_h_fft = np.exp(-2j * np.pi * vx) - 1.0
        D_v_fft = np.exp(-2j * np.pi * vy) - 1.0
        W_mean_h = np.mean(W_tilde_h)
        W_mean_v = np.mean(W_tilde_v)
        precond_denom = 1.0 + self.alpha * (W_mean_h * np.abs(D_h_fft) ** 2 + W_mean_v * np.abs(D_v_fft) ** 2)

        def apply_precond(r: np.ndarray) -> np.ndarray:
            return np.real(np.fft.ifft2(np.fft.fft2(r.reshape(h, w)) / precond_denom)).ravel()

        # A = I + α Σ D_d^T Diag(W̃_d) D_d  [PAPER_STRICT] Eq. (19)
        def matvec(x: np.ndarray) -> np.ndarray:
            t = x.reshape(h, w)
            Dt_h, Dt_v = self._compute_gradient(t)
            Dt_h *= W_tilde_h
            Dt_v *= W_tilde_v
            DTDt = self._adjoint_gradient(Dt_h, Dt_v)
            return (t + self.alpha * DTDt).ravel()

        A = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
        b = T_hat.ravel()
        # [ENGINEERING_ADAPTATION] FFT-preconditioned initial guess.
        t0 = apply_precond(b)

        M = LinearOperator((N, N), matvec=apply_precond, dtype=np.float64)
        t_flat, _ = cg(A, b, M=M, x0=t0, rtol=self.delta, maxiter=200)
        return t_flat.reshape(h, w)

    def _gamma_correction(self, T: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction to illumination map.
        [PAPER_STRICT] Section II-E: T <- T^gamma
        """
        return np.power(np.clip(T, 0, 1), self.gamma)

    def _denoise_and_recompose(self, R: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Denoise R on Y channel via BM3D and recompose via Eq. (24).
        [PAPER_STRICT] Section II-E, Algorithm 2 Step 5, Eq. (24).
        """
        import bm3d

        R_float = np.clip(R, 0, 1).astype(np.float32)
        R_yuv = cv2.cvtColor(R_float, cv2.COLOR_RGB2YUV)

        y_channel = R_yuv[:, :, 0]
        sigma = self.denoise_sigma / 255.0 if self.denoise_sigma > 1.0 else self.denoise_sigma
        y_denoised = bm3d.bm3d(y_channel, sigma, profile="np")
        y_denoised = np.clip(y_denoised, 0, 1)

        R_yuv[:, :, 0] = y_denoised

        R_d = cv2.cvtColor(R_yuv, cv2.COLOR_YUV2RGB)

        # Eq. (24): R_f = R ∘ T + R_d ∘ (1 - T)
        T_3d = T[:, :, np.newaxis]
        R_f = R * T_3d + R_d * (1.0 - T_3d)

        return np.clip(R_f, 0, 1)

    def process(self, L: np.ndarray) -> np.ndarray:
        """
        Main LIME algorithm.
        [PAPER_STRICT] Algorithm 2.

        Args:
            L: Low-light input image (float32, range [0, 1], RGB).

        Returns:
            Enhanced image (uint8, range [0, 255], RGB).
        """
        # Initialization: Compute weight matrix (Algorithm 2 initialization)
        T_hat = self._initial_estimate(L)
        W_h, W_v = self._compute_weight_map(T_hat)

        # Step 1: Initial illumination map (Eq. 2)
        # Step 2: Structure-aware refinement via solver
        if self.use_fast_solver:
            T = self._fast_solver(T_hat, W_h, W_v)
        else:
            T = self._exact_solver(T_hat, W_h, W_v)

        # Step 3: Gamma correction (Section II-E)
        T = self._gamma_correction(T)

        # Step 4: Enhance image (Eq. 3: R = L / T)
        T_3d = T[:, :, np.newaxis]
        R = L / (T_3d + self.eps)

        # Step 5: Denoise and recompose (Algorithm 2 Step 5, Eq. 24)
        if self.denoise_sigma is not None:
            R = self._denoise_and_recompose(R, T)

        R = np.clip(R, 0, 1)

        return (R * 255.0).astype(np.uint8)


def load_image(path: str) -> np.ndarray:
    """
    Load image and convert to RGB float32.
    [ENGINEERING_ADAPTATION] Consistent with other HDR modules.
    """
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Missing {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)

    return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LIME: Low-Light Image Enhancement via Illumination Map Estimation (Guo 2016)"
    )
    parser.add_argument("input", type=str, help="Input low-light image path")
    parser.add_argument("--alpha", type=float, default=0.15, help="Balance coefficient (default: 0.15)")
    parser.add_argument("--gamma", type=float, default=0.8, help="Gamma correction parameter (default: 0.8)")
    parser.add_argument(
        "--sigma", type=float, default=2.0, help="Gaussian kernel sigma for Strategy III (default: 2.0)"
    )
    parser.add_argument(
        "--strategy",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Weighting strategy: 1=TV, 2=Gradient, 3=RTV (default: 3)",
    )
    parser.add_argument("--fast", action="store_true", default=True, help="Use fast solver (default: True)")
    parser.add_argument("--exact", action="store_true", help="Use exact solver (ALM-ADM)")
    parser.add_argument(
        "--denoise", type=float, metavar="SIGMA", help="Apply BM3D denoising with given noise sigma (e.g. 0.02)"
    )
    parser.add_argument("--output", type=str, help="Output path")

    args = parser.parse_args()

    # Path resolution
    in_p = Path(args.input)
    if not in_p.exists():
        # Try to find in dataset directory
        base_dir = Path(__file__).resolve().parent
        in_p = base_dir / "dataset" / args.input
        if not in_p.exists():
            # Try common extensions
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                test_p = base_dir / "dataset" / f"{args.input}{ext}"
                if test_p.exists():
                    in_p = test_p
                    break

    if args.output is None:
        base_dir = Path(__file__).resolve().parent
        out_dir = base_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"{in_p.stem}_guo2016.png")

    try:
        L = load_image(str(in_p))
        if args.denoise is not None:
            print(f"--- Processing {in_p.name} with LIME (Guo 2016) + BM3D ---")
        else:
            print(f"--- Processing {in_p.name} with LIME (Guo 2016) ---")
        print(f"Parameters: alpha={args.alpha}, gamma={args.gamma}, sigma={args.sigma}, strategy={args.strategy}")
        solver_name = "Fast (CG + FFT preconditioner)" if args.fast and not args.exact else "Exact (ALM-ADM)"
        print(f"Solver: {solver_name}")
        if args.denoise is not None:
            print(f"  BM3D denoising sigma={args.denoise}")

        lime = Guo2016(
            alpha=args.alpha,
            gamma=args.gamma,
            sigma=args.sigma,
            strategy=args.strategy,
            use_fast_solver=args.fast and not args.exact,
            denoise_sigma=args.denoise,
        )
        result = lime.process(L)

        cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"✅ Result saved: {args.output}")
    except Exception:
        import traceback

        traceback.print_exc()
