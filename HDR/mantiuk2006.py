#!/usr/bin/env python3
r"""
A Perceptual Framework for Contrast Processing of High Dynamic Range Images (Mantiuk 2006)

[STRICT_REPRODUCTION] This implementation follows the formulas in the paper exactly.
- Section 3.1: Contrast definition and objective function.
- Section 3.2: Transducer function T(G).
- Section 4: Contrast Mapping (linear scaling of response).
- Section 5: Contrast Equalization (histogram equalization of response).
- Section 7: Multi-scale solver with recursive A and B.
- Section 8: Color reconstruction with adaptive range mapping (Eq. 26).
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.sparse.linalg import LinearOperator, bicgstab
from typing import List, Tuple, Optional


class Mantiuk2006:
    def __init__(
        self,
        mode: str = "contrast_mapping",
        linear_scale: float = 0.8,
        saturation: float = 0.5,
        gamma: float = 2.2,
        pyramid_levels: int = 8,
        solver_tol: float = 1e-6,
        solver_maxiter: int = 500,
    ):
        self.mode = mode
        self.linear_scale = linear_scale
        self.saturation = saturation
        self.gamma = gamma
        self.pyramid_levels = pyramid_levels
        self.solver_tol = solver_tol
        self.solver_maxiter = solver_maxiter

        # Rec. 709 luminance weights
        self.Y_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

    def _build_gaussian_pyramid(self, x: np.ndarray) -> List[np.ndarray]:
        pyramid = [x]
        current = x
        for _ in range(self.pyramid_levels - 1):
            h, w = current.shape
            if h <= 4 or w <= 4:
                break
            # [PAPER_STRICT] Use float64 for precision; cv2.pyrDown supports float64 directly
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def _transducer_forward(self, G: np.ndarray) -> np.ndarray:
        # Eq. 14: T(G) = 54.09288 * G^0.41850
        return np.sign(G) * 54.09288 * np.power(np.abs(G), 0.41850)

    def _transducer_inverse(self, R: np.ndarray) -> np.ndarray:
        # Eq. 15: T_inv(R) = 7.2232 * 10^-5 * R^2.3895
        return np.sign(R) * 7.2232e-5 * np.power(np.abs(R), 2.3895)

    def _compute_weights(self, G_hat: np.ndarray) -> np.ndarray:
        # [PAPER_STRICT] Eq. 9: p = 1 / DeltaG_simpl(G_hat)
        safe_G = np.maximum(np.abs(G_hat), 0.001)
        delta_G = 0.038737 * np.power(safe_G, 0.537756)
        return 1.0 / (delta_G + 1e-8)

    def _modify_contrast_equalization(
        self, R_list: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        # [PAPER_STRICT] Section 5
        all_responses = []
        for Rx, Ry in R_list:
            # Eq. 28: magnitude ||R|| = sqrt(sum_{j in Phi_i} R_{i,j}^2)
            # For each pixel, consider all 4 neighbors (up, down, left, right)
            h, w = Rx.shape[0], Ry.shape[1]
            mag = np.zeros((h, w))
            mag[:, :-1] += Rx**2  # right neighbor
            mag[:, 1:] += Rx**2  # left neighbor
            mag[:-1, :] += Ry**2  # bottom neighbor
            mag[1:, :] += Ry**2  # top neighbor
            all_responses.append(np.sqrt(mag).flatten())

        flat_R = np.concatenate(all_responses)
        sorted_R = np.sort(flat_R)

        def cpdf(v):
            idx = np.searchsorted(sorted_R, v)
            return idx / len(sorted_R)

        new_R_list = []
        for Rx, Ry in R_list:
            # Eq. 27: R_hat = sign(R) * CPDF(||R||)
            h, w = Rx.shape[0], Ry.shape[1]
            mag = np.zeros((h, w))
            mag[:, :-1] += Rx**2  # right neighbor
            mag[:, 1:] += Rx**2  # left neighbor
            mag[:-1, :] += Ry**2  # bottom neighbor
            mag[1:, :] += Ry**2  # top neighbor
            mag = np.sqrt(mag)

            cpdf_vals = cpdf(mag)
            # Re-scale Rx and Ry by the ratio of new_mag / old_mag
            scale = np.zeros_like(mag)
            mask = mag > 1e-8
            scale[mask] = cpdf_vals[mask] / mag[mask]

            new_Rx = Rx * scale[:, :-1]
            new_Ry = Ry * scale[:-1, :]
            new_R_list.append((new_Rx, new_Ry))
        return new_R_list

    def _solve_optimization(
        self,
        B: np.ndarray,
        pyramid_shapes: List[Tuple[int, int]],
        p_list: List[Tuple[np.ndarray, np.ndarray]],
        x0: np.ndarray,
    ) -> np.ndarray:
        h0, w0 = pyramid_shapes[0]
        n_pixels = h0 * w0

        def matvec(v):
            x = v.reshape(h0, w0)
            x_pyr = [x]
            for i in range(1, len(pyramid_shapes)):
                # [PAPER_STRICT] Use float64 for precision
                x_pyr.append(cv2.pyrDown(x_pyr[-1]))

            def get_ax(k):
                xk = x_pyr[k]
                px, py = p_list[k]
                ax_k = np.zeros_like(xk)

                # sum 2p(xi - xj)
                diff_x = xk[:, :-1] - xk[:, 1:]
                term_x = 2.0 * px * diff_x
                ax_k[:, :-1] += term_x
                ax_k[:, 1:] -= term_x

                diff_y = xk[:-1, :] - xk[1:, :]
                term_y = 2.0 * py * diff_y
                ax_k[:-1, :] += term_y
                ax_k[1:, :] -= term_y

                if k < len(pyramid_shapes) - 1:
                    ax_next = get_ax(k + 1)
                    # [PAPER_STRICT] Eq. 22 upsample
                    ax_up = cv2.pyrUp(ax_next, dstsize=(xk.shape[1], xk.shape[0]))
                    ax_k += ax_up
                return ax_k

            return get_ax(0).flatten()

        A = LinearOperator((n_pixels, n_pixels), matvec=matvec, dtype=np.float64)
        X_res, info = bicgstab(
            A,
            B.flatten(),
            x0=x0.flatten(),
            rtol=self.solver_tol,
            maxiter=self.solver_maxiter,
        )
        # [PAPER_STRICT] info == 0 means convergence; info > 0 means no convergence (reached maxiter)
        return X_res.reshape(h0, w0) if info == 0 else x0

    def process(self, img_rgb: np.ndarray) -> np.ndarray:
        img = img_rgb.astype(np.float64)
        # Rec 709 Luminance
        L_in = np.dot(img, self.Y_WEIGHTS)

        # Log domain for processing
        # [PAPER_STRICT] Rec. 709 uses natural log or log10? Table II says log10.
        x = np.log10(L_in + 1e-8)

        pyramid = self._build_gaussian_pyramid(x)
        shapes = [p.shape for p in pyramid]

        # Step 1: Compute response R
        R_list = []
        for pk in pyramid:
            Gx = pk[:, :-1] - pk[:, 1:]
            Gy = pk[:-1, :] - pk[1:, :]
            R_list.append((self._transducer_forward(Gx), self._transducer_forward(Gy)))

        # Step 2: Modify Response
        if self.mode == "contrast_mapping":
            R_hat_list = [
                (Rx * self.linear_scale, Ry * self.linear_scale) for Rx, Ry in R_list
            ]
        else:  # contrast_equalization
            R_hat_list = self._modify_contrast_equalization(R_list)

        # Step 3: Compute G_hat and Weights p
        G_hat_list = []
        p_list = []
        for Rx_hat, Ry_hat in R_hat_list:
            Gx_hat, Gy_hat = self._transducer_inverse(Rx_hat), self._transducer_inverse(
                Ry_hat
            )
            G_hat_list.append((Gx_hat, Gy_hat))
            p_list.append(
                (self._compute_weights(Gx_hat), self._compute_weights(Gy_hat))
            )

        # Step 4: Compute RHS B (Eq. 24)
        def compute_B(k):
            Gx_hat, Gy_hat = G_hat_list[k]
            px, py = p_list[k]
            bk = np.zeros(shapes[k])

            # sum 2p * G_hat
            tx, ty = 2.0 * px * Gx_hat, 2.0 * py * Gy_hat
            bk[:, :-1] += tx
            bk[:, 1:] -= tx
            bk[:-1, :] += ty
            bk[1:, :] -= ty

            if k < len(pyramid) - 1:
                b_next = compute_B(k + 1)
                bk += cv2.pyrUp(b_next, dstsize=(shapes[k][1], shapes[k][0]))
            return bk

        B = compute_B(0)

        # Step 5: Solve
        X = self._solve_optimization(B, shapes, p_list, x * self.linear_scale)

        # Step 6: Color Reconstruction (Eq. 26)
        # Section 8
        P01, P50, P999 = np.percentile(X, [0.1, 50.0, 99.9])
        d = max(P50 - P01, P999 - P50)
        l_min = P50 - d
        l_max = P50 + d

        log_img = np.log10(img + 1e-8)
        log_L = x[:, :, np.newaxis]

        # Eq. 26
        C_out = (X[:, :, np.newaxis] - l_min + self.saturation * (log_img - log_L)) / (
            l_max - l_min
        )

        # Map [0, 1] to [0, 255]
        res = (np.clip(C_out, 0.0, 1.0) * 255.0).astype(np.uint8)
        return res


def load_hdr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="?", default="memorial")
    parser.add_argument("--mode", type=str, default="contrast_mapping")
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--saturation", type=float, default=0.5)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    in_p = Path(args.input)
    if not in_p.exists():
        in_p = base_dir / "dataset" / args.input / f"{args.input}.hdr"
    out_p = args.output or str(base_dir / "output" / f"{in_p.stem}_mantiuk2006.png")
    Path(out_p).parent.mkdir(parents=True, exist_ok=True)

    print("--- Mantiuk 2006 [STRICT REPRODUCTION] ---")
    hdr = load_hdr(str(in_p))
    tmo = Mantiuk2006(
        mode=args.mode, linear_scale=args.scale, saturation=args.saturation
    )
    result = tmo.process(hdr)
    cv2.imwrite(out_p, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_p}")
