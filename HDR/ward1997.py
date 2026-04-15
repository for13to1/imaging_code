"""
Larson/Ward 1997 HDR Tone Reproduction Implementation.

Based on:
Larson, G.W., Rushmeier, H., & Piatko, C. (1997). 
A visibility matching tone reproduction operator for high dynamic range scenes.
IEEE Transactions on Visualization and Computer Graphics.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import traceback

class Ward1997:
    """
    Implements the Larson/Ward 1997 tone mapping operator with full 
    physiological and physical modeling.
    """
    def __init__(self, enable_glare=True, enable_acuity=True, enable_color=True, 
                 n_bins=512, ld_max=100.0, ld_min=1.0, scale=60.0):
        """
        Initialize the TMO with display and behavior parameters.
        
        Args:
            enable_glare: Include veiling glare in the adaptation map.
            enable_acuity: Apply spatially varying acuity loss blur.
            enable_color: Include mesopic/scotopic color sensitivity.
            n_bins: Number of histogram bins.
            ld_max: Maximum display luminance (Ld_max, default 100 cd/m2).
            ld_min: Minimum display luminance (Ld_min, default 1 cd/m2).
            scale: Calibration factor to map input data to absolute cd/m2 (default 60 for Memorial).
        """
        self.enable_glare = enable_glare
        self.enable_acuity = enable_acuity
        self.enable_color = enable_color
        self.n_bins = n_bins
        self.scale = scale
        
        # Display parameters
        self.ld_max = ld_max
        self.ld_min = ld_min
        self.log_ld_min = np.log10(self.ld_min)
        self.log_ld_max = np.log10(self.ld_max)
        self.log_ld_range = self.log_ld_max - self.log_ld_min

        # Physiological Constants
        self.VADAPT = 0.08      # Fraction of adaptation from glare
        self.TOP_MESOPIC = 5.62 # cd/m2
        self.BOT_MESOPIC = 0.00562

        # CIE XYZ (D65) Matrices
        self.rgb2xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        self.xyz2rgb = np.linalg.inv(self.rgb2xyz)

    def srgb_to_xyz(self, img_rgb: np.ndarray) -> np.ndarray:
        return np.dot(img_rgb, self.rgb2xyz.T)

    def xyz_to_srgb(self, img_xyz: np.ndarray) -> np.ndarray:
        return np.dot(img_xyz, self.xyz2rgb.T)

    def threshold_function_ferwerda(self, la: np.ndarray) -> np.ndarray:
        """
        TVI piecewise approximation from Larson/Ward 1997 Table 1.
        Returns the delta-luminance (JND) threshold.
        """
        log_la = np.log10(np.maximum(la, 1e-6))
        log_dl = np.zeros_like(log_la)
        
        m1 = log_la < -3.94
        log_dl[m1] = -2.86
        m2 = (log_la >= -3.94) & (log_la < -1.44)
        log_dl[m2] = np.power(0.405 * log_la[m2] + 1.6, 2.18) - 2.86
        m3 = (log_la >= -1.44) & (log_la < -0.0184)
        log_dl[m3] = log_la[m3] - 0.395
        m4 = (log_la >= -0.0184) & (log_la < 1.9)
        log_dl[m4] = np.power(0.249 * log_la[m4] + 0.65, 2.7) - 0.72
        m5 = log_la >= 1.9
        log_dl[m5] = log_la[m5] - 1.255

        return np.power(10.0, log_dl)

    def _prepare_adaptation_map(self, img_xyz, fov_deg):
        """Phase I: Compute foveal samples, glare veil, and adaptation (Steps 1-4)."""
        rows, cols, _ = img_xyz.shape
        rad_fov = np.radians(fov_deg)
        
        # Step 1: Compute foveal samples (Grid resolution: ~1 degree per pixel)
        # Section 4.1: "A grid of 50x50... provides a resolution of approximately one degree"
        target_size = (int(max(1, fov_deg)), int(max(1, fov_deg * (rows/cols))))
        foveal_xyz = cv2.resize(img_xyz, target_size, interpolation=cv2.INTER_AREA)
        foveal_y = foveal_xyz[:, :, 1]
        
        if not self.enable_glare:
            return foveal_y, img_xyz, 1.0

        # Step 2: Glare Calculation (Vectorized Holladay Integral)
        y_dir, x_dir = np.mgrid[0:target_size[1], 0:target_size[0]]
        vx = np.tan((x_dir / (target_size[0] - 1) - 0.5) * rad_fov)
        vy = np.tan((y_dir / (target_size[1] - 1) - 0.5) * rad_fov * (target_size[1]/target_size[0]))
        directions = np.stack([vx, vy, np.ones_like(vx)], axis=-1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        
        # Block-based physical summation for glare veiling (Holladay integral)
        xyz_flat = foveal_xyz.reshape(-1, 3); vecs = directions.reshape(-1, 3)
        veil_xyz_flat = np.zeros_like(xyz_flat); block_size = 1000
        
        # Solid angle per sample pixel (Omega)
        omega = (rad_fov / target_size[0])**2
        
        for i in range(0, len(xyz_flat), block_size):
            end_i = min(i + block_size, len(xyz_flat))
            # cos(theta) between directions
            cos_a = np.clip(np.dot(vecs[i:end_i], vecs.T), -1, 1)
            # theta in degrees
            theta_deg = np.degrees(np.arccos(cos_a))
            
            # Holladay model (Section 4.5): Valid for theta > 1 degree.
            # "For smaller angles, we simply omit these small angles."
            with np.errstate(divide='ignore', invalid='ignore'):
                 weights = 10.0 * omega / np.power(theta_deg, 2)
                 weights[theta_deg <= 1.0] = 0.0 # Strict 1-deg exclusion
                 weights[np.isinf(weights)] = 0.0
            
            # Veiling luminance (Lv) is a summation of absolute scattered light
            veil_xyz_flat[i:end_i] = np.dot(weights, xyz_flat)
        
        veil_foveal_xyz = veil_xyz_flat.reshape(target_size[1], target_size[0], 3)
        veil_full_xyz = cv2.resize(veil_foveal_xyz, (cols, rows), interpolation=cv2.INTER_LINEAR)
        
        # Step 3 & 4: Calculate adaptation map and veiled image (VADAPT = 0.08)
        la_foveal = (1.0 - self.VADAPT) * foveal_y + self.VADAPT * veil_foveal_xyz[:, :, 1]
        img_xyz_veiled = (1.0 - self.VADAPT) * img_xyz + self.VADAPT * veil_full_xyz
        return la_foveal, img_xyz_veiled, target_size[0]

    def _apply_physiological_models(self, img_xyz, la_foveal, cdm2_scale, foveal_w):
        """Phase II: Acuity and Color sensitivities (Steps 5-6)."""
        rows, cols, _ = img_xyz.shape
        la_abs_f = la_foveal * cdm2_scale
        
        # Step 5: Visual Acuity (Spatial Blur)
        if self.enable_acuity:
            log_la_abs_f = np.log10(np.maximum(la_abs_f, 1e-6))
            acuity_f = 17.25 * np.arctan(1.4 * log_la_abs_f + 0.35) + 25.72
            sigma_full_px = cv2.resize(1.0/(2.0*np.maximum(acuity_f,1.0)), (cols, rows)) * (cols/foveal_w)
            
            max_sigma = float(np.max(sigma_full_px))
            if max_sigma < 0.5:
                img_xyz = img_xyz
            else:
                pyramid = [img_xyz.astype(np.float32)]
                levels = min(6, int(np.ceil(np.log2(max_sigma + 1))) + 1)
                for l in range(1, levels): pyramid.append(cv2.GaussianBlur(pyramid[0], (0, 0), sigmaX=2.0**l))
                l_idx = np.clip(np.log2(np.maximum(sigma_full_px, 1e-3)), 0, levels-1)
                l_lo = l_idx.astype(int); l_hi = np.minimum(l_lo + 1, levels - 1)
                alpha = (l_idx - l_lo.astype(float))[:, :, np.newaxis]
                img_xyz = np.zeros_like(img_xyz, dtype=np.float32)
                for l in range(levels):
                    img_xyz += pyramid[l] * ((l_lo==l).astype(float)*(1-alpha) + (l_hi==l).astype(float)*alpha)
        
        # Step 6: Mesopic Color Conversion
        if self.enable_color:
            la_abs_full = cv2.resize(la_abs_f, (cols, rows))
            X, Y, Z = img_xyz[:,:,0], img_xyz[:,:,1], img_xyz[:,:,2]
            y_scot = np.maximum(Y * (1.33 * (1 + (Y+Z)/(X+1e-8)) - 1.68), 0)
            k = np.clip((np.log10(np.maximum(la_abs_full,1e-6)) - np.log10(self.BOT_MESOPIC)) / 3.0, 0, 1)
            img_xyz = np.stack([k*X+(1-k)*y_scot, k*Y+(1-k)*y_scot, k*Z+(1-k)*y_scot], axis=-1)
            
        return img_xyz

    def _build_tone_mapping_function(self, la_foveal, full_y, cdm2_scale):
        """Phase III: Histogram Adjustment & Visibility Matching (Steps 7-8)."""
        # Step 7: Hybrid Sampling
        f_samples = la_foveal.flatten() * cdm2_scale
        rn_idx = np.random.choice(full_y.size, size=min(f_samples.size*16, full_y.size), replace=False)
        all_samples = np.concatenate([f_samples, full_y.flatten()[rn_idx] * cdm2_scale])
        
        log_samples = np.log10(np.maximum(all_samples, 1e-7))
        hist, bin_edges = np.histogram(log_samples, bins=self.n_bins)
        widths = bin_edges[1:] - bin_edges[:-1]; centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        
        # Step 8: Histogram Adjustment (Proportional Redistribution)
        f = hist.astype(np.float32); total_n = np.sum(f)
        orig_f = f.copy()
        
        for _ in range(30):
            p = f / total_n
            p_cum = np.cumsum(p)
            p_cum_mid = p_cum - 0.5 * p 
            
            ld = np.power(10.0, self.log_ld_min + self.log_ld_range * p_cum_mid)
            lw = np.power(10.0, centers)
            
            delta_ld = self.threshold_function_ferwerda(ld)
            delta_lw = self.threshold_function_ferwerda(lw)
            
            # Larson 1997 Eq 7b Ceiling
            ceiling = (delta_ld / delta_lw) * (total_n * widths * lw / (self.log_ld_range * ld))
            
            over = f > (ceiling + 1e-4) # Epsilon for stability
            if not np.any(over): break
            
            # Redistribute excess count proportional to ORIGINAL counts of unclipped bins
            excess = np.sum(f[over] - ceiling[over])
            f[over] = ceiling[over]
            
            unclipped = ~over
            if np.any(unclipped):
                unclipped_orig_sum = np.sum(orig_f[unclipped])
                if unclipped_orig_sum > 1e-6:
                    f[unclipped] += excess * (orig_f[unclipped] / unclipped_orig_sum)
                else:
                    # Fallback to uniform if original counts are zero (unlikely but safe)
                    f[unclipped] += excess / np.sum(unclipped)
            
        p_mapping = np.insert(np.cumsum(f/np.sum(f)), 0, 0.0)
        return bin_edges, p_mapping

    def process(self, img_xyz, fov_deg=140.0) -> np.ndarray:
        """Execute the full 10-step Larson/Ward tone reproduction pipeline."""
        img_xyz = img_xyz.astype(np.float32)
        
        # I. Preparation (Steps 1-4)
        la_foveal, img_xyz_veiled, fov_w = self._prepare_adaptation_map(img_xyz, fov_deg)
        
        # Physical Calibration Scale
        # The paper assumes absolute luminance; .hdr files require a scale factor.
        la_median = np.median(la_foveal[la_foveal > 1e-6])
        actual_scale = self.scale / (la_median + 1e-8) if self.scale != 1.0 else 1.0
        
        # II. Physiological Modeling (Steps 5-6)
        img_xyz_processed = self._apply_physiological_models(img_xyz_veiled, la_foveal, actual_scale, fov_w)
        
        # III. Building Mapping (Steps 7-8)
        bin_edges, p_mapping = self._build_tone_mapping_function(la_foveal, img_xyz[:,:,1], actual_scale)
        
        # IV. Application & Display (Steps 9-10)
        py_abs = img_xyz_processed[:, :, 1] * actual_scale
        log_lw = np.log10(np.maximum(py_abs, 1e-7))
        
        # Histogram mapping applying Eq 7
        log_ld_vals = self.log_ld_min + self.log_ld_range * p_mapping
        log_ld_full = np.interp(log_lw, bin_edges, log_ld_vals)
        ld_full = np.power(10.0, log_ld_full)
        
        img_xyz_mapped = img_xyz_processed * (ld_full / (py_abs/actual_scale + 1e-8))[:, :, np.newaxis]
        img_rgb = np.clip(self.xyz_to_srgb(img_xyz_mapped) / self.ld_max, 0, 1)
        
        # Apply display gamma and return 8-bit image
        return (np.power(img_rgb, 1.0/2.2) * 255).astype(np.uint8)

def load_hdr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"Missing {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Larson/Ward 1997 Reference Implementation")
    parser.add_argument("input", type=str, nargs="?", default="HDR/dataset/memorial/memorial.hdr")
    parser.add_argument("--fov", type=float, default=140.0, help="Field of view in degrees")
    parser.add_argument("--scale", type=float, default=60.0, help="Absolute luminance scale factor")
    parser.add_argument("--ld-max", type=float, default=100.0, help="Max display luminance (Ld_max)")
    parser.add_argument("--ld-min", type=float, default=1.0, help="Min display luminance (Ld_min)")
    args = parser.parse_args()
    
    # Auto-output path: HDR/output/<input>_ward1997.png
    output_dir = Path("HDR/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"{Path(args.input).stem}_ward1997.png")

    print(f"--- Processing {args.input} (FOV={args.fov}, Scale={args.scale}) ---")
    try:
        img_hdr = load_hdr(args.input)
        tmo = Ward1997(ld_max=args.ld_max, ld_min=args.ld_min, scale=args.scale)
        result = tmo.process(tmo.srgb_to_xyz(img_hdr), fov_deg=args.fov)
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"✅ Result saved: {output_path}")
    except Exception:
        traceback.print_exc()
