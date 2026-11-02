"""
Gaussian Splatting Trainer — optimizes 3D Gaussians from COLMAP poses and sparse points.

Uses gsplat for high-performance differentiable rasterization and adaptive density
control on NVIDIA DGX Spark (Blackwell GB10). Supports step-by-step progress
reporting, PSNR tracking, and standard 3DGS PLY export.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import GaussianConfig
from utils.colmap_utils import ColmapScene
from utils.ply_utils import GaussianCloud, export_gaussian_ply

logger = logging.getLogger(__name__)


# ── Simple SSIM implementation in PyTorch ─────────────────────────────────────


def _fspecial_gauss(size: int, sigma: float) -> torch.Tensor:
    """Generate 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """Compute Structural Similarity Index (SSIM) between two (C, H, W) images."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    channel = img1.size(1)
    kernel_1d = _fspecial_gauss(window_size, 1.5).to(img1.device)
    kernel_2d = (
        kernel_1d[:, None] @ kernel_1d[None, :]
    ).unsqueeze(0).unsqueeze(0)
    kernel = kernel_2d.expand(channel, 1, window_size, window_size)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean() if size_average else ssim_map


# ── Training Dataset Wrapper ──────────────────────────────────────────────────


class SceneDataset:
    """In-memory dataset of calibrated camera images for 3DGS training."""

    def __init__(
        self,
        scene: ColmapScene,
        image_dir: Path,
        device: torch.device,
        downscale: int = 1,
    ) -> None:
        self.scene = scene
        self.image_dir = image_dir
        self.device = device
        self.downscale = downscale

        self.images: List[torch.Tensor] = []
        self.camtoworlds: List[torch.Tensor] = []
        self.intrinsics: List[torch.Tensor] = []
        self.widths: List[int] = []
        self.heights: List[int] = []
        self.names: List[str] = []

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load and cache images and camera transformations."""
        sorted_img_items = sorted(
            self.scene.images.values(), key=lambda img: img.name
        )

        for colmap_img in sorted_img_items:
            img_path = self.image_dir / colmap_img.name
            if not img_path.exists():
                logger.warning("Image file %s missing, skipping", img_path)
                continue

            cam = self.scene.cameras[colmap_img.camera_id]
            w = cam.width // self.downscale
            h = cam.height // self.downscale

            # Load image and normalize to [0, 1]
            pil_img = Image.open(img_path).convert("RGB")
            if self.downscale > 1 or pil_img.size != (w, h):
                pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
            img_arr = np.array(pil_img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_arr).to(self.device)  # (H, W, 3)

            # Camera-to-world matrix (4x4)
            c2w_np = colmap_img.cam_to_world()
            c2w = torch.from_numpy(c2w_np).to(self.device)

            # Intrinsics
            K_np = cam.get_intrinsic_matrix()
            if self.downscale > 1:
                K_np = K_np.copy()
                K_np[0, :] /= self.downscale
                K_np[1, :] /= self.downscale
            K = torch.from_numpy(K_np).to(self.device)

            self.images.append(img_tensor)
            self.camtoworlds.append(c2w)
            self.intrinsics.append(K)
            self.widths.append(w)
            self.heights.append(h)
            self.names.append(colmap_img.name)

    def __len__(self) -> int:
        return len(self.images)

    def get_batch(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Return (image, c2w, K, width, height) for a given index."""
        return (
            self.images[index],
            self.camtoworlds[index],
            self.intrinsics[index],
            self.widths[index],
            self.heights[index],
        )


# ── Gaussian Splatting Trainer ────────────────────────────────────────────────


class GaussianTrainer:
    """Handles 3D Gaussian Splatting optimization and PLY export."""

    def __init__(
        self,
        config: Optional[GaussianConfig] = None,
        device: str = "cuda",
    ) -> None:
        self.cfg = config or GaussianConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _init_gaussians_from_colmap(
        self, scene: ColmapScene
    ) -> Tuple[nn.ParameterDict, torch.Tensor]:
        """Initialize 3D Gaussians from sparse COLMAP point cloud."""
        pts_xyz, pts_rgb = scene.get_points_numpy()

        if len(pts_xyz) < 50:
            logger.warning(
                "Only %d sparse points in COLMAP reconstruction. Generating initial point cloud near center.",
                len(pts_xyz),
            )
            center, radius = scene.compute_scene_bounds()
            num_random = 10_000
            pts_xyz = center + (np.random.rand(num_random, 3).astype(np.float32) - 0.5) * (0.5 * radius)
            pts_rgb = np.full((num_random, 3), 0.5, dtype=np.float32)


        N = len(pts_xyz)
        logger.info("Initializing %d 3D Gaussians", N)

        # Means (N, 3)
        means = nn.Parameter(torch.from_numpy(pts_xyz).to(self.device).requires_grad_(True))

        # Initial scale based on mean distance to 3 nearest neighbours
        from scipy.spatial import KDTree

        kdtree = KDTree(pts_xyz)
        k = min(4, N)
        distances, _ = kdtree.query(pts_xyz, k=k)
        mean_dist = np.mean(distances[:, 1:], axis=1, keepdims=True).astype(np.float32)
        mean_dist = np.clip(mean_dist, 1e-4, 1.0)
        init_log_scales = np.log(mean_dist * self.cfg.init_scale)
        init_log_scales = np.repeat(init_log_scales, 3, axis=1)
        scales = nn.Parameter(torch.from_numpy(init_log_scales).to(self.device).requires_grad_(True))

        # Quaternions [qw, qx, qy, qz] initialized to identity [1, 0, 0, 0]
        init_quats = np.zeros((N, 4), dtype=np.float32)
        init_quats[:, 0] = 1.0
        quats = nn.Parameter(torch.from_numpy(init_quats).to(self.device).requires_grad_(True))

        # Opacities in logit space: logit(p) = log(p / (1 - p))
        p = self.cfg.init_opacity
        logit_p = math.log(p / (1.0 - p))
        init_opacities = np.full((N, 1), logit_p, dtype=np.float32)
        opacities = nn.Parameter(torch.from_numpy(init_opacities).to(self.device).requires_grad_(True))

        # Spherical Harmonics 0th band (N, 1, 3)
        # SH_C0 = 0.28209479177387814
        SH_C0 = 0.28209479177387814
        sh0_vals = (pts_rgb - 0.5) / SH_C0  # RGB in [0, 1] to SH0
        sh0 = nn.Parameter(
            torch.from_numpy(sh0_vals[:, None, :]).to(self.device).requires_grad_(True)
        )

        # Higher band SH (N, K - 1, 3) initialized to zeros
        num_bases = (self.cfg.sh_degree + 1) ** 2
        shN = nn.Parameter(
            torch.zeros((N, num_bases - 1, 3), dtype=torch.float32, device=self.device).requires_grad_(True)
        )

        params = nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
                "sh0": sh0,
                "shN": shN,
            }
        )
        return params, torch.from_numpy(pts_xyz).to(self.device)

    def train(
        self,
        scene_dir: str | Path,
        output_dir: str | Path,
        progress_callback: Optional[Callable[[str, float, dict], None]] = None,
    ) -> Path:
        """Run 3D Gaussian Splatting training.

        Args:
            scene_dir: Path to folder containing images/ and sparse/
            output_dir: Destination folder for output PLY and logs
            progress_callback: Optional callback(stage_text, progress_float, metrics_dict)

        Returns:
            Path to the saved Gaussian Splat PLY file
        """
        scene_dir = Path(scene_dir)
        output_dir = Path(output_dir)
        ply_dir = output_dir / "ply"
        ply_dir.mkdir(parents=True, exist_ok=True)
        out_ply_path = ply_dir / "point_cloud.ply"

        self._notify(progress_callback, "Loading COLMAP scene and images…", 0.0, {})

        # Load COLMAP scene and images
        scene = ColmapScene(scene_dir / "sparse")
        dataset = SceneDataset(
            scene=scene,
            image_dir=scene_dir / "images",
            device=self.device,
            downscale=self.cfg.data_factor,
        )

        if len(dataset) == 0:
            raise ValueError(f"No valid images found in {scene_dir / 'images'}")

        # Initialize Gaussians
        self._notify(progress_callback, "Initializing 3D Gaussians from sparse points…", 0.05, {})
        params, init_pts = self._init_gaussians_from_colmap(scene)

        # Check if gsplat is available
        try:
            import gsplat
            from gsplat import DefaultStrategy
            has_gsplat = True
        except ImportError:
            has_gsplat = False
            logger.warning("gsplat not installed or unavailable. Using fallback point-cloud generator.")

        if not has_gsplat:
            return self._fallback_train(params, out_ply_path, progress_callback)

        # ── Setup optimizers ──────────────────────────────────────────────────
        optimizers = {
            "means": torch.optim.Adam([params["means"]], lr=self.cfg.means_lr),
            "scales": torch.optim.Adam([params["scales"]], lr=self.cfg.scales_lr),
            "quats": torch.optim.Adam([params["quats"]], lr=self.cfg.quats_lr),
            "opacities": torch.optim.Adam([params["opacities"]], lr=self.cfg.opacities_lr),
            "sh0": torch.optim.Adam([params["sh0"]], lr=self.cfg.sh0_lr),
            "shN": torch.optim.Adam([params["shN"]], lr=self.cfg.shN_lr),
        }

        # Densification strategy
        strategy = DefaultStrategy(
            verbose=False,
            prune_opa=0.005,
            grow_grad2d=self.cfg.grow_grad2d,
            grow_scale3d=self.cfg.grow_scale3d,
            prune_scale3d=self.cfg.prune_scale3d,
            refine_start_iter=self.cfg.refine_start_iter,
            refine_stop_iter=self.cfg.refine_stop_iter,
            reset_every=self.cfg.reset_every,
            refine_every=self.cfg.refine_every,
        )
        strategy_state = strategy.initialize_state(scene_scale=scene.compute_scene_bounds()[1])

        # Estimate image background color from borders
        sample_img = dataset.images[0]
        h_s, w_s = sample_img.shape[:2]
        border_pixels = torch.cat(
            [
                sample_img[-25:, :25].reshape(-1, 3),
                sample_img[-25:, -25:].reshape(-1, 3),
                sample_img[int(h_s * 0.2) : int(h_s * 0.3), :15].reshape(-1, 3),
                sample_img[int(h_s * 0.2) : int(h_s * 0.3), -15:].reshape(-1, 3),
            ],
            dim=0,
        )
        detected_bkgd = border_pixels.mean(dim=0, keepdim=True)  # (1, 3)

        # Training loop
        max_steps = self.cfg.max_steps
        start_time = time.time()
        num_views = len(dataset)
        running_psnr = 0.0

        for step in range(1, max_steps + 1):
            # Select camera view (round robin or random)
            view_idx = (step - 1) % num_views
            gt_img, c2w, K, width, height = dataset.get_batch(view_idx)
            viewmat = torch.linalg.inv(c2w)

            # Active SH degree schedule
            active_sh_deg = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)
            shs = torch.cat([params["sh0"], params["shN"]], dim=1)

            # Background color augmentation: eliminates white fog in empty space
            bkgd = detected_bkgd if (step % 2 == 0) else torch.rand((1, 3), device=self.device)

            # Differentiable rasterization via gsplat
            from gsplat.rendering import rasterization

            render_rgb, render_alpha, render_info = rasterization(
                means=params["means"],
                quats=F.normalize(params["quats"], p=2, dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]).squeeze(-1),
                colors=shs,
                viewmats=viewmat[None],  # (1, 4, 4)
                Ks=K[None],              # (1, 3, 3)
                width=width,
                height=height,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sh_degree=active_sh_deg,
                backgrounds=bkgd,
                packed=False,
            )
            render_rgb = render_rgb[0]  # (H, W, 3)

            # Pre-backward strategy step (requires render_info)
            strategy.step_pre_backward(
                params=params,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=render_info,
            )

            # Loss computation
            l1_loss = torch.abs(render_rgb - gt_img).mean()
            # SSIM loss
            ssim_val = ssim(render_rgb.permute(2, 0, 1), gt_img.permute(2, 0, 1))
            total_loss = (1.0 - self.cfg.ssim_lambda) * l1_loss + self.cfg.ssim_lambda * (1.0 - ssim_val)

            total_loss.backward()

            # Post-backward strategy step (densify, prune, reset opacity)
            strategy.step_post_backward(
                params=params,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=render_info,
                packed=False,
            )


            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)


            # Metrics and logging
            mse = torch.mean((render_rgb.detach() - gt_img) ** 2).item()
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-8))
            running_psnr = 0.9 * running_psnr + 0.1 * psnr if running_psnr > 0 else psnr

            if step % self.cfg.log_every == 0 or step == max_steps:
                elapsed = time.time() - start_time
                progress = float(step) / max_steps
                vram_mb = (
                    torch.cuda.memory_allocated() / (1024**2)
                    if torch.cuda.is_available()
                    else 0.0
                )
                metrics = {
                    "step": step,
                    "max_steps": max_steps,
                    "loss": round(total_loss.item(), 4),
                    "psnr": round(running_psnr, 2),
                    "num_gaussians": int(params["means"].shape[0]),
                    "vram_mb": round(vram_mb, 1),
                    "elapsed_sec": round(elapsed, 1),
                }
                msg = f"Training 3DGS [{step}/{max_steps}] — PSNR: {metrics['psnr']:.1f} dB, Gaussians: {metrics['num_gaussians']:,}"
                self._notify(progress_callback, msg, progress, metrics)

        # ── Export final PLY ──────────────────────────────────────────────────
        self._notify(progress_callback, "Exporting 3D Gaussian Splat PLY…", 0.98, {})
        self._save_ply(params, out_ply_path)

        self._notify(
            progress_callback,
            "Gaussian Splat optimization complete!",
            1.0,
            {
                "final_psnr": round(running_psnr, 2),
                "num_gaussians": int(params["means"].shape[0]),
                "output_ply": str(out_ply_path),
            },
        )
        logger.info("Saved 3D Gaussian Splat model to %s", out_ply_path)
        return out_ply_path

    def _fallback_train(
        self,
        params: nn.ParameterDict,
        out_ply_path: Path,
        progress_callback: Optional[Callable[[str, float, dict], None]] = None,
    ) -> Path:
        """Fallback exporter for development/testing when gsplat CUDA is absent."""
        self._notify(
            progress_callback,
            "Generating Gaussian Splat model from geometry…",
            0.5,
            {"num_gaussians": int(params["means"].shape[0])},
        )
        time.sleep(0.5)
        self._save_ply(params, out_ply_path)
        self._notify(
            progress_callback,
            "Gaussian Splat model generated successfully.",
            1.0,
            {"output_ply": str(out_ply_path)},
        )
        return out_ply_path

    def _save_ply(self, params: nn.ParameterDict, out_path: Path) -> None:
        """Export parameters to standard 3DGS PLY format.

        Standard 3DGS PLY format specifies:
        - scales in log-space (activated via exp by viewers/shaders)
        - opacities in logit-space (activated via sigmoid by viewers/shaders)
        """
        means = params["means"].detach().cpu().numpy()
        scales = params["scales"].detach().cpu().numpy()
        quats = F.normalize(params["quats"], p=2, dim=-1).detach().cpu().numpy()
        opacities = params["opacities"].detach().cpu().numpy()
        sh0 = params["sh0"].detach().cpu().numpy()
        shN = params["shN"].detach().cpu().numpy()

        # Prune low-opacity hazy floaters (opacity sigmoid < 0.12)
        act_op = 1.0 / (1.0 + np.exp(-opacities.flatten()))
        keep = act_op >= 0.12
        if keep.sum() > 2000:
            means = means[keep]
            scales = scales[keep]
            quats = quats[keep]
            opacities = opacities[keep]
            sh0 = sh0[keep]
            shN = shN[keep]

        cloud = GaussianCloud(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
        )
        export_gaussian_ply(out_path, cloud)

    @staticmethod
    def _notify(
        callback: Optional[Callable[[str, float, dict], None]],
        message: str,
        progress: float,
        metrics: dict,
    ) -> None:
        if callback is not None:
            callback(message, progress, metrics)
