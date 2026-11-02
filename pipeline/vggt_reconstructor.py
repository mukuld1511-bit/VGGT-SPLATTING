"""
VGGT Reconstructor — camera-pose and depth estimation → COLMAP export.

Wraps Meta's VGGT (Visual Geometry Grounded Transformer) to predict cameras and
depth maps from multi-view frames, then converts the output to a standard COLMAP
sparse reconstruction that gsplat can consume directly.

The heavy-lifting code mirrors ``demo_colmap.py`` from the VGGT repo but is
refactored into a reusable class with progress-callback support.
"""

from __future__ import annotations

import copy
import glob
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import VGGTConfig

logger = logging.getLogger(__name__)


class VGGTReconstructor:
    """Run VGGT feed-forward inference and export COLMAP sparse reconstruction.

    The model is loaded once and kept in GPU memory for repeated use across
    sessions.  Call :meth:`reconstruct` for each scene.
    """

    def __init__(
        self,
        config: Optional[VGGTConfig] = None,
        device: str = "cuda",
    ) -> None:
        self.cfg = config or VGGTConfig()
        self.device = device
        self.model = None
        # bfloat16 on Ampere+ (Compute Capability ≥ 8); float16 otherwise
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

    # ── Model management ──────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load VGGT-1B weights (downloads on first call)."""
        if self.model is not None:
            return
        logger.info("Loading VGGT model (%s)…", self.cfg.model_name)
        from vggt.models.vggt import VGGT

        model = VGGT()
        state = torch.hub.load_state_dict_from_url(
            self.cfg.model_url, map_location="cpu"
        )
        model.load_state_dict(state)
        model.eval()
        model = model.to(self.device)
        self.model = model
        logger.info("VGGT model loaded on %s (dtype=%s)", self.device, self.dtype)

    def unload_model(self) -> None:
        """Free GPU memory occupied by the model."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    # ── Reconstruction entry point ────────────────────────────────────────────

    def reconstruct(
        self,
        scene_dir: str | Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """Run VGGT on ``scene_dir/images/`` and write COLMAP to ``scene_dir/sparse/``.

        Returns the path to the sparse directory.
        """
        scene_dir = Path(scene_dir)
        image_dir = scene_dir / "images"
        sparse_dir = scene_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        self._notify(progress_callback, "Loading VGGT model…", 0.0)
        self.load_model()

        # ── Discover and load images ──────────────────────────────────────
        self._notify(progress_callback, "Loading images…", 0.05)
        image_paths = sorted(glob.glob(str(image_dir / "*")))
        image_paths = [
            p for p in image_paths
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        base_names = [os.path.basename(p) for p in image_paths]
        logger.info("Found %d images in %s", len(image_paths), image_dir)

        from vggt.utils.load_fn import load_and_preprocess_images_square

        images, original_coords = load_and_preprocess_images_square(
            image_paths, self.cfg.load_resolution
        )
        images = images.to(self.device)
        original_coords = original_coords.to(self.device)

        # ── VGGT inference ────────────────────────────────────────────────
        self._notify(progress_callback, "Running VGGT inference…", 0.15)
        extrinsic, intrinsic, depth_map, depth_conf = self._run_vggt(images)

        self._notify(progress_callback, "Unprojecting depth to 3D…", 0.55)
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        points_3d = unproject_depth_map_to_point_map(
            depth_map, extrinsic, intrinsic
        )

        # ── Convert to COLMAP format ──────────────────────────────────────
        self._notify(progress_callback, "Building COLMAP reconstruction…", 0.65)

        if self.cfg.use_bundle_adjustment:
            reconstruction, recon_resolution = self._build_colmap_ba(
                images, extrinsic, intrinsic, depth_conf, points_3d
            )
        else:
            reconstruction, recon_resolution = self._build_colmap_ff(
                images, extrinsic, intrinsic, depth_conf, points_3d
            )

        # ── Rename images & rescale cameras to original resolution ────────
        self._notify(progress_callback, "Finalising camera parameters…", 0.80)
        reconstruction = self._rescale_reconstruction(
            reconstruction,
            base_names,
            original_coords.cpu().numpy(),
            recon_resolution,
        )

        # ── Write to disk ─────────────────────────────────────────────────
        self._notify(progress_callback, "Writing COLMAP files…", 0.90)
        if hasattr(reconstruction, "write"):
            reconstruction.write(str(sparse_dir))
        elif hasattr(reconstruction, "save"):
            reconstruction.save(sparse_dir)

        # Also write a quick PLY for visualisation
        try:
            import trimesh

            pts = np.array([p.xyz for p in reconstruction.points3D.values()], dtype=np.float32)
            cols = []
            for p in reconstruction.points3D.values():
                if hasattr(p, "color"):
                    cols.append(p.color)
                elif hasattr(p, "rgb"):
                    cols.append(p.rgb)
                else:
                    cols.append([128, 128, 128])
            cols = np.array(cols, dtype=np.uint8) if cols else np.zeros((0, 3), dtype=np.uint8)

            if len(pts) > 0:
                trimesh.PointCloud(pts, colors=cols).export(
                    str(sparse_dir / "points.ply")
                )
        except Exception as exc:
            logger.warning("Could not export points.ply: %s", exc)

        self._notify(progress_callback, "VGGT reconstruction complete.", 1.0)
        logger.info("Sparse reconstruction saved to %s", sparse_dir)
        return sparse_dir


    # ── VGGT inference ────────────────────────────────────────────────────────

    def _run_vggt(self, images: torch.Tensor):
        """Run the VGGT forward pass at 518×518 and return numpy arrays."""
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        res = self.cfg.inference_resolution
        imgs = F.interpolate(
            images, size=(res, res), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                imgs_batch = imgs[None]  # (1, N, 3, H, W)
                agg, ps_idx = self.model.aggregator(imgs_batch)

            pose_enc = self.model.camera_head(agg)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, imgs_batch.shape[-2:]
            )
            depth_map, depth_conf = self.model.depth_head(agg, imgs_batch, ps_idx)

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        return extrinsic, intrinsic, depth_map, depth_conf

    # ── Feed-forward COLMAP conversion (no BA) ────────────────────────────────

    def _build_colmap_ff(self, images, extrinsic, intrinsic, depth_conf, points_3d):
        """Build COLMAP reconstruction using feed-forward predictions only."""
        from vggt.utils.helper import (
            create_pixel_coordinate_grid,
            randomly_limit_trues,
        )

        res = self.cfg.inference_resolution

        image_size = np.array([res, res])
        num_frames, H, W, _ = points_3d.shape

        # Get RGB values for the 3D points
        points_rgb = F.interpolate(
            images, size=(res, res), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)  # (N, H, W, 3)

        # Pixel-coordinate grid with frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, H, W)

        # Filter by depth confidence
        # Background pixels on white/solid backdrops have conf == 1.000 and depth > 10m.
        # Foreground object pixels have conf >= 1.01 (up to 3.5) with compact physical depth.
        conf_threshold = max(self.cfg.confidence_threshold, 1.005)
        conf_mask = depth_conf >= conf_threshold
        if conf_mask.sum() < 2000:
            logger.info(
                "Adaptive confidence: %d points with conf >= %.3f; falling back to top 40%% confidence.",
                int(conf_mask.sum()),
                conf_threshold,
            )
            conf_mask = depth_conf >= np.percentile(depth_conf, 60.0)

        # Depth outlier filter: remove background points with extreme depth
        depth_map = np.linalg.norm(points_3d, axis=-1)
        valid_depths = depth_map[conf_mask]
        if len(valid_depths) > 0:
            depth_cap = float(np.percentile(valid_depths, 95.0)) * 1.5
            conf_mask = conf_mask & (depth_map <= depth_cap)

        conf_mask = randomly_limit_trues(conf_mask, self.cfg.max_colmap_points)

        p3d = points_3d[conf_mask]
        pxyf = points_xyf[conf_mask]
        prgb = points_rgb[conf_mask]

        logger.info("Feed-forward COLMAP: %d points selected from VGGT geometry", len(p3d))


        try:
            from vggt.dependency.np_to_pycolmap import (
                batch_np_matrix_to_pycolmap_wo_track,
            )

            reconstruction = batch_np_matrix_to_pycolmap_wo_track(
                p3d,
                pxyf,
                prgb,
                extrinsic,
                intrinsic,
                image_size,
                shared_camera=False,
                camera_type="PINHOLE",
            )
        except (ImportError, ModuleNotFoundError):
            logger.info("Using pure-Python colmap_utils builder (pycolmap not present)")
            from utils.colmap_utils import build_colmap_scene_from_arrays

            reconstruction = build_colmap_scene_from_arrays(
                points3d=p3d,
                points_xyf=pxyf,
                points_rgb=prgb,
                extrinsics=extrinsic,
                intrinsics=intrinsic,
                image_size=image_size,
            )

        return reconstruction, res


    # ── Bundle-adjusted COLMAP conversion ─────────────────────────────────────

    def _build_colmap_ba(self, images, extrinsic, intrinsic, depth_conf, points_3d):
        """Build COLMAP reconstruction with Bundle Adjustment via pycolmap."""
        import pycolmap
        from vggt.dependency.track_predict import predict_tracks
        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap

        load_res = self.cfg.load_resolution
        infer_res = self.cfg.inference_resolution
        image_size = np.array(images.shape[-2:])
        scale = load_res / infer_res

        with torch.cuda.amp.autocast(dtype=self.dtype):
            pred_tracks, pred_vis, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=self.cfg.max_query_pts,
                query_frame_num=self.cfg.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=True,
            )
            torch.cuda.empty_cache()

        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis > 0.2

        reconstruction, _ = batch_np_matrix_to_pycolmap(
            points_3d, extrinsic, intrinsic,
            pred_tracks, image_size,
            masks=track_mask,
            max_reproj_error=self.cfg.max_reproj_error,
            shared_camera=self.cfg.shared_camera,
            camera_type=self.cfg.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise RuntimeError("Bundle adjustment failed to produce a reconstruction.")

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        return reconstruction, load_res

    # ── Rescale cameras to original image resolution ──────────────────────────

    @staticmethod
    def _rescale_reconstruction(reconstruction, image_names, original_coords, img_size):
        """Rename COLMAP images and rescale camera intrinsics to original resolution."""
        rescale = True
        for pyimgid in reconstruction.images:
            pyimage = reconstruction.images[pyimgid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            idx = pyimgid - 1
            if idx < len(image_names):
                pyimage.name = image_names[idx]

            if rescale and idx < len(original_coords):
                pred_params = copy.deepcopy(pycamera.params)
                real_size = original_coords[idx, -2:]
                ratio = max(real_size) / img_size
                pred_params = pred_params * ratio
                real_pp = real_size / 2
                pred_params[-2:] = real_pp
                pycamera.params = pred_params
                pycamera.width = int(real_size[0])
                pycamera.height = int(real_size[1])

            # Shift 2D point observations to original resolution if present
            if hasattr(pyimage, "points2D") and pyimage.points2D and idx < len(original_coords):
                top_left = original_coords[idx, :2]
                ratio_img = max(original_coords[idx, -2:]) / img_size
                for pt2d in pyimage.points2D:
                    pt2d.xy = (pt2d.xy - top_left) * ratio_img

        return reconstruction



    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _notify(cb, msg, frac):
        if cb is not None:
            cb(msg, frac)
