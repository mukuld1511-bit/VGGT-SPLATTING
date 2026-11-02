"""
Viser 3D Viewer — Interactive WebGL viewer for 3D Gaussian Splats.

Serves an interactive 3D viewer on a dedicated port (default 8080) that can be
embedded directly into the Gradio UI via iframe or viewed standalone in full screen.
Supports real-time splat loading, background adjustment, camera controls, and auto-rotation.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config import ViewerConfig
from utils.ply_utils import GaussianCloud, read_gaussian_ply

logger = logging.getLogger(__name__)


class SplatViewer:
    """Manages the Viser 3D WebGL viewer server."""

    def __init__(self, config: Optional[ViewerConfig] = None) -> None:
        self.cfg = config or ViewerConfig()
        self.server = None
        self._splat_handle = None
        self._pcd_handle = None
        self._current_ply: Optional[Path] = None
        self._auto_rotate: bool = False
        self._rotate_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the Viser server in the background if not already running."""
        if self.server is not None:
            return

        import viser

        logger.info("Starting Viser 3D viewer server on %s:%d…", self.cfg.host, self.cfg.port)
        self.server = viser.ViserServer(host=self.cfg.host, port=self.cfg.port)

        # Setup GUI controls
        self._setup_gui()
        logger.info("Viser viewer ready at http://localhost:%d", self.cfg.port)

    def stop(self) -> None:
        """Stop the Viser server and background threads."""
        self._auto_rotate = False
        if self.server is not None:
            self.server.stop()
            self.server = None
            logger.info("Viser viewer stopped.")

    def _setup_gui(self) -> None:
        """Configure interactive sidebar controls in Viser."""
        if self.server is None:
            return

        with self.server.gui.add_folder("Controls"):
            # Splat scale slider
            scale_slider = self.server.gui.add_slider(
                "Splat Scale", min=0.1, max=2.0, step=0.05, initial_value=1.0
            )

            @scale_slider.on_update
            def _(_):
                if self._splat_handle is not None:
                    # Update scale property on splat handle
                    try:
                        self._splat_handle.scale = scale_slider.value
                    except Exception:
                        pass

            # Reset camera button
            reset_btn = self.server.gui.add_button("Reset Camera")

            @reset_btn.on_click
            def _(event):
                client = event.client
                if client is not None:
                    d = getattr(self, "_cam_dist", 1.8)
                    client.camera.position = (0.0, -d, d * 0.4)
                    client.camera.look_at = (0.0, 0.0, 0.0)
                    client.camera.up_direction = (0.0, 0.0, 1.0)

            # Auto turntable toggle
            turntable_cb = self.server.gui.add_checkbox("Turntable Rotation", initial_value=False)

            @turntable_cb.on_update
            def _(_):
                self._auto_rotate = turntable_cb.value
                if self._auto_rotate and (self._rotate_thread is None or not self._rotate_thread.is_alive()):
                    self._rotate_thread = threading.Thread(target=self._run_turntable, daemon=True)
                    self._rotate_thread.start()

        with self.server.gui.add_folder("Scene Info"):
            self._info_markdown = self.server.gui.add_markdown(
                "**No model loaded yet.**\nUpload a video to reconstruct."
            )

    def _run_turntable(self) -> None:
        """Gradually orbit cameras around the object origin."""
        angle = 0.0
        dist = 2.0
        height = 0.6
        while self._auto_rotate and self.server is not None:
            angle += 0.02
            cam_x = dist * np.sin(angle)
            cam_y = -dist * np.cos(angle)
            cam_z = height

            for client in self.server.get_clients().values():
                client.camera.position = (cam_x, cam_y, cam_z)
                client.camera.look_at = (0.0, 0.0, 0.0)
                client.camera.up_direction = (0.0, 0.0, 1.0)
            time.sleep(0.03)

    def load_splat(self, ply_path: str | Path) -> None:
        """Load and display a 3D Gaussian Splat PLY file."""
        self.start()
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file does not exist: {ply_path}")

        logger.info("Loading Gaussian Splat from %s into Viser…", ply_path)
        cloud = read_gaussian_ply(ply_path)

        with self._lock:
            # Remove previous splat/point cloud
            if self._splat_handle is not None:
                self._splat_handle.remove()
                self._splat_handle = None
            if self._pcd_handle is not None:
                self._pcd_handle.remove()
                self._pcd_handle = None

            centers = cloud.positions.astype(np.float32)
            covariances = cloud.compute_covariances().astype(np.float32)
            rgbs = cloud.colors_rgb.astype(np.float32)
            # Viser add_gaussian_splats requires (N, 1) opacities
            opacities = cloud.opacities_activated.reshape(-1, 1).astype(np.float32)

            # Center the model around coordinate origin using robust median
            scene_center = np.median(centers, axis=0)

            # Filter out elongated needle artifacts (scale ratio > 25.0 or axis > 0.10m)
            scales_act = cloud.scales_activated
            max_s = scales_act.max(axis=-1)
            min_s = scales_act.min(axis=-1)
            scale_ratios = max_s / (min_s + 1e-7)
            non_spiky = (max_s <= 0.10) & (scale_ratios <= 25.0)

            # Filter out extreme distant floaters
            dists = np.linalg.norm(centers - scene_center, axis=-1)
            r_98 = float(np.percentile(dists, 98.0))
            max_r = max(r_98 * 1.5, 1.2)
            valid = (dists <= max_r) & non_spiky
            if np.sum(valid) > 100:
                centers = centers[valid]
                covariances = covariances[valid]
                rgbs = rgbs[valid]
                opacities = opacities[valid]
                scene_center = np.median(centers, axis=0)

            centers_centered = centers - scene_center

            # Add Gaussian Splats to scene
            try:
                self._splat_handle = self.server.scene.add_gaussian_splats(
                    name="/reconstruction/splats",
                    centers=centers_centered,
                    covariances=covariances,
                    rgbs=rgbs,
                    opacities=opacities,
                )
                logger.info(
                    "Added %d Gaussian splats to Viser scene (centered at origin)",
                    centers_centered.shape[0],
                )
            except Exception as exc:
                logger.warning(
                    "add_gaussian_splats failed (%s), falling back to point cloud", exc
                )
                self._pcd_handle = self.server.scene.add_point_cloud(
                    name="/reconstruction/points",
                    points=centers_centered,
                    colors=rgbs,
                    point_size=0.015,
                )

            self._current_ply = ply_path

            # Update stats panel
            if hasattr(self, "_info_markdown"):
                mb = ply_path.stat().st_size / (1024**2)
                self._info_markdown.content = (
                    f"### Reconstruction Loaded\n"
                    f"- **Model**: `{ply_path.name}`\n"
                    f"- **Gaussians**: {cloud.num_gaussians:,}\n"
                    f"- **File Size**: {mb:.2f} MB\n"
                    f"- **SH Degree**: {cloud.sh_degree}\n"
                )

            # Auto-frame camera based on object radius
            obj_radius = float(np.percentile(np.linalg.norm(centers_centered, axis=-1), 90.0))
            self._cam_dist = max(obj_radius * 2.2, 0.8)
            for client in self.server.get_clients().values():
                client.camera.position = (0.0, -self._cam_dist, self._cam_dist * 0.4)
                client.camera.look_at = (0.0, 0.0, 0.0)
                client.camera.up_direction = (0.0, 0.0, 1.0)


# Global singleton instance for easy web app embedding
_viewer_instance: Optional[SplatViewer] = None


def get_viewer(config: Optional[ViewerConfig] = None) -> SplatViewer:
    """Return the global viewer instance, initializing if necessary."""
    global _viewer_instance
    if _viewer_instance is None:
        _viewer_instance = SplatViewer(config)
    return _viewer_instance
