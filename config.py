"""
Configuration for the Video-to-3D Reconstruction Pipeline.
All tunable parameters are defined here as dataclasses.
Defaults are optimized for NVIDIA DGX Spark (GB10, 128GB unified memory).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
EXTERNAL_DIR = PROJECT_DIR / "external"
VGGT_DIR = EXTERNAL_DIR / "vggt"


@dataclass
class VideoConfig:
    """Frame extraction and quality filtering settings."""

    max_duration_seconds: int = 35  # allow small margin over 30s
    target_fps: float = 3.0  # frames per second to extract
    max_frames: int = 100
    min_frames: int = 8
    max_resolution: int = 1024  # longest side
    blur_threshold: float = 50.0  # Laplacian variance; below = blurry
    min_overlap_score: float = 0.15  # histogram correlation threshold
    supported_formats: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    jpeg_quality: int = 95


@dataclass
class VGGTConfig:
    """VGGT model and inference settings."""

    model_name: str = "facebook/VGGT-1B"
    model_url: str = (
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    )
    inference_resolution: int = 518  # VGGT's native resolution
    load_resolution: int = 1024  # resolution for COLMAP bridge
    use_bundle_adjustment: bool = False  # BA adds ~60s but improves quality
    confidence_threshold: float = 1.01  # VGGT-1B depth confidence is in [1.0, 3.5] (background = 1.0, object >= 1.01)
    max_colmap_points: int = 100_000

    # BA-specific parameters (only used when use_bundle_adjustment=True)
    max_query_pts: int = 4096
    query_frame_num: int = 8
    max_reproj_error: float = 8.0
    shared_camera: bool = False
    camera_type: str = "SIMPLE_PINHOLE"


@dataclass
class GaussianConfig:
    """Gaussian Splatting training hyperparameters."""

    max_steps: int = 7_000  # 7K gives ~85% of 30K quality in ~3 min
    eval_steps: List[int] = field(default_factory=lambda: [7_000])
    save_ply_steps: List[int] = field(default_factory=lambda: [7_000])
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # activate next SH band every N steps
    ssim_lambda: float = 0.2
    # Learning rates
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    # Initialization
    init_opacity: float = 0.1
    init_scale: float = 1.0
    # Rendering
    near_plane: float = 0.01
    far_plane: float = 1e10
    # Densification (DefaultStrategy)
    refine_start_iter: int = 500
    refine_stop_iter: int = 5_000
    refine_every: int = 100
    reset_every: int = 3_000
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    # Image downscale factor (1 = original resolution)
    data_factor: int = 1
    # Progress reporting
    log_every: int = 100


@dataclass
class ViewerConfig:
    """Viser 3D viewer settings."""

    port: int = 8080
    host: str = "0.0.0.0"


@dataclass
class AppConfig:
    """Gradio web application settings."""

    port: int = 7860
    host: str = "0.0.0.0"
    lab_name: str = "AI Research Lab"
    lab_tagline: str = "Video to 3D — Powered by VGGT + Gaussian Splatting"
    share: bool = False  # set True for ngrok tunnel


@dataclass
class PipelineConfig:
    """Top-level configuration bundling all sub-configs."""

    video: VideoConfig = field(default_factory=VideoConfig)
    vggt: VGGTConfig = field(default_factory=VGGTConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    app: AppConfig = field(default_factory=AppConfig)
    project_dir: Path = field(default_factory=lambda: PROJECT_DIR)
    PROJECT_DIR: Path = field(default_factory=lambda: PROJECT_DIR)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)

    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    device: str = "cuda"
    seed: int = 42


    def scene_dir(self, session_id: str) -> Path:
        """Return the working directory for a given reconstruction session."""
        return self.data_dir / session_id

    def result_dir(self, session_id: str) -> Path:
        """Return the output directory for a given reconstruction session."""
        return self.results_dir / session_id

    def ensure_dirs(self, session_id: str) -> None:
        """Create all required directories for a session."""
        scene = self.scene_dir(session_id)
        result = self.result_dir(session_id)
        (scene / "images").mkdir(parents=True, exist_ok=True)
        (scene / "sparse").mkdir(parents=True, exist_ok=True)
        (result / "ply").mkdir(parents=True, exist_ok=True)
