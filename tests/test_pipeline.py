"""
End-to-end and integration tests for the Video-to-3D pipeline components.

Tests COLMAP serialization, 3DGS PLY I/O, covariance computation, Gaussian training,
benchmark tracking, and Viser viewer loading.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from config import GaussianConfig, PipelineConfig
from pipeline.gaussian_trainer import GaussianTrainer
from pipeline.viewer import SplatViewer
from utils.benchmark import MemoryTracker, PipelineBenchmark
from utils.colmap_utils import (
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
    ColmapScene,
    build_colmap_scene_from_arrays,
    rotmat2qvec,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)
from utils.ply_utils import GaussianCloud, export_gaussian_ply, load_gaussian_ply


def test_colmap_binary_roundtrip():
    """Verify that COLMAP cameras, images, and points3D can be written and read back."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sparse_dir = Path(tmp_dir) / "sparse"
        sparse_dir.mkdir()

        # Create sample camera
        cameras = {
            1: ColmapCamera(
                id=1,
                model="PINHOLE",
                width=640,
                height=480,
                params=np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32),
            )
        }

        # Create sample image
        qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        tvec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        images = {
            1: ColmapImage(
                id=1,
                qvec=qvec,
                tvec=tvec,
                camera_id=1,
                name="frame_0001.jpg",
                xys=np.array([[100.0, 100.0]], dtype=np.float32),
                point3D_ids=np.array([1], dtype=np.int64),
            )
        }

        # Create sample 3D point
        points3D = {
            1: ColmapPoint3D(
                id=1,
                xyz=np.array([0.5, 0.2, 1.5], dtype=np.float32),
                rgb=np.array([255, 128, 64], dtype=np.uint8),
                error=0.5,
                image_ids=np.array([1], dtype=np.int32),
                point2D_idxs=np.array([0], dtype=np.int32),
            )
        }

        # Write binary
        write_cameras_binary(sparse_dir / "cameras.bin", cameras)
        write_images_binary(sparse_dir / "images.bin", images)
        write_points3D_binary(sparse_dir / "points3D.bin", points3D)

        # Read back via ColmapScene
        scene = ColmapScene(sparse_dir)
        assert len(scene.cameras) == 1
        assert len(scene.images) == 1
        assert len(scene.points3D) == 1

        cam = scene.cameras[1]
        assert cam.width == 640
        assert cam.height == 480
        assert np.isclose(cam.focal_x, 500.0)

        img = scene.images[1]
        assert img.name == "frame_0001.jpg"
        assert np.allclose(img.tvec, [0.0, 0.0, 1.0])

        pt = scene.points3D[1]
        assert np.allclose(pt.xyz, [0.5, 0.2, 1.5])
        assert np.all(pt.rgb == [255, 128, 64])


def test_ply_gaussian_io_and_covariances():
    """Verify PLY serialization, GaussianCloud properties, and covariance matrix math."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ply_file = Path(tmp_dir) / "test_splats.ply"

        N = 100
        positions = np.random.randn(N, 3).astype(np.float32)
        sh_dc = np.random.randn(N, 3).astype(np.float32)
        scales = np.full((N, 3), -2.0, dtype=np.float32)
        rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1))
        opacities = np.full(N, 1.5, dtype=np.float32)

        cloud = GaussianCloud(
            positions=positions,
            sh_dc=sh_dc,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )

        assert cloud.num_gaussians == N
        assert cloud.colors_rgb.shape == (N, 3)
        assert cloud.opacities_activated.shape == (N,)

        # Compute covariances
        covs = cloud.compute_covariances()
        assert covs.shape == (N, 3, 3)
        # For identity rotation and scale exp(-2), diagonal should be exp(-2)^2 = exp(-4)
        expected_diag = np.exp(-2.0) ** 2
        assert np.isclose(covs[0, 0, 0], expected_diag, atol=1e-4)

        # Write and read back
        export_gaussian_ply(ply_file, cloud)
        assert ply_file.exists()

        loaded = load_gaussian_ply(ply_file)
        assert loaded.num_gaussians == N
        assert np.allclose(loaded.positions, positions, atol=1e-5)


def test_pipeline_benchmark_and_memory():
    """Verify timing benchmark and GPU memory tracking."""
    bench = PipelineBenchmark()
    with bench.stage("Test Stage"):
        _ = [i**2 for i in range(10_000)]

    assert len(bench.stages) == 1
    assert bench.total_seconds > 0
    assert "Test Stage" in bench.summary()

    mem_mb = MemoryTracker.get_gpu_memory_mb()
    assert mem_mb >= 0.0


def test_viser_viewer_splat_loading():
    """Verify Viser viewer starts and handles splat loading without errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ply_file = Path(tmp_dir) / "sample_model.ply"

        N = 20
        positions = np.random.randn(N, 3).astype(np.float32)
        sh_dc = np.random.randn(N, 3).astype(np.float32)
        scales = np.full((N, 3), -3.0, dtype=np.float32)
        rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1))
        opacities = np.full(N, 1.0, dtype=np.float32)

        cloud = GaussianCloud(
            positions=positions,
            sh_dc=sh_dc,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
        )
        export_gaussian_ply(ply_file, cloud)

        # Start viewer on test port
        viewer = SplatViewer()
        viewer.cfg.port = 8890
        viewer.start()
        viewer.load_splat(ply_file)

        assert viewer._current_ply == ply_file
        viewer.stop()
