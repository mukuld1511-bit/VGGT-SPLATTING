"""
Unit tests for the VideoProcessor module.

Tests frame extraction, blur filtering, overlap detection, and scene directory preparation.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from config import VideoConfig
from pipeline.video_processor import FrameQuality, VideoProcessor, VideoReport


def _generate_test_video(path: Path, num_frames: int = 30, fps: int = 10, blur: bool = False) -> Path:
    """Generate a test MP4 video with known patterns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    for i in range(num_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw high-contrast shapes to avoid blur
        cv2.circle(img, (160 + int(20 * np.sin(i * 0.5)), 120), 40, (0, 255, 255), -1)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 0, 0), -1)
        cv2.putText(img, f"Frame {i}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if blur:
            img = cv2.GaussianBlur(img, (25, 25), 0)

        out.write(img)

    out.release()
    return path


def test_video_processor_extraction():
    """Verify frames are extracted at target FPS and written to scene_dir/images."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        video_path = tmp_path / "test.mp4"
        scene_dir = tmp_path / "scene"

        _generate_test_video(video_path, num_frames=30, fps=10)

        cfg = VideoConfig(target_fps=5.0, blur_threshold=10.0)
        proc = VideoProcessor(cfg)
        report = proc.process_video(video_path, scene_dir)

        assert isinstance(report, VideoReport)
        assert report.frames_accepted > 0
        assert (scene_dir / "images").exists()

        images = list((scene_dir / "images").glob("*.jpg"))
        assert len(images) == report.frames_accepted


def test_blur_filtering():
    """Verify that deliberately blurred video frames are detected and rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sharp_vid = tmp_path / "sharp.mp4"
        blurry_vid = tmp_path / "blurry.mp4"

        _generate_test_video(sharp_vid, num_frames=15, fps=5, blur=False)
        _generate_test_video(blurry_vid, num_frames=15, fps=5, blur=True)

        cfg = VideoConfig(target_fps=5.0, blur_threshold=50.0)
        proc = VideoProcessor(cfg)

        sharp_report = proc.process_video(sharp_vid, tmp_path / "scene_sharp")
        blurry_report = proc.process_video(blurry_vid, tmp_path / "scene_blurry")

        assert sharp_report.frames_rejected_blur < blurry_report.frames_rejected_blur


def test_invalid_video_path():
    """Ensure handling of nonexistent video files raises or handles cleanly."""
    cfg = VideoConfig()
    proc = VideoProcessor(cfg)
    with pytest.raises((FileNotFoundError, ValueError)):
        proc.process_video(Path("/tmp/nonexistent_video_path_xyz.mp4"), Path("/tmp/scene"))
