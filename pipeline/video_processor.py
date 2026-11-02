"""
Video Processor — frame extraction, quality filtering, and scene preparation.

Accepts MP4/MOV videos, extracts frames at a target FPS, resizes to a maximum
resolution, detects motion blur and insufficient overlap, and writes the
surviving frames to a scene directory for downstream reconstruction.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from config import VideoConfig

logger = logging.getLogger(__name__)


@dataclass
class FrameQuality:
    """Quality assessment for a single extracted frame."""

    index: int
    path: Path
    blur_score: float
    is_blurry: bool
    overlap_score: float  # -1 means no previous frame to compare
    is_low_overlap: bool


@dataclass
class VideoReport:
    """Summary of video processing results."""

    video_path: str
    duration_seconds: float
    original_fps: float
    original_resolution: Tuple[int, int]
    total_frames_extracted: int
    frames_rejected_blur: int
    frames_rejected_overlap: int
    frames_accepted: int
    accepted_paths: List[str]
    warnings: List[str]

    @property
    def is_usable(self) -> bool:
        return self.frames_accepted >= 8

    @property
    def quality_rating(self) -> str:
        if self.frames_accepted >= 60:
            return "excellent"
        if self.frames_accepted >= 30:
            return "good"
        if self.frames_accepted >= 15:
            return "fair"
        return "poor"


class VideoProcessor:
    """Extract, filter, and prepare video frames for 3D reconstruction."""

    def __init__(self, config: Optional[VideoConfig] = None) -> None:
        self.cfg = config or VideoConfig()

    def process(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> VideoReport:
        """Full video -> frames pipeline."""
        return self._process_impl(video_path, output_dir, progress_callback)

    # Alias for flexible naming
    process_video = process

    def _process_impl(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> VideoReport:
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # ── Validate video ────────────────────────────────────────────────
        warnings: List[str] = []
        self._notify(progress_callback, "Opening video…", 0.0)


        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        if duration > self.cfg.max_duration_seconds:
            warnings.append(
                f"Video is {duration:.1f}s (max {self.cfg.max_duration_seconds}s). "
                "Only the first portion will be used."
            )
        if duration < 2.0:
            warnings.append("Video is very short — reconstruction quality may be poor.")

        logger.info(
            "Video: %dx%d @ %.1f FPS, %.1fs (%d frames)",
            width, height, fps, duration, total_frames,
        )

        # ── Extract frames at target FPS ──────────────────────────────────
        self._notify(progress_callback, "Extracting frames…", 0.05)
        frame_interval = max(1, int(round(fps / self.cfg.target_fps)))
        raw_frames: List[Tuple[int, np.ndarray]] = []

        frame_idx = 0
        usable_duration = min(duration, self.cfg.max_duration_seconds)
        max_raw = int(usable_duration * fps)

        while cap.isOpened() and frame_idx < max_raw:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0 and len(raw_frames) < self.cfg.max_frames:
                # Resize if needed
                frame = self._resize(frame)
                raw_frames.append((frame_idx, frame))
            frame_idx += 1

            if len(raw_frames) % 10 == 0:
                frac = 0.05 + 0.35 * (frame_idx / max(max_raw, 1))
                self._notify(
                    progress_callback,
                    f"Extracting frames… ({len(raw_frames)} so far)",
                    frac,
                )

        cap.release()

        if len(raw_frames) < self.cfg.min_frames:
            warnings.append(
                f"Only {len(raw_frames)} frames extracted (minimum {self.cfg.min_frames}). "
                "Try a longer or higher-FPS video."
            )

        # ── Shot Boundary / Scene Cut Detection ───────────────────────────
        # Edited promotional videos / YouTube shorts often contain hard cuts between shots.
        # Structure-from-Motion requires a continuous camera sequence.
        if len(raw_frames) > 1:
            shots: List[List[Tuple[int, np.ndarray]]] = []
            cur_shot = [raw_frames[0]]
            for i in range(1, len(raw_frames)):
                diff = float(
                    np.mean(
                        np.abs(
                            raw_frames[i][1].astype(float)
                            - raw_frames[i - 1][1].astype(float)
                        )
                    )
                )
                if diff > 35.0:
                    shots.append(cur_shot)
                    cur_shot = [raw_frames[i]]
                else:
                    cur_shot.append(raw_frames[i])
            if cur_shot:
                shots.append(cur_shot)

            if len(shots) > 1:
                longest = max(shots, key=len)
                if len(longest) >= self.cfg.min_frames:
                    warnings.append(
                        f"Detected {len(shots)} scene cuts in video. Isolated primary continuous sequence of {len(longest)} frames."
                    )
                    logger.info(
                        "Shot detection: isolated primary continuous shot (%d frames) out of %d shots",
                        len(longest),
                        len(shots),
                    )
                    raw_frames = longest

        # ── Quality assessment ────────────────────────────────────────────
        self._notify(progress_callback, "Analysing frame quality…", 0.45)
        assessments = self._assess_quality(raw_frames)

        accepted: List[FrameQuality] = []
        n_blur = 0
        n_overlap = 0

        for qa in assessments:
            if qa.is_blurry:
                n_blur += 1
            elif qa.is_low_overlap:
                n_overlap += 1
            else:
                accepted.append(qa)

        if n_blur > len(assessments) * 0.5:
            warnings.append(
                f"{n_blur}/{len(assessments)} frames detected as blurry — "
                "try recording with steadier motion or better lighting."
            )

        # ── Write accepted frames ─────────────────────────────────────────
        self._notify(progress_callback, "Saving accepted frames…", 0.75)
        accepted_paths: List[str] = []

        for i, qa in enumerate(accepted):
            fname = f"frame_{i:04d}.jpg"
            dst = images_dir / fname
            _, frame = raw_frames[qa.index]
            cv2.imwrite(
                str(dst),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality],
            )
            accepted_paths.append(str(dst))

        self._notify(progress_callback, "Frame extraction complete.", 1.0)

        report = VideoReport(
            video_path=str(video_path),
            duration_seconds=round(duration, 2),
            original_fps=round(fps, 2),
            original_resolution=(width, height),
            total_frames_extracted=len(raw_frames),
            frames_rejected_blur=n_blur,
            frames_rejected_overlap=n_overlap,
            frames_accepted=len(accepted),
            accepted_paths=accepted_paths,
            warnings=warnings,
        )
        logger.info(
            "Extraction done: %d/%d frames accepted (blur=%d, overlap=%d)",
            report.frames_accepted,
            report.total_frames_extracted,
            n_blur,
            n_overlap,
        )
        return report

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame so longest side ≤ max_resolution."""
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= self.cfg.max_resolution:
            return frame
        scale = self.cfg.max_resolution / longest
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _blur_score(self, frame: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _overlap_score(
        self, frame_a: np.ndarray, frame_b: np.ndarray
    ) -> float:
        """Histogram correlation ∈ [-1, 1]; higher = more similar."""
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        ha = cv2.calcHist([gray_a], [0], None, [64], [0, 256])
        hb = cv2.calcHist([gray_b], [0], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))

    def _assess_quality(
        self, frames: List[Tuple[int, np.ndarray]]
    ) -> List[FrameQuality]:
        """Assess blur and overlap for every extracted frame."""
        # Adaptive blur threshold: studio videos with plain/white backdrops naturally have lower
        # full-frame Laplacian variance. Scale dynamically to 50% of median sharpness (min 12.0).
        blur_scores = [self._blur_score(f) for _, f in frames]
        med_blur = float(np.median(blur_scores)) if blur_scores else 50.0
        active_blur_threshold = min(self.cfg.blur_threshold, max(12.0, med_blur * 0.5))

        results: List[FrameQuality] = []
        prev_frame: Optional[np.ndarray] = None

        for list_idx, (raw_idx, frame) in enumerate(frames):
            blur = self._blur_score(frame)
            is_blurry = blur < active_blur_threshold

            if prev_frame is not None:
                overlap = self._overlap_score(prev_frame, frame)
                is_low_overlap = overlap < self.cfg.min_overlap_score
            else:
                overlap = -1.0
                is_low_overlap = False

            results.append(
                FrameQuality(
                    index=list_idx,
                    path=Path(""),  # filled later when written
                    blur_score=round(blur, 2),
                    is_blurry=is_blurry,
                    overlap_score=round(overlap, 4),
                    is_low_overlap=is_low_overlap,
                )
            )
            if not is_blurry:
                prev_frame = frame

        return results

    @staticmethod
    def _notify(
        cb: Optional[Callable[[str, float], None]],
        message: str,
        fraction: float,
    ) -> None:
        if cb is not None:
            cb(message, fraction)
