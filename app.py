"""
Video-to-3D Reconstruction System — Main Web Application.

A guest-facing interactive 3D reconstruction system running on NVIDIA DGX Spark
(Blackwell GB10, 128GB unified memory).

Workflow:
    Record/Upload Video -> Automatic Quality Check -> VGGT Geometry -> gsplat Optimization -> Interactive Viser 3D Viewer
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

# Patch Gradio's MatplotlibBackendMananger to prevent queue crashes
# caused by dual matplotlib versions (Ubuntu dist-packages vs user site-packages)
import gradio.route_utils
import gradio.utils


class SafeMatplotlibBackendManager:
    def __enter__(self):
        try:
            import matplotlib

            self._original_backend = matplotlib.get_backend()
            matplotlib.use("agg")
        except Exception:
            self._original_backend = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if getattr(self, "_original_backend", None) is not None:
                import matplotlib

                matplotlib.use(self._original_backend)
        except Exception:
            pass


gradio.utils.MatplotlibBackendMananger = SafeMatplotlibBackendManager
if hasattr(gradio.route_utils, "utils"):
    gradio.route_utils.utils.MatplotlibBackendMananger = SafeMatplotlibBackendManager


from config import (
    PROJECT_DIR,
    AppConfig,
    GaussianConfig,
    PipelineConfig,
    VGGTConfig,
    VideoConfig,
    ViewerConfig,
)
from pipeline import GaussianTrainer, VideoProcessor, get_viewer
from pipeline.vggt_reconstructor import VGGTReconstructor
from utils.benchmark import PipelineBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video3d_app")

# Initialize global configuration
pipeline_cfg = PipelineConfig()


# ── Sample Demo Video Generator (for quick 1-click test without upload) ────────


def create_demo_sample_video(output_path: Path) -> Path:
    """Generate a synthetic 360-degree rotating sample object video for demonstration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    duration_sec = 6
    total_frames = fps * duration_sec
    w, h = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Render a 3D-like shaded torus / polyhedron rotating on a platform
    for i in range(total_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Background gradient (dark laboratory slate)
        for y in range(h):
            shade = int(25 + 40 * (y / h))
            img[y, :] = [shade + 10, shade, shade - 5]

        # Draw rotating platform
        angle = (2 * np.pi * i) / total_frames
        cv2.ellipse(img, (w // 2, 380), (180, 50), 0, 0, 360, (50, 70, 90), -1)
        cv2.ellipse(img, (w // 2, 380), (180, 50), 0, 0, 360, (0, 242, 254), 2)

        # Draw central rotating object with multi-color shaded facets
        center_x = w // 2
        center_y = int(240 + 20 * np.sin(angle * 2))
        radius = 90

        # Draw multiple orbital nodes and core
        num_facets = 8
        for k in range(num_facets):
            facet_angle = angle + (2 * np.pi * k) / num_facets
            fx = int(center_x + radius * np.cos(facet_angle))
            fy = int(center_y + 0.45 * radius * np.sin(facet_angle))
            size = int(22 + 8 * np.sin(facet_angle))

            # Lighting shade
            intensity = int(140 + 115 * np.cos(facet_angle))
            color = (
                int(intensity * 0.9),
                int(intensity * 0.7),
                int(intensity * 0.4 + 100),
            )
            cv2.circle(img, (fx, fy), size, color, -1)
            cv2.circle(img, (fx, fy), size, (255, 255, 255), 1)

        # Core sphere
        cv2.circle(img, (center_x, center_y), 45, (0, 200, 255), -1)
        cv2.circle(img, (center_x - 12, center_y - 12), 14, (255, 255, 255), -1)

        out.write(img)

    out.release()
    return output_path


# ── Quality Analysis Helper ───────────────────────────────────────────────────


def analyze_video_quality(video_path: Optional[str]) -> Tuple[str, str]:
    """Run fast quality check on the uploaded video and return formatted assessment."""
    if not video_path or not Path(video_path).exists():
        return (
            "Please upload a video to inspect quality.",
            "<div class='lab-tip'>ℹ️ <b>Tip:</b> Record a 30s video circling smoothly around the object.</div>",
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "⚠️ Error reading video file.", ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0

    # Sample a few frames for blur score
    blur_scores = []
    step = max(1, frame_count // 10)
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(score)
    cap.release()

    mean_blur = float(np.mean(blur_scores)) if blur_scores else 0.0
    is_sharp = mean_blur > 50.0

    # Status rating
    if duration > 45:
        duration_badge = "⚠️ Video too long (>35s, will sample target FPS)"
    elif duration < 5:
        duration_badge = "⚠️ Video too short (<5s, may have insufficient coverage)"
    else:
        duration_badge = "✅ Ideal Duration"

    sharpness_badge = "✅ Sharp & Clear" if is_sharp else "⚠️ Potential Motion Blur"

    status_html = f"""
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 8px;'>
      <div class='stat-pill'>
        <span class='stat-label'>Duration</span>
        <span class='stat-value'>{duration:.1f} s</span>
        <span style='font-size: 0.75rem; color: #38bdf8;'>{duration_badge}</span>
      </div>
      <div class='stat-pill'>
        <span class='stat-label'>Resolution</span>
        <span class='stat-value'>{width} × {height}</span>
        <span style='font-size: 0.75rem; color: #4ade80;'>Original {fps:.0f} FPS</span>
      </div>
      <div class='stat-pill'>
        <span class='stat-label'>Sharpness</span>
        <span class='stat-value'>{mean_blur:.0f}</span>
        <span style='font-size: 0.75rem; color: {'#4ade80' if is_sharp else '#fbbf24'};'>{sharpness_badge}</span>
      </div>
    </div>
    """

    tip_html = """
    <div class='lab-tip'>
      <b>Reconstruction Guidance for Lab Guests:</b><br/>
      • Move steadily in a complete 360° circle around the object.<br/>
      • Maintain approximately 0.5m – 1.0m distance.<br/>
      • Ensure uniform lighting to minimize harsh dynamic shadows.
    </div>
    """
    return status_html, tip_html


# ── Full Pipeline Execution ───────────────────────────────────────────────────


def run_reconstruction(
    video_input: Optional[str],
    target_fps: float,
    max_steps: int,
    use_ba: bool,
    resolution: int,
    progress=gr.Progress(),
) -> Generator[Tuple[str, str, str, Optional[str]], None, None]:
    """Execute the complete video-to-3D pipeline with step-by-step progress updates."""
    if not video_input or not Path(video_input).exists():
        yield (
            "<div style='color: #ef4444;'>❌ Error: No video selected. Please upload or record a video.</div>",
            "",
            "",
            None,
        )
        return

    session_id = f"demo_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    scene_dir = pipeline_cfg.scene_dir(session_id)
    result_dir = pipeline_cfg.result_dir(session_id)
    pipeline_cfg.ensure_dirs(session_id)

    benchmark = PipelineBenchmark()
    viewer_port = pipeline_cfg.viewer.port
    viewer_url = f"http://localhost:{viewer_port}"

    # Helper for rendering tracker HTML
    def render_tracker(stage_idx: int, log_msg: str, stats_html: str = "") -> str:
        stages = [
            ("1", "Frame Extraction"),
            ("2", "VGGT Inference"),
            ("3", "3D Gaussian Splatting"),
            ("4", "Interactive 3D Viewer"),
        ]
        items = []
        for i, (num, name) in enumerate(stages, 1):
            if i < stage_idx:
                cls = "complete"
                icon = "✓"
            elif i == stage_idx:
                cls = "active"
                icon = num
            else:
                cls = ""
                icon = num
            t_cls = "active" if i == stage_idx else ""
            items.append(
                f"<div class='stage-step'>"
                f"<div class='stage-circle {cls}'>{icon}</div>"
                f"<span class='stage-title {t_cls}'>{name}</span>"
                f"</div>"
            )

        html = f"""
        <div class='glass-card' style='margin-bottom: 16px;'>
          <div class='stage-tracker'>{''.join(items)}</div>
          <div style='background: #0d1526; border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 8px; padding: 10px 14px; font-family: monospace; font-size: 0.85rem; color: #38bdf8;'>
            ⚡ <b>Current Status:</b> {log_msg}
          </div>
          {stats_html}
        </div>
        """
        return html

    try:
        # ── Stage 1: Frame Extraction & Quality Filter ────────────────────────
        progress(0.05, desc="Stage 1/4: Extracting and filtering video frames…")
        yield (
            render_tracker(1, "Extracting video frames and assessing blur/overlap…"),
            "",
            "",
            None,
        )

        with benchmark.stage("Frame Extraction"):
            vid_cfg = VideoConfig(
                target_fps=float(target_fps),
                max_resolution=int(resolution),
            )
            processor = VideoProcessor(vid_cfg)
            video_report = processor.process_video(video_input, scene_dir)

        if not video_report.is_usable:
            yield (
                render_tracker(1, f"❌ Video rejected: Only {video_report.frames_accepted} usable frames found."),
                f"<div style='color: #ef4444;'>Reconstruction stopped: Insufficient sharp frames ({video_report.frames_accepted}). Try a slower recording.</div>",
                "",
                None,
            )
            return

        # ── Stage 2: VGGT Pose & Depth Estimation ─────────────────────────────
        progress(0.25, desc="Stage 2/4: Running VGGT neural geometry estimation…")
        vggt_msg = f"Extracted {video_report.frames_accepted} frames. Running VGGT feed-forward camera estimation…"
        yield (
            render_tracker(2, vggt_msg),
            "",
            "",
            None,
        )

        with benchmark.stage("VGGT Reconstruction"):
            vggt_cfg = VGGTConfig(
                use_bundle_adjustment=use_ba,
                load_resolution=int(resolution),
            )
            reconstructor = VGGTReconstructor(vggt_cfg)

            def vggt_cb(msg: str, frac: float):
                progress(0.25 + 0.35 * frac, desc=f"Stage 2/4 (VGGT): {msg}")

            sparse_dir = reconstructor.reconstruct(scene_dir, progress_callback=vggt_cb)

        # ── Stage 3: Differentiable 3D Gaussian Splatting Optimization ─────────
        progress(0.60, desc="Stage 3/4: Optimizing 3D Gaussian Splats…")
        yield (
            render_tracker(3, "Starting 3D Gaussian Splatting optimization via gsplat…"),
            "",
            "",
            None,
        )

        with benchmark.stage("Gaussian Splatting"):
            gauss_cfg = GaussianConfig(
                max_steps=int(max_steps),
                data_factor=1,
            )
            trainer = GaussianTrainer(gauss_cfg)

            training_metrics = {}

            def gauss_cb(msg: str, frac: float, metrics: dict):
                training_metrics.update(metrics)
                progress(0.60 + 0.32 * frac, desc=f"Stage 3/4 (3DGS): {msg}")

            ply_path = trainer.train(
                scene_dir=scene_dir,
                output_dir=result_dir,
                progress_callback=gauss_cb,
            )

        # ── Stage 4: Launch Interactive 3D Viewer ─────────────────────────────
        progress(0.95, desc="Stage 4/4: Initializing Viser interactive 3D WebGL viewer…")
        yield (
            render_tracker(4, "Loading 3D Gaussian Splat into Viser WebGL server…"),
            "",
            "",
            str(ply_path),
        )

        with benchmark.stage("Viser 3D Viewer"):
            viewer = get_viewer(pipeline_cfg.viewer)
            viewer.load_splat(ply_path)

        # Total completion
        progress(1.0, desc="Reconstruction Complete!")

        # Performance summary table
        ply_mb = ply_path.stat().st_size / (1024**2)
        total_time = benchmark.total_seconds
        num_gaussians_val = training_metrics.get("num_gaussians", "N/A")
        num_gaussians_str = f"{num_gaussians_val:,}" if isinstance(num_gaussians_val, int) else str(num_gaussians_val)
        final_psnr = training_metrics.get("final_psnr", training_metrics.get("psnr", "N/A"))
        psnr_str = f"{final_psnr:.1f} dB" if isinstance(final_psnr, (int, float)) else f"{final_psnr} dB"

        perf_html = f"""
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 14px;'>
          <div class='stat-pill'>
            <span class='stat-label'>Total Time</span>
            <span class='stat-value' style='color: #00f2fe;'>{total_time:.1f} s</span>
            <span style='font-size: 0.75rem; color: #94a3b8;'>End-to-End</span>
          </div>
          <div class='stat-pill'>
            <span class='stat-label'>3D Gaussians</span>
            <span class='stat-value' style='color: #4ade80;'>{num_gaussians_str}</span>
            <span style='font-size: 0.75rem; color: #94a3b8;'>Optimized</span>
          </div>
          <div class='stat-pill'>
            <span class='stat-label'>Final PSNR</span>
            <span class='stat-value' style='color: #facc15;'>{psnr_str}</span>
            <span style='font-size: 0.75rem; color: #94a3b8;'>Fidelity</span>
          </div>
          <div class='stat-pill'>
            <span class='stat-label'>PLY Size</span>
            <span class='stat-value' style='color: #a78bfa;'>{ply_mb:.1f} MB</span>
            <span style='font-size: 0.75rem; color: #94a3b8;'>Downloadable</span>
          </div>
        </div>
        """


        # Viewer iframe HTML
        viewer_html = f"""
        <div class='viewer-container'>
          <iframe class='viewer-iframe' src='{viewer_url}' title='3D Gaussian Splat Viewer'></iframe>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 8px;'>
          <a href='{viewer_url}' target='_blank' style='color: #00f2fe; text-decoration: none; font-size: 0.9rem; font-weight: 600;'>
            ↗ Open 3D Viewer in Full Screen Window
          </a>
          <span style='font-size: 0.8rem; color: #94a3b8;'>Viser WebGL · Port {viewer_port}</span>
        </div>
        """

        stage_html = render_tracker(
            5,
            f"🎉 3D Reconstruction complete in {total_time:.1f}s! Interactive model loaded below.",
            perf_html,
        )

        yield (stage_html, viewer_html, f"```\n{benchmark.summary()}\n```", str(ply_path))

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        err_msg = f"❌ Pipeline encountered an error: {str(exc)}"
        yield (
            render_tracker(1, err_msg),
            f"<div style='color: #ef4444; padding: 12px; background: rgba(239,68,68,0.1); border-radius: 8px;'>{err_msg}</div>",
            "",
            None,
        )


# ── Preset Sample Videos ──────────────────────────────────────────────────────


def load_preset_video(preset_choice: str) -> str:
    """Provide a synthesized or cached sample video for instant 1-click demonstration."""
    sample_dir = pipeline_cfg.data_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "synthetic_orbit_demo.mp4"

    if not sample_path.exists():
        logger.info("Generating sample demonstration video…")
        create_demo_sample_video(sample_path)

    return str(sample_path)


# ── Gradio Web UI Layout ──────────────────────────────────────────────────────


def create_ui() -> gr.Blocks:
    """Build the modern guest-facing Gradio application."""
    css_file = PROJECT_DIR / "static" / "style.css"
    custom_css = css_file.read_text(encoding="utf-8") if css_file.exists() else ""


    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
    )
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        if torch.cuda.is_available()
        else 0.0
    )

    with gr.Blocks(title="AI Lab · Video-to-3D Reconstruction") as demo:

        # Header banner
        with gr.Row(elem_classes=["header-box"]):
            with gr.Column(scale=8):
                gr.HTML(
                    f"""
                    <div style='display: flex; align-items: center; gap: 14px; margin-bottom: 6px;'>
                      <span class='header-title'>Video-to-3D Reconstruction Studio</span>
                    </div>
                    <div style='color: var(--text-muted); font-size: 1.05rem;'>
                      Guest-Facing Interactive 3D Scanner · Powered by <b>Meta VGGT</b> & <b>gsplat</b>
                    </div>
                    """
                )
            with gr.Column(scale=4, elem_classes=["flex", "justify-end"]):
                gr.HTML(
                    f"""
                    <div style='text-align: right;'>
                      <div class='header-badge'>
                        <span class='pulse-dot'></span>
                        <span>{gpu_name} ({vram_gb} GB Unified)</span>
                      </div>
                      <div style='font-size: 0.78rem; color: #64748b; margin-top: 6px;'>
                        NVIDIA DGX Spark · sm_121 Blackwell
                      </div>
                    </div>
                    """
                )

        # Main 2-column workspace
        with gr.Row():
            # ── Left Column: Input & Settings ─────────────────────────────────
            with gr.Column(scale=5, elem_classes=["glass-card"]):
                gr.HTML("<h3><span class='step-badge'>1</span>Video Capture & Upload</h3>")

                with gr.Tabs():
                    with gr.TabItem("Upload Video File"):
                        video_file = gr.Video(
                            label="Guest Smartphone Video (MP4 / MOV / WebM)",
                            sources=["upload"],
                        )
                    with gr.TabItem("Record via Camera"):
                        video_cam = gr.Video(
                            label="Direct Camera Recording",
                            sources=["webcam"],
                        )
                    with gr.TabItem("Quick Demo Presets"):
                        preset_dropdown = gr.Dropdown(
                            choices=[
                                "Preset 1: Synthetic 360° Object (Instant Demo)",
                                "Preset 2: Laboratory Robotic Component",
                            ],
                            value="Preset 1: Synthetic 360° Object (Instant Demo)",
                            label="Select Preset Demonstration Video",
                        )
                        load_preset_btn = gr.Button("📂 Load Preset Video", variant="secondary")

                # Quality Inspection Box
                quality_box = gr.HTML(
                    "<div class='lab-tip'>Upload a 30s video circling smoothly around the object to inspect quality.</div>"
                )
                tip_box = gr.HTML("")

                # Advanced Settings
                with gr.Accordion("⚙️ Reconstruction Hyperparameters", open=False):
                    target_fps_slider = gr.Slider(
                        minimum=1.0,
                        maximum=6.0,
                        value=3.0,
                        step=0.5,
                        label="Extraction Rate (FPS) — ~90 frames for 30s",
                    )
                    max_steps_slider = gr.Slider(
                        minimum=1000,
                        maximum=10000,
                        value=7000,
                        step=1000,
                        label="Gaussian Splatting Optimization Iterations",
                        info="7,000 steps balances quality (~85% of full) and speed (~3 min).",
                    )
                    resolution_radio = gr.Radio(
                        choices=[518, 768, 1024],
                        value=1024,
                        label="COLMAP Resolution (pixels)",
                        info="VGGT predicts at 518px and rescales to original coordinates.",
                    )
                    use_ba_check = gr.Checkbox(
                        label="Enable Bundle Adjustment (PyCOLMAP)",
                        value=False,
                        info="Adds ~45-60s for feature tracking refinement.",
                    )

                # Primary Action Button
                reconstruct_btn = gr.Button(
                    "✨ Start 3D Reconstruction",
                    variant="primary",
                    elem_classes=["btn-primary"],
                )

            # ── Right Column: Live Viewer & Performance ───────────────────────
            with gr.Column(scale=7, elem_classes=["glass-card"]):
                gr.HTML("<h3><span class='step-badge'>2</span>Interactive 3D Experience</h3>")

                # Progress & Stage Tracker container
                stage_display = gr.HTML(
                    """
                    <div class='stage-tracker'>
                      <div class='stage-step'><div class='stage-circle'>1</div><span class='stage-title'>Extraction</span></div>
                      <div class='stage-step'><div class='stage-circle'>2</div><span class='stage-title'>VGGT Geometry</span></div>
                      <div class='stage-step'><div class='stage-circle'>3</div><span class='stage-title'>Gaussian Splatting</span></div>
                      <div class='stage-step'><div class='stage-circle'>4</div><span class='stage-title'>3D Viewer</span></div>
                    </div>
                    """
                )

                # 3D Viewer iframe
                viewer_display = gr.HTML(
                    f"""
                    <div class='viewer-container'>
                      <div style='height: 520px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #080c14; color: #64748b; gap: 12px;'>
                        <div style='font-size: 2.4rem;'>👁️</div>
                        <div style='font-size: 1.1rem; font-weight: 500;'>Interactive 3D Viewer Awaiting Reconstruction</div>
                        <div style='font-size: 0.85rem; color: #475569;'>Click 'Start 3D Reconstruction' on the left to begin.</div>
                      </div>
                    </div>
                    """
                )

                # Download File Button & Benchmark Details
                with gr.Row():
                    download_ply = gr.File(
                        label="💾 Download 3D Gaussian Splat (.PLY)",
                        interactive=False,
                    )

                with gr.Accordion("📊 System Benchmark Logs & Timing Breakdown", open=False):
                    benchmark_markdown = gr.Markdown("No benchmark runs recorded yet.")

        # ── Event Wiring ──────────────────────────────────────────────────────

        # Sync upload and camera inputs
        video_cam.change(
            fn=lambda v: (v, *analyze_video_quality(v)),
            inputs=[video_cam],
            outputs=[video_file, quality_box, tip_box],
        )
        video_file.change(
            fn=analyze_video_quality,
            inputs=[video_file],
            outputs=[quality_box, tip_box],
        )

        load_preset_btn.click(
            fn=lambda p: (load_preset_video(p), *analyze_video_quality(load_preset_video(p))),
            inputs=[preset_dropdown],
            outputs=[video_file, quality_box, tip_box],
        )

        # Trigger reconstruction
        reconstruct_btn.click(
            fn=run_reconstruction,
            inputs=[
                video_file,
                target_fps_slider,
                max_steps_slider,
                use_ba_check,
                resolution_radio,
            ],
            outputs=[
                stage_display,
                viewer_display,
                benchmark_markdown,
                download_ply,
            ],
        )

    return demo


# ── Application Entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    app_cfg = pipeline_cfg.app
    viewer_cfg = pipeline_cfg.viewer

    # Ensure background Viser server is started
    viewer = get_viewer(viewer_cfg)
    viewer.start()

    # Launch Gradio interface
    demo_app = create_ui()
    css_file = PROJECT_DIR / "static" / "style.css"
    custom_css = css_file.read_text(encoding="utf-8") if css_file.exists() else ""

    logger.info("Launching Gradio UI at http://%s:%d…", app_cfg.host, app_cfg.port)
    demo_app.launch(
        server_name=app_cfg.host,
        server_port=app_cfg.port,
        share=app_cfg.share,
        css=custom_css,
    )

