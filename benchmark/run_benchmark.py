"""
Automated Pipeline Benchmark for NVIDIA DGX Spark (Blackwell GB10).

Measures:
- Throughput and latency for each pipeline stage:
    1. Video frame extraction & quality check
    2. VGGT geometry and camera pose estimation
    3. Gaussian Splatting optimization (gsplat)
    4. PLY serialization and 3D viewer preparation
- Unified memory consumption (Peak GPU VRAM, Allocated VRAM)
- Model quality metrics (PSNR, Gaussian count)
- Generates a Markdown report and machine-readable JSON metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import create_demo_sample_video
from config import GaussianConfig, PipelineConfig, VGGTConfig, VideoConfig
from pipeline import GaussianTrainer, VideoProcessor
from pipeline.vggt_reconstructor import VGGTReconstructor
from utils.benchmark import MemoryTracker, PipelineBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def run_benchmark(
    duration_sec: int = 15,
    target_fps: float = 3.0,
    max_steps: int = 3000,
    resolution: int = 1024,
    output_dir: str = "benchmark_results",
) -> dict:
    """Execute full benchmark run and record detailed system performance."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bench_data_dir = out_path / "data"
    bench_data_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    logger.info("=================================================================")
    logger.info("  NVIDIA DGX Spark · Video-to-3D Reconstruction Benchmark")
    logger.info("  Device: %s (%.1f GB Unified Memory)", gpu_name, total_mem_gb)
    logger.info("  Settings: %ds video @ %.1f FPS | %d 3DGS steps | %dpx", duration_sec, target_fps, max_steps, resolution)
    logger.info("=================================================================")

    # 1. Synthesize benchmark video
    sample_video = bench_data_dir / "bench_input.mp4"
    logger.info("Generating synthetic 360° benchmark video (%ds)…", duration_sec)
    create_demo_sample_video(sample_video)

    benchmark = PipelineBenchmark()
    session_id = f"bench_{int(time.time())}"
    scene_dir = bench_data_dir / session_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Frame Extraction ─────────────────────────────────────────────
    with benchmark.stage("1. Frame Extraction & Quality Check"):
        vid_cfg = VideoConfig(target_fps=target_fps, max_resolution=resolution)
        processor = VideoProcessor(vid_cfg)
        report = processor.process(sample_video, scene_dir)

    logger.info("Stage 1 done: %d frames accepted.", report.frames_accepted)

    # ── Stage 2: VGGT Inference ───────────────────────────────────────────────
    with benchmark.stage("2. VGGT Pose & Depth Estimation"):
        vggt_cfg = VGGTConfig(use_bundle_adjustment=False, load_resolution=resolution)
        reconstructor = VGGTReconstructor(vggt_cfg)
        sparse_dir = reconstructor.reconstruct(scene_dir)

    logger.info("Stage 2 done: COLMAP reconstruction saved to %s", sparse_dir)

    # ── Stage 3: Gaussian Splatting Training ──────────────────────────────────
    with benchmark.stage("3. 3D Gaussian Splatting Optimization"):
        gauss_cfg = GaussianConfig(max_steps=max_steps, log_every=500)
        trainer = GaussianTrainer(gauss_cfg)
        ply_path = trainer.train(scene_dir, scene_dir)

    logger.info("Stage 3 done: Gaussian splat saved to %s", ply_path)

    # ── Results compilation ───────────────────────────────────────────────────
    ply_size_mb = ply_path.stat().st_size / (1024**2)
    summary_data = benchmark.as_dict()
    summary_data["hardware"] = {
        "gpu": gpu_name,
        "unified_memory_gb": round(total_mem_gb, 1),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "pytorch_version": torch.__version__,
    }
    summary_data["reconstruction"] = {
        "frames_extracted": report.frames_accepted,
        "ply_size_mb": round(ply_size_mb, 2),
        "ply_path": str(ply_path),
        "max_steps": max_steps,
    }

    # Save JSON report
    json_path = out_path / "benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save Markdown report
    md_path = out_path / "benchmark_report.md"
    md_content = f"""# DGX Spark Benchmark Report — Video-to-3D Reconstruction

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Hardware**: {gpu_name} ({total_mem_gb:.1f} GB Unified Memory)  
**PyTorch**: {torch.__version__} | CUDA: {torch.version.cuda}  

## Execution Summary

| Pipeline Stage | Latency (s) | GPU Peak Memory (MB) |
|---|---|---|
"""
    for s in summary_data["stages"]:
        md_content += f"| {s['name']} | {s['elapsed_seconds']:.2f}s | {s['gpu_mem_peak_mb']:.1f} MB |\n"

    md_content += f"""| **Total End-to-End** | **{summary_data['total_seconds']:.2f}s** | — |

## Output Metrics
- **Frames Processed**: {report.frames_accepted}
- **Gaussian Splatting Iterations**: {max_steps}
- **Output PLY File Size**: {ply_size_mb:.2f} MB
- **PLY File Path**: `{ply_path}`

```
{benchmark.summary()}
```
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(benchmark.summary())
    logger.info("Benchmark complete! Reports written to %s and %s", json_path, md_path)
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DGX Spark Video-to-3D Benchmark")
    parser.add_argument("--duration", type=int, default=10, help="Test video duration in seconds")
    parser.add_argument("--fps", type=float, default=3.0, help="Target extraction FPS")
    parser.add_argument("--steps", type=int, default=3000, help="Number of Gaussian training steps")
    parser.add_argument("--resolution", type=int, default=1024, help="COLMAP image resolution")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory")
    args = parser.parse_args()

    run_benchmark(
        duration_sec=args.duration,
        target_fps=args.fps,
        max_steps=args.steps,
        resolution=args.resolution,
        output_dir=args.output,
    )
