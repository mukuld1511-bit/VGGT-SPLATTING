"""
Benchmark utilities — timing, memory tracking, and report generation.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, Optional

import torch


@dataclass
class StageMetrics:
    """Timing and memory snapshot for one pipeline stage."""

    name: str
    elapsed_seconds: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    gpu_mem_allocated_mb: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)


class Timer:
    """Simple wall-clock timer context manager."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._t0


class MemoryTracker:
    """Tracks CPU and GPU memory usage on DGX Spark."""

    @staticmethod
    def get_gpu_memory_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0

    @staticmethod
    def get_gpu_peak_memory_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return 0.0

    @staticmethod
    def reset_peak() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class PipelineBenchmark:


    """Accumulates timing + GPU-memory metrics across pipeline stages.

    Usage::

        bench = PipelineBenchmark()
        with bench.stage("Frame Extraction"):
            extract_frames(...)
        with bench.stage("VGGT Inference"):
            run_vggt(...)
        print(bench.summary())
    """

    def __init__(self) -> None:
        self.stages: list[StageMetrics] = []
        self._wall_start: Optional[float] = None

    def start(self) -> None:
        self._wall_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Generator[StageMetrics, None, None]:
        """Context manager that records wall-clock time and GPU peak memory."""
        if self._wall_start is None:
            self.start()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        metrics = StageMetrics(name=name)
        t0 = time.perf_counter()

        yield metrics  # caller runs their code here

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            metrics.gpu_mem_peak_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
            )
            metrics.gpu_mem_allocated_mb = (
                torch.cuda.memory_allocated() / 1024**2
            )

        metrics.elapsed_seconds = time.perf_counter() - t0
        self.stages.append(metrics)

    @property
    def total_seconds(self) -> float:
        return sum(s.elapsed_seconds for s in self.stages)

    def summary(self) -> str:
        """Return a human-readable benchmark summary table."""
        lines = [
            "",
            "┌──────────────────────────────┬───────────┬──────────────┐",
            "│ Stage                        │  Time (s) │ GPU Peak MB  │",
            "├──────────────────────────────┼───────────┼──────────────┤",
        ]
        for s in self.stages:
            name = s.name[:28].ljust(28)
            t = f"{s.elapsed_seconds:8.1f}s"
            m = f"{s.gpu_mem_peak_mb:10.0f}" if s.gpu_mem_peak_mb else "       N/A"
            lines.append(f"│ {name} │ {t} │ {m}  │")
        lines.append(
            "├──────────────────────────────┼───────────┼──────────────┤"
        )
        total = f"{self.total_seconds:8.1f}s"
        lines.append(
            f"│ {'TOTAL':28s} │ {total} │              │"
        )
        lines.append(
            "└──────────────────────────────┴───────────┴──────────────┘"
        )
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, object]:
        """Serializable representation for logs / JSON."""
        return {
            "total_seconds": round(self.total_seconds, 2),
            "stages": [
                {
                    "name": s.name,
                    "elapsed_seconds": round(s.elapsed_seconds, 2),
                    "gpu_mem_peak_mb": round(s.gpu_mem_peak_mb, 1),
                    **s.extra,
                }
                for s in self.stages
            ],
        }


# Convenient alias
Benchmark = PipelineBenchmark

