"""
PLY utilities for 3D Gaussian Splat I/O.

Supports reading and writing the standard 3DGS PLY format used by gsplat,
the original 3D Gaussian Splatting, and compatible viewers like Viser and
SuperSplat.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Spherical-harmonic DC normalization constant (band 0)
SH_C0 = 0.28209479177387814


@dataclass
class GaussianCloud:
    """In-memory representation of a 3D Gaussian Splat point cloud."""

    positions: np.ndarray  # (N, 3) float32 — xyz
    sh_dc: np.ndarray  # (N, 3) float32 — SH band-0 coefficients
    sh_rest: Optional[np.ndarray] = None  # (N, R, 3) float32 — higher SH bands
    opacities: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))  # (N,) or (N,1) logit
    scales: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))  # (N, 3) log-space
    rotations: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))  # (N, 4) quaternion wxyz

    def __init__(
        self,
        positions: Optional[np.ndarray] = None,
        sh_dc: Optional[np.ndarray] = None,
        sh_rest: Optional[np.ndarray] = None,
        opacities: Optional[np.ndarray] = None,
        scales: Optional[np.ndarray] = None,
        rotations: Optional[np.ndarray] = None,
        # Keyword aliases
        means: Optional[np.ndarray] = None,
        sh0: Optional[np.ndarray] = None,
        shN: Optional[np.ndarray] = None,
        quats: Optional[np.ndarray] = None,
    ) -> None:
        self.positions = positions if positions is not None else means
        if self.positions is None:
            raise ValueError("Either 'positions' or 'means' must be provided")

        if sh_dc is not None:
            self.sh_dc = sh_dc
        elif sh0 is not None:
            # Handle sh0 shaped (N, 1, 3) or (N, 3)
            self.sh_dc = sh0.squeeze(1) if (sh0.ndim == 3 and sh0.shape[1] == 1) else sh0
        else:
            self.sh_dc = np.zeros_like(self.positions)

        self.sh_rest = sh_rest if sh_rest is not None else shN
        if opacities is not None:
            self.opacities = opacities.squeeze(-1) if opacities.ndim == 2 else opacities
        else:
            self.opacities = np.full(self.positions.shape[0], 2.0, dtype=np.float32)

        self.scales = scales if scales is not None else np.full((self.positions.shape[0], 3), -5.0, dtype=np.float32)
        self.rotations = (
            rotations
            if rotations is not None
            else (
                quats
                if quats is not None
                else np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (self.positions.shape[0], 1),
                )
            )
        )

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def sh_degree(self) -> int:
        if self.sh_rest is None or self.sh_rest.shape[1] == 0:
            return 0
        rest_per_channel = self.sh_rest.shape[1]
        total = rest_per_channel + 1  # +1 for DC
        degree = int(np.sqrt(total)) - 1
        return degree

    # ── Derived visual properties (for viewers) ──────────────────────────────

    @property
    def colors_rgb(self) -> np.ndarray:
        """Base RGB colors in [0, 1] from SH DC."""
        return np.clip(self.sh_dc * SH_C0 + 0.5, 0.0, 1.0)

    @property
    def opacities_activated(self) -> np.ndarray:
        """Opacity in [0, 1] via sigmoid."""
        return 1.0 / (1.0 + np.exp(-self.opacities))

    @property
    def scales_activated(self) -> np.ndarray:
        """Actual scales via exp."""
        return np.exp(self.scales)

    @property
    def rotations_normalized(self) -> np.ndarray:
        """Unit quaternions (wxyz)."""
        norms = np.linalg.norm(self.rotations, axis=-1, keepdims=True)
        return self.rotations / np.maximum(norms, 1e-8)

    def compute_covariances(self) -> np.ndarray:
        """Compute full 3x3 covariance matrices Sigma = R * S * S^T * R^T for each Gaussian.

        Returns:
            (N, 3, 3) float32 array suitable for Viser or WebGL renderers.
        """
        N = self.num_gaussians
        q = self.rotations_normalized
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = np.zeros((N, 3, 3), dtype=np.float32)
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        # Scale transformation matrix M = R * S
        s = self.scales_activated
        M = R * s[:, None, :]
        cov = M @ M.transpose(0, 2, 1)
        return cov.astype(np.float32)


def write_gaussian_ply(path: str | Path, cloud: GaussianCloud) -> None:
    """Write a GaussianCloud to a standard 3DGS PLY file.

    The output is binary-little-endian and compatible with gsplat, the original
    3DGS implementation, SuperSplat, and Viser.
    """
    if isinstance(path, GaussianCloud):
        path, cloud = cloud, path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


    N = cloud.num_gaussians
    num_rest = 0 if cloud.sh_rest is None else cloud.sh_rest.shape[1]

    # Build header
    props = [
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(num_rest * 3):
        props.append(f"property float f_rest_{i}")
    props.append("property float opacity")
    props.append("property float scale_0")
    props.append("property float scale_1")
    props.append("property float scale_2")
    props.append("property float rot_0")
    props.append("property float rot_1")
    props.append("property float rot_2")
    props.append("property float rot_3")

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        *props,
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"

    normals = np.zeros((N, 3), dtype=np.float32)

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(N):
            # xyz
            f.write(struct.pack("<3f", *cloud.positions[i]))
            # normals
            f.write(struct.pack("<3f", *normals[i]))
            # SH DC
            f.write(struct.pack("<3f", *cloud.sh_dc[i]))
            # SH rest (interleaved: f_rest_0 = sh_rest[0,0], f_rest_1 = sh_rest[0,1], ...)
            if cloud.sh_rest is not None and num_rest > 0:
                rest_flat = cloud.sh_rest[i].T.flatten()  # (3, R) -> (3*R,)
                f.write(struct.pack(f"<{len(rest_flat)}f", *rest_flat))
            # opacity
            f.write(struct.pack("<f", float(cloud.opacities[i])))
            # scales
            f.write(struct.pack("<3f", *cloud.scales[i]))
            # rotations (wxyz)
            f.write(struct.pack("<4f", *cloud.rotations[i]))


def read_gaussian_ply(path: str | Path) -> GaussianCloud:
    """Read a standard 3DGS PLY file into a GaussianCloud.

    Handles both binary-little-endian and ASCII PLY files.
    """
    from plyfile import PlyData

    path = Path(path)
    plydata = PlyData.read(str(path))
    v = plydata["vertex"]
    prop_names = set(p.name for p in v.properties)

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # SH DC coefficients
    if "f_dc_0" in prop_names:
        sh_dc = np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1
        ).astype(np.float32)
    elif "red" in prop_names:
        rgb = (
            np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(
                np.float32
            )
            / 255.0
        )
        sh_dc = (rgb - 0.5) / SH_C0
    else:
        raise ValueError("PLY has neither SH DC nor RGB vertex properties")

    # SH higher-order coefficients
    rest_keys = sorted(
        [k for k in prop_names if k.startswith("f_rest_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    if rest_keys:
        rest_flat = np.stack([v[k] for k in rest_keys], axis=-1).astype(
            np.float32
        )
        num_rest = len(rest_keys) // 3
        sh_rest = rest_flat.reshape(-1, 3, num_rest).transpose(0, 2, 1)
    else:
        sh_rest = None

    # Opacity (logit-space)
    if "opacity" in prop_names:
        opacities = np.asarray(v["opacity"], dtype=np.float32)
    else:
        opacities = np.full(positions.shape[0], 2.0, dtype=np.float32)

    # Scales (log-space)
    if "scale_0" in prop_names:
        scales = np.stack(
            [v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1
        ).astype(np.float32)
    else:
        scales = np.full((positions.shape[0], 3), -5.0, dtype=np.float32)

    # Rotations (wxyz quaternion)
    if "rot_0" in prop_names:
        rotations = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
        ).astype(np.float32)
    else:
        rotations = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (positions.shape[0], 1),
        )


    return GaussianCloud(
        positions=positions,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )


# Aliases for flexible naming
export_gaussian_ply = write_gaussian_ply
load_gaussian_ply = read_gaussian_ply
