"""
COLMAP utility functions for reading and parsing camera poses and sparse 3D points.

Supports both binary (.bin) and text (.txt) COLMAP formats, as well as pycolmap
Reconstruction objects. Provides camera matrix conversions (world-to-camera,
camera-to-world) and projection utilities.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch


# ── Data structures matching COLMAP specifications ─────────────────────────────


class CameraModel(NamedTuple):
    model_id: int
    model_name: str
    num_params: int


CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_NAMES = {v.model_name: v for v in CAMERA_MODELS.values()}


@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def focal_x(self) -> float:
        return float(self.params[0])

    @property
    def focal_y(self) -> float:
        if self.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            return float(self.params[1])
        return float(self.params[0])

    @property
    def cx(self) -> float:
        idx = 2 if self.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV") else 1
        return float(self.params[idx])

    @property
    def cy(self) -> float:
        idx = 3 if self.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV") else 2
        return float(self.params[idx])

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic calibration matrix K."""
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self.focal_x
        K[1, 1] = self.focal_y
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray  # [qw, qx, qy, qz]
    tvec: np.ndarray  # [tx, ty, tz]
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

    def qvec2rotmat(self) -> np.ndarray:
        """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix R (world to cam)."""
        qw, qx, qy, qz = self.qvec
        return np.array(
            [
                [
                    1 - 2 * qy**2 - 2 * qz**2,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ],
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx**2 - 2 * qz**2,
                    2 * qy * qz - 2 * qx * qw,
                ],
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx**2 - 2 * qy**2,
                ],
            ],
            dtype=np.float32,
        )

    def world_to_cam(self) -> np.ndarray:
        """Return 4x4 [R | t] world-to-camera transformation matrix."""
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = self.qvec2rotmat()
        w2c[:3, 3] = self.tvec
        return w2c

    def cam_to_world(self) -> np.ndarray:
        """Return 4x4 [R^T | -R^T * t] camera-to-world transformation matrix."""
        w2c = self.world_to_cam()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = w2c[:3, :3].T
        c2w[:3, 3] = -w2c[:3, :3].T @ w2c[:3, 3]
        return c2w


@dataclass
class ColmapPoint3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


# ── Binary Reading Functions ───────────────────────────────────────────────────


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file: str | Path) -> Dict[int, ColmapCamera]:
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = properties[0]
            model_id = properties[1]
            model_name = CAMERA_MODELS[model_id].model_name
            width = properties[2]
            height = properties[3]
            num_params = CAMERA_MODELS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params, dtype=np.float32),
            )
    return cameras


def read_images_binary(path_to_model_file: str | Path) -> Dict[int, ColmapImage]:
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = properties[0]
            qvec = np.array(properties[1:5], dtype=np.float32)
            tvec = np.array(properties[5:8], dtype=np.float32)
            camera_id = properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(
                fid, 24 * num_points2D, "ddq" * num_points2D
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_binary(path_to_model_file: str | Path) -> Dict[int, ColmapPoint3D]:
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = properties[0]

            xyz = np.array(properties[1:4], dtype=np.float32)
            rgb = np.array(properties[4:7], dtype=np.uint8)
            error = properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(
                fid, 8 * track_length, "ii" * track_length
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = ColmapPoint3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


# ── High-Level Scene Loader ───────────────────────────────────────────────────


class ColmapScene:
    """Convenient loader and accessor for a COLMAP reconstruction directory."""

    def __init__(self, sparse_dir: str | Path) -> None:
        self.sparse_dir = Path(sparse_dir)
        self.cameras: Dict[int, ColmapCamera] = {}
        self.images: Dict[int, ColmapImage] = {}
        self.points3D: Dict[int, ColmapPoint3D] = {}
        self.load()

    def load(self) -> None:
        """Load cameras, images, and points3D from binary or text files."""
        # Try binary first
        cam_bin = self.sparse_dir / "cameras.bin"
        img_bin = self.sparse_dir / "images.bin"
        pts_bin = self.sparse_dir / "points3D.bin"

        if cam_bin.exists() and img_bin.exists():
            self.cameras = read_cameras_binary(cam_bin)
            self.images = read_images_binary(img_bin)
            if pts_bin.exists():
                self.points3D = read_points3D_binary(pts_bin)
            return

        # Fallback to pycolmap if installed and files exist in standard directory
        try:
            import pycolmap

            rec = pycolmap.Reconstruction(str(self.sparse_dir))
            for cam_id, cam in rec.cameras.items():
                self.cameras[cam_id] = ColmapCamera(
                    id=cam_id,
                    model=str(cam.model_name),
                    width=cam.width,
                    height=cam.height,
                    params=np.array(cam.params, dtype=np.float32),
                )
            for img_id, img in rec.images.items():
                qvec = np.array(img.cam_from_world.rotation.quat, dtype=np.float32)
                tvec = np.array(img.cam_from_world.translation, dtype=np.float32)
                self.images[img_id] = ColmapImage(
                    id=img_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=img.camera_id,
                    name=img.name,
                    xys=np.array([p.xy for p in img.points2D], dtype=np.float32),
                    point3D_ids=np.array([p.point3D_id for p in img.points2D]),
                )
            for pt_id, pt in rec.points3D.items():
                self.points3D[pt_id] = ColmapPoint3D(
                    id=pt_id,
                    xyz=np.array(pt.xyz, dtype=np.float32),
                    rgb=np.array(pt.color, dtype=np.uint8),
                    error=pt.error,
                    image_ids=np.array([el.image_id for el in pt.track.elements]),
                    point2D_idxs=np.array([el.point2D_idx for el in pt.track.elements]),
                )
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not load COLMAP reconstruction from {self.sparse_dir}: {exc}"
            )

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_points(self) -> int:
        return len(self.points3D)

    def get_points_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (N, 3) XYZ and (N, 3) RGB normalized to [0, 1]."""
        if not self.points3D:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        xyz = np.array([p.xyz for p in self.points3D.values()], dtype=np.float32)
        rgb = np.array([p.rgb for p in self.points3D.values()], dtype=np.float32) / 255.0
        return xyz, rgb

    def get_camera_centers(self) -> np.ndarray:
        """Return (N, 3) array of camera center positions in world space."""
        centers = []
        for img in self.images.values():
            c2w = img.cam_to_world()
            centers.append(c2w[:3, 3])
        return np.array(centers, dtype=np.float32)

    def compute_scene_bounds(self) -> Tuple[np.ndarray, float]:
        """Compute scene center (3,) and approximate radius float."""
        centers = self.get_camera_centers()
        if len(centers) == 0:
            return np.zeros(3, dtype=np.float32), 1.0
        center = np.mean(centers, axis=0)
        radius = float(np.max(np.linalg.norm(centers - center, axis=-1))) * 1.1
        return center, radius

    def save(self, sparse_dir: Optional[str | Path] = None) -> None:
        """Save scene to binary COLMAP files (cameras.bin, images.bin, points3D.bin)."""
        target_dir = Path(sparse_dir) if sparse_dir is not None else self.sparse_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        write_cameras_binary(target_dir / "cameras.bin", self.cameras)
        write_images_binary(target_dir / "images.bin", self.images)
        write_points3D_binary(target_dir / "points3D.bin", self.points3D)


def write_cameras_binary(path_to_model_file: str | Path, cameras: Dict[int, ColmapCamera]) -> None:
    """Write cameras dictionary to COLMAP binary cameras.bin."""
    with open(path_to_model_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for cam_id, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES.get(cam.model, CAMERA_MODELS[0]).model_id
            fid.write(struct.pack("<iiQQ", int(cam.id), int(model_id), int(cam.width), int(cam.height)))
            for p in cam.params:
                fid.write(struct.pack("<d", float(p)))


def write_images_binary(path_to_model_file: str | Path, images: Dict[int, ColmapImage]) -> None:
    """Write images dictionary to COLMAP binary images.bin."""
    with open(path_to_model_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for img_id, img in images.items():
            fid.write(struct.pack("<i", int(img.id)))
            fid.write(struct.pack("<4d", *[float(x) for x in img.qvec]))
            fid.write(struct.pack("<3d", *[float(x) for x in img.tvec]))
            fid.write(struct.pack("<i", int(img.camera_id)))
            fid.write(img.name.encode("utf-8") + b"\x00")
            num_points2D = len(img.xys)
            fid.write(struct.pack("<Q", num_points2D))
            for (x, y), p3d_id in zip(img.xys, img.point3D_ids):
                fid.write(struct.pack("<ddq", float(x), float(y), int(p3d_id)))


def write_points3D_binary(path_to_model_file: str | Path, points3D: Dict[int, ColmapPoint3D]) -> None:
    """Write points3D dictionary to COLMAP binary points3D.bin."""
    with open(path_to_model_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(points3D)))
        for pt_id, pt in points3D.items():
            fid.write(struct.pack("<Q", int(pt.id)))
            fid.write(struct.pack("<3d", *[float(x) for x in pt.xyz]))
            fid.write(struct.pack("<3B", *[int(x) for x in pt.rgb]))
            fid.write(struct.pack("<d", float(pt.error)))
            track_len = len(pt.image_ids)
            fid.write(struct.pack("<Q", track_len))
            for img_id, p2d_idx in zip(pt.image_ids, pt.point2D_idxs):
                fid.write(struct.pack("<ii", int(img_id), int(p2d_idx)))


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz], dtype=np.float32)


def build_colmap_scene_from_arrays(
    points3d: np.ndarray,
    points_xyf: np.ndarray,
    points_rgb: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int] | np.ndarray,
    image_names: Optional[List[str]] = None,
) -> ColmapScene:
    """Build a ColmapScene directly from numpy predictions without pycolmap."""
    N = len(extrinsics)
    W, H = int(image_size[0]), int(image_size[1])

    # Construct empty scene
    scene = ColmapScene.__new__(ColmapScene)
    scene.sparse_dir = Path(".")
    scene.cameras = {}
    scene.images = {}
    scene.points3D = {}

    # Cameras and Images
    for fidx in range(N):
        cam_id = fidx + 1
        fx = float(intrinsics[fidx, 0, 0])
        fy = float(intrinsics[fidx, 1, 1])
        cx = float(intrinsics[fidx, 0, 2])
        cy = float(intrinsics[fidx, 1, 2])
        camera_params = np.array([fx, fy, cx, cy], dtype=np.float32)

        scene.cameras[cam_id] = ColmapCamera(
            id=cam_id,
            model="PINHOLE",
            width=W,
            height=H,
            params=camera_params,
        )

        R = extrinsics[fidx][:3, :3]
        t = extrinsics[fidx][:3, 3]
        qvec = rotmat2qvec(R)
        name = image_names[fidx] if (image_names and fidx < len(image_names)) else f"image_{fidx + 1:04d}.jpg"

        scene.images[cam_id] = ColmapImage(
            id=cam_id,
            qvec=qvec,
            tvec=t.astype(np.float32),
            camera_id=cam_id,
            name=name,
            xys=np.zeros((0, 2), dtype=np.float32),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )

    # 3D Points
    P = len(points3d)
    for pidx in range(P):
        pt_id = pidx + 1
        scene.points3D[pt_id] = ColmapPoint3D(
            id=pt_id,
            xyz=points3d[pidx].astype(np.float32),
            rgb=points_rgb[pidx].astype(np.uint8),
            error=1.0,
            image_ids=np.zeros((0,), dtype=np.int32),
            point2D_idxs=np.zeros((0,), dtype=np.int32),
        )

    return scene


