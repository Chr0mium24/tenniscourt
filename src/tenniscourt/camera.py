from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    k: np.ndarray
    d: np.ndarray
    model: str = "pinhole"

    def as_json(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "k": self.k.tolist(),
            "d": self.d.reshape(-1).tolist(),
            "model": self.model,
        }


def default_intrinsics(width: int, height: int, fov_deg: float, model: str) -> CameraIntrinsics:
    fx = width / (2.0 * tan(radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    d_len = 4 if model == "fisheye" else 5
    d = np.zeros((d_len, 1), dtype=np.float64)
    return CameraIntrinsics(width=width, height=height, k=k, d=d, model=model)


def look_at_rvec_tvec(
    position: np.ndarray,
    target: np.ndarray,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)

    if abs(roll_deg) > 1e-6:
        angle = radians(roll_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        right, down = cos_a * right + sin_a * down, -sin_a * right + cos_a * down

    rotation = np.stack([right, down, forward], axis=0)
    tvec = -rotation @ position.reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec.reshape(3, 1), tvec.reshape(3, 1), rotation


def project_points(
    points_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    if intrinsics.model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(points.reshape(1, -1, 3), rvec, tvec, intrinsics.k, intrinsics.d)
        return projected.reshape(-1, 2)

    projected, _ = cv2.projectPoints(points, rvec, tvec, intrinsics.k, intrinsics.d)
    return projected.reshape(-1, 2)


def points_in_front(points_world: np.ndarray, rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    camera_points = (rotation @ points_world.T + tvec).T
    return camera_points[:, 2] > 0.05
