from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tenniscourt.camera import CameraIntrinsics


NEAR_Z_M = 0.05


def world_to_camera(points_world: np.ndarray, rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    return (rotation @ points.T + tvec).T


def project_pinhole_camera(points_camera: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    z = points[:, 2].reshape(-1, 1)
    xy = points[:, :2] / z
    projected = np.empty((len(points), 2), dtype=np.float64)
    projected[:, 0] = intrinsics.k[0, 0] * xy[:, 0] + intrinsics.k[0, 2]
    projected[:, 1] = intrinsics.k[1, 1] * xy[:, 1] + intrinsics.k[1, 2]
    return projected


def clip_polygon_near(points_camera: np.ndarray, near_z: float = NEAR_Z_M) -> np.ndarray:
    points = _as_point_list(points_camera)
    clipped = _clip_polygon(points, lambda p: p[2] >= near_z, lambda a, b: _intersect_z(a, b, near_z))
    return np.asarray(clipped, dtype=np.float64)


def clip_polyline_near(points_camera: np.ndarray, near_z: float = NEAR_Z_M) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    output: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:], strict=False):
        clipped = _clip_segment_near(start, end, near_z)
        if clipped is None:
            continue
        a, b = clipped
        if not output or not np.allclose(output[-1], a):
            output.append(a)
        output.append(b)
    return np.asarray(output, dtype=np.float64)


def clip_polygon_to_image(points_2d: np.ndarray, width: int, height: int) -> np.ndarray:
    points = _as_point_list(points_2d)
    max_x = float(width - 1)
    max_y = float(height - 1)
    boundaries = [
        (lambda p: p[0] >= 0.0, lambda a, b: _intersect_x(a, b, 0.0)),
        (lambda p: p[0] <= max_x, lambda a, b: _intersect_x(a, b, max_x)),
        (lambda p: p[1] >= 0.0, lambda a, b: _intersect_y(a, b, 0.0)),
        (lambda p: p[1] <= max_y, lambda a, b: _intersect_y(a, b, max_y)),
    ]
    clipped = points
    for inside, intersect in boundaries:
        clipped = _clip_polygon(clipped, inside, intersect)
        if len(clipped) < 3:
            return np.empty((0, 2), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def _clip_segment_near(
    start: np.ndarray,
    end: np.ndarray,
    near_z: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    start_inside = start[2] >= near_z
    end_inside = end[2] >= near_z
    if start_inside and end_inside:
        return start, end
    if not start_inside and not end_inside:
        return None
    point = _intersect_z(start, end, near_z)
    return (point, end) if end_inside else (start, point)


def _clip_polygon(
    points: list[np.ndarray],
    inside: Callable[[np.ndarray], bool],
    intersect: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> list[np.ndarray]:
    if not points:
        return []
    output: list[np.ndarray] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def _intersect_z(start: np.ndarray, end: np.ndarray, z_value: float) -> np.ndarray:
    denom = end[2] - start[2]
    if abs(float(denom)) < 1e-12:
        return start.copy()
    t = (z_value - start[2]) / denom
    return start + t * (end - start)


def _intersect_x(start: np.ndarray, end: np.ndarray, x_value: float) -> np.ndarray:
    denom = end[0] - start[0]
    if abs(float(denom)) < 1e-12:
        return start.copy()
    t = (x_value - start[0]) / denom
    return start + t * (end - start)


def _intersect_y(start: np.ndarray, end: np.ndarray, y_value: float) -> np.ndarray:
    denom = end[1] - start[1]
    if abs(float(denom)) < 1e-12:
        return start.copy()
    t = (y_value - start[1]) / denom
    return start + t * (end - start)


def _as_point_list(points: np.ndarray) -> list[np.ndarray]:
    array = np.asarray(points, dtype=np.float64)
    if len(array) == 0:
        return []
    return [point.copy() for point in array]
