from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tenniscourt.camera import CameraIntrinsics, look_at_rvec_tvec, points_in_front, project_points
from tenniscourt.court import sample_line_points, tennis_court_lines


@dataclass(frozen=True)
class RenderBounds:
    height_min_m: float = 0.35
    height_max_m: float = 0.45
    x_min_m: float = -4.8
    x_max_m: float = 4.8
    y_min_m: float = -12.8
    y_max_m: float = -2.5
    lookahead_min_m: float = 5.0
    lookahead_max_m: float = 17.0


def render_sample(
    rng: np.random.Generator,
    intrinsics: CameraIntrinsics,
    bounds: RenderBounds | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    bounds = bounds or RenderBounds()
    image = _court_background(rng, intrinsics.width, intrinsics.height)
    mask = np.zeros((intrinsics.height, intrinsics.width), dtype=np.uint8)

    position, target, roll = _sample_camera_pose(rng, bounds)
    rvec, tvec, rotation = look_at_rvec_tvec(position, target, roll)
    visible_lines = _draw_projected_lines(rng, image, mask, intrinsics, rvec, tvec, rotation)
    _apply_shadows(rng, image)
    _apply_occluders(rng, image, mask)
    image = _apply_photo_noise(rng, image)

    label = {
        "camera": intrinsics.as_json(),
        "position_world_m": position.round(6).tolist(),
        "target_world_m": target.round(6).tolist(),
        "roll_deg": float(roll),
        "rvec": rvec.reshape(-1).round(8).tolist(),
        "tvec": tvec.reshape(-1).round(8).tolist(),
        "lines": visible_lines,
    }
    return image, mask, label


def _sample_camera_pose(
    rng: np.random.Generator,
    bounds: RenderBounds,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = rng.uniform(bounds.x_min_m, bounds.x_max_m)
    y = rng.uniform(bounds.y_min_m, bounds.y_max_m)
    z = rng.uniform(bounds.height_min_m, bounds.height_max_m)
    lookahead = rng.uniform(bounds.lookahead_min_m, bounds.lookahead_max_m)
    target = np.array(
        [
            x + rng.normal(0.0, 1.8),
            min(y + lookahead, 12.0),
            rng.uniform(-0.05, 0.25),
        ],
        dtype=np.float64,
    )
    position = np.array([x, y, z], dtype=np.float64)
    roll = rng.normal(0.0, 2.0)
    return position, target, roll


def _draw_projected_lines(
    rng: np.random.Generator,
    image: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    rotation: np.ndarray,
) -> list[dict[str, object]]:
    line_color = int(rng.integers(210, 255))
    rgb_line = (line_color, line_color, line_color)
    thickness = int(rng.integers(2, 5))
    visible_lines: list[dict[str, object]] = []

    for line in tennis_court_lines():
        points_3d = sample_line_points(line, samples=96)
        in_front = points_in_front(points_3d, rotation, tvec)
        points_2d = project_points(points_3d, intrinsics, rvec, tvec)
        inside = _inside_image(points_2d, intrinsics.width, intrinsics.height, margin=16)
        valid = in_front & inside
        polyline = _draw_valid_polyline(image, mask, points_2d, valid, rgb_line, thickness)
        if len(polyline) >= 2:
            visible_lines.append({"name": line.name, "polyline": polyline})

    return visible_lines


def _inside_image(points: np.ndarray, width: int, height: int, margin: int) -> np.ndarray:
    return (
        (points[:, 0] >= -margin)
        & (points[:, 0] < width + margin)
        & (points[:, 1] >= -margin)
        & (points[:, 1] < height + margin)
    )


def _draw_valid_polyline(
    image: np.ndarray,
    mask: np.ndarray,
    points_2d: np.ndarray,
    valid: np.ndarray,
    rgb_line: tuple[int, int, int],
    thickness: int,
) -> list[list[float]]:
    visible: list[list[float]] = []
    last_point: tuple[int, int] | None = None

    for point, is_valid in zip(points_2d, valid, strict=True):
        current = (int(round(point[0])), int(round(point[1])))
        if is_valid:
            visible.append([round(float(point[0]), 3), round(float(point[1]), 3)])
            if last_point is not None and _segment_reasonable(last_point, current):
                cv2.line(image, last_point, current, rgb_line, thickness, lineType=cv2.LINE_AA)
                cv2.line(mask, last_point, current, 255, thickness, lineType=cv2.LINE_AA)
            last_point = current
        else:
            last_point = None

    return visible


def _segment_reasonable(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy < 200 * 200


def _court_background(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    palettes = np.array(
        [
            [54, 118, 75],
            [50, 116, 150],
            [129, 91, 60],
            [72, 105, 86],
        ],
        dtype=np.int16,
    )
    base = palettes[int(rng.integers(0, len(palettes)))].reshape(1, 1, 3)
    noise = rng.normal(0.0, 9.0, size=(height, width, 3))
    gradient = np.linspace(-10, 10, width, dtype=np.float32).reshape(1, width, 1)
    image = np.clip(base + noise + gradient, 0, 255).astype(np.uint8)
    return image


def _apply_shadows(rng: np.random.Generator, image: np.ndarray) -> None:
    if rng.random() > 0.65:
        return
    overlay = image.copy()
    height, width = image.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        x0 = int(rng.integers(-width // 2, width))
        x1 = int(x0 + rng.integers(width // 3, width))
        y0 = int(rng.integers(0, height))
        y1 = int(y0 + rng.integers(height // 4, height))
        polygon = np.array([[x0, y0], [x1, y0], [x1 + 80, y1], [x0 + 80, y1]], dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], (20, 20, 20))
    alpha = float(rng.uniform(0.12, 0.28))
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _apply_occluders(rng: np.random.Generator, image: np.ndarray, mask: np.ndarray) -> None:
    height, width = image.shape[:2]
    for _ in range(int(rng.integers(0, 4))):
        color = tuple(int(v) for v in rng.integers(20, 180, size=3))
        x = int(rng.integers(0, width))
        y = int(rng.integers(height // 4, height))
        w = int(rng.integers(width // 20, max(width // 8, width // 20 + 1)))
        h = int(rng.integers(height // 15, max(height // 4, height // 15 + 1)))
        p1 = (max(0, x - w // 2), max(0, y - h))
        p2 = (min(width - 1, x + w // 2), min(height - 1, y))
        cv2.rectangle(image, p1, p2, color, thickness=-1)
        cv2.rectangle(mask, p1, p2, 0, thickness=-1)


def _apply_photo_noise(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32)
    gamma = float(rng.uniform(0.8, 1.25))
    output = 255.0 * np.power(np.clip(output / 255.0, 0.0, 1.0), gamma)
    output += rng.normal(0.0, rng.uniform(1.0, 5.0), size=output.shape)

    if rng.random() < 0.35:
        kernel = int(rng.choice([3, 5]))
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)

    output = np.clip(output, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        quality = int(rng.integers(55, 95))
        ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return output
