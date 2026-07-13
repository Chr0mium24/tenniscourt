from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tenniscourt.camera import CameraIntrinsics, look_at_rvec_tvec, points_in_front, project_points, scale_intrinsics
from tenniscourt.court import DOUBLES_WIDTH_M, sample_line_points, sample_line_strip, tennis_court_lines
from tenniscourt.projection_clip import (
    clip_polygon_near,
    clip_polygon_to_image,
    clip_polyline_near,
    project_pinhole_camera,
    world_to_camera,
)


NET_HEIGHT_CENTER_M = 0.914
NET_HEIGHT_POST_M = 1.07
DEFAULT_SUPERSAMPLE = 3


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
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    bounds = bounds or RenderBounds()
    work_intrinsics = scale_intrinsics(intrinsics, supersample)
    image = _court_background(rng, work_intrinsics.width, work_intrinsics.height)
    mask = np.zeros((work_intrinsics.height, work_intrinsics.width), dtype=np.uint8)

    position, target, roll = _sample_camera_pose(rng, bounds)
    rvec, tvec, rotation = look_at_rvec_tvec(position, target, roll)
    visible_lines = _draw_projected_lines(rng, image, mask, work_intrinsics, rvec, tvec, rotation)
    net_segments = _draw_projected_net(image, mask, work_intrinsics, rvec, tvec, rotation)
    _apply_shadows(rng, image)
    _apply_occluders(rng, image, mask)
    image, mask = _downsample_render(image, mask, intrinsics.width, intrinsics.height)
    image = _apply_photo_noise(rng, image)

    label = {
        "camera": intrinsics.as_json(),
        "position_world_m": position.round(6).tolist(),
        "target_world_m": target.round(6).tolist(),
        "roll_deg": float(roll),
        "rvec": rvec.reshape(-1).round(8).tolist(),
        "tvec": tvec.reshape(-1).round(8).tolist(),
        "lines": visible_lines,
        "net_segments": net_segments,
    }
    _scale_label_points(label, 1.0 / supersample)
    return image, mask, label


def render_camera_view(
    intrinsics: CameraIntrinsics,
    position: np.ndarray,
    target: np.ndarray,
    roll_deg: float = 0.0,
    background_bgr: tuple[int, int, int] = (54, 118, 75),
    line_bgr: tuple[int, int, int] = (245, 245, 245),
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    work_intrinsics = scale_intrinsics(intrinsics, supersample)
    image = np.full((work_intrinsics.height, work_intrinsics.width, 3), background_bgr, dtype=np.uint8)
    mask = np.zeros((work_intrinsics.height, work_intrinsics.width), dtype=np.uint8)
    rvec, tvec, rotation = look_at_rvec_tvec(position, target, roll_deg)
    visible_lines = _draw_projected_lines_styled(
        image=image,
        mask=mask,
        intrinsics=work_intrinsics,
        rvec=rvec,
        tvec=tvec,
        rotation=rotation,
        line_bgr=line_bgr,
    )
    net_segments = _draw_projected_net(image, mask, work_intrinsics, rvec, tvec, rotation)
    image, mask = _downsample_render(image, mask, intrinsics.width, intrinsics.height)
    label = {
        "camera": intrinsics.as_json(),
        "position_world_m": np.asarray(position).round(6).tolist(),
        "target_world_m": np.asarray(target).round(6).tolist(),
        "roll_deg": float(roll_deg),
        "rvec": rvec.reshape(-1).round(8).tolist(),
        "tvec": tvec.reshape(-1).round(8).tolist(),
        "lines": visible_lines,
        "net_segments": net_segments,
    }
    _scale_label_points(label, 1.0 / supersample)
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
    visible_lines: list[dict[str, object]] = []

    for line in tennis_court_lines():
        polyline = _draw_projected_line_strip(image, mask, line, intrinsics, rvec, tvec, rotation, rgb_line)
        if len(polyline) >= 2:
            visible_lines.append({"name": line.name, "polyline": polyline})

    return visible_lines


def _draw_projected_lines_styled(
    image: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    rotation: np.ndarray,
    line_bgr: tuple[int, int, int],
) -> list[dict[str, object]]:
    visible_lines: list[dict[str, object]] = []
    for line in tennis_court_lines():
        polyline = _draw_projected_line_strip(image, mask, line, intrinsics, rvec, tvec, rotation, line_bgr)
        if len(polyline) >= 2:
            visible_lines.append({"name": line.name, "polyline": polyline})
    return visible_lines


def _draw_projected_line_strip(
    image: np.ndarray,
    mask: np.ndarray,
    line: object,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    rotation: np.ndarray,
    line_bgr: tuple[int, int, int],
) -> list[list[float]]:
    if _projects_as_straight_strip(intrinsics):
        return _draw_pinhole_line_strip(image, mask, line, intrinsics, rotation, tvec, line_bgr)

    center = sample_line_points(line, samples=128)
    edge_a, edge_b = sample_line_strip(line, samples=128)
    valid = _valid_projected_points(center, rotation, tvec)
    center_2d = project_points(center, intrinsics, rvec, tvec)
    edge_a_2d = project_points(edge_a, intrinsics, rvec, tvec)
    edge_b_2d = project_points(edge_b, intrinsics, rvec, tvec)
    valid &= (
        np.isfinite(center_2d).all(axis=1)
        & np.isfinite(edge_a_2d).all(axis=1)
        & np.isfinite(edge_b_2d).all(axis=1)
    )

    for run in _valid_runs(valid):
        if len(run) < 2:
            continue
        polygon = np.vstack([edge_a_2d[run], edge_b_2d[run][::-1]])
        _fill_projected_polygon(image, mask, polygon, intrinsics.width, intrinsics.height, line_bgr)

    return _collect_clipped_polyline(center_2d, valid, intrinsics.width, intrinsics.height)


def _draw_pinhole_line_strip(
    image: np.ndarray,
    mask: np.ndarray,
    line: object,
    intrinsics: CameraIntrinsics,
    rotation: np.ndarray,
    tvec: np.ndarray,
    line_bgr: tuple[int, int, int],
) -> list[list[float]]:
    center = sample_line_points(line, samples=2)
    edge_a, edge_b = sample_line_strip(line, samples=2)
    strip_world = np.array([edge_a[0], edge_a[-1], edge_b[-1], edge_b[0]], dtype=np.float64)
    strip_camera = clip_polygon_near(world_to_camera(strip_world, rotation, tvec))
    if len(strip_camera) >= 3:
        polygon_2d = project_pinhole_camera(strip_camera, intrinsics)
        _fill_projected_polygon(image, mask, polygon_2d, intrinsics.width, intrinsics.height, line_bgr)

    center_camera = clip_polyline_near(world_to_camera(center, rotation, tvec))
    if len(center_camera) < 2:
        return []
    center_2d = project_pinhole_camera(center_camera, intrinsics)
    valid = np.ones(len(center_2d), dtype=bool)
    return _collect_clipped_polyline(center_2d, valid, intrinsics.width, intrinsics.height)


def _fill_projected_polygon(
    image: np.ndarray,
    mask: np.ndarray,
    polygon_2d: np.ndarray,
    width: int,
    height: int,
    line_bgr: tuple[int, int, int],
) -> None:
    clipped = clip_polygon_to_image(polygon_2d, width, height)
    if len(clipped) < 3:
        return
    polygon_i32 = np.round(clipped).astype(np.int32)
    cv2.fillPoly(image, [polygon_i32], line_bgr, lineType=cv2.LINE_AA)
    cv2.fillPoly(mask, [polygon_i32], 255, lineType=cv2.LINE_AA)


def _projects_as_straight_strip(intrinsics: CameraIntrinsics) -> bool:
    return intrinsics.model == "pinhole" and np.allclose(intrinsics.d, 0.0)


def _draw_projected_net(
    image: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    rotation: np.ndarray,
) -> list[list[list[float]]]:
    segments_3d = _net_segments_3d()
    visible_segments: list[list[list[float]]] = []
    for points_3d in segments_3d:
        valid = _valid_projected_points(points_3d, rotation, tvec)
        points_2d = project_points(points_3d, intrinsics, rvec, tvec)
        valid &= np.isfinite(points_2d).all(axis=1)
        clipped = _draw_clipped_polyline(
            image=image,
            mask=mask,
            points_2d=points_2d,
            valid=valid,
            color=(35, 35, 35),
            mask_value=0,
            thickness=1,
        )
        if len(clipped) >= 2:
            visible_segments.append(clipped)
    return visible_segments


def _net_segments_3d() -> list[np.ndarray]:
    half_width = DOUBLES_WIDTH_M / 2.0
    x_values = np.linspace(-half_width, half_width, 23, dtype=np.float32)
    segments: list[np.ndarray] = []

    top = np.array([[x, 0.0, _net_top_z(float(x), half_width)] for x in x_values], dtype=np.float32)
    segments.append(top)

    for x in x_values:
        top_z = _net_top_z(float(x), half_width)
        segments.append(np.array([[x, 0.0, 0.04], [x, 0.0, top_z]], dtype=np.float32))

    for ratio in [0.25, 0.5, 0.75]:
        row = np.array([[x, 0.0, _net_top_z(float(x), half_width) * ratio] for x in x_values], dtype=np.float32)
        segments.append(row)
    return segments


def _net_top_z(x: float, half_width: float) -> float:
    edge_weight = min(1.0, abs(x) / max(half_width, 1e-6))
    return NET_HEIGHT_CENTER_M + (NET_HEIGHT_POST_M - NET_HEIGHT_CENTER_M) * edge_weight**2


def _valid_projected_points(points_3d: np.ndarray, rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    in_front = points_in_front(points_3d, rotation, tvec)
    return in_front & np.isfinite(points_3d).all(axis=1)


def _valid_runs(valid: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            runs.append(np.arange(start, index))
            start = None
    if start is not None:
        runs.append(np.arange(start, len(valid)))
    return runs


def _collect_clipped_polyline(
    points_2d: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> list[list[float]]:
    return _draw_clipped_polyline(None, None, points_2d, valid, (0, 0, 0), 0, 1, width, height)


def _draw_clipped_polyline(
    image: np.ndarray | None,
    mask: np.ndarray | None,
    points_2d: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    mask_value: int,
    thickness: int,
    width: int | None = None,
    height: int | None = None,
) -> list[list[float]]:
    if width is None or height is None:
        if image is None:
            raise ValueError("width and height are required without an image")
        height, width = image.shape[:2]

    visible: list[list[float]] = []
    rect = (0, 0, int(width), int(height))

    for index in range(len(points_2d) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        p0 = _safe_int_point(points_2d[index], width, height)
        p1 = _safe_int_point(points_2d[index + 1], width, height)
        if not _segment_reasonable(p0, p1, width, height):
            continue
        ok, clipped0, clipped1 = cv2.clipLine(rect, p0, p1)
        if not ok:
            continue
        if image is not None:
            cv2.line(image, clipped0, clipped1, color, thickness, lineType=cv2.LINE_AA)
        if mask is not None:
            cv2.line(mask, clipped0, clipped1, mask_value, thickness, lineType=cv2.LINE_AA)
        _append_visible_point(visible, clipped0)
        _append_visible_point(visible, clipped1)

    return visible


def _safe_int_point(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    limit = max(width, height) * 8
    clipped = np.clip(point, -limit, limit)
    return int(round(float(clipped[0]))), int(round(float(clipped[1])))


def _segment_reasonable(a: tuple[int, int], b: tuple[int, int], width: int, height: int) -> bool:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    max_len = max(width, height) * 4
    return dx * dx + dy * dy < max_len * max_len


def _append_visible_point(polyline: list[list[float]], point: tuple[int, int]) -> None:
    current = [float(point[0]), float(point[1])]
    if not polyline or polyline[-1] != current:
        polyline.append(current)


def _downsample_render(
    image: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    if image.shape[1] == width and image.shape[0] == height:
        return image, mask
    image_small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    return image_small, mask_small


def _scale_label_points(label: dict[str, object], scale: float) -> None:
    for line in label.get("lines", []):
        _scale_polyline(line.get("polyline", []), scale)
    for segment in label.get("net_segments", []):
        _scale_polyline(segment, scale)


def _scale_polyline(polyline: list[list[float]], scale: float) -> None:
    for point in polyline:
        point[0] = round(float(point[0]) * scale, 3)
        point[1] = round(float(point[1]) * scale, 3)


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
