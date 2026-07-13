from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tenniscourt.court import COURT_LENGTH_M, DOUBLES_WIDTH_M, SERVICE_LINE_FROM_NET_M, SINGLES_WIDTH_M


@dataclass(frozen=True)
class CourtKeypoint:
    name: str
    point: tuple[float, float, float]


def court_keypoints() -> list[CourtKeypoint]:
    half_len = COURT_LENGTH_M / 2.0
    half_double = DOUBLES_WIDTH_M / 2.0
    half_single = SINGLES_WIDTH_M / 2.0
    service = SERVICE_LINE_FROM_NET_M
    return [
        CourtKeypoint("near_left_doubles_corner", (-half_double, -half_len, 0.0)),
        CourtKeypoint("near_right_doubles_corner", (half_double, -half_len, 0.0)),
        CourtKeypoint("far_left_doubles_corner", (-half_double, half_len, 0.0)),
        CourtKeypoint("far_right_doubles_corner", (half_double, half_len, 0.0)),
        CourtKeypoint("near_left_singles_corner", (-half_single, -half_len, 0.0)),
        CourtKeypoint("near_right_singles_corner", (half_single, -half_len, 0.0)),
        CourtKeypoint("far_left_singles_corner", (-half_single, half_len, 0.0)),
        CourtKeypoint("far_right_singles_corner", (half_single, half_len, 0.0)),
        CourtKeypoint("left_near_service_corner", (-half_single, -service, 0.0)),
        CourtKeypoint("right_near_service_corner", (half_single, -service, 0.0)),
        CourtKeypoint("left_far_service_corner", (-half_single, service, 0.0)),
        CourtKeypoint("right_far_service_corner", (half_single, service, 0.0)),
        CourtKeypoint("center_near_service_t", (0.0, -service, 0.0)),
        CourtKeypoint("center_far_service_t", (0.0, service, 0.0)),
    ]


def keypoint_names() -> list[str]:
    return [keypoint.name for keypoint in court_keypoints()]


def project_keypoints_from_label(label: dict[str, object]) -> list[dict[str, object]]:
    camera = label["camera"]
    k = np.asarray(camera["k"], dtype=np.float64)
    d = np.asarray(camera["d"], dtype=np.float64).reshape(-1, 1)
    model = str(camera["model"])
    rvec = np.asarray(label["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(label["tvec"], dtype=np.float64).reshape(3, 1)
    width = int(camera["width"])
    height = int(camera["height"])
    points_3d = np.asarray([kp.point for kp in court_keypoints()], dtype=np.float64)

    if model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(points_3d.reshape(1, -1, 3), rvec, tvec, k, d)
    else:
        projected, _ = cv2.projectPoints(points_3d, rvec, tvec, k, d)

    camera_points = _camera_points(points_3d, rvec, tvec)
    result = []
    for kp, point_2d, point_cam in zip(court_keypoints(), projected.reshape(-1, 2), camera_points, strict=True):
        x = float(point_2d[0])
        y = float(point_2d[1])
        visible = bool(point_cam[2] > 0.05 and 0.0 <= x < width and 0.0 <= y < height)
        result.append({"name": kp.name, "xy": [round(x, 3), round(y, 3)], "visible": visible})
    return result


def heatmaps_from_keypoints(
    keypoints: list[dict[str, object]],
    width: int,
    height: int,
    sigma_px: float = 3.0,
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    heatmaps = np.zeros((len(keypoints), height, width), dtype=np.float32)
    denom = 2.0 * sigma_px * sigma_px
    for index, keypoint in enumerate(keypoints):
        if not keypoint.get("visible", False):
            continue
        x, y = keypoint["xy"]
        heatmaps[index] = np.exp(-((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / denom)
    return heatmaps


def _camera_points(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    return (rotation @ points_3d.T + tvec).T
