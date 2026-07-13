from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CourtLine:
    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]


COURT_LENGTH_M = 23.77
DOUBLES_WIDTH_M = 10.97
SINGLES_WIDTH_M = 8.23
SERVICE_LINE_FROM_NET_M = 6.40
CENTER_MARK_LENGTH_M = 0.10


def tennis_court_lines() -> list[CourtLine]:
    half_len = COURT_LENGTH_M / 2.0
    half_double = DOUBLES_WIDTH_M / 2.0
    half_single = SINGLES_WIDTH_M / 2.0
    service_y = SERVICE_LINE_FROM_NET_M

    lines = [
        CourtLine("near_baseline", (-half_double, -half_len, 0), (half_double, -half_len, 0)),
        CourtLine("far_baseline", (-half_double, half_len, 0), (half_double, half_len, 0)),
        CourtLine("left_doubles_sideline", (-half_double, -half_len, 0), (-half_double, half_len, 0)),
        CourtLine("right_doubles_sideline", (half_double, -half_len, 0), (half_double, half_len, 0)),
        CourtLine("left_singles_sideline", (-half_single, -half_len, 0), (-half_single, half_len, 0)),
        CourtLine("right_singles_sideline", (half_single, -half_len, 0), (half_single, half_len, 0)),
        CourtLine("near_service_line", (-half_single, -service_y, 0), (half_single, -service_y, 0)),
        CourtLine("far_service_line", (-half_single, service_y, 0), (half_single, service_y, 0)),
        CourtLine("center_service_line", (0, -service_y, 0), (0, service_y, 0)),
        CourtLine(
            "near_center_mark",
            (0, -half_len, 0),
            (0, -half_len + CENTER_MARK_LENGTH_M, 0),
        ),
        CourtLine(
            "far_center_mark",
            (0, half_len, 0),
            (0, half_len - CENTER_MARK_LENGTH_M, 0),
        ),
    ]
    return lines


def sample_line_points(line: CourtLine, samples: int = 64) -> np.ndarray:
    start = np.asarray(line.start, dtype=np.float32)
    end = np.asarray(line.end, dtype=np.float32)
    weights = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    return start[None, :] * (1.0 - weights) + end[None, :] * weights
