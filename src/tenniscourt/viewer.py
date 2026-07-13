from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import cv2
import numpy as np

from tenniscourt.camera import default_intrinsics
from tenniscourt.keypoints import keypoint_names
from tenniscourt.render import render_camera_view

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install viewer dependencies first: uv sync --extra viewer") from exc


@dataclass
class ViewerState:
    position: np.ndarray
    yaw_deg: float
    pitch_deg: float
    speed_mps: float
    mouse_sensitivity: float


def main() -> None:
    args = _parse_args()
    state = ViewerState(
        position=np.array([args.start_x, args.start_y, args.start_z], dtype=np.float64),
        yaw_deg=args.yaw_deg,
        pitch_deg=args.pitch_deg,
        speed_mps=args.speed,
        mouse_sensitivity=args.mouse_sensitivity,
    )
    if args.headless:
        _run_headless(args, state)
    else:
        _run_window(args, state)


def _run_window(args: argparse.Namespace, state: ViewerState) -> None:
    pygame.init()
    pygame.display.set_caption("Tennis Court Camera Viewer")
    screen = pygame.display.set_mode((args.width, args.height))
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(not args.no_grab)
    intrinsics = default_intrinsics(args.width, args.height, args.fov_deg, "pinhole")
    predictor = _make_predictor(args)
    prediction = None

    running = True
    frames = 0
    predict_every = max(args.predict_every, 1)
    while running:
        dt = clock.tick(args.fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                _apply_mouse_delta(state, event.rel[0], event.rel[1])

        _apply_keyboard(state, dt)
        image, _, label = _render_state(intrinsics, state, args.supersample)
        if predictor is not None and frames % predict_every == 0:
            if args.reload_checkpoint:
                predictor.reload_if_changed()
            prediction = predictor.predict(image)
        if prediction is not None:
            _draw_prediction_overlay(image, prediction, args)
        screen.blit(_image_surface(image), (0, 0))
        _draw_status(screen, state, len(label["lines"]), prediction, predictor)
        pygame.display.flip()
        frames += 1
        if args.max_frames is not None and frames >= args.max_frames:
            running = False

    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    pygame.quit()


def _run_headless(args: argparse.Namespace, state: ViewerState) -> None:
    intrinsics = default_intrinsics(args.width, args.height, args.fov_deg, "pinhole")
    predictor = _make_predictor(args)
    image = None
    label = None
    prediction = None
    frames = args.max_frames or 1
    for _ in range(frames):
        image, _, label = _render_state(intrinsics, state, args.supersample)
        if predictor is not None:
            if args.reload_checkpoint:
                predictor.reload_if_changed()
            prediction = predictor.predict(image)
            _draw_prediction_overlay(image, prediction, args)
        state.position += _forward_vector(state.yaw_deg) * (state.speed_mps / max(args.fps, 1))

    if args.save_frame is not None and image is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_frame), image)

    visible = len(label["lines"]) if label else 0
    status = f"frames={frames} visible_lines={visible} position={state.position.round(3).tolist()}"
    if prediction is not None:
        status += f" selected_keypoints={int(prediction.selected.sum())} infer_ms={prediction.inference_ms:.1f}"
    print(status)


def _render_state(
    intrinsics: object,
    state: ViewerState,
    supersample: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    direction = _look_direction(state.yaw_deg, state.pitch_deg)
    target = state.position + direction
    return render_camera_view(intrinsics, state.position, target, supersample=supersample)


def _apply_keyboard(state: ViewerState, dt: float) -> None:
    keys = pygame.key.get_pressed()
    movement = np.zeros(3, dtype=np.float64)
    forward = _forward_vector(state.yaw_deg)
    right = _right_vector(state.yaw_deg)

    if keys[pygame.K_w]:
        movement += forward
    if keys[pygame.K_s]:
        movement -= forward
    if keys[pygame.K_d]:
        movement += right
    if keys[pygame.K_a]:
        movement -= right
    if keys[pygame.K_SPACE]:
        movement[2] += 1.0
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        movement[2] -= 1.0

    norm = np.linalg.norm(movement)
    if norm > 1e-6:
        state.position += movement / norm * state.speed_mps * dt
        state.position[2] = max(0.15, state.position[2])


def _apply_mouse_delta(state: ViewerState, dx: int, dy: int) -> None:
    state.yaw_deg += dx * state.mouse_sensitivity
    state.pitch_deg -= dy * state.mouse_sensitivity
    state.pitch_deg = float(np.clip(state.pitch_deg, -85.0, 85.0))


def _look_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = radians(yaw_deg)
    pitch = radians(pitch_deg)
    return np.array([sin(yaw) * cos(pitch), cos(yaw) * cos(pitch), sin(pitch)], dtype=np.float64)


def _forward_vector(yaw_deg: float) -> np.ndarray:
    yaw = radians(yaw_deg)
    return np.array([sin(yaw), cos(yaw), 0.0], dtype=np.float64)


def _right_vector(yaw_deg: float) -> np.ndarray:
    yaw = radians(yaw_deg)
    return np.array([cos(yaw), -sin(yaw), 0.0], dtype=np.float64)


def _image_surface(image_bgr: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")


def _make_predictor(args: argparse.Namespace) -> object | None:
    if args.checkpoint is None:
        return None
    from tenniscourt.prediction import CourtKeypointPredictor

    return CourtKeypointPredictor(
        args.checkpoint,
        device_name=args.device,
        require_cuda=args.require_cuda,
        base_channels=args.base_channels,
        selection_score=args.selection_score,
        score_threshold=args.score_threshold,
        subpixel=args.subpixel,
    )


def _draw_prediction_overlay(image: np.ndarray, prediction: object, args: argparse.Namespace) -> None:
    names = keypoint_names()
    for index, (xy, score, selected) in enumerate(
        zip(prediction.xy, prediction.selection_scores, prediction.selected, strict=True)
    ):
        if not args.show_all_keypoints and not selected:
            continue
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            continue
        color = (50, 255, 80) if selected else (110, 110, 110)
        radius = 5 if selected else 3
        cv2.circle(image, (x, y), radius + 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        if args.overlay_labels:
            text = f"{index}:{score:.2f}" if args.short_labels else f"{names[index]} {score:.2f}"
            cv2.putText(image, text, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _draw_status(
    screen: pygame.Surface,
    state: ViewerState,
    visible_lines: int,
    prediction: object | None = None,
    predictor: object | None = None,
) -> None:
    font = pygame.font.SysFont("monospace", 16)
    lines = [
        f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f}) "
        f"yaw={state.yaw_deg:.1f} pitch={state.pitch_deg:.1f} lines={visible_lines}"
    ]
    if prediction is not None:
        lines.append(
            f"model={prediction.device} keypoints={int(prediction.selected.sum())}/14 "
            f"infer={prediction.inference_ms:.1f}ms"
        )
    if predictor is not None and getattr(predictor, "warning", None):
        lines.append(str(predictor.warning)[:120])

    for row, text in enumerate(lines):
        surface = font.render(text, True, (245, 245, 245))
        screen.blit(surface, (12, 10 + row * 20))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive OpenCV-rendered tennis court camera viewer.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov-deg", type=float, default=105.0)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--mouse-sensitivity", type=float, default=0.08)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=-13.0)
    parser.add_argument("--start-z", type=float, default=0.4)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=-2.0)
    parser.add_argument("--no-grab", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-frame", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--predict-every", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--selection-score", choices=["peak", "visibility", "combined"], default="combined")
    parser.add_argument("--show-all-keypoints", action="store_true")
    parser.add_argument("--short-labels", action="store_true")
    parser.add_argument("--no-overlay-labels", dest="overlay_labels", action="store_false")
    parser.add_argument("--reload-checkpoint", action="store_true")
    parser.add_argument("--no-subpixel", dest="subpixel", action="store_false")
    parser.set_defaults(overlay_labels=True, subpixel=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
