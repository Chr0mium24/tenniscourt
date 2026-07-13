from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

import cv2
import numpy as np

from tenniscourt.camera import default_intrinsics
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

    running = True
    frames = 0
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
        image, _, label = _render_state(intrinsics, state)
        screen.blit(_image_surface(image), (0, 0))
        _draw_status(screen, state, len(label["lines"]))
        pygame.display.flip()
        frames += 1
        if args.max_frames is not None and frames >= args.max_frames:
            running = False

    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    pygame.quit()


def _run_headless(args: argparse.Namespace, state: ViewerState) -> None:
    intrinsics = default_intrinsics(args.width, args.height, args.fov_deg, "pinhole")
    image = None
    label = None
    frames = args.max_frames or 1
    for _ in range(frames):
        image, _, label = _render_state(intrinsics, state)
        state.position += _forward_vector(state.yaw_deg) * (state.speed_mps / max(args.fps, 1))

    if args.save_frame is not None and image is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_frame), image)

    visible = len(label["lines"]) if label else 0
    print(f"frames={frames} visible_lines={visible} position={state.position.round(3).tolist()}")


def _render_state(
    intrinsics: object,
    state: ViewerState,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    direction = _look_direction(state.yaw_deg, state.pitch_deg)
    target = state.position + direction
    return render_camera_view(intrinsics, state.position, target)


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


def _draw_status(screen: pygame.Surface, state: ViewerState, visible_lines: int) -> None:
    font = pygame.font.SysFont("monospace", 16)
    text = (
        f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f}) "
        f"yaw={state.yaw_deg:.1f} pitch={state.pitch_deg:.1f} lines={visible_lines}"
    )
    surface = font.render(text, True, (245, 245, 245))
    screen.blit(surface, (12, 10))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive OpenCV-rendered tennis court camera viewer.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov-deg", type=float, default=105.0)
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
