from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tenniscourt.camera import default_intrinsics
from tenniscourt.config import GenerateSettings, load_generate_settings, override_settings
from tenniscourt.render import RenderBounds, render_sample


def main() -> None:
    args = _parse_args()
    settings = override_settings(load_generate_settings(args.config), args)
    _validate(settings)
    generate_dataset(settings)


def generate_dataset(settings: GenerateSettings) -> None:
    rng = np.random.default_rng(settings.seed)
    intrinsics = default_intrinsics(settings.width, settings.height, settings.fov_deg, settings.camera_model)
    bounds = RenderBounds(height_min_m=settings.height_min_m, height_max_m=settings.height_max_m)
    images_dir, masks_dir, labels_dir = _prepare_dirs(settings.out)

    metadata = {
        "count": settings.count,
        "width": settings.width,
        "height": settings.height,
        "seed": settings.seed,
        "camera_model": settings.camera_model,
        "fov_deg": settings.fov_deg,
        "supersample": settings.supersample,
    }
    (settings.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for index in range(settings.count):
        image, mask, label = render_sample(rng, intrinsics, bounds, supersample=settings.supersample)
        stem = f"{index:06d}"
        cv2.imwrite(str(images_dir / f"{stem}.png"), image)
        cv2.imwrite(str(masks_dir / f"{stem}.png"), mask)
        (labels_dir / f"{stem}.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
        if (index + 1) % max(1, settings.count // 10) == 0 or index == settings.count - 1:
            print(f"generated {index + 1}/{settings.count}")


def _prepare_dirs(out: Path) -> tuple[Path, Path, Path]:
    images_dir = out / "images"
    masks_dir = out / "masks"
    labels_dir = out / "labels"
    for directory in [out, images_dir, masks_dir, labels_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir, labels_dir


def _validate(settings: GenerateSettings) -> None:
    if settings.count <= 0:
        raise ValueError("count must be positive")
    if settings.width <= 0 or settings.height <= 0:
        raise ValueError("width and height must be positive")
    if settings.camera_model not in {"pinhole", "fisheye"}:
        raise ValueError("camera_model must be 'pinhole' or 'fisheye'")
    if settings.height_min_m <= 0 or settings.height_max_m <= settings.height_min_m:
        raise ValueError("invalid camera height range")
    if settings.supersample <= 0:
        raise ValueError("supersample must be positive")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic tennis court line images.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--supersample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fov-deg", type=float, default=None)
    parser.add_argument("--camera-model", choices=["pinhole", "fisheye"], default=None)
    parser.add_argument("--height-min-m", type=float, default=None)
    parser.add_argument("--height-max-m", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
