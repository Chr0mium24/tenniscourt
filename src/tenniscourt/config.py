from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerateSettings:
    out: Path = Path("outputs/synth")
    count: int = 1000
    width: int = 640
    height: int = 360
    seed: int = 7
    fov_deg: float = 105.0
    camera_model: str = "pinhole"
    height_min_m: float = 0.35
    height_max_m: float = 0.45


def load_generate_settings(config_path: Path | None) -> GenerateSettings:
    if config_path is None:
        return GenerateSettings()

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    image = data.get("image", {})
    camera = data.get("camera", {})
    output = data.get("output", {})
    return GenerateSettings(
        out=Path(output.get("dir", "outputs/synth")),
        count=int(image.get("count", 1000)),
        width=int(image.get("width", 640)),
        height=int(image.get("height", 360)),
        seed=int(output.get("seed", 7)),
        fov_deg=float(camera.get("fov_deg", 105.0)),
        camera_model=str(camera.get("model", "pinhole")),
        height_min_m=float(camera.get("height_min_m", 0.35)),
        height_max_m=float(camera.get("height_max_m", 0.45)),
    )


def override_settings(settings: GenerateSettings, args: object) -> GenerateSettings:
    values = settings.__dict__.copy()
    for key in values:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    return GenerateSettings(**values)
