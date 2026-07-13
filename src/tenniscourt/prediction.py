from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tenniscourt.keypoints import keypoint_names
from tenniscourt.model import TinyUNet


@dataclass
class KeypointPrediction:
    xy: np.ndarray
    peak_scores: np.ndarray
    visibility_probs: np.ndarray | None
    selection_scores: np.ndarray
    selected: np.ndarray
    inference_ms: float
    device: str


class CourtKeypointPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
        require_cuda: bool = False,
        base_channels: int = 16,
        selection_score: str = "combined",
        score_threshold: float = 0.5,
        subpixel: bool = True,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.base_channels = base_channels
        self.selection_score = selection_score
        self.score_threshold = score_threshold
        self.subpixel = subpixel
        self.warning: str | None = None
        self._torch = _import_torch()
        self.device = _select_device(self._torch, device_name, require_cuda)
        self.model: TinyUNet | None = None
        self._mtime_ns: int | None = None
        self._has_trained_visibility = False
        self.reload()

    @property
    def has_trained_visibility(self) -> bool:
        return self._has_trained_visibility

    def reload(self) -> None:
        checkpoint = _torch_load(self._torch, self.checkpoint_path, self.device)
        checkpoint_args = checkpoint.get("args", {})
        base_channels = self.base_channels
        if isinstance(checkpoint_args, dict) and "base_channels" in checkpoint_args:
            base_channels = int(checkpoint_args["base_channels"])

        model = TinyUNet(base_channels=base_channels, keypoint_channels=len(keypoint_names()))
        model = model.to(self.device)
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()
        self.model = model
        self.base_channels = base_channels
        self._mtime_ns = self.checkpoint_path.stat().st_mtime_ns
        self._has_trained_visibility = not any(
            key.startswith("visibility_head") for key in incompatible.missing_keys
        )
        self.warning = None
        if incompatible.missing_keys or incompatible.unexpected_keys:
            self.warning = (
                f"partial checkpoint load missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def reload_if_changed(self) -> bool:
        try:
            mtime_ns = self.checkpoint_path.stat().st_mtime_ns
            if self._mtime_ns == mtime_ns:
                return False
            self.reload()
            return True
        except Exception as exc:  # pragma: no cover - defensive for live checkpoint writes
            self.warning = f"checkpoint reload failed: {exc}"
            return False

    def predict(self, image_bgr: np.ndarray) -> KeypointPrediction:
        if self.model is None:
            raise RuntimeError("model is not loaded")

        start = time.perf_counter()
        tensor = _image_tensor(self._torch, image_bgr, self.device)
        with self._torch.inference_mode():
            outputs = self.model(tensor)
            heatmaps = self._torch.sigmoid(outputs["keypoints"])[0].detach().cpu().numpy()
            visibility_probs = None
            if self._has_trained_visibility:
                visibility_probs = self._torch.sigmoid(outputs["visibility"])[0].detach().cpu().numpy()

        xy, peak_scores = decode_heatmap_peaks(heatmaps, self.subpixel)
        selection_scores = _selection_scores(peak_scores, visibility_probs, self.selection_score)
        selected = selection_scores >= self.score_threshold
        inference_ms = (time.perf_counter() - start) * 1000.0
        return KeypointPrediction(
            xy=xy,
            peak_scores=peak_scores,
            visibility_probs=visibility_probs,
            selection_scores=selection_scores,
            selected=selected,
            inference_ms=inference_ms,
            device=str(self.device),
        )


def decode_heatmap_peaks(heatmaps: np.ndarray, subpixel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    channels, height, width = heatmaps.shape
    xy = np.zeros((channels, 2), dtype=np.float64)
    scores = np.zeros((channels,), dtype=np.float64)
    for channel in range(channels):
        heatmap = heatmaps[channel]
        index = int(np.argmax(heatmap))
        y, x = divmod(index, width)
        scores[channel] = float(heatmap[y, x])
        if subpixel and 0 < x < width - 1 and 0 < y < height - 1:
            dx = _quadratic_offset(float(heatmap[y, x - 1]), float(heatmap[y, x]), float(heatmap[y, x + 1]))
            dy = _quadratic_offset(float(heatmap[y - 1, x]), float(heatmap[y, x]), float(heatmap[y + 1, x]))
            xy[channel] = [x + dx, y + dy]
        else:
            xy[channel] = [x, y]
    return xy, scores


def _selection_scores(
    peak_scores: np.ndarray,
    visibility_probs: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    if mode == "peak" or visibility_probs is None:
        return peak_scores
    if mode == "visibility":
        return visibility_probs
    if mode == "combined":
        return peak_scores * visibility_probs
    raise ValueError(f"unsupported selection score: {mode}")


def _quadratic_offset(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-8:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def _image_tensor(torch: object, image_bgr: np.ndarray, device: object) -> object:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()[None, :, :, :]
    return tensor.to(device, non_blocking=True)


def _select_device(torch: object, device_name: str, require_cuda: bool) -> object:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if require_cuda and not str(device_name).startswith("cuda"):
        raise SystemExit("--require-cuda was set, but the selected device is not CUDA")
    if require_cuda and not torch.cuda.is_available():
        raise SystemExit("--require-cuda was set, but torch.cuda.is_available() is false")
    return torch.device(device_name)


def _torch_load(torch: object, path: Path, device: object) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location=device)


def _import_torch() -> object:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install training dependencies first: uv sync --extra train") from exc
    return torch
