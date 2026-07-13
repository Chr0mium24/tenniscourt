from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from tenniscourt.keypoints import heatmaps_from_keypoints, project_keypoints_from_label, refine_keypoint_visibility_with_mask


class LineMaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pairs: list[tuple[Path, Path, Path]],
        image_size: tuple[int, int] | None = None,
        heatmap_sigma: float = 3.0,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, mask_path, label_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        if mask is None:
            raise FileNotFoundError(mask_path)

        if self.image_size is not None:
            width, height = self.image_size
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            height, width = mask.shape

        label = json.loads(label_path.read_text(encoding="utf-8"))
        keypoints = _keypoints_from_label(label, width, height)
        keypoints = refine_keypoint_visibility_with_mask(keypoints, mask)
        heatmaps = heatmaps_from_keypoints(keypoints, width=width, height=height, sigma_px=self.heatmap_sigma)
        visible = np.asarray([bool(keypoint.get("visible", False)) for keypoint in keypoints], dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0) > 0.5
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, :, :]
        heatmap_tensor = torch.from_numpy(heatmaps).contiguous()
        visible_tensor = torch.from_numpy(visible)
        return image_tensor, mask_tensor, heatmap_tensor, visible_tensor


def list_image_mask_pairs(data_dir: Path) -> list[tuple[Path, Path, Path]]:
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    labels_dir = data_dir / "labels"
    image_paths = sorted(images_dir.glob("*.png"))
    pairs = [(image_path, masks_dir / image_path.name, labels_dir / f"{image_path.stem}.json") for image_path in image_paths]
    missing = [path for _, mask_path, label_path in pairs for path in [mask_path, label_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dataset file: {missing[0]}")
    if not pairs:
        raise FileNotFoundError(f"no PNG images found in {images_dir}")
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path, Path]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path, Path]], list[tuple[Path, Path, Path]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(pairs) * val_ratio))) if len(pairs) > 1 else 0
    val_indices = set(indices[:val_count].tolist())
    train = [pair for idx, pair in enumerate(pairs) if idx not in val_indices]
    val = [pair for idx, pair in enumerate(pairs) if idx in val_indices]
    if not train and val:
        train, val = val, []
    return train, val


def _heatmaps_from_label(label: dict[str, object], width: int, height: int, sigma_px: float) -> np.ndarray:
    keypoints = _keypoints_from_label(label, width, height)
    return heatmaps_from_keypoints(keypoints, width=width, height=height, sigma_px=sigma_px)


def _keypoints_from_label(label: dict[str, object], width: int, height: int) -> list[dict[str, object]]:
    keypoints = label.get("keypoints")
    if keypoints is None:
        keypoints = project_keypoints_from_label(label)
    if label["camera"]["width"] != width or label["camera"]["height"] != height:
        keypoints = _rescale_keypoints(keypoints, label["camera"]["width"], label["camera"]["height"], width, height)
    return keypoints


def _rescale_keypoints(
    keypoints: list[dict[str, object]],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> list[dict[str, object]]:
    sx = dst_width / src_width
    sy = dst_height / src_height
    scaled = []
    for keypoint in keypoints:
        x, y = keypoint["xy"]
        scaled.append({**keypoint, "xy": [float(x) * sx, float(y) * sy]})
    return scaled
