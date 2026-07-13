from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LineMaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: tuple[int, int] | None = None) -> None:
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) / 255.0) > 0.5
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, :, :]
        return image_tensor, mask_tensor


def list_image_mask_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    image_paths = sorted(images_dir.glob("*.png"))
    pairs = [(image_path, masks_dir / image_path.name) for image_path in image_paths]
    missing = [mask_path for _, mask_path in pairs if not mask_path.exists()]
    if missing:
        raise FileNotFoundError(f"missing mask for {missing[0]}")
    if not pairs:
        raise FileNotFoundError(f"no PNG images found in {images_dir}")
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
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
