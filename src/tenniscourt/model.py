from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, base_channels: int = 16, keypoint_channels: int = 14) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.enc1 = ConvBlock(3, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)
        self.mask_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.keypoint_head = nn.Conv2d(c1, keypoint_channels, kernel_size=1)
        self.visibility_pool = nn.AdaptiveAvgPool2d(1)
        self.visibility_head = nn.Linear(c1, keypoint_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        visibility_features = self.visibility_pool(d1).flatten(1)
        return {
            "mask": self.mask_head(d1),
            "keypoints": self.keypoint_head(d1),
            "visibility": self.visibility_head(visibility_features),
        }


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def mask_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    preds = torch.sigmoid(logits) > threshold
    targets_bool = targets > 0.5
    intersection = (preds & targets_bool).sum(dim=(1, 2, 3)).float()
    union = (preds | targets_bool).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
    return (intersection / union).mean()


@torch.no_grad()
def heatmap_peak_error(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits).flatten(2)
    target = targets.flatten(2)
    pred_idx = pred.argmax(dim=2)
    target_idx = target.argmax(dim=2)
    width = logits.shape[-1]
    pred_x = pred_idx % width
    pred_y = pred_idx // width
    target_x = target_idx % width
    target_y = target_idx // width
    visible = target.amax(dim=2) > 0.1
    distance = torch.sqrt((pred_x - target_x).float() ** 2 + (pred_y - target_y).float() ** 2)
    return distance[visible].mean() if visible.any() else torch.tensor(0.0, device=logits.device)
