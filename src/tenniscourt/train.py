from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install training dependencies first: uv sync --extra train") from exc

from tenniscourt.data import LineMaskDataset, list_image_mask_pairs, split_pairs
from tenniscourt.keypoints import keypoint_names
from tenniscourt.model import TinyUNet, dice_loss, heatmap_peak_error, mask_iou


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device, args.require_cuda)
    train(args, device)


def train(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = list_image_mask_pairs(args.data)
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    image_size = (args.width, args.height) if args.width and args.height else None

    train_loader = DataLoader(
        LineMaskDataset(train_pairs, image_size=image_size, heatmap_sigma=args.heatmap_sigma),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        LineMaskDataset(val_pairs, image_size=image_size, heatmap_sigma=args.heatmap_sigma),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    ) if val_pairs else None

    model = TinyUNet(base_channels=args.base_channels, keypoint_channels=len(keypoint_names())).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_iou = -1.0

    metrics_path = args.out / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, bce, scaler, device, use_amp, args)
        val_iou, val_kp_error = _validate(model, val_loader, device) if val_loader else (0.0, 0.0)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_iou": val_iou,
            "val_keypoint_peak_error_px": val_kp_error,
            "device": str(device),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        checkpoint = {"model": model.state_dict(), "args": vars(args), "metrics": row}
        torch.save(checkpoint, args.out / "last.pt")
        if val_iou >= best_iou:
            best_iou = val_iou
            torch.save(checkpoint, args.out / "best.pt")
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_iou={val_iou:.4f} val_kp_px={val_kp_error:.2f} device={device}"
        )


def _train_epoch(
    model: TinyUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    iterator = tqdm(loader, desc="train", leave=False)
    for images, masks, heatmaps in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            mask_logits = outputs["mask"]
            heatmap_logits = outputs["keypoints"]
            mask_loss = bce(mask_logits, masks) + dice_loss(mask_logits, masks)
            heatmap_loss = bce(heatmap_logits, heatmaps)
            loss = mask_loss + args.heatmap_loss_weight * heatmap_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        steps += 1
        iterator.set_postfix(loss=total_loss / steps)
        if args.max_steps is not None and steps >= args.max_steps:
            break
    return total_loss / max(steps, 1)


@torch.no_grad()
def _validate(model: TinyUNet, loader: DataLoader | None, device: torch.device) -> tuple[float, float]:
    if loader is None:
        return 0.0, 0.0
    model.eval()
    total_iou = 0.0
    total_kp_error = 0.0
    steps = 0
    for images, masks, heatmaps in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        outputs = model(images)
        total_iou += float(mask_iou(outputs["mask"], masks).cpu())
        total_kp_error += float(heatmap_peak_error(outputs["keypoints"], heatmaps).cpu())
        steps += 1
    return total_iou / max(steps, 1), total_kp_error / max(steps, 1)


def _select_device(name: str, require_cuda: bool) -> torch.device:
    if name == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = name
    if require_cuda and selected != "cuda":
        raise RuntimeError("CUDA is required but not available")
    device = torch.device(selected)
    if require_cuda and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but PyTorch cannot access it")
    return device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small line-mask segmentation model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/line-seg"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--heatmap-loss-weight", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
