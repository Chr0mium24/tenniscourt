from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
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
    best_score = _initial_best_score(args.best_metric)
    start_epoch = 1
    if args.resume is not None:
        start_epoch, best_score = _load_checkpoint(args.resume, model, optimizer, scaler, device, args)

    metrics_path = args.out / "metrics.jsonl"
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, mask_loss, heatmap_loss = _train_epoch(model, train_loader, optimizer, bce, scaler, device, use_amp, args)
        val_iou, val_kp_error = _validate(model, val_loader, device) if val_loader else (0.0, 0.0)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "mask_loss": mask_loss,
            "heatmap_loss": heatmap_loss,
            "val_iou": val_iou,
            "val_keypoint_peak_error_px": val_kp_error,
            "device": str(device),
            "heatmap_loss_name": args.heatmap_loss,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": _serializable_args(args),
            "metrics": row,
            "best_metric": args.best_metric,
            "best_score": best_score,
        }
        torch.save(checkpoint, args.out / "last.pt")
        score = row[args.best_metric]
        if _is_better(score, best_score, args.best_metric):
            best_score = score
            checkpoint["best_score"] = best_score
            torch.save(checkpoint, args.out / "best.pt")
        if args.viz_count > 0 and val_loader is not None:
            viz_dir = (args.viz_dir or args.out / "viz") / f"epoch_{epoch:04d}"
            _write_keypoint_visualizations(model, val_loader, device, viz_dir, args.viz_count)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"mask_loss={mask_loss:.4f} heatmap_loss={heatmap_loss:.4f} "
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
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_mask_loss = 0.0
    total_heatmap_loss = 0.0
    steps = 0
    iterator = tqdm(loader, desc="train", leave=False)
    for images, masks, heatmaps, keypoint_visible in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        keypoint_visible = keypoint_visible.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            mask_logits = outputs["mask"]
            heatmap_logits = outputs["keypoints"]
            mask_loss = bce(mask_logits, masks) + dice_loss(mask_logits, masks)
            heatmap_loss = _heatmap_loss(heatmap_logits, heatmaps, keypoint_visible, args)
            loss = mask_loss + args.heatmap_loss_weight * heatmap_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        total_mask_loss += float(mask_loss.detach().cpu())
        total_heatmap_loss += float(heatmap_loss.detach().cpu())
        steps += 1
        iterator.set_postfix(loss=total_loss / steps)
        if args.max_steps is not None and steps >= args.max_steps:
            break
    return (
        total_loss / max(steps, 1),
        total_mask_loss / max(steps, 1),
        total_heatmap_loss / max(steps, 1),
    )


@torch.no_grad()
def _validate(model: TinyUNet, loader: DataLoader | None, device: torch.device) -> tuple[float, float]:
    if loader is None:
        return 0.0, 0.0
    model.eval()
    total_iou = 0.0
    total_kp_error = 0.0
    steps = 0
    for images, masks, heatmaps, _keypoint_visible in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        outputs = model(images)
        total_iou += float(mask_iou(outputs["mask"], masks).cpu())
        total_kp_error += float(heatmap_peak_error(outputs["keypoints"], heatmaps).cpu())
        steps += 1
    return total_iou / max(steps, 1), total_kp_error / max(steps, 1)


def _heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    visible: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    visibility = visible[:, :, None, None].to(dtype=targets.dtype)
    weights = (1.0 + (args.heatmap_pos_weight - 1.0) * targets) * visibility
    denom = weights.sum().clamp_min(1.0)
    if args.heatmap_loss == "weighted-mse":
        loss = (torch.sigmoid(logits) - targets).square()
    elif args.heatmap_loss == "weighted-bce":
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    else:  # pragma: no cover
        raise ValueError(f"unsupported heatmap loss: {args.heatmap_loss}")
    return (loss * weights).sum() / denom


def _load_checkpoint(
    path: Path,
    model: TinyUNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[int, float]:
    checkpoint = _torch_load(path, device)
    model.load_state_dict(checkpoint["model"])
    if not args.reset_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if not args.reset_optimizer and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    last_epoch = int(checkpoint.get("epoch") or checkpoint.get("metrics", {}).get("epoch", 0))
    best_score = _initial_best_score(args.best_metric)
    if checkpoint.get("best_metric") == args.best_metric:
        best_score = float(checkpoint.get("best_score", best_score))
    print(f"resumed checkpoint={path} start_epoch={last_epoch + 1}")
    return last_epoch + 1, best_score


def _torch_load(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location=device)


def _initial_best_score(metric: str) -> float:
    return float("inf") if metric == "val_keypoint_peak_error_px" else -float("inf")


def _is_better(score: float, best_score: float, metric: str) -> bool:
    return score <= best_score if metric == "val_keypoint_peak_error_px" else score >= best_score


def _serializable_args(args: argparse.Namespace) -> dict[str, object]:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


@torch.no_grad()
def _write_keypoint_visualizations(
    model: TinyUNet,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_images: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    written = 0
    for images, _masks, heatmaps, visible in loader:
        outputs = model(images.to(device, non_blocking=True))
        pred_heatmaps = torch.sigmoid(outputs["keypoints"]).cpu()
        for index in range(images.shape[0]):
            canvas = _draw_keypoint_overlay(images[index], pred_heatmaps[index], heatmaps[index], visible[index])
            cv2.imwrite(str(out_dir / f"sample_{written:03d}.png"), canvas)
            written += 1
            if written >= max_images:
                return


def _draw_keypoint_overlay(
    image: torch.Tensor,
    predicted: torch.Tensor,
    target: torch.Tensor,
    visible: torch.Tensor,
) -> np.ndarray:
    rgb = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pred_xy = _peak_xy(predicted)
    target_xy = _peak_xy(target)
    for channel, is_visible in enumerate(visible.numpy() > 0.5):
        if not is_visible:
            continue
        px, py = pred_xy[channel]
        tx, ty = target_xy[channel]
        cv2.circle(canvas, (tx, ty), 4, (0, 0, 255), -1)
        cv2.circle(canvas, (px, py), 5, (0, 255, 0), 1)
        cv2.line(canvas, (tx, ty), (px, py), (0, 255, 255), 1)
        cv2.putText(canvas, str(channel), (px + 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(canvas, "GT red / pred green", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def _peak_xy(heatmaps: torch.Tensor) -> list[tuple[int, int]]:
    flat = heatmaps.flatten(1)
    indices = flat.argmax(dim=1)
    width = heatmaps.shape[-1]
    xs = (indices % width).tolist()
    ys = (indices // width).tolist()
    return [(int(x), int(y)) for x, y in zip(xs, ys, strict=True)]


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
    parser.add_argument("--heatmap-loss", choices=["weighted-mse", "weighted-bce"], default="weighted-mse")
    parser.add_argument("--heatmap-loss-weight", type=float, default=5.0)
    parser.add_argument("--heatmap-pos-weight", type=float, default=50.0)
    parser.add_argument("--best-metric", choices=["val_iou", "val_keypoint_peak_error_px"], default="val_keypoint_peak_error_px")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--viz-count", type=int, default=4)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
