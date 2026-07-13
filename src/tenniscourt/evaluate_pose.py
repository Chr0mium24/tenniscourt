from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install training dependencies first: uv sync --extra train") from exc

from tenniscourt.camera import CameraIntrinsics, project_points
from tenniscourt.data import LineMaskDataset, list_image_mask_pairs, split_pairs
from tenniscourt.keypoints import court_keypoints, keypoint_names, project_keypoints_from_label
from tenniscourt.model import TinyUNet


@dataclass(frozen=True)
class PoseResult:
    ok: bool
    mode: str
    used_points: int
    inliers: int
    reject_reason: str | None
    reproj_error_px: float | None
    position_error_m: float | None
    rotation_error_deg: float | None


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device, args.require_cuda)
    summary = evaluate(args, device)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def evaluate(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = _select_pairs(args)
    dataset = LineMaskDataset(pairs, image_size=_image_size(args), heatmap_sigma=args.heatmap_sigma)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = _load_model(args.checkpoint, args.base_channels, device)
    rows = _evaluate_batches(model, loader, pairs, args, device)
    _write_jsonl(args.out / "pose_eval.jsonl", rows)
    summary = _summarize(rows, args)
    (args.out / "pose_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def _evaluate_batches(
    model: TinyUNet,
    loader: DataLoader,
    pairs: list[tuple[Path, Path, Path]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    keypoints_3d = np.asarray([kp.point for kp in court_keypoints()], dtype=np.float64)
    offset = 0
    for images, _masks, _heatmaps, _visible in loader:
        outputs = model(images.to(device, non_blocking=True))
        heatmaps = torch.sigmoid(outputs["keypoints"]).cpu().numpy()
        visibility_probs = torch.sigmoid(outputs["visibility"]).cpu().numpy() if "visibility" in outputs else None
        for item in range(images.shape[0]):
            label_path = pairs[offset + item][2]
            row = _evaluate_sample(
                label_path,
                heatmaps[item],
                None if visibility_probs is None else visibility_probs[item],
                keypoints_3d,
                args,
            )
            rows.append(row)
            if args.max_samples is not None and len(rows) >= args.max_samples:
                return rows
        offset += images.shape[0]
    return rows


def _evaluate_sample(
    label_path: Path,
    heatmaps: np.ndarray,
    visibility_probs: np.ndarray | None,
    keypoints_3d: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    label = json.loads(label_path.read_text(encoding="utf-8"))
    decoded_xy, scores = decode_heatmap_peaks(heatmaps, subpixel=args.subpixel)
    gt_keypoints = label.get("keypoints") or project_keypoints_from_label(label)
    gt_xy = np.asarray([kp["xy"] for kp in gt_keypoints], dtype=np.float64)
    gt_visible = np.asarray([bool(kp.get("visible", False)) for kp in gt_keypoints], dtype=bool)
    all_errors = np.linalg.norm(decoded_xy - gt_xy, axis=1)
    visible_errors = all_errors[gt_visible]
    camera = _camera_from_label(label)
    gt_rvec = np.asarray(label["rvec"], dtype=np.float64).reshape(3, 1)
    gt_tvec = np.asarray(label["tvec"], dtype=np.float64).reshape(3, 1)
    selection_scores = _selection_scores(scores, visibility_probs, args.selection_score)

    modes = {}
    if args.pnp_mode in {"oracle-visible", "both"}:
        modes["oracle_visible"] = _solve_pose(
            keypoints_3d,
            decoded_xy,
            gt_visible,
            "oracle_visible",
            camera,
            gt_rvec,
            gt_tvec,
            args,
        )
    if args.pnp_mode in {"score-gated", "both"}:
        selected = selection_scores >= args.peak_threshold
        modes["score_gated"] = _solve_pose(
            keypoints_3d,
            decoded_xy,
            selected,
            "score_gated",
            camera,
            gt_rvec,
            gt_tvec,
            args,
        )

    return {
        "label": str(label_path),
        "visible_keypoints": int(gt_visible.sum()),
        "score_selected_keypoints": int((selection_scores >= args.peak_threshold).sum()),
        "keypoint_error_mean_px": float(visible_errors.mean()) if visible_errors.size else None,
        "keypoint_error_median_px": float(np.median(visible_errors)) if visible_errors.size else None,
        "keypoint_error_p95_px": float(np.percentile(visible_errors, 95)) if visible_errors.size else None,
        "keypoint_errors_px": [float(error) if visible else None for error, visible in zip(all_errors, gt_visible, strict=True)],
        "peak_scores": [float(score) for score in scores],
        "visibility_probs": None if visibility_probs is None else [float(value) for value in visibility_probs],
        "selection_scores": [float(score) for score in selection_scores],
        "peak_score_mean_visible": float(scores[gt_visible].mean()) if gt_visible.any() else None,
        "peak_score_min_visible": float(scores[gt_visible].min()) if gt_visible.any() else None,
        "pnp": {name: result.__dict__ for name, result in modes.items()},
    }


def _selection_scores(peak_scores: np.ndarray, visibility_probs: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == "peak" or visibility_probs is None:
        return peak_scores
    if mode == "visibility":
        return visibility_probs
    if mode == "combined":
        return peak_scores * visibility_probs
    raise ValueError(f"unsupported selection score: {mode}")


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


def _quadratic_offset(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-8:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def _solve_pose(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    selected: np.ndarray,
    mode: str,
    camera: CameraIntrinsics,
    gt_rvec: np.ndarray,
    gt_tvec: np.ndarray,
    args: argparse.Namespace,
) -> PoseResult:
    indices = np.flatnonzero(selected)
    if len(indices) < args.min_points:
        return PoseResult(False, mode, int(len(indices)), 0, "not_enough_points", None, None, None)
    object_points = points_3d[indices].astype(np.float64)
    image_points = points_2d[indices].astype(np.float64)
    k, d, image_points = _pnp_camera_inputs(camera, image_points)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        k,
        d,
        iterationsCount=args.ransac_iterations,
        reprojectionError=args.ransac_reproj_error,
        confidence=args.ransac_confidence,
        flags=_pnp_flag(args.pnp_solver),
    )
    if not ok or rvec is None or tvec is None:
        return PoseResult(False, mode, int(len(indices)), 0, "pnp_failed", None, None, None)
    inlier_count = 0 if inliers is None else int(len(inliers))
    if inliers is not None and len(inliers) >= args.min_points:
        inlier_object = object_points[inliers.reshape(-1)]
        inlier_image = image_points[inliers.reshape(-1)]
        refined, rvec, tvec = cv2.solvePnP(inlier_object, inlier_image, k, d, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        ok = bool(refined)
    if not ok:
        return PoseResult(False, mode, int(len(indices)), inlier_count, "refine_failed", None, None, None)
    if not _pose_in_bounds(rvec, tvec, args):
        return PoseResult(False, mode, int(len(indices)), inlier_count, "pose_out_of_bounds", None, None, None)
    reproj_indices = indices if inliers is None or len(inliers) < args.min_points else indices[inliers.reshape(-1)]
    reproj_error = _reprojection_error(camera, points_3d[reproj_indices], points_2d[reproj_indices], rvec, tvec)
    return PoseResult(
        True,
        mode,
        int(len(indices)),
        inlier_count,
        None,
        reproj_error,
        _position_error(gt_rvec, gt_tvec, rvec, tvec),
        _rotation_error_deg(gt_rvec, rvec),
    )


def _pnp_camera_inputs(
    camera: CameraIntrinsics,
    points_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera.model == "fisheye":
        normalized = cv2.fisheye.undistortPoints(points_2d.reshape(-1, 1, 2), camera.k, camera.d).reshape(-1, 2)
        return np.eye(3, dtype=np.float64), np.zeros((4, 1), dtype=np.float64), normalized
    return camera.k, camera.d, points_2d


def _pnp_flag(name: str) -> int:
    flags = {
        "epnp": cv2.SOLVEPNP_EPNP,
        "ippe": cv2.SOLVEPNP_IPPE,
        "iterative": cv2.SOLVEPNP_ITERATIVE,
        "sqpnp": cv2.SOLVEPNP_SQPNP,
    }
    return flags[name]


def _pose_in_bounds(rvec: np.ndarray, tvec: np.ndarray, args: argparse.Namespace) -> bool:
    position = _camera_center(rvec, tvec)
    return bool(
        abs(position[0]) <= args.max_abs_x_m
        and abs(position[1]) <= args.max_abs_y_m
        and args.min_z_m <= position[2] <= args.max_z_m
    )


def _reprojection_error(
    camera: CameraIntrinsics,
    points_3d: np.ndarray,
    observed_2d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected = project_points(points_3d, camera, rvec, tvec)
    return float(np.linalg.norm(projected - observed_2d, axis=1).mean())


def _position_error(gt_rvec: np.ndarray, gt_tvec: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> float:
    return float(np.linalg.norm(_camera_center(rvec, tvec) - _camera_center(gt_rvec, gt_tvec)))


def _camera_center(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    return (-rotation.T @ tvec.reshape(3, 1)).reshape(3)


def _rotation_error_deg(gt_rvec: np.ndarray, rvec: np.ndarray) -> float:
    gt_rotation, _ = cv2.Rodrigues(gt_rvec)
    rotation, _ = cv2.Rodrigues(rvec)
    delta = rotation @ gt_rotation.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _camera_from_label(label: dict[str, object]) -> CameraIntrinsics:
    camera = label["camera"]
    return CameraIntrinsics(
        width=int(camera["width"]),
        height=int(camera["height"]),
        k=np.asarray(camera["k"], dtype=np.float64),
        d=np.asarray(camera["d"], dtype=np.float64).reshape(-1, 1),
        model=str(camera["model"]),
    )


def _load_model(checkpoint_path: Path, base_channels: int, device: torch.device) -> TinyUNet:
    checkpoint = _torch_load(checkpoint_path, device)
    checkpoint_args = checkpoint.get("args", {})
    if isinstance(checkpoint_args, dict) and "base_channels" in checkpoint_args:
        base_channels = int(checkpoint_args["base_channels"])
    model = TinyUNet(base_channels=base_channels, keypoint_channels=len(keypoint_names())).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"partial checkpoint load missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}")
    return model


def _torch_load(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location=device)


def _select_pairs(args: argparse.Namespace) -> list[tuple[Path, Path, Path]]:
    pairs = list_image_mask_pairs(args.data)
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    if args.split == "train":
        selected = train_pairs
    elif args.split == "val":
        selected = val_pairs
    else:
        selected = pairs
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    if not selected:
        raise FileNotFoundError(f"no samples selected for split={args.split}")
    return selected


def _summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    summary: dict[str, object] = {
        "samples": len(rows),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "peak_threshold": args.peak_threshold,
        "selection_score": args.selection_score,
        "pnp_solver": args.pnp_solver,
        "keypoint_error_px": _stats([row["keypoint_error_mean_px"] for row in rows]),
        "visible_keypoints": _stats([row["visible_keypoints"] for row in rows]),
        "score_selected_keypoints": _stats([row["score_selected_keypoints"] for row in rows]),
        "per_keypoint_error_px": _per_keypoint_stats(rows),
    }
    for mode in ["oracle_visible", "score_gated"]:
        mode_rows = [row["pnp"][mode] for row in rows if mode in row["pnp"]]
        if mode_rows:
            ok_rows = [row for row in mode_rows if row["ok"]]
            summary[mode] = {
                "success_rate": len(ok_rows) / max(len(mode_rows), 1),
                "reject_reasons": _reject_reason_counts(mode_rows),
                "used_points": _stats([row["used_points"] for row in mode_rows]),
                "inliers": _stats([row["inliers"] for row in ok_rows]),
                "reproj_error_px": _stats([row["reproj_error_px"] for row in ok_rows]),
                "position_error_m": _stats([row["position_error_m"] for row in ok_rows]),
                "rotation_error_deg": _stats([row["rotation_error_deg"] for row in ok_rows]),
            }
    return summary


def _per_keypoint_stats(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int | None]]:
    names = keypoint_names()
    result = {}
    for index, name in enumerate(names):
        result[name] = _stats([row["keypoint_errors_px"][index] for row in rows])
    return result


def _reject_reason_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row["ok"]:
            continue
        reason = str(row.get("reject_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _stats(values: list[object]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(numeric),
        "mean": mean(numeric),
        "median": median(numeric),
        "p95": float(np.percentile(numeric, 95)),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _image_size(args: argparse.Namespace) -> tuple[int, int] | None:
    return (args.width, args.height) if args.width and args.height else None


def _select_device(name: str, require_cuda: bool) -> torch.device:
    selected = "cuda" if name == "auto" and torch.cuda.is_available() else name
    if name == "auto" and selected != "cuda":
        selected = "cpu"
    if require_cuda and selected != "cuda":
        raise RuntimeError("CUDA is required but not available")
    return torch.device(selected)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode keypoint heatmaps and evaluate court-relative camera pose.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/pose-eval"))
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--peak-threshold", type=float, default=0.7)
    parser.add_argument("--selection-score", choices=["peak", "visibility", "combined"], default="peak")
    parser.add_argument("--pnp-mode", choices=["oracle-visible", "score-gated", "both"], default="both")
    parser.add_argument("--pnp-solver", choices=["epnp", "ippe", "iterative", "sqpnp"], default="ippe")
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--ransac-reproj-error", type=float, default=12.0)
    parser.add_argument("--ransac-iterations", type=int, default=100)
    parser.add_argument("--ransac-confidence", type=float, default=0.99)
    parser.add_argument("--max-abs-x-m", type=float, default=20.0)
    parser.add_argument("--max-abs-y-m", type=float, default=30.0)
    parser.add_argument("--min-z-m", type=float, default=-0.2)
    parser.add_argument("--max-z-m", type=float, default=3.0)
    parser.add_argument("--no-subpixel", dest="subpixel", action="store_false")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.set_defaults(subpixel=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
