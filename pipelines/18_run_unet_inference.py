"""Run U-Net checkpoint inference over generated Sentinel-1/2 patch datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Torch is intentionally imported inside `run_inference` before NumPy so the
# Windows Conda geospatial stack does not load NumPy/MKL DLLs ahead of Torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config

from ml.unet import ARCHITECTURE_SUMMARY


def read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"U-Net config not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def select_patch_records(
    manifest: dict[str, Any],
    split: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records = manifest.get("patches", [])
    if split:
        records = [record for record in records if record["split"] == split]
    if limit is not None:
        records = records[:limit]
    return records


def run_inference(
    config: dict[str, Any],
    settings: dict[str, Any],
    split: str | None = None,
    limit: int | None = None,
) -> Path:
    """Write per-patch probabilities, dominant class, and entropy outputs."""
    try:
        import torch
        from ml.unet import build_unet
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError("PyTorch is required before running U-Net inference.") from exc

    import numpy as np

    ensure_output_dirs(config)
    checkpoint_path = Path(settings["outputs"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"U-Net checkpoint not found: {checkpoint_path}. Run `python pipelines/15_train_unet_baseline.py --fit` first."
        )

    patch_manifest = read_json(settings["training"]["patch_manifest"])
    records = select_patch_records(patch_manifest, split=split, limit=limit)
    if not records:
        raise RuntimeError("No patch records matched the requested inference filters.")

    configured_device = settings["training"]["device"]
    if configured_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(configured_device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_unet(
        input_channels=int(settings["model"]["input_channels"]),
        output_classes=int(settings["model"]["output_classes"]),
        base_channels=int(settings["model"]["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(settings["outputs"]["inference_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = settings["model"]["class_names"]
    class_ids = list(range(1, int(settings["model"]["output_classes"])))
    class_counts = {str(class_id): 0 for class_id in class_ids}
    entropy_values: list[float] = []
    confidence_values: list[float] = []
    inference_records = []

    with torch.no_grad():
        for record in records:
            with np.load(record["path"]) as data:
                features = np.nan_to_num(
                    data["features"].astype("float32"),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

            tensor = torch.from_numpy(features).unsqueeze(0).to(device)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype("float32")

            monitored_probabilities = probabilities[1:, :, :]
            dominant_zero_based = np.argmax(monitored_probabilities, axis=0)
            dominant_class = (dominant_zero_based + 1).astype("uint8")
            max_probability = np.max(monitored_probabilities, axis=0).astype("float32")
            clipped = np.clip(monitored_probabilities, 1.0e-7, 1.0)
            entropy = (-np.sum(clipped * np.log(clipped), axis=0) / np.log(len(class_ids))).astype("float32")

            output_path = output_dir / f"{record['patch_id']}_unet_inference.npz"
            np.savez_compressed(
                output_path,
                class_probabilities=probabilities.astype("float16"),
                dominant_class=dominant_class,
                max_probability=max_probability.astype("float16"),
                entropy=entropy.astype("float16"),
            )

            unique, counts = np.unique(dominant_class, return_counts=True)
            patch_counts = {str(int(key)): int(value) for key, value in zip(unique, counts, strict=True)}
            for key, value in patch_counts.items():
                class_counts[key] = class_counts.get(key, 0) + value
            entropy_values.append(float(np.mean(entropy)))
            confidence_values.append(float(np.mean(max_probability)))

            inference_records.append(
                {
                    "patch_id": record["patch_id"],
                    "source_patch": record["path"],
                    "path": str(output_path).replace("\\", "/"),
                    "training_tag": record["training_tag"],
                    "year": int(record["year"]),
                    "split": record["split"],
                    "row": int(record["row"]),
                    "col": int(record["col"]),
                    "height": int(record["height"]),
                    "width": int(record["width"]),
                    "transform": record["transform"],
                    "dominant_class_counts": patch_counts,
                    "mean_max_probability": round(float(np.mean(max_probability)), 6),
                    "mean_entropy": round(float(np.mean(entropy)), 6),
                }
            )

    manifest = {
        "phase": "phase_33_unet_patch_inference",
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
        "source_patch_manifest": settings["training"]["patch_manifest"],
        "architecture": ARCHITECTURE_SUMMARY,
        "class_names": class_names,
        "class_ids": class_ids,
        "requested_split": split or "all",
        "record_count": len(inference_records),
        "class_pixel_counts": class_counts,
        "mean_patch_max_probability": round(float(np.mean(confidence_values)), 6),
        "mean_patch_entropy": round(float(np.mean(entropy_values)), 6),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "outputs": {
            "inference_dir": str(output_dir).replace("\\", "/"),
            "inference_manifest": settings["outputs"]["inference_manifest"],
        },
        "notes": [
            "Outputs are patch-level model products for downstream mosaicking and WebGIS publication.",
            "Dominant classes exclude background_or_unlabeled because class 0 is an ignore label during training.",
            "Entropy is normalized from 0 to 1 across monitored land-cover classes.",
        ],
        "records": inference_records,
    }

    manifest_path = Path(settings["outputs"]["inference_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run U-Net inference over generated patch records.")
    parser.add_argument("--split", choices=["train", "validation", "test"], help="Optional patch split to infer.")
    parser.add_argument("--limit", type=int, help="Optional limit for smoke tests or quick local checks.")
    args = parser.parse_args()

    config = load_config()
    settings = load_yaml("configs/model_unet.yaml")
    manifest_path = run_inference(config, settings, split=args.split, limit=args.limit)
    print(f"U-Net inference manifest: {manifest_path}")


if __name__ == "__main__":
    main()
