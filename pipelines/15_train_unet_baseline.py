"""Train or preflight the weakly supervised U-Net segmentation baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Torch must initialize before NumPy in this Windows Conda geospatial stack.
# Otherwise PyTorch can fail to load fbgemm.dll after NumPy/MKL DLLs are loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from ml.unet import ARCHITECTURE_SUMMARY

import yaml

try:
    from common import ensure_output_dirs, load_config
except ModuleNotFoundError:
    from pipelines.common import ensure_output_dirs, load_config


def load_yaml(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"U-Net config not found: {settings_path}")
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def load_patch_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Patch manifest not found: {manifest_path}. Run Phase 27 patch generation first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def summarize_manifest(manifest: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Return dataset checks that are useful before fitting any model."""
    patches = manifest.get("patches", [])
    split_counts = manifest.get("split_counts", {})
    training_pixels = int(manifest.get("total_training_pixel_count", 0))
    minimum_pixels = int(settings["training"]["minimum_training_pixels"])
    feature_stack = manifest.get("feature_stack", [])
    expected_channels = int(settings["model"]["input_channels"])

    return {
        "patch_count": int(manifest.get("patch_count", len(patches))),
        "split_counts": split_counts,
        "total_training_pixel_count": training_pixels,
        "feature_stack": feature_stack,
        "expected_input_channels": expected_channels,
        "feature_channel_count": len(feature_stack),
        "has_required_channels": len(feature_stack) == expected_channels,
        "has_train_and_validation": split_counts.get("train", 0) > 0 and split_counts.get("validation", 0) > 0,
        "has_enough_training_pixels": training_pixels >= minimum_pixels,
    }


def normalized_effective_number_weights(
    class_pixel_counts: dict[int, int],
    output_classes: int,
    beta: float,
    ignore_index: int,
    max_class_weight: float | None = None,
    class_weight_overrides: dict[int, float] | None = None,
) -> list[float]:
    """Compute conservative class weights for imbalanced weak-label pixels."""
    weights = [0.0 for _ in range(output_classes)]
    active_weights = []
    for class_id in range(output_classes):
        if class_id == ignore_index:
            continue
        count = int(class_pixel_counts.get(class_id, 0))
        if count <= 0:
            weights[class_id] = 0.0
            continue
        effective_number = 1.0 - (beta**count)
        weight = (1.0 - beta) / effective_number if effective_number > 0 else 0.0
        weights[class_id] = weight
        active_weights.append(weight)

    if active_weights:
        mean_weight = sum(active_weights) / len(active_weights)
        weights = [round(weight / mean_weight, 6) if weight > 0 else 0.0 for weight in weights]
    if max_class_weight is not None:
        weights = [min(weight, max_class_weight) if weight > 0 else 0.0 for weight in weights]
    for class_id, override in (class_weight_overrides or {}).items():
        if 0 <= class_id < len(weights):
            weights[class_id] = max(weights[class_id], float(override))
    return weights


def record_class_counts(records: list[dict[str, Any]]) -> dict[int, int]:
    """Aggregate weak-label class histograms from patch manifest records."""
    counts: dict[int, int] = {}
    for record in records:
        for key, value in record.get("class_pixel_counts", {}).items():
            class_id = int(key)
            counts[class_id] = counts.get(class_id, 0) + int(value)
    return counts


def record_sampling_weights(
    records: list[dict[str, Any]],
    class_weights: list[float],
    minority_boost: float,
    minimum_minority_weight: float = 1.0,
) -> list[float]:
    """Assign higher sampling probability to patches containing rare classes."""
    weights = []
    for record in records:
        patch_weights = [
            class_weights[int(class_id)]
            for class_id in record.get("class_pixel_counts", {})
            if int(class_id) < len(class_weights) and class_weights[int(class_id)] > 0
        ]
        boosted = max(patch_weights) * minority_boost if patch_weights else 0.0
        weights.append(max(minimum_minority_weight if patch_weights else 1.0, 1.0 + boosted))
    return weights


def write_report(config: dict[str, Any], settings: dict[str, Any], status: str, details: dict[str, Any]) -> Path:
    ensure_output_dirs(config)
    output_path = Path(settings["outputs"]["report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "phase_31_unet_segmentation_baseline",
        "status": status,
        "architecture": ARCHITECTURE_SUMMARY,
        "model_config": settings["model"],
        "training_config": settings["training"],
        "outputs": settings["outputs"],
        "details": details,
        "accuracy_claim": "none; weakly supervised preflight/baseline only",
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def run_preflight(config: dict[str, Any], settings: dict[str, Any]) -> Path:
    """Check the patch manifest and write a model-readiness report."""
    manifest = load_patch_manifest(settings["training"]["patch_manifest"])
    summary = summarize_manifest(manifest, settings)
    ok = (
        summary["has_required_channels"]
        and summary["has_train_and_validation"]
        and summary["has_enough_training_pixels"]
    )
    return write_report(config, settings, "ready_for_training" if ok else "needs_attention", summary)


def run_training(config: dict[str, Any], settings: dict[str, Any]) -> Path:
    """Train with imbalance-aware sampling, weighted CE, and Dice loss."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from ml.unet import build_segmentation_model
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            "PyTorch is not installed. Run preflight now, then install PyTorch "
            "before fitting the U-Net baseline."
        ) from exc
    import numpy as np

    manifest = load_patch_manifest(settings["training"]["patch_manifest"])
    train_records = [record for record in manifest.get("patches", []) if record["split"] == "train"]
    validation_records = [record for record in manifest.get("patches", []) if record["split"] == "validation"]
    if not train_records:
        raise RuntimeError("No train patches found in patch manifest.")
    if not validation_records:
        raise RuntimeError("No validation patches found in patch manifest.")

    class PatchDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> tuple["torch.Tensor", "torch.Tensor"]:
            record = self.records[index]
            with np.load(record["path"]) as data:
                features = np.nan_to_num(
                    data["features"].astype("float32"),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                labels = data["agreement_labels"].astype("int64")
                mask = data["training_mask"].astype("bool")
            labels = np.where(mask, labels, int(settings["training"]["ignore_index"]))
            return torch.from_numpy(features), torch.from_numpy(labels)

    configured_device = settings["training"]["device"]
    if configured_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(configured_device)

    output_classes = int(settings["model"]["output_classes"])
    ignore_index = int(settings["training"]["ignore_index"])
    loss_settings = settings["training"].get("loss", {})
    train_class_counts = record_class_counts(train_records)
    # Weak labels are highly imbalanced in peri-urban scenes; effective-number
    # weights reduce majority-class dominance without making rare noisy classes
    # explode the loss.
    class_weights = normalized_effective_number_weights(
        train_class_counts,
        output_classes=output_classes,
        beta=float(loss_settings.get("effective_number_beta", 0.999)),
        ignore_index=ignore_index,
        max_class_weight=float(loss_settings["max_class_weight"]) if "max_class_weight" in loss_settings else None,
        class_weight_overrides={
            int(class_id): float(weight)
            for class_id, weight in loss_settings.get("class_weight_overrides", {}).items()
        },
    )
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = build_segmentation_model(
        architecture=settings["model"].get("architecture", "lightweight_unet"),
        input_channels=int(settings["model"]["input_channels"]),
        output_classes=output_classes,
        base_channels=int(settings["model"]["base_channels"]),
    ).to(device)
    sampler = None
    sampling_settings = settings["training"].get("sampling", {})
    if bool(sampling_settings.get("class_aware", False)):
        # Patch-level sampling complements pixel-level loss weighting: rare
        # classes must appear in enough batches before the loss can help them.
        sampler = WeightedRandomSampler(
            weights=record_sampling_weights(
                train_records,
                class_weights=class_weights,
                minority_boost=float(sampling_settings.get("minority_boost", 1.0)),
                minimum_minority_weight=float(sampling_settings.get("minimum_minority_weight", 1.0)),
            ),
            num_samples=max(1, round(len(train_records) * float(sampling_settings.get("samples_per_epoch_multiplier", 1.0)))),
            replacement=True,
        )
    train_loader = DataLoader(
        PatchDataset(train_records),
        batch_size=int(settings["training"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(settings["training"].get("num_workers", 0)),
    )
    validation_loader = DataLoader(
        PatchDataset(validation_records),
        batch_size=int(settings["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(settings["training"].get("num_workers", 0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["training"]["learning_rate"]),
        weight_decay=float(settings["training"]["weight_decay"]),
    )
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor, ignore_index=ignore_index)

    def generalized_dice_loss(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        """Dice component over monitored classes, excluding ignored pixels."""
        valid_mask = labels != ignore_index
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0
        probabilities = torch.softmax(logits, dim=1)
        dice_terms = []
        for class_id in range(1, output_classes):
            if class_weights[class_id] <= 0:
                continue
            target = ((labels == class_id) & valid_mask).float()
            predicted = probabilities[:, class_id, :, :] * valid_mask.float()
            intersection = torch.sum(predicted * target)
            denominator = torch.sum(predicted) + torch.sum(target)
            dice_terms.append(1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)))
        if not dice_terms:
            return logits.sum() * 0.0
        return torch.stack(dice_terms).mean()

    def tversky_loss(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        """Tversky component to emphasize rare-class recall under imbalance."""
        valid_mask = labels != ignore_index
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0
        probabilities = torch.softmax(logits, dim=1)
        alpha = float(loss_settings.get("tversky_alpha", 0.3))
        beta = float(loss_settings.get("tversky_beta", 0.7))
        terms = []
        for class_id in range(1, output_classes):
            if class_weights[class_id] <= 0:
                continue
            target = ((labels == class_id) & valid_mask).float()
            predicted = probabilities[:, class_id, :, :] * valid_mask.float()
            true_positive = torch.sum(predicted * target)
            false_positive = torch.sum(predicted * (1.0 - target) * valid_mask.float())
            false_negative = torch.sum((1.0 - predicted) * target)
            terms.append(
                1.0
                - (
                    (true_positive + 1.0)
                    / (true_positive + alpha * false_positive + beta * false_negative + 1.0)
                )
            )
        if not terms:
            return logits.sum() * 0.0
        return torch.stack(terms).mean()

    def focal_loss(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        """Focal component so rare, hard pixels still influence the weakly supervised fit."""
        valid_mask = labels != ignore_index
        if not bool(valid_mask.any()):
            return logits.sum() * 0.0
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
        probabilities = torch.exp(log_probabilities)
        safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        label_log_probabilities = log_probabilities.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        label_probabilities = probabilities.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        pixel_weights = class_weight_tensor[safe_labels]
        focal_gamma = float(loss_settings.get("focal_gamma", 2.0))
        focal = -pixel_weights * ((1.0 - label_probabilities) ** focal_gamma) * label_log_probabilities
        return focal[valid_mask].mean()

    def combined_loss(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        # The loss is deliberately modular so the project can compare weighted
        # CE, Dice, Focal, and Tversky without changing the training loop.
        ce_loss = ce_loss_fn(logits, labels)
        dice_loss = generalized_dice_loss(logits, labels)
        focal = focal_loss(logits, labels)
        tversky = tversky_loss(logits, labels)
        return (
            float(loss_settings.get("cross_entropy_weight", 1.0)) * ce_loss
            + float(loss_settings.get("dice_weight", 0.5)) * dice_loss
            + float(loss_settings.get("focal_weight", 0.0)) * focal
            + float(loss_settings.get("tversky_weight", 0.0)) * tversky
        )

    def validation_metrics() -> dict[str, Any]:
        """Calculate per-class weak-label IoU, Dice, and recall."""
        model.eval()
        metrics = {
            str(class_id): {
                "true_pixels": 0,
                "predicted_pixels": 0,
                "intersection": 0,
                "union": 0,
            }
            for class_id in range(1, output_classes)
        }
        with torch.no_grad():
            for features, labels in validation_loader:
                features = features.to(device)
                labels = labels.to(device)
                predictions = torch.argmax(model(features), dim=1)
                valid_mask = labels != ignore_index
                for class_id in range(1, output_classes):
                    label_mask = (labels == class_id) & valid_mask
                    prediction_mask = (predictions == class_id) & valid_mask
                    intersection = label_mask & prediction_mask
                    union = label_mask | prediction_mask
                    item = metrics[str(class_id)]
                    item["true_pixels"] += int(label_mask.sum().detach().cpu())
                    item["predicted_pixels"] += int(prediction_mask.sum().detach().cpu())
                    item["intersection"] += int(intersection.sum().detach().cpu())
                    item["union"] += int(union.sum().detach().cpu())
        for class_id in range(1, output_classes):
            item = metrics[str(class_id)]
            true_pixels = item["true_pixels"]
            predicted_pixels = item["predicted_pixels"]
            intersection = item["intersection"]
            union = item["union"]
            item["iou"] = round(intersection / union, 6) if union else None
            dice_denominator = true_pixels + predicted_pixels
            item["dice"] = round((2 * intersection) / dice_denominator, 6) if dice_denominator else None
            item["recall"] = round(intersection / true_pixels, 6) if true_pixels else None
        return metrics

    train_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    patience = int(settings["training"].get("early_stopping_patience", 0))
    min_delta = float(settings["training"].get("min_delta", 0.0))
    epochs_without_improvement = 0
    model.train()
    for epoch in range(int(settings["training"]["epochs"])):
        epoch_losses = []
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = combined_loss(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_losses.append(round(float(np.mean(epoch_losses)), 6))

        model.eval()
        validation_epoch_losses = []
        with torch.no_grad():
            for features, labels in validation_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                validation_loss = combined_loss(logits, labels)
                validation_epoch_losses.append(float(validation_loss.detach().cpu()))
        validation_losses.append(round(float(np.mean(validation_epoch_losses)), 6))
        current_validation_loss = validation_losses[-1]
        if current_validation_loss < best_validation_loss - min_delta:
            # Keep the best validation checkpoint instead of the last epoch so
            # weak-label overfitting does not silently become the deployed model.
            best_validation_loss = current_validation_loss
            best_epoch = epoch + 1
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        model.train()
        print(
            f"epoch {epoch + 1}: train_loss={train_losses[-1]} "
            f"validation_loss={validation_losses[-1]}",
            flush=True,
        )
        if patience and epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch + 1}; best_epoch={best_epoch}", flush=True)
            break

    checkpoint_path = Path(settings["outputs"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": settings["model"],
            "training_config": settings["training"],
            "architecture": ARCHITECTURE_SUMMARY,
        },
        checkpoint_path,
    )

    details = summarize_manifest(manifest, settings)
    details["train_patch_count"] = len(train_records)
    details["validation_patch_count"] = len(validation_records)
    details["train_class_pixel_counts"] = {str(key): int(value) for key, value in sorted(train_class_counts.items())}
    details["class_weights"] = {
        str(index): float(weight)
        for index, weight in enumerate(class_weights)
        if index != ignore_index
    }
    details["imbalance_strategy"] = {
        "class_weight_method": loss_settings.get("class_weight_method", "effective_number"),
        "loss": "weighted_cross_entropy_plus_generalized_dice_plus_focal",
        "tversky_loss": float(loss_settings.get("tversky_weight", 0.0)) > 0,
        "class_aware_sampling": bool(sampling_settings.get("class_aware", False)),
        "samples_per_epoch_multiplier": float(sampling_settings.get("samples_per_epoch_multiplier", 1.0)),
        "balanced_manifest": bool(sampling_settings.get("balanced_manifest", False)),
        "minority_classes": sampling_settings.get("minority_classes", []),
    }
    details["training_loss_by_epoch"] = train_losses
    details["validation_loss_by_epoch"] = validation_losses
    details["best_validation_loss"] = best_validation_loss if best_validation_loss < float("inf") else None
    details["best_epoch"] = best_epoch
    details["early_stopping"] = {
        "patience": patience,
        "min_delta": min_delta,
        "stopped_early": len(train_losses) < int(settings["training"]["epochs"]),
    }
    details["validation_metrics_by_class"] = validation_metrics()
    details["final_training_loss"] = train_losses[-1] if train_losses else None
    details["final_validation_loss"] = validation_losses[-1] if validation_losses else None
    details["device"] = str(device)
    details["cuda_available"] = bool(torch.cuda.is_available())
    details["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    details["checkpoint_path"] = str(checkpoint_path).replace("\\", "/")
    return write_report(config, settings, "trained", details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight or train the weakly supervised U-Net baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/model_unet.yaml"), help="Model config YAML path.")
    parser.add_argument("--fit", action="store_true", help="Run training. Default writes a readiness report only.")
    args = parser.parse_args()

    config = load_config()
    settings = load_yaml(args.config)
    report_path = run_training(config, settings) if args.fit else run_preflight(config, settings)
    print(f"U-Net baseline report: {report_path}")


if __name__ == "__main__":
    main()
