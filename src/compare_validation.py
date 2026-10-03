"""Validation-only comparison of baseline vs fine-tuned models (milestone 9).

Scope
-----
Evaluates TWO saved models on the **identical** validation pipeline (same
manifest, same deterministic manifest order, same image size and batch size)
and applies the selection rule below. The test manifest is never opened.

Selection rule — fixed in code BEFORE any comparison is run
-----------------------------------------------------------
1. If the fine-tuned model's macro F1 is strictly higher than the
   baseline's -> select the fine-tuned model.
2. If macro F1 is equal, select the fine-tuned model only when its
   validation accuracy is strictly higher.
3. Otherwise (macro F1 lower, or both metrics equal) -> keep the baseline.

Within a single run the best checkpoint is still chosen by lowest `val_loss`
(that rule is unchanged and never compares across models). The test set is
never consulted by this rule. If the fine-tuned model does not exist yet, the
report is written with status `pending` and NO metrics — nothing is invented.

Metrics are computed in NumPy (no scikit-learn), so the same code runs in both
project environments. `loss` is the mean sparse categorical cross-entropy of
the softmax outputs; ties in `argmax` resolve to the lowest class index.

Usage
-----
    .venv-infer\\Scripts\\python.exe src\\compare_validation.py --help
    (the baseline artifact was saved by Keras 3, so use the .venv-infer env;
     synthetic tests under .venv work the same way with their own models)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

import data_pipeline
import finetune
import train

DEFAULT_CONFIG = data_pipeline.DEFAULT_TRAINING_CONFIG
DEFAULT_METADATA_DIR = data_pipeline.DEFAULT_METADATA_DIR
DEFAULT_BASELINE_MODEL = finetune.DEFAULT_PARENT_MODEL
DEFAULT_OUTPUT = (
    data_pipeline.REPO_ROOT / "models" / "metadata" / "comparison" / "validation_comparison.json"
)

SELECTION_RULE: dict[str, Any] = {
    "id": "validation_macro_f1_then_accuracy",
    "steps": [
        "1. If the fine-tuned model's macro F1 is strictly higher than the "
        "baseline's, select the fine-tuned model.",
        "2. If macro F1 is equal, select the fine-tuned model only when its "
        "validation accuracy is strictly higher.",
        "3. Otherwise (macro F1 lower, or both metrics equal), keep the baseline.",
    ],
    "within_a_run": (
        "best checkpoint = lowest val_loss (unchanged; this rule never "
        "compares checkpoints inside one run)"
    ),
    "test_set_used": False,
    "fixed_before_running": (
        "This rule lives in src/compare_validation.py and was written before "
        "any comparison ran; measured metrics can never change it."
    ),
}


# ---------------------------------------------------------------------------
# Metrics (pure NumPy so both environments compute the same way)
# ---------------------------------------------------------------------------

def metrics_from_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
    class_order: list[str],
) -> dict[str, Any]:
    """Loss, accuracy, macro F1, per-class P/R/F1/support, confusion matrix.

    Zero-division policy: precision with no predictions, recall with no
    support, and F1 with precision+recall == 0 are all reported as 0.0.
    Confusion matrix rows = true class, columns = predicted class.
    """
    probs = np.asarray(probs, dtype="float64")
    labels = np.asarray(labels, dtype="int64")
    if probs.ndim != 2 or probs.shape[1] != len(class_order):
        raise ValueError(
            f"probabilities shape {probs.shape} does not match "
            f"{len(class_order)} classes {class_order}"
        )
    if labels.ndim != 1 or len(labels) != probs.shape[0]:
        raise ValueError(
            f"labels shape {labels.shape} does not match probabilities {probs.shape}"
        )
    if len(labels) == 0:
        raise ValueError("cannot compute metrics from zero samples")

    num_classes = len(class_order)
    predictions = probs.argmax(axis=1)

    true_probs = probs[np.arange(len(labels)), labels]
    loss = float(-np.log(np.clip(true_probs, 1e-12, 1.0)).mean())
    accuracy = float((predictions == labels).mean())

    confusion = np.zeros((num_classes, num_classes), dtype="int64")
    for true_index, predicted_index in zip(labels, predictions):
        confusion[true_index, predicted_index] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(class_order):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "predicted_count": predicted_count,
        }
        f1_values.append(float(f1))

    return {
        "images": int(len(labels)),
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": {
            "order": list(class_order),
            "rows_true_columns_predicted": confusion.tolist(),
        },
    }


def collect_predictions(
    model: tf.keras.Model,
    records: list[dict],
    *,
    image_size: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full deterministic pass in manifest order -> (probabilities, labels).

    The dataset is built with training=False (no shuffle, augmentation off),
    so row i of the result corresponds to records[i]. That alignment is
    verified, not assumed: a mismatch raises instead of producing a report
    whose precision/recall rows point at the wrong images.
    """
    dataset = data_pipeline.make_dataset(
        records,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        seed=seed,
    )
    probability_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for images, labels in dataset:
        probability_batches.append(np.asarray(model(images, training=False)))
        label_batches.append(np.asarray(labels))
    probs = np.concatenate(probability_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)

    expected = [int(record["class_index"]) for record in records]
    if labels.astype("int64").tolist() != expected:
        raise RuntimeError(
            "prediction/label alignment broken: the evaluated label sequence "
            "does not match the validation manifest order — refusing to "
            "compute metrics from misaligned rows"
        )
    if not np.all(np.isfinite(probs)):
        raise RuntimeError("model produced non-finite probabilities")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise RuntimeError(
            "model outputs do not sum to 1 per image (softmax expected); "
            "loss would be meaningless"
        )
    return probs, labels


def evaluate_model_file(
    model_path: Path,
    *,
    records: list[dict],
    class_order: list[str],
    image_size: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Load one saved model and evaluate it on the shared validation records."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # noqa: BLE001 - deserialization errors vary by version
        raise RuntimeError(
            f"Could not load model {model_path} ({type(exc).__name__}: {exc}). "
            "Use the environment that matches how the file was saved "
            "(.venv-infer for the Keras 3 baseline artifact)."
        ) from exc

    input_shape = getattr(model, "input_shape", None)
    if not (
        isinstance(input_shape, tuple)
        and len(input_shape) == 4
        and int(input_shape[1]) == image_size
    ):
        raise ValueError(
            f"model {model_path.name} expects input {input_shape!r}, but the "
            f"validation pipeline uses image_size={image_size}"
        )
    num_outputs = int(model.output_shape[-1])
    if num_outputs != len(class_order):
        raise ValueError(
            f"model {model_path.name} outputs {num_outputs} classes but "
            f"class_order has {len(class_order)}: {class_order}"
        )

    probs, labels = collect_predictions(
        model, records, image_size=image_size, batch_size=batch_size, seed=seed
    )
    metrics = metrics_from_predictions(probs, labels, class_order)
    return {
        "model": {
            "path": model_path.as_posix(),
            "sha256": train.sha256_file(model_path),
            "size_bytes": int(model_path.stat().st_size),
        },
        "images_evaluated": len(records),
        "metrics": metrics,
    }


def decide_model(baseline_metrics: dict, finetuned_metrics: dict) -> dict[str, Any]:
    """Apply the selection rule. Pure function — tested for ties and worse cases."""
    baseline_f1 = float(baseline_metrics["macro_f1"])
    finetuned_f1 = float(finetuned_metrics["macro_f1"])
    baseline_accuracy = float(baseline_metrics["accuracy"])
    finetuned_accuracy = float(finetuned_metrics["accuracy"])

    if finetuned_f1 > baseline_f1:
        selected = "finetuned"
        reason = (
            f"rule step 1: macro F1 improved "
            f"{baseline_f1:.6f} -> {finetuned_f1:.6f}"
        )
    elif finetuned_f1 == baseline_f1 and finetuned_accuracy > baseline_accuracy:
        selected = "finetuned"
        reason = (
            f"rule step 2: macro F1 tied at {baseline_f1:.6f} and accuracy "
            f"improved {baseline_accuracy:.6f} -> {finetuned_accuracy:.6f}"
        )
    else:
        selected = "baseline"
        if finetuned_f1 == baseline_f1 and finetuned_accuracy == baseline_accuracy:
            reason = (
                f"rule step 3: macro F1 tied at {baseline_f1:.6f} and accuracy "
                f"tied at {baseline_accuracy:.6f} — keep the baseline"
            )
        else:
            reason = (
                f"rule step 3: macro F1 {finetuned_f1:.6f} vs baseline "
                f"{baseline_f1:.6f} and accuracy {finetuned_accuracy:.6f} vs "
                f"{baseline_accuracy:.6f} — the fine-tuned model did not "
                "improve, keep the baseline"
            )

    return {
        "selected": selected,
        "reason": reason,
        "measured": {
            "baseline": {"macro_f1": baseline_f1, "accuracy": baseline_accuracy},
            "finetuned": {"macro_f1": finetuned_f1, "accuracy": finetuned_accuracy},
        },
    }


# ---------------------------------------------------------------------------
# Comparison workflow
# ---------------------------------------------------------------------------

def _model_identity(model_path: Path) -> dict[str, Any]:
    """Path + checksum of a model file (or an honest note when it is absent)."""
    if model_path.is_file():
        return {
            "path": model_path.as_posix(),
            "sha256": train.sha256_file(model_path),
            "size_bytes": int(model_path.stat().st_size),
        }
    return {"path": model_path.as_posix(), "sha256": None, "note": "file not found"}


def run_comparison(
    *,
    baseline_model: Path = DEFAULT_BASELINE_MODEL,
    finetune_model: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    path_root: Path = data_pipeline.REPO_ROOT,
    output: Path = DEFAULT_OUTPUT,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Compare baseline vs fine-tuned on validation; write the report; return it.

    Without a fine-tuned model file the report is `pending` — no metrics are
    measured or invented. The report always carries the selection rule so a
    later reader sees the rule the decision was based on.
    """
    config = data_pipeline.load_training_config(config_path)
    class_order = data_pipeline.load_class_order(
        train.resolve_config_path(path_root, config["class_mapping_path"])
    )
    image_size = config["image_size"]
    batch_size = config["batch_size"] if batch_size is None else batch_size
    seed = config["seed"]

    baseline_model = Path(baseline_model)
    finetune_model = Path(finetune_model) if finetune_model is not None else None

    report: dict[str, Any] = {
        "report_type": "validation_model_selection",
        "status": "pending",
        "created_at_utc": train.utc_now_iso(),
        "git": train.collect_git_info(path_root),
        "environment": train.collect_environment_info(),
        "selection_rule": SELECTION_RULE,
        "evaluation": {
            "split": "validation",
            "manifest": None,
            "image_size": image_size,
            "batch_size": batch_size,
            "seed": seed,
            "pipeline": (
                "identical deterministic pass for both models: manifest order, "
                "training=False, no augmentation, full split including the "
                "final partial batch"
            ),
            "test_manifest_used": False,
            "class_order": list(class_order),
            "baseline": {"model": _model_identity(baseline_model)},
            "finetuned": None,
        },
        "decision": {"selected": None, "reason": None, "measured": None},
        "honesty_notes": [
            "status=pending means the fine-tuned model does not exist yet: no "
            "metrics were measured, and none are guessed.",
            "The selection rule was fixed in code before this report; it is "
            "never changed after seeing results.",
            "No test image is opened by this workflow (test_manifest_used=false).",
        ],
    }

    finetune_missing = finetune_model is None or not finetune_model.is_file()
    if finetune_missing:
        if finetune_model is None:
            missing = "--finetune-model was not provided"
        else:
            missing = f"fine-tuned model file not found: {finetune_model}"
        report["decision"]["reason"] = (
            f"pending: {missing}. Run src/finetune.py (or the Colab notebook) "
            "first, then re-run this comparison."
        )
    else:
        # Both models exist -> measure both on the identical pipeline.
        manifest_path = Path(metadata_dir) / "validation.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        splits = data_pipeline.load_splits(
            metadata_dir=metadata_dir,
            class_order=class_order,
            path_root=path_root,
            splits=("validation",),
        )
        records = splits["validation"]
        report["evaluation"]["manifest"] = {
            "path": manifest_path.relative_to(path_root).as_posix()
            if manifest_path.is_relative_to(path_root)
            else manifest_path.as_posix(),
            "sha256": train.sha256_file(manifest_path),
            "rows": len(records),
        }

        shared = dict(
            records=records,
            class_order=class_order,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
        )
        baseline_result = evaluate_model_file(baseline_model, **shared)
        finetuned_result = evaluate_model_file(finetune_model, **shared)

        report["status"] = "complete"
        report["evaluation"]["baseline"] = baseline_result
        report["evaluation"]["finetuned"] = finetuned_result
        report["evaluation"]["images_evaluated"] = len(records)
        report["decision"] = decide_model(
            baseline_result["metrics"], finetuned_result["metrics"]
        )
        report["honesty_notes"] = [
            "Every metric comes from evaluating the two saved models on the "
            "identical validation manifest in this single run; no test image "
            "was opened.",
            "The selection rule was fixed in code before this comparison ran; "
            "it is never tuned after seeing results.",
            "Both model checksums are recorded so a later reader can prove "
            "which files were compared.",
        ]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["_output_path"] = output.as_posix()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline and fine-tuned models on the identical "
            "validation pipeline, apply the documented selection rule, and "
            "write a model-selection report (no test-set access)."
        )
    )
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL,
                        help="Baseline artifact (default: the verified baseline).")
    parser.add_argument("--finetune-model", type=Path, default=None,
                        help="Fine-tuned best_model.keras; omitted -> status 'pending'.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to configs/training.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write validation_comparison.json.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Evaluation batch size (default: config batch_size).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Validation comparison (milestone 9: validation only, no test set) ===")

    try:
        report = run_comparison(
            baseline_model=args.baseline_model,
            finetune_model=args.finetune_model,
            config_path=args.config,
            output=args.output,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[error] Comparison did not run. Fix the cause above and re-run.",
            file=sys.stderr,
        )
        return 1

    evaluation = report["evaluation"]
    print(f"Report: {args.output}")
    print(f"Status: {report['status']}")
    images = evaluation.get("images_evaluated")
    split_note = f"{images} images" if images else "not evaluated yet"
    print(f"Split : validation ({split_note}), "
          f"test_manifest_used={evaluation['test_manifest_used']}")
    if report["status"] == "complete":
        baseline_metrics = evaluation["baseline"]["metrics"]
        finetuned_metrics = evaluation["finetuned"]["metrics"]
        print(f"Baseline  loss={baseline_metrics['loss']:.4f} "
              f"accuracy={baseline_metrics['accuracy']:.4f} "
              f"macro_f1={baseline_metrics['macro_f1']:.4f}")
        print(f"Fine-tuned loss={finetuned_metrics['loss']:.4f} "
              f"accuracy={finetuned_metrics['accuracy']:.4f} "
              f"macro_f1={finetuned_metrics['macro_f1']:.4f}")
    print(f"Decision : {report['decision']['selected']}")
    print(f"Reason   : {report['decision']['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
