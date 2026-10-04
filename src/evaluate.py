"""Held-out test-set evaluation for the SELECTED model (milestone 10).

Scope
-----
- Evaluates **exactly one** checkpoint: the model recorded in the selection
  record (`models/metadata/selection/selected_model.json`). That record was
  written from the validation comparison *before* this script existed and
  before any test image was opened, so the test split cannot influence which
  model is measured.
- There is **no fitting, no model choice and no threshold tuning** here: the
  probabilities come from one saved model and the prediction is a plain
  argmax. Whatever the test numbers are, they are reported as measured.
- Reuses the shared deterministic pipeline (`src/data_pipeline.py`) and the
  metric helpers of `src/compare_validation.py`, so validation and test
  metrics are produced by the same code.

Protocol (why the numbers are comparable to validation)
--------------------------------------------------------
- `training=False`: manifest order, no shuffle, augmentation layers inert.
- Pixels stay float32 **0-255 RGB**; MobileNetV2 preprocessing runs once,
  inside the model (already verified by the compatibility report).
- `drop_remainder=False`: the final partial batch is kept, so all 459 test
  images are scored.
- Loss / accuracy / macro F1 / per-class P-R-F1 / confusion matrix come from
  `compare_validation.metrics_from_predictions` (pure NumPy).

Guard rails: the test manifest checksum is checked against the committed
`split_summary.json`, the class order must agree across the selection record,
the config and the model bundle, and the model file must hash to the checksum
in the selection record. Any disagreement stops the run.

Misclassified-image grids are written **outside** the tracked tree
(`models/runs/...`, which `.gitignore` blocks) because they embed dataset
photographs.

Usage
-----
    .venv-infer\\Scripts\\python.exe src\\evaluate.py --help
    (run with .venv-infer: the selected artifact was saved by Keras 3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

import compare_validation
import data_pipeline
import train

REPO_ROOT = data_pipeline.REPO_ROOT
DEFAULT_CONFIG = data_pipeline.DEFAULT_TRAINING_CONFIG
DEFAULT_METADATA_DIR = data_pipeline.DEFAULT_METADATA_DIR
DEFAULT_SELECTION = REPO_ROOT / "models" / "metadata" / "selection" / "selected_model.json"
DEFAULT_SPLIT_SUMMARY = DEFAULT_METADATA_DIR / "split_summary.json"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "metadata" / "evaluation" / "test_evaluation.json"
# Dataset photographs must never enter Git: models/* is ignored except README
# and models/metadata/**, and the file name also matches the sample_grid rule.
DEFAULT_GRIDS_DIR = REPO_ROOT / "models" / "runs" / "misclassified_grids"
MAX_GRID_IMAGES = 16


# ---------------------------------------------------------------------------
# Selection record and pre-flight checks
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Selection record not found: {path}. Run the import/selection "
            "step first (models/metadata/selection/selected_model.json)."
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_selected_model(selection: dict, path_root: Path) -> dict[str, Any]:
    """The single checkpoint this script is allowed to evaluate.

    The path comes from the selection record, not from a command-line flag:
    evaluating "whatever model the user passed" would reopen the model choice
    this record exists to freeze.
    """
    chosen = selection.get("selected_model")
    if not isinstance(chosen, dict) or "path" not in chosen or "sha256" not in chosen:
        raise ValueError(
            "selection record has no usable 'selected_model' block "
            f"(keys: {sorted(selection)})"
        )
    raw_path = Path(chosen["path"])
    model_path = raw_path if raw_path.is_absolute() else Path(path_root) / raw_path
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Selected model not found: {model_path}. Restore the model bundle "
            "(models/<run_id>/best_model.keras) before evaluating."
        )
    return {
        "path": model_path,
        "sha256": chosen["sha256"],
        "selection_recorded": selection.get("selected"),
        "selection_reason": selection.get("selection_reason"),
        "selection_rule": selection.get("selection_rule"),
        "run_id": (selection.get("run") or {}).get("run_id"),
        "class_order": list(selection.get("class_order") or []),
        "record_created_at": selection.get("created_at_utc"),
    }


def verify_model_checksum(model_path: Path, expected_sha: str) -> str:
    actual = train.sha256_file(model_path)
    if actual != expected_sha:
        raise RuntimeError(
            f"model checksum mismatch: the selection record expects "
            f"{expected_sha} but {model_path} hashes to {actual}. This is not "
            "the selected artifact — refusing to evaluate it."
        )
    return actual


def verify_test_manifest(
    manifest_path: Path,
    split_summary_path: Path,
    class_order: list[str],
) -> dict[str, Any]:
    """Test manifest must be byte-identical to what the split committed."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test manifest not found: {manifest_path}")
    summary = read_json(split_summary_path)

    expected_sha = summary["manifest_checksums_sha256"]["test.csv"]
    expected_rows = int(summary["split_totals"]["test"])
    actual_sha = train.sha256_file(manifest_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"test manifest checksum mismatch: split_summary.json records "
            f"{expected_sha} but {manifest_path} hashes to {actual_sha}. The "
            "manifest changed after the split — refusing to evaluate."
        )
    if summary.get("class_order") != class_order:
        raise RuntimeError(
            f"class order mismatch: split_summary.json has "
            f"{summary.get('class_order')} but the selection record says "
            f"{class_order}"
        )
    return {
        "path": manifest_path,
        "sha256": actual_sha,
        "expected_sha256": expected_sha,
        "expected_rows": expected_rows,
        "split_summary": split_summary_path,
    }


def verify_class_order_sources(
    selection_order: list[str],
    config_order: list[str],
    model_path: Path,
) -> dict[str, Any]:
    """Every source that names the classes must name them identically."""
    sources: dict[str, list[str]] = {
        "selection_record": list(selection_order),
        "configs/class_mapping.json": list(config_order),
    }
    bundle = model_path.parent / "class_order.json"
    if bundle.is_file():
        payload = read_json(bundle)
        if isinstance(payload, list):
            payload = {"class_order": payload}
        sources["model_bundle/class_order.json"] = list(payload.get("class_order") or [])

    distinct = {tuple(order) for order in sources.values()}
    if len(distinct) != 1:
        raise RuntimeError(f"class order disagrees across sources: {sources}")
    if not selection_order:
        raise RuntimeError("class order is empty")
    return {"verified": True, "sources": sources, "class_order": list(selection_order)}


def assert_every_image_once(
    records: list[dict],
    predictions_count: int,
    expected_rows: int,
) -> dict[str, Any]:
    """All expected rows, one prediction each, no duplicated image path."""
    paths = [record["path"] for record in records]
    problems: list[str] = []
    if len(records) != expected_rows:
        problems.append(
            f"manifest holds {len(records)} rows but split_summary.json "
            f"commits {expected_rows}"
        )
    if predictions_count != len(records):
        problems.append(
            f"{predictions_count} predictions for {len(records)} manifest rows"
        )
    duplicates = sorted({p for p in paths if paths.count(p) > 1})
    if duplicates:
        problems.append(f"duplicated manifest paths: {duplicates[:5]}")
    if problems:
        raise RuntimeError(
            "test coverage is not 'every image exactly once': " + "; ".join(problems)
        )
    return {
        "expected_images": expected_rows,
        "manifest_rows": len(records),
        "predictions_made": predictions_count,
        "unique_image_paths": len(set(paths)),
        "each_image_evaluated_exactly_once": True,
    }


# ---------------------------------------------------------------------------
# Metrics helpers built on the shared implementation
# ---------------------------------------------------------------------------

def row_normalized_confusion(raw: list[list[int]]) -> list[list[float]]:
    """Each true-class row scaled to fractions (row sums to 1, or 0 if empty)."""
    normalized: list[list[float]] = []
    for row in raw:
        total = float(sum(row))
        normalized.append([value / total if total else 0.0 for value in row])
    return normalized


def build_prediction_rows(
    records: list[dict],
    probabilities: np.ndarray,
    class_order: list[str],
) -> list[dict[str, Any]]:
    """Per-image rows: relative path, truth, prediction, and every class score."""
    predicted = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        target_index = int(record["class_index"])
        rows.append(
            {
                "path": record["path"],
                "target": record["target"],
                "target_index": target_index,
                "predicted_index": int(predicted[index]),
                "predicted": class_order[int(predicted[index])],
                "correct": bool(predicted[index] == target_index),
                "scores": [float(value) for value in probabilities[index]],
            }
        )
    return rows


def write_misclassified_grid(
    prediction_rows: list[dict[str, Any]],
    path_root: Path,
    destination_dir: Path,
    max_images: int = MAX_GRID_IMAGES,
) -> str | None:
    """Small local grid of wrong predictions (dataset photos -> never committed)."""
    wrong = [row for row in prediction_rows if not row["correct"]]
    if not wrong:
        return None
    import matplotlib

    matplotlib.use("Agg")  # headless: never opens a window
    import matplotlib.pyplot as plt
    from PIL import Image

    shown = wrong[:max_images]
    columns = 4
    rows_count = (len(shown) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns, figsize=(3 * columns, 3 * rows_count))
    axes = np.atleast_1d(axes).ravel()

    for axis in axes:
        axis.axis("off")
    for axis, row in zip(axes, shown):
        image_path = data_pipeline.resolve_image_path(row["path"], path_root)
        axis.imshow(Image.open(image_path).convert("RGB"))
        axis.set_title(
            f"true: {row['target']}\npred: {row['predicted']}",
            fontsize=9,
        )
        axis.axis("off")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "sample_grid_test_misclassified.png"
    figure.tight_layout()
    figure.savefig(destination, dpi=110)
    plt.close(figure)
    return destination.as_posix()


# ---------------------------------------------------------------------------
# Evaluation workflow
# ---------------------------------------------------------------------------

def run_evaluation(
    *,
    selection_path: Path = DEFAULT_SELECTION,
    config_path: Path = DEFAULT_CONFIG,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    split_summary_path: Path | None = None,
    path_root: Path = REPO_ROOT,
    output: Path = DEFAULT_OUTPUT,
    grids_dir: Path | None = DEFAULT_GRIDS_DIR,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate the selected checkpoint on the committed test split once."""
    split_summary_path = (
        Path(split_summary_path)
        if split_summary_path is not None
        else Path(metadata_dir) / "split_summary.json"
    )

    config = data_pipeline.load_training_config(config_path)
    selection = read_json(selection_path)
    chosen = resolve_selected_model(selection, path_root)

    config_order = data_pipeline.load_class_order(
        train.resolve_config_path(path_root, config["class_mapping_path"])
    )
    order_check = verify_class_order_sources(
        chosen["class_order"], config_order, chosen["path"]
    )
    class_order = order_check["class_order"]

    model_sha = verify_model_checksum(chosen["path"], chosen["sha256"])
    manifest = verify_test_manifest(
        Path(metadata_dir) / "test.csv", split_summary_path, class_order
    )

    # The test split is loaded here and only here — nothing else opens it.
    test_records = data_pipeline.load_manifest(
        manifest["path"], class_order, path_root=path_root, check_files=True
    )

    image_size = config["image_size"]
    batch_size = config["batch_size"] if batch_size is None else batch_size

    model = tf.keras.models.load_model(chosen["path"], compile=False, safe_mode=True)
    input_shape = tuple(model.input_shape)
    if not (len(input_shape) == 4 and int(input_shape[1]) == image_size):
        raise ValueError(
            f"selected model expects input {input_shape!r} but the pipeline "
            f"uses image_size={image_size}"
        )
    if int(model.output_shape[-1]) != len(class_order):
        raise ValueError(
            f"selected model outputs {model.output_shape[-1]} classes but "
            f"class_order has {len(class_order)}: {class_order}"
        )

    # Shared deterministic pass: manifest order, no shuffle, final batch kept.
    probabilities, labels = compare_validation.collect_predictions(
        model,
        test_records,
        image_size=image_size,
        batch_size=batch_size,
        seed=config["seed"],
    )
    coverage = assert_every_image_once(
        test_records, predictions_count=int(probabilities.shape[0]),
        expected_rows=manifest["expected_rows"],
    )

    metrics = compare_validation.metrics_from_predictions(
        probabilities, labels, class_order
    )
    raw_confusion = metrics["confusion_matrix"]["rows_true_columns_predicted"]
    prediction_rows = build_prediction_rows(test_records, probabilities, class_order)

    grid_path: str | None = None
    if grids_dir is not None:
        grid_path = write_misclassified_grid(prediction_rows, path_root, Path(grids_dir))

    misclassified = sum(1 for row in prediction_rows if not row["correct"])
    report: dict[str, Any] = {
        "report_type": "test_set_evaluation",
        "status": "ok",
        "created_at_utc": train.utc_now_iso(),
        "split": "test",
        "selection": {
            "record": Path(selection_path).relative_to(path_root).as_posix()
            if Path(selection_path).is_relative_to(path_root)
            else str(selection_path),
            "selected": selection.get("selected"),
            "selection_reason": selection.get("selection_reason"),
            "selection_rule": chosen["selection_rule"],
            "record_created_at": chosen["record_created_at"],
            "recorded_before_test_evaluation": selection.get(
                "created_before_test_evaluation"
            ),
        },
        "model": {
            "path": chosen["path"].relative_to(path_root).as_posix()
            if chosen["path"].is_relative_to(path_root)
            else str(chosen["path"]),
            "sha256": model_sha,
            "sha256_matches_selection_record": True,
            "run_id": chosen["run_id"],
            "size_bytes": int(chosen["path"].stat().st_size),
            "load_options": {"compile": False, "safe_mode": True},
        },
        "manifest": {
            "path": manifest["path"].relative_to(path_root).as_posix()
            if manifest["path"].is_relative_to(path_root)
            else str(manifest["path"]),
            "sha256": manifest["sha256"],
            "sha256_matches_split_summary": True,
            "rows": len(test_records),
            "split_summary": manifest["split_summary"].name,
        },
        "class_order_check": order_check,
        "protocol": {
            "pipeline": (
                "src/data_pipeline.make_dataset with training=False: manifest "
                "order, no shuffle, augmentation inert, float32 0-255 RGB, "
                "preprocessing applied once inside the model"
            ),
            "image_size": image_size,
            "batch_size": batch_size,
            "seed": config["seed"],
            "final_partial_batch": "kept (drop_remainder=False)",
            "extra_normalization": "none",
            "augmentation": "none (training=False)",
            "fitting": "none — weights are loaded, never updated",
            "threshold_tuning": "none — prediction is argmax of softmax",
            "model_selection": (
                "none at evaluation time — the checkpoint was fixed by the "
                "validation selection record written before this run"
            ),
        },
        "coverage": coverage,
        "metrics": metrics,
        "confusion_matrix_row_normalized": {
            "order": class_order,
            "rows_true_columns_predicted": row_normalized_confusion(raw_confusion),
            "note": "each row scaled by that class's support; rows sum to 1",
        },
        "summary": {
            "images": metrics["images"],
            "correct": metrics["images"] - misclassified,
            "misclassified": misclassified,
        },
        "misclassified_grid": {
            "path": grid_path,
            "tracked_in_git": False,
            "reason": (
                "grid images are dataset photographs; .gitignore blocks "
                "models/* and **/sample_grid*"
            ),
        },
        "environment": train.collect_environment_info(),
        "git": train.collect_git_info(path_root),
        "predictions": prediction_rows,
        "honesty_notes": [
            "These are HELD-OUT TEST metrics for the four selected RealWaste "
            "classes only (metal, organic, paper, plastic). They are separate "
            "from the validation numbers used to pick this model.",
            "The test split was opened for the first time by this evaluation; "
            "no training, fine-tuning or model selection used it.",
            "Nothing here was fitted, tuned or chosen after seeing test "
            "results — the checkpoint comes from the selection record "
            "committed beforehand.",
            "These numbers do not measure rejection of unknown objects, "
            "out-of-distribution inputs, or deployment accuracy in the wild.",
            "Per-image scores are raw softmax outputs; no calibration was "
            "applied.",
        ],
    }
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
            "Evaluate the previously SELECTED model on the held-out test "
            "split exactly once (no fitting, no tuning, no test-set access "
            "during training or selection)."
        )
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION,
                        help="Selection record written before test evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to configs/training.json.")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR,
                        help="Directory holding test.csv and split_summary.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write test_evaluation.json.")
    parser.add_argument("--grids-dir", type=Path, default=DEFAULT_GRIDS_DIR,
                        help="Local (Git-ignored) folder for misclassification grids.")
    parser.add_argument("--no-grids", action="store_true",
                        help="Skip writing the misclassified-image grid.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Evaluation batch size (default: config batch_size).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, path_root: Path = REPO_ROOT) -> int:
    args = parse_args(argv)
    print("=== Test-set evaluation (milestone 10: selected model, one pass) ===")
    try:
        report = run_evaluation(
            selection_path=args.selection,
            config_path=args.config,
            metadata_dir=args.metadata_dir,
            path_root=path_root,
            output=args.output,
            grids_dir=None if args.no_grids else args.grids_dir,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[error] Evaluation did not run. Fix the cause above; do not "
            "re-run to chase better numbers.",
            file=sys.stderr,
        )
        return 1

    metrics = report["metrics"]
    print(f"Report : {args.output}")
    print(f"Model  : {report['model']['path']} sha256={report['model']['sha256'][:16]}...")
    print(f"Split  : test ({report['coverage']['manifest_rows']} images, "
          f"each evaluated exactly once)")
    print(f"Loss   : {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    for name in report["class_order_check"]["class_order"]:
        cell = metrics["per_class"][name]
        print(f"  {name:<8} precision={cell['precision']:.4f} "
              f"recall={cell['recall']:.4f} f1={cell['f1']:.4f} "
              f"support={cell['support']}")
    print(f"Misclassified: {report['summary']['misclassified']} "
          f"(grid: {report['misclassified_grid']['path'] or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
