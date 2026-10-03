"""Baseline frozen-base training for the waste classifier (milestone 6).

Scope
-----
Trains ONLY the classification head on a **frozen** ImageNet MobileNetV2
using `train.csv` + `validation.csv`. This module deliberately:

- reuses `src/data_pipeline.py` (manifests + tf.data) and `src/model.py`
  (model builder) instead of duplicating their logic;
- never opens the test manifest or test images (test evaluation belongs to
  a later milestone);
- writes every artifact of a run into a unique directory
  (`models/runs/baseline_<timestamp>/` by default) so the baseline checkpoint
  stays separate from future fine-tuned models.

Reproducibility
---------------
Seeds are applied to Python, NumPy, and TensorFlow, and deterministic ops
are enabled where TensorFlow supports them. Bitwise-identical results are
still only expected on the same hardware/OS/library versions — limits are
recorded in each run's `run_metadata.json`.

Usage
-----
    .venv\\Scripts\\python.exe src\\train.py --help
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

import data_pipeline
import model as model_builder

REPO_ROOT = data_pipeline.REPO_ROOT
DEFAULT_CONFIG = data_pipeline.DEFAULT_TRAINING_CONFIG
DEFAULT_METADATA_DIR = data_pipeline.DEFAULT_METADATA_DIR
DEFAULT_RUNS_DIR = REPO_ROOT / "models" / "runs"
DEFAULT_REPORTS_ROOT = REPO_ROOT / "models" / "metadata" / "runs"

# Every baseline run uses this prefix so baseline and future fine-tuned
# checkpoints can never land in the same folder by accident.
RUN_PROFILE = "baseline"

EARLY_STOPPING_PATIENCE = 3
MANIFEST_NAMES = ("train.csv", "validation.csv")  # test.csv is intentionally absent

DETERMINISM_LIMITS = [
    "Bitwise-identical results are expected only on the same hardware, OS, and library versions.",
    "Different GPU/CPU models, drivers, or oneDNN/cuDNN builds can change floating-point results slightly.",
    "PYTHONHASHSEED only affects hash randomization when set before the interpreter starts; "
    "the CLI sets it for child processes it launches, but a manually started Python needs it in the environment.",
    "Augmentation draws from TensorFlow's RNGs each epoch by design, so augmented pixel values differ per epoch.",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Seeds and environment
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, and TensorFlow; enable deterministic ops if possible.

    Returns a description of what was enabled (recorded in run metadata).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    op_determinism = "disabled"
    try:
        tf.config.experimental.enable_op_determinism()
        op_determinism = "enabled"
    except Exception as exc:  # noqa: BLE001 - availability differs by build/platform
        op_determinism = f"unavailable ({type(exc).__name__}: {exc})"

    return {
        "seed": seed,
        "seeded_rngs": ["python.random", "numpy", "tensorflow"],
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "enable_op_determinism": op_determinism,
        "limits": DETERMINISM_LIMITS,
    }


def collect_environment_info() -> dict[str, Any]:
    """Python/Keras/versions + available hardware (measured, not assumed)."""
    import keras  # standalone keras gives the reliable version string on TF 2.15

    devices = tf.config.list_physical_devices()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "physical_devices": [f"{d.device_type}:{d.name}" for d in devices],
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
    }


def collect_git_info(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit + whether the checkout has uncommitted changes."""
    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git command failed")
        return result.stdout.strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run_git("status", "--porcelain"))
        return {"commit": commit, "branch": branch, "uncommitted_changes": dirty, "status": "ok"}
    except Exception as exc:  # noqa: BLE001 - git may be absent or repo may not exist
        return {
            "commit": None,
            "branch": None,
            "uncommitted_changes": None,
            "status": f"unavailable ({type(exc).__name__}: {exc})",
        }


def collect_pip_freeze() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return result.stdout
        return f"# pip freeze failed: {result.stderr.strip()}"
    except Exception as exc:  # noqa: BLE001
        return f"# pip freeze failed: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Class weights and callbacks
# ---------------------------------------------------------------------------

def compute_class_weights(records: list[dict], class_order: list[str]) -> dict[str, float]:
    """Balanced weights computed from TRAINING labels only.

    weight(c) = n_train / (n_classes * n_train_count(c))

    The caller must pass training records; validation labels must never be
    used to compute weights (that would leak validation distribution into
    training decisions).
    """
    if not records:
        raise ValueError("Cannot compute class weights from an empty record list")
    counts = {name: 0 for name in class_order}
    for record in records:
        target = record["target"]
        if target not in counts:
            raise ValueError(f"Unknown target {target!r} in training records")
        counts[target] += 1

    zero = [name for name, count in counts.items() if count == 0]
    if zero:
        raise ValueError(
            f"Cannot compute class weights: no training images for class(es) {zero}"
        )

    total = len(records)
    num_classes = len(class_order)
    return {name: total / (num_classes * count) for name, count in counts.items()}


class HistoryCsvCallback(tf.keras.callbacks.Callback):
    """Append every completed epoch to history.csv immediately.

    Writing after each epoch means completed epochs survive if training is
    interrupted (Ctrl+C, Colab disconnect, power loss).
    """

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.fieldnames: list[str] | None = None

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = dict(logs or {})
        if self.fieldnames is None:
            # Fresh run directory -> history.csv does not exist yet.
            self.fieldnames = ["epoch"] + sorted(logs.keys())
            with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerow({"epoch": epoch + 1, **logs})
                # Flush + fsync so the row survives an immediate interruption.
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, lineterminator="\n")
                writer.writerow({"epoch": epoch + 1, **logs})
                handle.flush()
                os.fsync(handle.fileno())


def build_callbacks(
    run_dir: Path,
    *,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    use_early_stopping: bool = True,
) -> list[tf.keras.callbacks.Callback]:
    """EarlyStopping on val_loss + best-model checkpoint + incremental history."""
    callbacks: list[tf.keras.callbacks.Callback] = [HistoryCsvCallback(run_dir / "history.csv")]
    if use_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )
    return callbacks


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_training_plots(history_csv: Path, run_dir: Path) -> list[str]:
    """Loss and accuracy curves from history.csv -> run_dir/plots/*.png."""
    import matplotlib

    matplotlib.use("Agg")  # headless: never opens a window (Colab/CI safe)
    import matplotlib.pyplot as plt

    with open(history_csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    epochs = [int(row["epoch"]) for row in rows]
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def plot_series(keys: tuple[str, str], title: str, ylabel: str, filename: str) -> None:
        present = [key for key in keys if key in rows[0]]
        if not present:
            return
        plt.figure(figsize=(7, 4))
        for key in present:
            plt.plot(epochs, [float(row[key]) for row in rows], label=key)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        target = plots_dir / filename
        plt.savefig(target, dpi=120)
        plt.close()
        written.append(str(target.relative_to(run_dir).as_posix()))

    plot_series(("loss", "val_loss"), "Training vs validation loss", "loss", "loss_curve.png")
    plot_series(
        ("accuracy", "val_accuracy"),
        "Training vs validation accuracy",
        "accuracy",
        "accuracy_curve.png",
    )
    return written


SMALL_REPORT_FILES = (
    "run_metadata.json",
    "class_order.json",
    "history.csv",
    "environment_freeze.txt",
)


def export_run_reports(
    run_dir: Path,
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
) -> Path:
    """Copy the small, Git-trackable reports of a run (never the model).

    Copies: run_metadata.json, class_order.json, history.csv,
    environment_freeze.txt, plots/*.png — into models/metadata/runs/<run_id>/.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    destination = Path(reports_root) / run_dir.name
    destination.mkdir(parents=True, exist_ok=True)

    for name in SMALL_REPORT_FILES:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)

    plots_dir = run_dir / "plots"
    if plots_dir.is_dir():
        (destination / "plots").mkdir(exist_ok=True)
        for plot in sorted(plots_dir.glob("*.png")):
            shutil.copy2(plot, destination / "plots" / plot.name)

    if not (destination / "run_metadata.json").exists():
        raise FileNotFoundError(
            f"Nothing to export: {run_dir} has no run_metadata.json"
        )
    return destination


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def resolve_config_path(path_root: Path, configured: str) -> Path:
    candidate = Path(configured)
    return candidate if candidate.is_absolute() else path_root / candidate


def new_run_dir(output_dir: Path, profile: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{profile}_{stamp}"
    run_dir = base
    counter = 2
    while run_dir.exists():
        run_dir = output_dir / f"{profile}_{stamp}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def history_to_rows(history: dict[str, list[float]]) -> list[dict[str, float]]:
    keys = sorted(history.keys())
    length = max((len(history[key]) for key in keys), default=0)
    return [
        {"epoch": index + 1, **{key: float(history[key][index]) for key in keys}}
        for index in range(length)
    ]


def run_baseline_training(
    *,
    config_path: Path = DEFAULT_CONFIG,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    path_root: Path = REPO_ROOT,
    output_dir: Path = DEFAULT_RUNS_DIR,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    image_size: int | None = None,
    dropout: float | None = None,
    seed: int | None = None,
    class_weights: str = "off",
    weights: str | None = "imagenet",
    use_early_stopping: bool = True,
    export_reports: bool = True,
) -> dict[str, Any]:
    """Run one baseline training job. Returns the run summary dict.

    Raises clear errors when configuration, manifests, or ImageNet weights
    are unavailable — never falls back silently.
    """
    if class_weights not in ("off", "balanced"):
        raise ValueError(f"--class-weights must be 'off' or 'balanced', got {class_weights!r}")

    config = data_pipeline.load_training_config(config_path)  # KeyError/ValueError if broken
    seed = config["seed"] if seed is None else seed
    epochs = config["baseline_max_epochs"] if epochs is None else epochs
    batch_size = config["batch_size"] if batch_size is None else batch_size
    learning_rate = config["learning_rate"] if learning_rate is None else learning_rate
    image_size = config["image_size"] if image_size is None else image_size
    dropout = config["dropout"] if dropout is None else dropout
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    class_order = data_pipeline.load_class_order(
        resolve_config_path(path_root, config["class_mapping_path"])
    )

    # ONLY train + validation. The test manifest is never opened here.
    splits = data_pipeline.load_splits(
        metadata_dir=metadata_dir,
        class_order=class_order,
        path_root=path_root,
        splits=("train", "validation"),
    )
    train_records = splits["train"]
    validation_records = splits["validation"]

    determinism = set_seeds(seed)

    weights_values: dict[str, float] | None = None
    if class_weights == "balanced":
        weights_values = compute_class_weights(train_records, class_order)
    class_weights_info = {
        "enabled": weights_values is not None,
        "mode": class_weights,
        "method": (
            "balanced: n_train / (n_classes * n_train_count)"
            if weights_values is not None
            else "none"
        ),
        "computed_from": "train.csv labels only (validation labels never used)",
        "passed_to_model_fit": weights_values is not None,
        "values": weights_values,
    }

    train_dataset = data_pipeline.make_dataset(
        train_records,
        image_size=image_size,
        batch_size=batch_size,
        training=True,
        seed=seed,
    )
    validation_dataset = data_pipeline.make_dataset(
        validation_records,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        seed=seed,
    )

    # Raises RuntimeError with instructions if ImageNet weights are unavailable.
    classifier = model_builder.build_waste_classifier(
        num_classes=len(class_order),
        image_size=image_size,
        weights=weights,
        dropout_rate=dropout,
        learning_rate=learning_rate,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = new_run_dir(output_dir, RUN_PROFILE)
    started_at = utc_now_iso()
    started_monotonic = time.perf_counter()

    callbacks = build_callbacks(run_dir, use_early_stopping=use_early_stopping)

    try:
        history = classifier.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2,
            # No steps_per_epoch / validation_steps: every training and
            # validation sample is used, including the final partial batch.
        )
    except BaseException as exc:
        # Keep whatever history.csv already recorded; mark the run honestly.
        failure_metadata = {
            "run_id": run_dir.name,
            "status": f"interrupted ({type(exc).__name__}: {exc})",
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "note": "history.csv holds any epochs that completed before the interruption.",
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(failure_metadata, indent=2) + "\n", encoding="utf-8"
        )
        raise

    train_seconds = time.perf_counter() - started_monotonic
    history_rows = history_to_rows(history.history)
    if not history_rows:
        raise RuntimeError("Training finished without recording any epoch history")

    best_row = min(history_rows, key=lambda row: row.get("val_loss", float("inf")))
    best_model_path = run_dir / "best_model.keras"
    if not best_model_path.exists():
        raise RuntimeError(
            f"ModelCheckpoint did not write {best_model_path}; "
            "val_loss was never produced — check the training logs."
        )

    plots = write_training_plots(run_dir / "history.csv", run_dir)
    model_checksum = sha256_file(best_model_path)

    # Prove the saved artifact reloads in this environment (no data involved).
    reloaded = tf.keras.models.load_model(best_model_path)
    reload_ok = reloaded.output_shape == classifier.output_shape

    manifest_info = {}
    for name in MANIFEST_NAMES:
        manifest_path = Path(metadata_dir) / name
        manifest_info[name] = {
            "path": manifest_path.relative_to(path_root).as_posix()
            if manifest_path.is_relative_to(path_root)
            else manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
        }

    run_metadata: dict[str, Any] = {
        "run_id": run_dir.name,
        "profile": RUN_PROFILE,
        "status": "ok",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "wall_time_seconds": round(train_seconds, 2),
        "git": collect_git_info(path_root),
        "configuration": {
            "config_path": str(config_path),
            "epochs_requested": epochs,
            "epochs_completed": len(history_rows),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "dropout": dropout,
            "seed": seed,
            "backbone": "MobileNetV2",
            "backbone_weights": weights,
            "backbone_frozen": model_builder.base_model_is_frozen(classifier),
            "optimizer": "Adam",
            "loss": "SparseCategoricalCrossentropy",
            "early_stopping": {
                "enabled": use_early_stopping,
                "monitor": "val_loss",
                "patience": EARLY_STOPPING_PATIENCE,
                "restore_best_weights": True,
            },
        },
        "seed_and_determinism": determinism,
        "class_order": class_order,
        "class_index": {name: index for index, name in enumerate(class_order)},
        "class_weights": class_weights_info,
        "data": {
            "train_images": len(train_records),
            "validation_images": len(validation_records),
            "manifests": manifest_info,
            "validation_pass": (
                "full pass over every validation image including the final "
                "partial batch (validation_steps not set)"
            ),
            "test_manifest_used": False,
        },
        "environment": collect_environment_info(),
        "training": {
            "history": history_rows,
            "best": {
                key: best_row[key]
                for key in ("epoch", "loss", "accuracy", "val_loss", "val_accuracy")
                if key in best_row
            },
        },
        "model": {
            "file": "best_model.keras",
            "sha256": model_checksum,
            "reload_ok_in_this_environment": reload_ok,
            "parameter_counts": model_builder.parameter_counts(classifier),
        },
        "artifacts": {
            "plots": plots,
            "history_csv": "history.csv",
            "environment_freeze": "environment_freeze.txt",
            "class_order": "class_order.json",
        },
        "honesty_notes": [
            "All numbers in this file come from this actual training run only.",
            "No test-set metrics are computed by the baseline workflow.",
            "If this run used weights='none', it was a synthetic/offline check, not ImageNet training.",
        ],
    }

    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "class_order.json").write_text(
        json.dumps(
            {"class_order": class_order, "class_index": run_metadata["class_index"]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "environment_freeze.txt").write_text(collect_pip_freeze(), encoding="utf-8")

    exported_to: str | None = None
    if export_reports:
        destination = export_run_reports(run_dir, reports_root=reports_root)
        exported_to = (
            str(destination.relative_to(REPO_ROOT))
            if destination.is_relative_to(REPO_ROOT)
            else str(destination)
        )

    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "run_metadata": run_metadata,
        "exported_reports_to": exported_to,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline frozen-base MobileNetV2 training on train/validation "
            "manifests (no test-set access, no fine-tuning)."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to configs/training.json.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max epochs (default: baseline_max_epochs from config).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: config batch_size).")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Adam learning rate (default: config learning_rate).")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Input size (default: config image_size).")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Head dropout (default: config dropout).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: config seed).")
    parser.add_argument("--class-weights", choices=("off", "balanced"), default="off",
                        help="Compute balanced class weights from TRAINING labels only.")
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet",
                        help="'none' is only for offline/synthetic checks, never the baseline.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR,
                        help="Where run directories are created (keep on Drive in Colab).")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable EarlyStopping (synthetic smoke checks only).")
    parser.add_argument("--no-export-reports", action="store_true",
                        help="Skip copying small reports into models/metadata/runs/.")
    parser.add_argument("--export-reports", type=Path, default=None, metavar="RUN_DIR",
                        help="Export small reports from an existing RUN_DIR and exit (no training).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.export_reports is not None:
        try:
            destination = export_run_reports(args.export_reports)
        except (FileNotFoundError, OSError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        print(f"Exported small reports to {destination}")
        print("best_model.keras is NOT copied — model binaries stay out of Git.")
        return 0

    print("=== Baseline training (milestone 6: frozen base, no test-set access) ===")
    weights = None if args.weights == "none" else "imagenet"
    if weights is None:
        print("[warn] --weights none: this is a SYNTHETIC/offline check, "
              "not ImageNet-pretrained baseline training.")

    try:
        result = run_baseline_training(
            config_path=args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            dropout=args.dropout,
            seed=args.seed,
            class_weights=args.class_weights,
            weights=weights,
            use_early_stopping=not args.no_early_stopping,
            export_reports=not args.no_export_reports,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[error] Training did not start/finish. Fix the cause above and re-run.",
            file=sys.stderr,
        )
        return 1

    metadata = result["run_metadata"]
    best = metadata["training"]["best"]
    print(f"\nRun directory: {result['run_dir']}")
    print(f"Epochs completed: {metadata['configuration']['epochs_completed']} "
          f"(max {metadata['configuration']['epochs_requested']})")
    print(f"Best epoch: {best.get('epoch')} "
          f"val_loss={best.get('val_loss'):.4f} val_accuracy={best.get('val_accuracy'):.4f}")
    print(f"Model: {result['run_dir']}/best_model.keras "
          f"sha256={metadata['model']['sha256'][:16]}...")
    print(f"Class weights: {metadata['class_weights']['mode']} "
          f"(computed from train.csv only)")
    print(f"Git: {metadata['git']['commit']} "
          f"uncommitted_changes={metadata['git']['uncommitted_changes']}")
    if result["exported_reports_to"]:
        print(f"Small reports exported to: {result['exported_reports_to']}")
    print("No test-set evaluation, no fine-tuning in this milestone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
