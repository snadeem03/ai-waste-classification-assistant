"""Controlled fine-tuning of the verified baseline (milestone 9).

Scope
-----
- Loads a trained **parent** model (the baseline `best_model.keras`), unfreezes
  ONLY the MobileNetV2 backbone layers from `block_13` onward (excluding every
  BatchNormalization layer), keeps the classification head trainable, and
  trains with a **fresh** Adam optimizer at learning rate 0.00001 for at most
  10 epochs (EarlyStopping + best-checkpoint on `val_loss`, patience 3).
- Uses the same manifests and class-weight policy as the baseline
  (`train.csv` + `validation.csv` only). The test manifest is **never opened**.
- Writes a separate run directory (`models/runs/finetune_<timestamp>/`) that
  records the parent run id and parent checksum, so a fine-tuned checkpoint can
  never overwrite or silently replace the baseline.

Important choices
-----------------
- **Why block_13:** MobileNetV2's last blocks hold the most task-specific
  features; the early blocks are generic edges/textures that ~3k images would
  degrade. Training only the last ~25 backbone layers keeps the risk low.
  The selection is index-based (first layer whose name starts with the block
  prefix), because MobileNetV2's first block is named `expanded_conv`, not
  `block_1`.
- **Why BatchNorm stays frozen and the backbone keeps `training=False`:**
  BatchNorm updates its running mean/variance from the current batch whenever
  it runs in training mode — on small batches that corrupts the statistics the
  rest of the network relies on. Two protections are applied: every BN layer is
  frozen (no trainable weights), and the baseline graph already calls the
  backbone with `training=False`, which is verified after loading — if the flag
  cannot be verified as `False`, fine-tuning refuses to start.
- **Why a fresh optimizer:** Adam keeps per-weight momentum from the baseline's
  lr 0.001 training. Reusing it at lr 0.00001 would make the first steps wrong,
  so `compile()` builds a brand-new Adam and records that fact.
- **Why the parent is checksummed before and after:** training must never
  modify the baseline artifact. `parent.unchanged_after_training` in
  `run_metadata.json` proves the file bytes stayed identical.

Usage
-----
    .venv\\Scripts\\python.exe src\\finetune.py --help          # help (any env)
    .venv-infer\\Scripts\\python.exe src\\finetune.py           # Keras 3 env (matches Colab)
Real fine-tuning runs happen on Colab via notebooks/finetune_colab.ipynb.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

import data_pipeline
import model as model_builder
import train

DEFAULT_CONFIG = data_pipeline.DEFAULT_TRAINING_CONFIG
DEFAULT_METADATA_DIR = data_pipeline.DEFAULT_METADATA_DIR
DEFAULT_RUNS_DIR = train.DEFAULT_RUNS_DIR
DEFAULT_REPORTS_ROOT = train.DEFAULT_REPORTS_ROOT
DEFAULT_PARENT_MODEL = (
    data_pipeline.REPO_ROOT / "models" / "baseline_20261003_172906" / "best_model.keras"
)

# Every fine-tune run uses this prefix so fine-tuned checkpoints and baseline
# checkpoints can never land in the same folder by accident.
RUN_PROFILE = "finetune"

# Defaults for the fine-tune knobs. They can be overridden by the optional
# "finetune" block in configs/training.json and by CLI flags; the resolved
# values are recorded in run_metadata.json.
FINETUNE_DEFAULTS: dict[str, Any] = {
    "max_epochs": 10,
    "learning_rate": 0.00001,
    "start_block": 13,
}
FINETUNE_CONFIG_KEYS = frozenset(FINETUNE_DEFAULTS)

POLICY_TEXT = (
    "Backbone MobileNetV2: only non-BatchNorm layers from block_{start_block} "
    "onward are trainable; all earlier backbone layers and every "
    "BatchNormalization layer stay frozen. The backbone keeps being called "
    "with training=False, so BN running statistics never change. The "
    "classification head (global_average_pooling, dropout, predictions) "
    "remains trainable."
)


# ---------------------------------------------------------------------------
# Fine-tune settings and layer policy
# ---------------------------------------------------------------------------

def finetune_settings(config: dict) -> dict[str, Any]:
    """Resolve fine-tune knobs from the optional "finetune" config block.

    Missing block -> FINETUNE_DEFAULTS. Unknown keys fail loudly, because a
    typo (e.g. "learning_rates") would otherwise silently keep the default.
    """
    block = config.get("finetune", {})
    if not isinstance(block, dict):
        raise ValueError("training config: 'finetune' must be a JSON object")
    unknown = sorted(set(block) - FINETUNE_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"training config [finetune] has unknown keys: {unknown}; "
            f"allowed keys: {sorted(FINETUNE_CONFIG_KEYS)}"
        )

    settings: dict[str, Any] = {**FINETUNE_DEFAULTS, **block}
    if not isinstance(settings["max_epochs"], int) or settings["max_epochs"] <= 0:
        raise ValueError(
            f"training config [finetune].max_epochs must be a positive integer, "
            f"got {settings['max_epochs']!r}"
        )
    if not isinstance(settings["learning_rate"], (int, float)) or settings["learning_rate"] <= 0:
        raise ValueError(
            f"training config [finetune].learning_rate must be > 0, "
            f"got {settings['learning_rate']!r}"
        )
    if not isinstance(settings["start_block"], int) or settings["start_block"] < 1:
        raise ValueError(
            f"training config [finetune].start_block must be an integer >= 1, "
            f"got {settings['start_block']!r}"
        )
    return settings


def find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Locate the nested MobileNetV2 backbone; fail clearly if structure differs."""
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower()
    ]
    if len(backbones) != 1:
        raise ValueError(
            f"expected exactly one nested MobileNetV2 backbone in model "
            f"{model.name!r}, found {len(backbones)} "
            f"({[backbone.name for backbone in backbones]})"
        )
    return backbones[0]


def backbone_training_flag(model: tf.keras.Model) -> bool | None:
    """The `training` kwarg recorded in the graph for the backbone call.

    The baseline graph calls the backbone with `training=False`, which keeps
    BatchNorm in inference mode. Keras 3 records the call in
    `node.arguments.kwargs`; Keras 2 records it in `node.call_kwargs`.
    Returns None when the flag is not recorded at all.
    """
    backbone = find_backbone(model)
    nodes = getattr(backbone, "_inbound_nodes", [])
    if not nodes:
        return None
    node = nodes[0]

    kwargs: dict | None = None
    arguments = getattr(node, "arguments", None)
    if arguments is not None and hasattr(arguments, "kwargs"):
        kwargs = arguments.kwargs
    elif hasattr(node, "call_kwargs"):
        kwargs = node.call_kwargs
    if not isinstance(kwargs, dict):
        return None

    value = kwargs.get("training")
    return None if value is None else bool(value)


def apply_finetune_policy(
    model: tf.keras.Model,
    *,
    start_block: int = FINETUNE_DEFAULTS["start_block"],
) -> dict[str, Any]:
    """Apply the selective-unfreeze policy and verify the structure.

    Order matters: `backbone.trainable = True` is set FIRST (it re-enables
    every sub-layer), then each layer gets its individual flag. Returns a
    description dict that is recorded in run metadata.

    Raises a clear error when the backbone, the start block, the head, or the
    recorded `training=False` flag cannot be found — never continues with an
    unexpected structure.
    """
    backbone = find_backbone(model)

    # 1. The backbone must already be called with training=False in the graph.
    flag = backbone_training_flag(model)
    if flag is None:
        raise RuntimeError(
            f"cannot verify that {backbone.name} is called with training=False "
            "(the recorded call has no 'training' kwarg); refusing to fine-tune "
            "a graph whose BatchNorm mode is unknown"
        )
    if flag is not False:
        raise RuntimeError(
            f"{backbone.name} is called with training=True; fine-tuning would "
            "update BatchNorm running statistics and corrupt them. The parent "
            "model must keep the baseline's training=False backbone call."
        )

    # 2. The requested start block must exist.
    prefix = f"block_{start_block}"
    start_index = next(
        (index for index, layer in enumerate(backbone.layers) if layer.name.startswith(prefix)),
        None,
    )
    if start_index is None:
        present = sorted(
            {
                int(layer.name.split("_")[1])
                for layer in backbone.layers
                if layer.name.startswith("block_") and layer.name.split("_")[1].isdigit()
            }
        )
        raise ValueError(
            f"{backbone.name} has no layer starting with {prefix!r}; "
            f"blocks present: {present}"
        )

    # 3. The classification head must exist.
    try:
        head = model.get_layer("predictions")
    except ValueError as exc:
        raise ValueError(
            f"parent model {model.name!r} has no 'predictions' head layer — "
            "expected the waste-classifier structure from src/model.py"
        ) from exc

    # 4. Apply the flags (parent first, then per-layer — see docstring).
    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= start_index and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    head.trainable = True

    # 5. Self-check: fail loudly if anything unexpected slipped through.
    batch_norm_trainable = [
        layer.name
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable
    ]
    if batch_norm_trainable:
        raise RuntimeError(
            f"BatchNorm layers must never train, but these are trainable: "
            f"{batch_norm_trainable}"
        )
    wrongly_frozen = [
        layer.name
        for index, layer in enumerate(backbone.layers)
        if index >= start_index
        and not layer.trainable
        and not isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    if wrongly_frozen:
        raise RuntimeError(
            f"layers from {prefix} onward must train, but these are frozen: "
            f"{wrongly_frozen}"
        )
    if not head.trainable:
        raise RuntimeError("classification head ('predictions') must stay trainable")

    trainable_backbone = [layer.name for layer in backbone.layers if layer.trainable]
    frozen_backbone = [layer.name for layer in backbone.layers if not layer.trainable]
    head_layers = [layer.name for layer in model.layers if not isinstance(layer, tf.keras.Model)]
    trainable_with_weights = [
        layer.name
        for layer in list(backbone.layers) + [head]
        if layer.trainable and layer.weights
    ]
    batch_norm_total = sum(
        1 for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
    )

    return {
        "start_block": start_block,
        "start_layer": backbone.layers[start_index].name,
        "backbone": backbone.name,
        "backbone_layers": len(backbone.layers),
        "backbone_call_training": flag,
        "backbone_trainable_layers": trainable_backbone,
        "backbone_frozen_layers_count": len(frozen_backbone),
        "batch_norm_layers": batch_norm_total,
        "batch_norm_layers_trainable": 0,
        "trainable_layers_with_weights": trainable_with_weights,
        "head_layers": head_layers,
        "head_trainable": head.trainable,
        "policy": POLICY_TEXT.format(start_block=start_block),
    }


def compile_finetune(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Fresh optimizer: new Adam at the fine-tune learning rate.

    The parent was loaded with compile=False, so no baseline optimizer state
    exists to reuse; compile() builds a brand-new Adam instance.
    """
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Parent (baseline) loading and verification
# ---------------------------------------------------------------------------

def load_parent_model(parent_model: Path) -> tf.keras.Model:
    """Load the parent artifact for training (no optimizer state needed)."""
    parent_model = Path(parent_model)
    if not parent_model.is_file():
        raise FileNotFoundError(
            f"Parent model not found: {parent_model}. "
            "Pass --parent-model pointing at the baseline best_model.keras "
            "(see models/README.md for where it lives after restoring the Colab zip)."
        )
    try:
        return tf.keras.models.load_model(parent_model, compile=False)
    except Exception as exc:  # noqa: BLE001 - deserialization errors vary by version
        raise RuntimeError(
            f"Could not load parent model {parent_model} "
            f"({type(exc).__name__}: {exc}). If the file was saved by a different "
            "TensorFlow/Keras major version, load it in the matching environment "
            "(.venv = TF 2.15/Keras 2 tests; .venv-infer = TF 2.20/Keras 3 for the "
            "Keras 3 baseline artifact; Colab = the fine-tuning environment)."
        ) from exc


def inspect_parent(
    parent_model: Path,
    *,
    sha256: str,
    class_order: list[str],
) -> dict[str, Any]:
    """Verify the parent against its own sidecar reports (when they exist).

    - run_metadata.json must record the same model checksum we just computed
      (otherwise the artifact was modified since training — refuse to start).
    - class_order.json / run_metadata class_order must match the configured
      class order (otherwise labels would silently shift).
    Missing sidecar files are recorded honestly as None, not invented.
    """
    info: dict[str, Any] = {
        "run_id": None,
        "code_commit": None,
        "baseline_best": None,
        "metadata_file": None,
        "checksum_verified": None,
        "class_order_verified": None,
        "class_order_source": None,
    }

    metadata_path = parent_model.parent / "run_metadata.json"
    if metadata_path.is_file():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        info["metadata_file"] = metadata_path.as_posix()
        info["run_id"] = payload.get("run_id")
        git = payload.get("git") or {}
        info["code_commit"] = git.get("commit")
        info["baseline_best"] = (payload.get("training") or {}).get("best")

        recorded = (payload.get("model") or {}).get("sha256")
        if recorded is not None:
            if recorded != sha256:
                raise ValueError(
                    f"parent model checksum mismatch: {metadata_path.name} records "
                    f"sha256 {recorded} but the file hashes to {sha256}. The artifact "
                    "changed since training — refusing to fine-tune it."
                )
            info["checksum_verified"] = True

        recorded_order = payload.get("class_order")
        if recorded_order is not None:
            if recorded_order != class_order:
                raise ValueError(
                    f"class order mismatch: {metadata_path.name} records "
                    f"{recorded_order} but configs/class_mapping.json says "
                    f"{class_order}"
                )
            info["class_order_verified"] = True
            info["class_order_source"] = metadata_path.name

    class_order_path = parent_model.parent / "class_order.json"
    if class_order_path.is_file():
        recorded_order = json.loads(class_order_path.read_text(encoding="utf-8")).get(
            "class_order"
        )
        if recorded_order != class_order:
            raise ValueError(
                f"class order mismatch: {class_order_path.name} records "
                f"{recorded_order} but configs/class_mapping.json says {class_order}"
            )
        info["class_order_verified"] = True
        info["class_order_source"] = class_order_path.name

    return info


# ---------------------------------------------------------------------------
# Fine-tuning run
# ---------------------------------------------------------------------------

def run_finetune(
    *,
    parent_model: Path = DEFAULT_PARENT_MODEL,
    config_path: Path = DEFAULT_CONFIG,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    path_root: Path = data_pipeline.REPO_ROOT,
    output_dir: Path = DEFAULT_RUNS_DIR,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    start_block: int | None = None,
    class_weights: str = "off",
    use_early_stopping: bool = True,
    export_reports: bool = True,
) -> dict[str, Any]:
    """Run one controlled fine-tuning job. Returns the run summary dict.

    Raises clear errors for missing/broken parents, unexpected structure, or
    missing manifests — never falls back silently.
    """
    if class_weights not in ("off", "balanced"):
        raise ValueError(f"--class-weights must be 'off' or 'balanced', got {class_weights!r}")

    config = data_pipeline.load_training_config(config_path)
    settings = finetune_settings(config)
    seed = config["seed"] if seed is None else seed
    epochs = settings["max_epochs"] if epochs is None else epochs
    batch_size = config["batch_size"] if batch_size is None else batch_size
    learning_rate = settings["learning_rate"] if learning_rate is None else learning_rate
    start_block = settings["start_block"] if start_block is None else start_block
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if start_block < 1:
        raise ValueError(f"start_block must be >= 1, got {start_block}")

    class_order = data_pipeline.load_class_order(
        train.resolve_config_path(path_root, config["class_mapping_path"])
    )

    # Parent artifact: existence + checksum + sidecar verification, first.
    parent_model = Path(parent_model)
    if not parent_model.is_file():
        raise FileNotFoundError(
            f"Parent model not found: {parent_model}. "
            "Pass --parent-model pointing at the baseline best_model.keras."
        )
    parent_sha_before = train.sha256_file(parent_model)
    parent_info = inspect_parent(parent_model, sha256=parent_sha_before, class_order=class_order)

    classifier = load_parent_model(parent_model)

    # Structure checks: the pipeline and head must match this parent.
    input_shape = getattr(classifier, "input_shape", None)
    if not (isinstance(input_shape, tuple) and len(input_shape) == 4):
        raise ValueError(
            f"parent model input shape {input_shape!r} is not (batch, H, W, C); "
            "unexpected structure"
        )
    image_size = int(input_shape[1])
    if image_size != config["image_size"]:
        raise ValueError(
            f"parent model expects image_size {image_size} but "
            f"{config_path.name} says {config['image_size']} — they must match "
            "or every image would be fed at the wrong resolution"
        )
    num_outputs = int(classifier.output_shape[-1])
    if num_outputs != len(class_order):
        raise ValueError(
            f"parent model outputs {num_outputs} classes but class_order has "
            f"{len(class_order)}: {class_order}"
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

    determinism = train.set_seeds(seed)

    weights_values: dict[str, float] | None = None
    if class_weights == "balanced":
        weights_values = train.compute_class_weights(train_records, class_order)
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

    # Selective unfreeze + verification, then a fresh optimizer.
    policy = apply_finetune_policy(classifier, start_block=start_block)
    compile_finetune(classifier, learning_rate)

    dropout_rate: float | None = None
    try:
        dropout_rate = float(classifier.get_layer("dropout").rate)
    except (ValueError, AttributeError):
        dropout_rate = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = train.new_run_dir(output_dir, RUN_PROFILE)
    started_at = train.utc_now_iso()
    started_monotonic = time.perf_counter()

    callbacks = train.build_callbacks(run_dir, use_early_stopping=use_early_stopping)

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
            "finished_at_utc": train.utc_now_iso(),
            "parent_model": str(parent_model),
            "parent_sha256": parent_sha_before,
            "note": "history.csv holds any epochs that completed before the interruption.",
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(failure_metadata, indent=2) + "\n", encoding="utf-8"
        )
        raise

    train_seconds = time.perf_counter() - started_monotonic
    history_rows = train.history_to_rows(history.history)
    if not history_rows:
        raise RuntimeError("Fine-tuning finished without recording any epoch history")

    best_row = min(history_rows, key=lambda row: row.get("val_loss", float("inf")))
    best_model_path = run_dir / "best_model.keras"
    if not best_model_path.exists():
        raise RuntimeError(
            f"ModelCheckpoint did not write {best_model_path}; "
            "val_loss was never produced — check the training logs."
        )

    plots = train.write_training_plots(run_dir / "history.csv", run_dir)
    model_checksum = train.sha256_file(best_model_path)

    # Prove the saved artifact reloads in this environment (no data involved).
    reloaded = tf.keras.models.load_model(best_model_path)
    reload_ok = reloaded.output_shape == classifier.output_shape

    # The baseline artifact must be byte-identical after training.
    parent_sha_after = train.sha256_file(parent_model)
    if parent_sha_after != parent_sha_before:
        raise RuntimeError(
            f"parent model file changed during fine-tuning "
            f"({parent_sha_before} -> {parent_sha_after}); the baseline artifact "
            "must never be modified"
        )

    manifest_info = {}
    for name in train.MANIFEST_NAMES:
        manifest_path = Path(metadata_dir) / name
        manifest_info[name] = {
            "path": manifest_path.relative_to(path_root).as_posix()
            if manifest_path.is_relative_to(path_root)
            else manifest_path.as_posix(),
            "sha256": train.sha256_file(manifest_path),
        }

    run_metadata: dict[str, Any] = {
        "run_id": run_dir.name,
        "profile": RUN_PROFILE,
        "status": "ok",
        "started_at_utc": started_at,
        "finished_at_utc": train.utc_now_iso(),
        "wall_time_seconds": round(train_seconds, 2),
        "git": train.collect_git_info(path_root),
        "parent": {
            "model_file": parent_model.as_posix(),
            "sha256": parent_sha_before,
            "size_bytes": int(parent_model.stat().st_size),
            "unchanged_after_training": True,
            **parent_info,
        },
        "fine_tuning_policy": policy,
        "configuration": {
            "config_path": str(config_path),
            "finetune_defaults": FINETUNE_DEFAULTS,
            "epochs_requested": epochs,
            "epochs_completed": len(history_rows),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "seed": seed,
            "start_block": start_block,
            "dropout_loaded_from_parent": dropout_rate,
            "optimizer": "Adam (fresh instance; baseline optimizer state not reused)",
            "loss": "SparseCategoricalCrossentropy",
            "early_stopping": {
                "enabled": use_early_stopping,
                "monitor": "val_loss",
                "patience": train.EARLY_STOPPING_PATIENCE,
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
        "environment": train.collect_environment_info(),
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
            "All numbers in this file come from this fine-tuning run only "
            "(validation metrics only; no test-set metrics).",
            "The parent model is whatever --parent-model pointed at; its checksum "
            "and sidecar reports were verified before training and the file is "
            "re-checked afterwards (parent.unchanged_after_training).",
            "Validation metrics from this run are NOT directly comparable to the "
            "baseline's recorded best: run src/compare_validation.py, which "
            "evaluates both models on the identical validation pipeline.",
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
    (run_dir / "environment_freeze.txt").write_text(
        train.collect_pip_freeze(), encoding="utf-8"
    )

    exported_to: str | None = None
    if export_reports:
        destination = train.export_run_reports(run_dir, reports_root=reports_root)
        exported_to = (
            str(destination.relative_to(data_pipeline.REPO_ROOT))
            if destination.is_relative_to(data_pipeline.REPO_ROOT)
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
            "Controlled fine-tuning of a trained baseline model: MobileNetV2 "
            "block_13+ (BatchNorm excluded), fresh Adam at 1e-5, "
            "train/validation manifests only (no test-set access)."
        )
    )
    parser.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT_MODEL,
                        help="Trained parent artifact (default: the verified baseline).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to configs/training.json.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max epochs (default: finetune.max_epochs, 10).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: config batch_size, 16).")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Fresh Adam learning rate (default: 0.00001).")
    parser.add_argument("--start-block", type=int, default=None,
                        help="First MobileNetV2 block to unfreeze (default: 13).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: config seed, 42).")
    parser.add_argument("--class-weights", choices=("off", "balanced"), default="off",
                        help="Compute balanced class weights from TRAINING labels only.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR,
                        help="Where run directories are created (keep on Drive in Colab).")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable EarlyStopping (synthetic smoke checks only).")
    parser.add_argument("--no-export-reports", action="store_true",
                        help="Skip copying small reports into models/metadata/runs/.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Controlled fine-tuning (milestone 9: block_13+, no test-set access) ===")

    try:
        result = run_finetune(
            parent_model=args.parent_model,
            config_path=args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            start_block=args.start_block,
            class_weights=args.class_weights,
            use_early_stopping=not args.no_early_stopping,
            export_reports=not args.no_export_reports,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[error] Fine-tuning did not start/finish. Fix the cause above and re-run.",
            file=sys.stderr,
        )
        return 1

    metadata = result["run_metadata"]
    best = metadata["training"]["best"]
    policy = metadata["fine_tuning_policy"]
    print(f"\nRun directory: {result['run_dir']}")
    print(f"Parent        : {metadata['parent']['model_file']}")
    print(f"Parent sha256 : {metadata['parent']['sha256'][:16]}... "
          f"(unchanged: {metadata['parent']['unchanged_after_training']})")
    print(f"Epochs        : {metadata['configuration']['epochs_completed']} "
          f"(max {metadata['configuration']['epochs_requested']})")
    print(f"Best epoch     : {best.get('epoch')} "
          f"val_loss={best.get('val_loss'):.4f} val_accuracy={best.get('val_accuracy'):.4f}")
    print(f"Learning rate  : {metadata['configuration']['learning_rate']}")
    print(f"Unfrozen from  : {policy['start_layer']} "
          f"({len(policy['backbone_trainable_layers'])} backbone layers trainable, "
          f"{policy['backbone_frozen_layers_count']} frozen, "
          f"{policy['batch_norm_layers']} BatchNorm frozen)")
    print(f"Model          : {result['run_dir']}/best_model.keras "
          f"sha256={metadata['model']['sha256'][:16]}...")
    print(f"Git            : {metadata['git']['commit']} "
          f"uncommitted_changes={metadata['git']['uncommitted_changes']}")
    if result["exported_reports_to"]:
        print(f"Small reports exported to: {result['exported_reports_to']}")
    print("No test-set evaluation here — compare both models on validation with "
          "src/compare_validation.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
