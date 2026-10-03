"""Post-training compatibility verification for the baseline model.

Run this with the **inference** environment only:

    .venv-infer\\Scripts\\python.exe src\\verify_baseline_inference.py

Why a separate environment: the baseline artifact was saved by Keras 3.13.2
(TensorFlow 2.20, the Colab run). The project's main ``.venv`` ships Keras 2.15
and cannot deserialize it (see docs/load_failure_baseline_20261003_172906.txt).
``.venv-infer`` mirrors the training versions; ``.venv`` stays untouched.

What this script does (in order):
1. Checks the model file's SHA-256 against the original run metadata.
2. Loads ``best_model.keras`` with ``compile=False, safe_mode=True``
   (unsafe deserialization is never enabled).
3. Verifies input/output shapes and the class order from all three sources.
4. Runs a fixed synthetic RGB batch (0-255, float32) and checks the numbers:
   finite, four scores per image, softmax sums ~1, repeated inference stable.
5. Probes the graph to confirm MobileNetV2 preprocessing (0-255 -> [-1, 1])
   is embedded before the backbone and no extra normalization is applied
   outside the frozen pretrained base.
6. Runs a small explicitly recorded sample of *training* images through the
   shared preprocessing pipeline (src/data_pipeline.py) to prove end-to-end
   execution and index-to-label mapping.

Honesty notes baked into the report:
- Synthetic predictions prove execution only; they are NOT accuracy numbers.
- No test image is ever opened (only train.csv is read).
- No accuracy is computed from the real-image sample either; predictions are
  recorded so the index -> class_order label mapping can be inspected.
- The optional --save-predictions / --compare-predictions flags exist for a
  cross-machine (Colab vs Windows) comparison; parity is only claimable when
  both files exist, and this script never claims it on its own.

Exit code: 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_pipeline  # noqa: E402  (same-dir import, like src/train.py)

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "baseline_20261003_172906" / "best_model.keras"
DEFAULT_RUN_DIR = REPO_ROOT / "models" / "metadata" / "runs" / "baseline_20261003_172906"
DEFAULT_RUN_METADATA = DEFAULT_RUN_DIR / "run_metadata.json"
DEFAULT_MODEL_BUNDLE_DIR = REPO_ROOT / "models" / "baseline_20261003_172906"
DEFAULT_REPORT_PATH = (
    REPO_ROOT / "models" / "metadata" / "verification"
    / "post_training_compatibility_baseline_20261003_172906.json"
)
DEFAULT_TRAIN_MANIFEST = REPO_ROOT / "data" / "metadata" / "train.csv"

# Tolerances: generous enough for float32, tight enough to catch real problems.
SOFTMAX_SUM_ATOL = 1e-4
STABILITY_MAX_DIFF = 1e-6
PROBE_ATOL = 1e-4


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks (the model is ~10 MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fixed_synthetic_batch(batch_size: int = 2, image_size: int = 224, channels: int = 3) -> np.ndarray:
    """Deterministic (2, 224, 224, 3) float32 batch on the 0-255 scale.

    Built from a plain arithmetic pattern instead of a random generator so the
    exact same bytes can be produced on any machine (e.g. in Colab) for a
    like-for-like comparison — no RNG/version differences to worry about.
    Each batch member gets a distinct per-image offset: one full image
    (224*224*3 = 150528) is divisible by 256, so a single shared `arange % 256`
    would make every image identical and hide input-dependent bugs.
    """
    per_image = image_size * image_size * channels
    images = []
    for index in range(batch_size):
        values = np.arange(per_image, dtype=np.float32) + float(index * 97)
        images.append(np.mod(values, 256.0).reshape(image_size, image_size, channels))
    return np.stack(images).astype(np.float32)


def check(name: str, passed: bool, details) -> dict:
    """One row of the report: a check name, its verdict, and the evidence."""
    return {"name": name, "passed": bool(passed), "details": details}


def load_model_safely(model_path: Path):
    """Load with compile=False, safe_mode=True; return (model, warnings, error).

    Unsafe deserialization is never enabled. Any failure is captured as text
    (full traceback) so the report can show the actual error.
    """
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            model = tf.keras.models.load_model(
                str(model_path), compile=False, safe_mode=True
            )
            caught = [str(record.message) for record in records]
        return model, caught, None
    except Exception:  # noqa: BLE001 - the report must capture any load error
        return None, caught, traceback.format_exc()


def round_robin_sample(records: list[dict], class_order: list[str], sample_size: int) -> list[dict]:
    """Pick up to sample_size records, cycling through the classes.

    train.csv is sorted by source folder, so the plain first N rows would all
    be one class ("organic"). Round-robin across class_index values instead,
    so the sample exercises every slot of the class order — which is the point
    of the label-order check. Manifest rows themselves are never modified.
    """
    by_class: dict[int, list[dict]] = {}
    for record in records:
        by_class.setdefault(record["class_index"], []).append(record)
    sample: list[dict] = []
    position = 0
    while len(sample) < sample_size:
        progressed = False
        for class_index in sorted(by_class):
            bucket = by_class[class_index]
            if position < len(bucket):
                sample.append(bucket[position])
                progressed = True
                if len(sample) == sample_size:
                    break
        if not progressed:
            break
        position += 1
    return sample


def verify_shapes(model, class_order: list[str]) -> list[dict]:
    """Input/output shape checks plus the four-class assumption."""
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    return [
        check(
            "input_shape_is_224x224x3",
            input_shape == (None, 224, 224, 3),
            {"actual": list(input_shape)},
        ),
        check(
            "output_shape_has_four_classes",
            output_shape == (None, len(class_order)),
            {"actual": list(output_shape), "class_order": class_order},
        ),
    ]


def verify_synthetic(model, batch: np.ndarray) -> tuple[list[dict], dict]:
    """Feed the fixed batch three times and check every numerical property."""
    checks: list[dict] = []
    checks.append(
        check(
            "synthetic_batch_shape_dtype_range",
            batch.shape == (2, 224, 224, 3)
            and batch.dtype == np.float32
            and float(batch.min()) >= 0.0
            and float(batch.max()) <= 255.0,
            {
                "shape": list(batch.shape),
                "dtype": str(batch.dtype),
                "min": float(batch.min()),
                "max": float(batch.max()),
            },
        )
    )

    runs = [np.asarray(model(batch, training=False), dtype=np.float32) for _ in range(3)]
    predictions = runs[0]

    checks.append(
        check(
            "predictions_shape_2x4",
            predictions.shape == (2, 4),
            {"actual": list(predictions.shape)},
        )
    )
    checks.append(
        check(
            "predictions_finite",
            bool(np.isfinite(predictions).all()),
            {"nonfinite_count": int((~np.isfinite(predictions)).sum())},
        )
    )

    sums = predictions.sum(axis=1)
    max_sum_deviation = float(np.abs(sums - 1.0).max())
    checks.append(
        check(
            "softmax_sums_close_to_one",
            max_sum_deviation <= SOFTMAX_SUM_ATOL,
            {"sums": [float(s) for s in sums], "max_abs_deviation": max_sum_deviation},
        )
    )
    checks.append(
        check(
            "scores_within_0_1",
            bool((predictions >= 0.0).all() and (predictions <= 1.0).all()),
            {"min": float(predictions.min()), "max": float(predictions.max())},
        )
    )

    stability_diff = float(max(np.abs(run - predictions).max() for run in runs[1:]))
    checks.append(
        check(
            "repeated_inference_stable_training_false",
            stability_diff <= STABILITY_MAX_DIFF,
            {"runs": len(runs), "max_abs_diff_between_runs": stability_diff},
        )
    )

    evidence = {
        "input_shape": list(batch.shape),
        "input_scale": "float32 0-255",
        "predictions": predictions.tolist(),
        "note": (
            "Predictions on a synthetic pattern image prove execution only. "
            "They are not accuracy measurements."
        ),
    }
    return checks, evidence


def find_backbone(model):
    """The nested MobileNetV2 model (identified by name, like src/model.py)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    return None


def verify_preprocessing(model) -> list[dict]:
    """Prove 0-255 -> [-1, 1] happens inside the graph, exactly once, before
    the backbone — and that nothing normalizes again after the backbone."""
    checks: list[dict] = []

    backbone = find_backbone(model)
    checks.append(
        check(
            "backbone_layer_present",
            backbone is not None,
            {"layers": [f"{l.name} ({type(l).__name__})" for l in model.layers]},
        )
    )

    # Functional probe: model input -> the outer-graph tensor that feeds the
    # backbone (the preprocessing ops are graph ops, not layers, so the
    # backbone's own .input placeholder is not reachable this way). If
    # MobileNetV2 preprocessing is embedded, constant 0 maps to -1, 127.5 to
    # 0 and 255 to +1. A second application would shift 0 to ~-1.008.
    if backbone is not None:
        try:
            feeding = backbone._inbound_nodes[0].input_tensors
            if isinstance(feeding, list):
                feeding = feeding[0]
            probe = tf.keras.Model(model.input, feeding)
            probe_results = {}
            probe_ok = True
            for value, expected in ((0.0, -1.0), (127.5, 0.0), (255.0, 1.0)):
                feed = np.full((1, 224, 224, 3), value, dtype=np.float32)
                out = np.asarray(probe(feed, training=False), dtype=np.float32)
                stats = {
                    "input_value": value,
                    "expected": expected,
                    "min": float(out.min()),
                    "max": float(out.max()),
                    "mean": float(out.mean()),
                }
                if not np.allclose(out, expected, atol=PROBE_ATOL):
                    probe_ok = False
                probe_results[str(value)] = stats
            checks.append(
                check(
                    "mobilenetv2_preprocessing_embedded_and_applied_once",
                    probe_ok,
                    probe_results,
                )
            )
        except Exception:  # noqa: BLE001 - record the real error instead of dying
            checks.append(
                check(
                    "mobilenetv2_preprocessing_embedded_and_applied_once",
                    False,
                    {"error": traceback.format_exc()},
                )
            )

    # Structural: no normalization layer outside the frozen pretrained base.
    # (BatchNorm inside MobileNetV2 is part of the pretrained architecture.)
    stray = []
    for layer in model.layers:
        if backbone is not None and layer.name == backbone.name:
            continue
        if isinstance(
            layer,
            (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization,
             tf.keras.layers.Rescaling),
        ):
            stray.append(f"{layer.name} ({type(layer).__name__})")
    checks.append(
        check(
            "no_normalization_outside_pretrained_backbone",
            not stray,
            {"stray_layers": stray},
        )
    )

    # The preprocessing ops are serialized in the config; keep that as evidence.
    config_text = json.dumps(model.get_config())
    checks.append(
        check(
            "preprocessing_ops_in_saved_config",
            '"TrueDivide"' in config_text and '"Subtract"' in config_text,
            {
                "TrueDivide": '"TrueDivide"' in config_text,
                "Subtract": '"Subtract"' in config_text,
            },
        )
    )
    return checks


def verify_real_image_sample(
    model, class_order: list[str], manifest_path: Path, sample_size: int
) -> tuple[list[dict], dict]:
    """Run the first N rows of train.csv through the shared pipeline.

    Only train.csv is opened — the test manifest stays untouched. No accuracy
    is computed; the records exist to show execution and the index -> label
    mapping.
    """
    checks: list[dict] = []
    records = data_pipeline.load_manifest(
        manifest_path, class_order, path_root=REPO_ROOT, check_files=True
    )
    sample = round_robin_sample(records, class_order, sample_size)

    # training=False keeps the manifest order, so records line up with batches.
    dataset = data_pipeline.make_dataset(
        sample,
        image_size=224,
        batch_size=len(sample),
        training=False,
        seed=42,
    )
    images, labels = next(iter(dataset))
    images = np.asarray(images, dtype=np.float32)
    labels = [int(v) for v in labels.numpy()]

    checks.append(
        check(
            "sample_images_shape_dtype",
            images.shape == (len(sample), 224, 224, 3) and images.dtype == np.float32,
            {"shape": list(images.shape), "dtype": str(images.dtype)},
        )
    )

    order_ok = all(
        labels[i] == class_order.index(sample[i]["target"])
        for i in range(len(sample))
    )
    checks.append(
        check(
            "manifest_labels_match_class_order",
            order_ok,
            {
                "class_order": class_order,
                "sample": [
                    {"target": r["target"], "class_index": r["class_index"]}
                    for r in sample
                ],
            },
        )
    )

    predictions = np.asarray(model(images, training=False), dtype=np.float32)
    finite = bool(np.isfinite(predictions).all())
    sums_ok = bool(
        np.abs(predictions.sum(axis=1) - 1.0).max() <= SOFTMAX_SUM_ATOL
    )
    checks.append(
        check(
            "real_sample_execution_ok",
            predictions.shape == (len(sample), 4) and finite and sums_ok,
            {
                "predictions_shape": list(predictions.shape),
                "finite": finite,
                "max_sum_deviation": float(
                    np.abs(predictions.sum(axis=1) - 1.0).max()
                ),
            },
        )
    )

    rows = []
    for i, record in enumerate(sample):
        predicted_index = int(np.argmax(predictions[i]))
        rows.append(
            {
                "path": record["path"],
                "target": record["target"],
                "class_index": record["class_index"],
                "predicted_index": predicted_index,
                "predicted_label": class_order[predicted_index],
                "scores": [float(v) for v in predictions[i]],
            }
        )

    evidence = {
        "source_manifest": str(manifest_path.relative_to(REPO_ROOT).as_posix()),
        "test_manifest_opened": False,
        "sample_size": len(sample),
        "sample_rows": rows,
        "note": (
            "Recorded to verify execution and index -> label mapping only. "
            "No accuracy metric is computed from this training sample."
        ),
    }
    return checks, evidence


def export_predictions(path: Path, batch: np.ndarray, predictions: np.ndarray) -> None:
    """Save synthetic input fingerprint + outputs for a cross-machine compare."""
    payload = {
        "purpose": "cross-machine synthetic-inference comparison",
        "input_sha256": hashlib.sha256(batch.tobytes()).hexdigest(),
        "input_shape": list(batch.shape),
        "input_scale": "float32 0-255",
        "predictions": predictions.tolist(),
        "environment": environment_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[export] Synthetic predictions written to {path}")


def compare_predictions(path: Path, batch: np.ndarray, predictions: np.ndarray, tolerance: float) -> dict:
    """Compare this run's synthetic outputs against another machine's file.

    Parity may only be stated when BOTH files exist and the input fingerprints
    match; the result is reported as numbers (max abs diff), never asserted
    here as 'bitwise identical'.
    """
    other = read_json(path)
    if other.get("input_sha256") != hashlib.sha256(batch.tobytes()).hexdigest():
        return {"compared": False, "error": "input fingerprints differ; inputs are not identical"}
    theirs = np.asarray(other["predictions"], dtype=np.float32)
    if theirs.shape != predictions.shape:
        return {"compared": False, "error": f"shape mismatch {theirs.shape} vs {predictions.shape}"}
    max_diff = float(np.abs(theirs - predictions).max())
    return {
        "compared": True,
        "other_file": str(path),
        "other_environment": other.get("environment"),
        "max_abs_diff": max_diff,
        "tolerance": tolerance,
        "within_tolerance": max_diff <= tolerance,
        "note": (
            "Both outputs exist; compare max_abs_diff against the tolerance. "
            "Only claim numerical parity if this result is within tolerance."
        ),
    }


def environment_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "numpy": np.__version__,
        "physical_devices": [str(d) for d in tf.config.list_physical_devices()],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-training compatibility verification for the baseline model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--run-metadata", type=Path, default=DEFAULT_RUN_METADATA)
    parser.add_argument("--model-bundle-dir", type=Path, default=DEFAULT_MODEL_BUNDLE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=DEFAULT_TRAIN_MANIFEST,
        help="Manifest for the real-image sample (default: train.csv — the "
        "test manifest is never used).",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Write this run's synthetic inputs+outputs to a JSON file "
        "(run the same command elsewhere to produce a matching file).",
    )
    parser.add_argument(
        "--compare-predictions",
        type=Path,
        default=None,
        help="Compare this run's synthetic outputs against a JSON file produced "
        "by --save-predictions on another machine.",
    )
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=1e-3,
        help="Max abs diff tolerated by --compare-predictions (default 1e-3).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks: list[dict] = []
    errors: list[str] = []
    load_warnings: list[str] = []

    print("=== Baseline model post-training compatibility verification ===")
    print(f"environment: tensorflow {tf.__version__}, keras {tf.keras.__version__}")

    # --- 1. model checksum against the ORIGINAL run metadata ---------------
    run_metadata = read_json(args.run_metadata)
    expected_sha = run_metadata["model"]["sha256"]
    model_exists = args.model.exists()
    actual_sha = sha256_of(args.model) if model_exists else None
    checks.append(
        check(
            "model_file_exists",
            model_exists,
            {"path": str(args.model.relative_to(REPO_ROOT).as_posix())},
        )
    )
    checks.append(
        check(
            "sha256_matches_original_run_metadata",
            actual_sha == expected_sha,
            {"expected": expected_sha, "actual": actual_sha},
        )
    )

    # --- 2. safe load -------------------------------------------------------
    model, load_warnings, load_error = load_model_safely(args.model)
    if load_error is not None:
        errors.append(load_error)
    checks.append(
        check(
            "model_loads_compile_false_safe_mode_true",
            model is not None,
            {
                "load_options": {"compile": False, "safe_mode": True},
                "unsafe_deserialization_used": False,
                "warnings": load_warnings,
                "error": load_error,
            },
        )
    )

    # --- 3. shapes + class order from all sources ---------------------------
    class_order = data_pipeline.load_class_order()
    bundle_class_order = read_json(args.model_bundle_dir / "class_order.json")
    if isinstance(bundle_class_order, list):
        bundle_class_order = {"class_order": bundle_class_order}
    run_class_order = run_metadata["class_order"]
    checks.append(
        check(
            "class_order_consistent_across_sources",
            class_order == run_class_order == bundle_class_order["class_order"],
            {
                "configs/class_mapping.json": class_order,
                "run_metadata.json": run_class_order,
                "model_bundle/class_order.json": bundle_class_order["class_order"],
            },
        )
    )

    synthetic_evidence: dict = {}
    real_evidence: dict = {}
    compare_result: dict = {}

    if model is not None:
        checks.extend(verify_shapes(model, class_order))

        # --- 4. synthetic batch numerical checks ---------------------------
        batch = fixed_synthetic_batch()
        synthetic_checks, synthetic_evidence = verify_synthetic(model, batch)
        checks.extend(synthetic_checks)
        predictions = np.asarray(model(batch, training=False), dtype=np.float32)

        # --- 5. embedded preprocessing inspection --------------------------
        checks.extend(verify_preprocessing(model))

        # --- 6. small training-image sample via the shared pipeline ---------
        try:
            real_checks, real_evidence = verify_real_image_sample(
                model, class_order, args.sample_manifest, args.sample_size
            )
            checks.extend(real_checks)
        except Exception:  # noqa: BLE001 - record and mark failed
            errors.append(traceback.format_exc())
            checks.append(
                check("real_sample_execution_ok", False, {"error": traceback.format_exc()})
            )

        if args.save_predictions is not None:
            export_predictions(args.save_predictions, batch, predictions)
        if args.compare_predictions is not None:
            compare_result = compare_predictions(
                args.compare_predictions, batch, predictions, args.parity_tolerance
            )
            checks.append(
                check(
                    "cross_machine_comparison",
                    bool(compare_result.get("compared")) and bool(compare_result.get("within_tolerance")),
                    compare_result,
                )
            )

    # --- report -------------------------------------------------------------
    failed = [c["name"] for c in checks if not c["passed"]]
    status = "pass" if not failed else "fail"
    report = {
        "report_type": "post_training_compatibility_verification",
        "run_id": run_metadata.get("run_id"),
        "purpose": (
            "Verify the original baseline artifact loads and executes in the "
            "separate inference environment (.venv-infer). No retraining, no "
            "model conversion, no test-set access."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_versions(),
        "model": {
            "path": str(args.model.relative_to(REPO_ROOT).as_posix()),
            "sha256_expected": expected_sha,
            "sha256_actual": actual_sha,
            "sha256_match": actual_sha == expected_sha,
            "load_options": {"compile": False, "safe_mode": True},
            "unsafe_deserialization_used": False,
            "load_warnings": load_warnings,
        },
        "class_order": class_order,
        "input_shape": list(model.input_shape) if model is not None else None,
        "output_shape": list(model.output_shape) if model is not None else None,
        "synthetic_check": synthetic_evidence,
        "real_image_sample_check": real_evidence,
        "cross_machine_comparison": compare_result,
        "checks": checks,
        "errors": errors,
        "verification_status": status,
        "failed_checks": failed,
        "notes": [
            "Synthetic predictions verify execution only; they are not accuracy.",
            "No test image or test manifest was opened by this script.",
            "run_metadata.json and all original run artifacts were read, never written.",
            "Numerical parity with the Colab runtime is NOT claimed unless a "
            "cross-machine comparison file exists and passes the tolerance.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nchecks: {len(checks) - len(failed)}/{len(checks)} passed")
    for name in failed:
        print(f"  FAILED: {name}")
    if errors:
        print("errors captured in the report (see 'errors' field)")
    print(f"verification_status: {status}")
    print(f"report: {args.report}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
