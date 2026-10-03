"""Keras 3 checks for the controlled fine-tuning workflow.

Run with the Keras 3 environment (TensorFlow 2.20 / Keras 3 — the versions
that match the Colab training run and can load the baseline artifact):

    .venv-infer\\Scripts\\python.exe -m pytest tests\\test_finetune_keras3.py

Under TF 2.15 / Keras 2 (.venv) every test in this file skips cleanly,
because the Keras 3 baseline artifact cannot be loaded there.

These tests touch the REAL baseline artifact read-only (each test re-checks
its sha256 afterwards) and the REAL validation manifest, but they claim no
accuracy: the behavioral fit uses synthetic inputs, and the validation pass
checks only label alignment and softmax shape — no metric is recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import compare_validation  # noqa: E402
import data_pipeline  # noqa: E402
import finetune  # noqa: E402
import model as model_builder  # noqa: E402
from test_finetune import run_tiny_finetune, sha256_of  # noqa: E402

BASELINE = ROOT / "models" / "baseline_20261003_172906" / "best_model.keras"
VALIDATION_MANIFEST = ROOT / "data" / "metadata" / "validation.csv"

import keras

KERAS3 = int(keras.__version__.split(".")[0]) >= 3

pytestmark = pytest.mark.skipif(
    not KERAS3 or not BASELINE.is_file(),
    reason=(
        "requires Keras 3 (.venv-infer) and the restored baseline artifact "
        "models/baseline_20261003_172906/best_model.keras"
    ),
)


def real_validation_available() -> bool:
    if not VALIDATION_MANIFEST.is_file():
        return False
    first = VALIDATION_MANIFEST.read_text(encoding="utf-8").splitlines()
    if len(first) < 2:
        return False
    return (ROOT / first[1].split(",")[0]).exists()


def load_real_baseline() -> tuple[tf.keras.Model, str]:
    """Load the baseline and prove the file matches its committed checksum."""
    sha_before = sha256_of(BASELINE)
    run_metadata = ROOT / "models" / "metadata" / "runs" / "baseline_20261003_172906" / "run_metadata.json"
    recorded = json.loads(run_metadata.read_text(encoding="utf-8"))["model"]["sha256"]
    assert sha_before == recorded, "artifact differs from run_metadata.json"
    model = finetune.load_parent_model(BASELINE)
    return model, sha_before


def optimizer_lr(model: tf.keras.Model) -> float:
    lr = model.optimizer.learning_rate
    return float(lr.numpy() if hasattr(lr, "numpy") else lr)


# ---------------------------------------------------------------------------
# Real baseline: policy + recompile
# ---------------------------------------------------------------------------

def test_real_baseline_policy_and_recompile() -> None:
    model, sha_before = load_real_baseline()

    # The recorded backbone call must be the safe training=False (Keras 3
    # stores it in node.arguments.kwargs — this exercises that code path).
    assert finetune.backbone_training_flag(model) is False

    policy = finetune.apply_finetune_policy(model)
    assert policy["start_block"] == 13
    assert policy["start_layer"] == "block_13_expand"
    assert policy["backbone_layers"] == 154
    assert policy["backbone_call_training"] is False
    assert policy["batch_norm_layers_trainable"] == 0
    assert policy["head_trainable"] is True
    assert "predictions" in policy["trainable_layers_with_weights"]

    counts = model_builder.parameter_counts(model)
    assert counts["trainable"] > 0 and counts["non_trainable"] > 0

    # Fresh Adam at the fine-tune learning rate; no baseline optimizer reused.
    finetune.compile_finetune(model, 0.00001)
    assert optimizer_lr(model) == pytest.approx(1e-5)

    assert sha256_of(BASELINE) == sha_before  # file untouched by all of this


# ---------------------------------------------------------------------------
# Real baseline: training behavior + save/load round trip
# ---------------------------------------------------------------------------

def test_real_baseline_training_behavior_and_roundtrip(tmp_path: Path) -> None:
    model, sha_before = load_real_baseline()
    finetune.apply_finetune_policy(model)
    finetune.compile_finetune(model, 0.00001)

    backbone = finetune.find_backbone(model)
    bn_layer = backbone.get_layer("block_13_expand_BN")
    frozen_layer = backbone.get_layer("block_5_expand")
    trained_layer = backbone.get_layer("block_13_expand")
    head = model.get_layer("predictions")

    bn_before = [w.copy() for w in bn_layer.get_weights()]
    frozen_before = [w.copy() for w in frozen_layer.get_weights()]
    trained_before = [w.copy() for w in trained_layer.get_weights()]
    head_before = [w.copy() for w in head.get_weights()]

    # Synthetic behavioral check only (inputs/labels are random, NOT accuracy):
    # does one optimizer step update exactly the layers the policy allows?
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(0)
    images = rng.uniform(0.0, 255.0, size=(4, 224, 224, 3)).astype("float32")
    labels = rng.integers(0, 4, size=4).astype("int64")
    model.fit(images, labels, batch_size=2, epochs=1, verbose=0)

    bn_after = bn_layer.get_weights()
    assert all(
        np.array_equal(before, after) for before, after in zip(bn_before, bn_after)
    ), "BatchNorm running stats must not move (backbone call is training=False)"
    assert all(
        np.array_equal(before, after)
        for before, after in zip(frozen_before, frozen_layer.get_weights())
    ), "frozen block_5 must not change"
    assert not all(
        np.array_equal(before, after)
        for before, after in zip(trained_before, trained_layer.get_weights())
    ), "block_13_expand must receive updates"
    assert not all(
        np.array_equal(before, after)
        for before, after in zip(head_before, head.get_weights())
    ), "the head must receive updates"

    # Save -> reload -> identical predictions and a still-verifiable graph.
    probe = np.random.default_rng(1).uniform(0.0, 255.0, (1, 224, 224, 3)).astype(
        "float32"
    )
    expected = np.asarray(model(probe, training=False))
    roundtrip = tmp_path / "roundtrip.keras"
    model.save(roundtrip)
    reloaded = tf.keras.models.load_model(roundtrip, compile=False)
    actual = np.asarray(reloaded(probe, training=False))
    assert np.allclose(expected, actual, atol=1e-4), "predictions changed after reload"
    assert finetune.backbone_training_flag(reloaded) is False
    reloaded_policy = finetune.apply_finetune_policy(reloaded)
    assert reloaded_policy["start_layer"] == "block_13_expand"

    assert sha256_of(BASELINE) == sha_before  # baseline file never modified


# ---------------------------------------------------------------------------
# Full synthetic fine-tune workflow inside the Keras 3 environment
# ---------------------------------------------------------------------------

def test_synthetic_finetune_smoke_in_keras3(tmp_path: Path) -> None:
    result = run_tiny_finetune(tmp_path)
    assert result["status"] == "ok"

    metadata = json.loads(
        (Path(result["run_dir"]) / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["environment"]["tensorflow"] == tf.__version__
    assert metadata["fine_tuning_policy"]["backbone_call_training"] is False
    assert metadata["fine_tuning_policy"]["batch_norm_layers_trainable"] == 0
    assert metadata["parent"]["unchanged_after_training"] is True
    assert metadata["data"]["test_manifest_used"] is False
    assert metadata["model"]["reload_ok_in_this_environment"] is True
    assert metadata["model"]["sha256"] == sha256_of(
        Path(result["run_dir"]) / "best_model.keras"
    )


# ---------------------------------------------------------------------------
# Real validation manifest: prediction/label alignment (no metrics claimed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not real_validation_available(),
    reason="RealWaste validation images/manifest not present",
)
def test_real_validation_alignment_smoke() -> None:
    config = data_pipeline.load_training_config()
    class_order = data_pipeline.load_class_order()
    splits = data_pipeline.load_splits(
        metadata_dir=ROOT / "data" / "metadata",
        class_order=class_order,
        path_root=ROOT,
        splits=("validation",),
    )
    records = splits["validation"][:4]

    model = finetune.load_parent_model(BASELINE)
    probs, labels = compare_validation.collect_predictions(
        model,
        records,
        image_size=config["image_size"],
        batch_size=2,
        seed=config["seed"],
    )

    # Shape/alignment/softmax checks only — deliberately no accuracy or loss
    # is computed or stored here (real metrics come from compare_validation).
    assert probs.shape == (4, len(class_order))
    assert labels.astype("int64").tolist() == [
        int(record["class_index"]) for record in records
    ]
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
