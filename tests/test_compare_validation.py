"""Tests for src/compare_validation.py (validation comparison + model selection).

Synthetic checks only: hand-computed metric values, prediction/label
alignment, the documented selection rule (including ties and worse cases),
report contents, pending behavior when no fine-tuned model exists, and proof
that the test manifest is never opened. No accuracy is claimed anywhere.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
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
import model as model_builder  # noqa: E402
from test_train import CLASS_ORDER, build_tiny_manifests, write_configs  # noqa: E402


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_tiny_model(tmp_path: Path, name: str, seed: int) -> Path:
    """A saved synthetic model; different seed -> different weights/metrics."""
    tf.keras.utils.set_random_seed(seed)
    model = model_builder.build_waste_classifier(weights=None, image_size=32)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.keras"
    model.save(path)
    return path


def validation_records(tmp_path: Path) -> list[dict]:
    metadata = build_tiny_manifests(tmp_path)
    splits = data_pipeline.load_splits(
        metadata_dir=metadata,
        class_order=CLASS_ORDER,
        path_root=tmp_path,
        splits=("validation",),
    )
    return splits["validation"]


# ---------------------------------------------------------------------------
# Metric math (hand-computed expectations)
# ---------------------------------------------------------------------------

def test_metrics_from_predictions_hand_computed() -> None:
    order = ["a", "b", "c", "d"]
    labels = np.array([0, 1, 2, 3])
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],   # predicted a -> correct
            [0.1, 0.2, 0.6, 0.1],   # predicted c -> wrong (true b)
            [0.05, 0.05, 0.8, 0.1], # predicted c -> correct
            [0.05, 0.05, 0.1, 0.8], # predicted d -> correct
        ],
        dtype="float64",
    )
    metrics = compare_validation.metrics_from_predictions(probs, labels, order)

    assert metrics["images"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["loss"] == pytest.approx(
        float(-np.log(np.array([0.7, 0.2, 0.8, 0.8])).mean())
    )

    # confusion: rows = true, columns = predicted
    assert metrics["confusion_matrix"]["order"] == order
    assert metrics["confusion_matrix"]["rows_true_columns_predicted"] == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    assert metrics["per_class"]["a"] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "support": 1, "predicted_count": 1,
    }
    assert metrics["per_class"]["b"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "support": 1, "predicted_count": 0,
    }
    assert metrics["per_class"]["c"]["precision"] == pytest.approx(0.5)
    assert metrics["per_class"]["c"]["recall"] == pytest.approx(1.0)
    assert metrics["per_class"]["c"]["f1"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["c"]["predicted_count"] == 2
    assert metrics["per_class"]["d"]["f1"] == pytest.approx(1.0)

    assert metrics["macro_f1"] == pytest.approx((1 + 0 + 2 / 3 + 1) / 4)


def test_metrics_zero_division_policy_is_zero() -> None:
    order = ["a", "b"]
    labels = np.array([0, 0])
    probs = np.array([[0.1, 0.9], [0.1, 0.9]], dtype="float64")  # always predicts b
    metrics = compare_validation.metrics_from_predictions(probs, labels, order)

    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["macro_f1"] == 0.0  # both classes have f1 = 0
    # a: never predicted -> precision 0; recall 0 (tp=0)
    assert metrics["per_class"]["a"]["precision"] == 0.0
    assert metrics["per_class"]["a"]["recall"] == 0.0
    # b: predicted twice, never correct -> precision 0; no support -> recall 0
    assert metrics["per_class"]["b"]["precision"] == 0.0
    assert metrics["per_class"]["b"]["recall"] == 0.0
    assert metrics["per_class"]["b"]["support"] == 0
    assert metrics["per_class"]["b"]["predicted_count"] == 2


def test_metrics_reject_bad_shapes() -> None:
    # 3 probability rows but class_order lists only 3 classes.
    with pytest.raises(ValueError, match="does not match"):
        compare_validation.metrics_from_predictions(
            np.ones((3, 4)) / 4, np.array([0, 1, 2]), ["a", "b", "c"]
        )
    # 3 probability rows but only 2 labels.
    with pytest.raises(ValueError, match="does not match"):
        compare_validation.metrics_from_predictions(
            np.ones((3, 4)) / 4, np.array([0, 1]), ["a", "b", "c", "d"]
        )
    with pytest.raises(ValueError, match="zero samples"):
        compare_validation.metrics_from_predictions(
            np.ones((0, 4)) / 4, np.array([], dtype="int64"), ["a", "b", "c", "d"]
        )


# ---------------------------------------------------------------------------
# Prediction / label alignment
# ---------------------------------------------------------------------------

def test_collect_predictions_aligns_with_manifest_order(tmp_path: Path) -> None:
    records = validation_records(tmp_path)
    model = model_builder.build_waste_classifier(weights=None, image_size=32)
    probs, labels = compare_validation.collect_predictions(
        model, records, image_size=32, batch_size=2, seed=42
    )
    assert probs.shape == (len(records), 4)
    assert labels.astype("int64").tolist() == [
        int(record["class_index"]) for record in records
    ]
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_collect_predictions_rejects_non_softmax_outputs(tmp_path: Path) -> None:
    records = validation_records(tmp_path)
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(4, activation=None, name="predictions")(x)
    linear_model = tf.keras.Model(inputs, outputs)
    with pytest.raises(RuntimeError, match="sum to 1"):
        compare_validation.collect_predictions(
            linear_model, records, image_size=32, batch_size=2, seed=42
        )


def test_alignment_guard_detects_swapped_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = validation_records(tmp_path)
    model = model_builder.build_waste_classifier(weights=None, image_size=32)

    def fake_make_dataset(records_arg, *, image_size, batch_size, training, seed, **kwargs):
        del training, seed, kwargs
        labels = [int(record["class_index"]) for record in records_arg][::-1]
        images = tf.zeros(
            (len(records_arg), image_size, image_size, 3), dtype=tf.float32
        )
        return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)

    monkeypatch.setattr(data_pipeline, "make_dataset", fake_make_dataset)
    with pytest.raises(RuntimeError, match="alignment"):
        compare_validation.collect_predictions(
            model, records, image_size=32, batch_size=2, seed=42
        )


# ---------------------------------------------------------------------------
# Selection rule (fixed before running: macro F1, then accuracy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("baseline", "finetuned", "expected", "step"),
    [
        # macro F1 wins even when accuracy drops -> fine-tuned (step 1)
        ({"macro_f1": 0.5, "accuracy": 0.8}, {"macro_f1": 0.6, "accuracy": 0.7},
         "finetuned", 1),
        # macro F1 tied, accuracy higher -> fine-tuned (step 2)
        ({"macro_f1": 0.5, "accuracy": 0.8}, {"macro_f1": 0.5, "accuracy": 0.9},
         "finetuned", 2),
        # both tied -> keep baseline (step 3)
        ({"macro_f1": 0.5, "accuracy": 0.8}, {"macro_f1": 0.5, "accuracy": 0.8},
         "baseline", 3),
        # macro F1 tied, accuracy lower -> keep baseline (step 3)
        ({"macro_f1": 0.5, "accuracy": 0.8}, {"macro_f1": 0.5, "accuracy": 0.7},
         "baseline", 3),
        # macro F1 lower even though accuracy jumps -> keep baseline (step 3)
        ({"macro_f1": 0.5, "accuracy": 0.8}, {"macro_f1": 0.4, "accuracy": 0.99},
         "baseline", 3),
    ],
)
def test_decide_model_applies_documented_rule(
    baseline: dict, finetuned: dict, expected: str, step: int
) -> None:
    decision = compare_validation.decide_model(baseline, finetuned)
    assert decision["selected"] == expected
    assert f"rule step {step}" in decision["reason"]
    assert decision["measured"]["baseline"] == baseline
    assert decision["measured"]["finetuned"] == finetuned
    # The reason quotes the measured numbers it was based on.
    assert f"{baseline['macro_f1']:.6f}" in decision["reason"]


# ---------------------------------------------------------------------------
# Reports: complete and pending
# ---------------------------------------------------------------------------

def test_complete_comparison_writes_report_with_checksums(tmp_path: Path) -> None:
    config_path = write_configs(tmp_path)
    build_tiny_manifests(tmp_path)
    baseline_path = save_tiny_model(tmp_path, "baseline", seed=11)
    finetuned_path = save_tiny_model(tmp_path, "finetuned", seed=22)
    output = tmp_path / "out" / "validation_comparison.json"

    report = compare_validation.run_comparison(
        baseline_model=baseline_path,
        finetune_model=finetuned_path,
        config_path=config_path,
        metadata_dir=tmp_path / "data" / "metadata",
        path_root=tmp_path,
        output=output,
    )

    assert report["status"] == "complete"
    evaluation = report["evaluation"]
    assert evaluation["split"] == "validation"
    assert evaluation["images_evaluated"] == 4
    assert evaluation["test_manifest_used"] is False
    assert evaluation["class_order"] == CLASS_ORDER
    assert evaluation["manifest"]["sha256"] == sha256_of(
        tmp_path / "data" / "metadata" / "validation.csv"
    )

    # Both checksums recorded and correct — the report proves which files ran.
    assert evaluation["baseline"]["model"]["sha256"] == sha256_of(baseline_path)
    assert evaluation["finetuned"]["model"]["sha256"] == sha256_of(finetuned_path)
    assert evaluation["baseline"]["model"]["sha256"] != evaluation["finetuned"]["model"][
        "sha256"
    ]

    for side in ("baseline", "finetuned"):
        metrics = evaluation[side]["metrics"]
        assert set(metrics) == {
            "images", "loss", "accuracy", "macro_f1", "per_class", "confusion_matrix",
        }
        assert set(metrics["per_class"]) == set(CLASS_ORDER)
        confusion = metrics["confusion_matrix"]["rows_true_columns_predicted"]
        assert sum(sum(row) for row in confusion) == 4
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    decision = report["decision"]
    assert decision["selected"] in {"baseline", "finetuned"}
    assert decision["reason"].startswith("rule step")
    assert set(decision["measured"]) == {"baseline", "finetuned"}

    # The rule travels with the report so a later reader sees what decided.
    assert report["selection_rule"] == compare_validation.SELECTION_RULE
    assert report["selection_rule"]["test_set_used"] is False
    assert len(report["selection_rule"]["steps"]) == 3
    assert "macro F1" in " ".join(report["selection_rule"]["steps"])

    # Written file matches the returned report (minus the helper key).
    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert on_disk["status"] == "complete"
    assert on_disk["decision"] == decision
    assert "_output_path" not in on_disk


def test_pending_report_invents_no_metrics(tmp_path: Path) -> None:
    config_path = write_configs(tmp_path)
    baseline_path = save_tiny_model(tmp_path, "baseline", seed=11)
    missing = tmp_path / "models" / "not_trained_yet.keras"
    output = tmp_path / "out" / "pending.json"

    report = compare_validation.run_comparison(
        baseline_model=baseline_path,
        finetune_model=missing,
        config_path=config_path,
        output=output,
    )

    assert report["status"] == "pending"
    assert report["decision"]["selected"] is None
    assert report["decision"]["measured"] is None
    assert "pending" in report["decision"]["reason"]
    assert str(missing) in report["decision"]["reason"]
    # Baseline identity is still recorded (checksum of the real file)...
    assert report["evaluation"]["baseline"]["model"]["sha256"] == sha256_of(
        baseline_path
    )
    # ...but no measured metric values exist anywhere in the report
    # (search for metric keys, not the word "metrics" in rule prose).
    assert report["evaluation"]["finetuned"] is None
    assert "metrics" not in report["evaluation"]["baseline"]
    dumped = json.dumps(report)
    for word in ('"macro_f1"', '"per_class"', '"confusion_matrix"'):
        assert word not in dumped

    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert on_disk["status"] == "pending"


def test_pending_when_finetune_model_not_provided(tmp_path: Path) -> None:
    report = compare_validation.run_comparison(
        baseline_model=tmp_path / "anything.keras",
        finetune_model=None,
        config_path=write_configs(tmp_path),
        output=tmp_path / "pending.json",
    )
    assert report["status"] == "pending"
    assert "--finetune-model was not provided" in report["decision"]["reason"]
    # No baseline file either -> honest note, not an invented checksum.
    assert report["evaluation"]["baseline"]["model"]["sha256"] is None
    assert report["evaluation"]["baseline"]["model"]["note"] == "file not found"


# ---------------------------------------------------------------------------
# Test manifest must never be consumed
# ---------------------------------------------------------------------------

def test_comparison_never_loads_test_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_configs(tmp_path)
    metadata = build_tiny_manifests(tmp_path)
    (metadata / "test.csv").write_text("not,a,valid,manifest\n", encoding="utf-8")
    baseline_path = save_tiny_model(tmp_path, "baseline", seed=11)
    finetuned_path = save_tiny_model(tmp_path, "finetuned", seed=22)

    captured: dict = {}
    original = data_pipeline.load_splits

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(data_pipeline, "load_splits", spy)
    report = compare_validation.run_comparison(
        baseline_model=baseline_path,
        finetune_model=finetuned_path,
        config_path=config_path,
        metadata_dir=metadata,
        path_root=tmp_path,
        output=tmp_path / "out" / "comparison.json",
    )

    assert report["status"] == "complete"
    assert captured["splits"] == ("validation",)
    assert (metadata / "test.csv").read_text(encoding="utf-8") == "not,a,valid,manifest\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_help_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src" / "compare_validation.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert completed.returncode == 0
    # argparse wraps help text across lines -> compare on normalized whitespace.
    flat = " ".join(completed.stdout.split())
    assert "validation pipeline" in flat
    assert "no test-set access" in flat
    assert "pending" in flat


def test_main_pending_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "report.json"
    exit_code = compare_validation.main(
        [
            "--baseline-model", str(tmp_path / "missing_baseline.keras"),
            "--finetune-model", str(tmp_path / "missing_tuned.keras"),
            "--config", str(write_configs(tmp_path)),
            "--output", str(output),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Status: pending" in captured.out
    assert "not evaluated yet" in captured.out
    assert output.is_file()
