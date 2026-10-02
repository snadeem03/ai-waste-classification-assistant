"""Tests for the refactored src/train.py (baseline training workflow).

Everything runs on tiny synthetic images in pytest's temporary directory
(outside tracked repo paths) with weights=None, so no network access and no
RealWaste download are required. These are *synthetic workflow checks* — not
pretrained baseline training.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import data_pipeline  # noqa: E402
import model as model_builder  # noqa: E402
import train  # noqa: E402

CLASS_ORDER = ["metal", "organic", "paper", "plastic"]
SOURCE_TO_TARGET = {
    "Metal": "metal",
    "Paper": "paper",
    "Plastic": "plastic",
    "Food Organics": "organic",
    "Vegetation": "organic",
}


def write_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, format="JPEG", quality=95)


def write_configs(tmp_path: Path, **overrides) -> Path:
    """Create configs/class_mapping.json + configs/training.json under tmp_path."""
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "class_mapping.json").write_text(
        json.dumps(
            {
                "dataset": "Synthetic",
                "source_to_target": SOURCE_TO_TARGET,
                "excluded_source_categories": ["Glass"],
                "class_order": CLASS_ORDER,
            }
        ),
        encoding="utf-8",
    )
    training = {
        "seed": 42,
        "image_size": 32,
        "batch_size": 2,
        "learning_rate": 0.001,
        "baseline_max_epochs": 1,
        "dropout": 0.2,
        "class_mapping_path": "configs/class_mapping.json",
    }
    training.update(overrides)
    config_path = configs / "training.json"
    config_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
    return config_path


def build_tiny_manifests(tmp_path: Path) -> Path:
    """train.csv + validation.csv with 4 classes. No test.csv on purpose."""
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    columns = ["path", "target", "class_index", "file_sha256", "decoded_sha256", "group_id"]
    metadata = tmp_path / "data" / "metadata"

    for split, per_class in (("train", 2), ("validation", 1)):
        rows = []
        for index, target in enumerate(CLASS_ORDER):
            for copy_index in range(per_class):
                relative = f"data/raw/tiny/{target}/{split}_{copy_index}.jpg"
                write_jpeg(tmp_path / relative, colors[index])
                rows.append(
                    {
                        "path": relative,
                        "target": target,
                        "class_index": CLASS_ORDER.index(target),
                        "file_sha256": "0" * 64,
                        "decoded_sha256": "1" * 64,
                        "group_id": "",
                    }
                )
        metadata.mkdir(parents=True, exist_ok=True)
        with open(metadata / f"{split}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return metadata


def run_tiny_training(tmp_path: Path, **kwargs) -> dict:
    """Shared helper: one synthetic baseline run in a temp directory."""
    config_path = kwargs.pop("config_path", None) or write_configs(tmp_path)
    build_tiny_manifests(tmp_path)
    defaults = dict(
        config_path=config_path,
        metadata_dir=tmp_path / "data" / "metadata",
        path_root=tmp_path,
        output_dir=tmp_path / "runs",
        reports_root=tmp_path / "reports",
        epochs=1,
        weights=None,
        use_early_stopping=False,
    )
    defaults.update(kwargs)
    return train.run_baseline_training(**defaults)


# ---------------------------------------------------------------------------
# Full synthetic run + artifacts
# ---------------------------------------------------------------------------

def test_synthetic_baseline_run_writes_all_artifacts(tmp_path: Path) -> None:
    """SYNTHETIC check (weights=None) — workflow only, not pretrained training."""
    result = run_tiny_training(tmp_path)
    assert result["status"] == "ok"

    run_dir = Path(result["run_dir"])
    assert run_dir.name.startswith("baseline_")  # baseline stays separate from fine-tunes
    for artifact in (
        "best_model.keras",
        "history.csv",
        "run_metadata.json",
        "class_order.json",
        "environment_freeze.txt",
        "plots/loss_curve.png",
        "plots/accuracy_curve.png",
    ):
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    # history.csv has a header plus one completed epoch.
    with open(run_dir / "history.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert {"epoch", "loss", "val_loss"} <= set(rows[0])

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "ok"
    assert metadata["data"]["test_manifest_used"] is False
    assert set(metadata["data"]["manifests"]) == {"train.csv", "validation.csv"}
    assert metadata["class_order"] == CLASS_ORDER
    assert metadata["class_weights"]["enabled"] is False
    assert metadata["class_weights"]["computed_from"].startswith("train.csv")
    # Best-epoch metrics are real floats from this run (never placeholders).
    best = metadata["training"]["best"]
    assert isinstance(best["val_loss"], float) and best["val_loss"] >= 0.0
    assert isinstance(best["val_accuracy"], float) and 0.0 <= best["val_accuracy"] <= 1.0
    assert len(metadata["model"]["sha256"]) == 64
    assert metadata["model"]["reload_ok_in_this_environment"] is True
    assert metadata["git"]["commit"] is not None or metadata["git"]["status"].startswith("unavailable")
    assert metadata["environment"]["tensorflow"]
    assert metadata["seed_and_determinism"]["seed"] == 42
    assert metadata["configuration"]["early_stopping"]["patience"] == 3

    class_order_payload = json.loads((run_dir / "class_order.json").read_text(encoding="utf-8"))
    assert class_order_payload["class_order"] == CLASS_ORDER

    # Small reports were exported; the model binary was not.
    exported = Path(result["exported_reports_to"])
    assert (exported / "run_metadata.json").exists()
    assert (exported / "history.csv").exists()
    assert (exported / "plots" / "loss_curve.png").exists()
    assert not (exported / "best_model.keras").exists()


# ---------------------------------------------------------------------------
# Test manifest must never be consumed
# ---------------------------------------------------------------------------

def test_baseline_never_loads_test_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_configs(tmp_path)
    metadata = build_tiny_manifests(tmp_path)
    # A test.csv that would explode if anything tried to read it.
    (metadata / "test.csv").write_text("not,a,valid,manifest\n", encoding="utf-8")

    captured: dict = {}
    original = data_pipeline.load_splits

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(data_pipeline, "load_splits", spy)
    result = run_tiny_training(tmp_path, config_path=config_path)

    assert result["status"] == "ok"
    assert captured["splits"] == ("train", "validation")
    # The broken test.csv survived untouched and unread.
    assert (metadata / "test.csv").read_text(encoding="utf-8") == "not,a,valid,manifest\n"


def test_baseline_works_without_any_test_manifest(tmp_path: Path) -> None:
    """No test.csv at all -> training must still succeed."""
    result = run_tiny_training(tmp_path)
    assert result["status"] == "ok"
    assert not (tmp_path / "data" / "metadata" / "test.csv").exists()


# ---------------------------------------------------------------------------
# Shared pipeline / model builder
# ---------------------------------------------------------------------------

def test_uses_shared_pipeline_and_model_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert train.data_pipeline is data_pipeline
    assert train.model_builder is model_builder

    calls = {"make_dataset": 0, "build_waste_classifier": 0}
    original_make = data_pipeline.make_dataset
    original_build = model_builder.build_waste_classifier

    def spy_make(*args, **kwargs):
        calls["make_dataset"] += 1
        return original_make(*args, **kwargs)

    def spy_build(*args, **kwargs):
        calls["build_waste_classifier"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(data_pipeline, "make_dataset", spy_make)
    monkeypatch.setattr(model_builder, "build_waste_classifier", spy_build)

    result = run_tiny_training(tmp_path)
    assert result["status"] == "ok"
    assert calls["make_dataset"] == 2  # one train dataset + one validation dataset
    assert calls["build_waste_classifier"] == 1


# ---------------------------------------------------------------------------
# Clear failures
# ---------------------------------------------------------------------------

def test_missing_manifests_fail_clearly(tmp_path: Path) -> None:
    config_path = write_configs(tmp_path)
    with pytest.raises(FileNotFoundError, match="train.csv"):
        train.run_baseline_training(
            config_path=config_path,
            metadata_dir=tmp_path / "does_not_exist",
            path_root=tmp_path,
            output_dir=tmp_path / "runs",
            weights=None,
        )


def test_invalid_config_fails_clearly(tmp_path: Path) -> None:
    bad_config = tmp_path / "training.json"
    bad_config.write_text(json.dumps({"seed": 42}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        train.run_baseline_training(config_path=bad_config, weights=None)


def test_imagenet_weight_failure_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_mobilenet_v2(**kwargs):  # noqa: ANN003 - mirrors Keras signature
        raise OSError("simulated offline download failure")

    monkeypatch.setattr(tf.keras.applications, "MobileNetV2", failing_mobilenet_v2)
    with pytest.raises(RuntimeError, match="ImageNet weights"):
        run_tiny_training(tmp_path, weights="imagenet")


# ---------------------------------------------------------------------------
# Callbacks, history, class weights, CLI
# ---------------------------------------------------------------------------

def test_build_callbacks_configuration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    callbacks = train.build_callbacks(run_dir)

    early_stopping = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)]
    checkpoint = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ModelCheckpoint)]
    history_writer = [cb for cb in callbacks if isinstance(cb, train.HistoryCsvCallback)]

    assert len(early_stopping) == 1
    assert early_stopping[0].monitor == "val_loss"
    assert early_stopping[0].patience == 3
    assert early_stopping[0].restore_best_weights is True

    assert len(checkpoint) == 1
    assert checkpoint[0].monitor == "val_loss"
    assert checkpoint[0].save_best_only is True
    assert checkpoint[0].save_weights_only is False
    assert Path(checkpoint[0].filepath).name == "best_model.keras"
    assert Path(checkpoint[0].filepath).parent == run_dir

    assert len(history_writer) == 1
    # Simulate two completed epochs -> header + two rows written immediately.
    callback = history_writer[0]
    callback.on_epoch_end(0, {"loss": 1.0, "accuracy": 0.5, "val_loss": 0.9, "val_accuracy": 0.6})
    callback.on_epoch_end(1, {"loss": 0.5, "accuracy": 0.8, "val_loss": 0.7, "val_accuracy": 0.7})
    with open(run_dir / "history.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[-1]["val_loss"] == "0.7"


def test_build_callbacks_can_disable_early_stopping(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    callbacks = train.build_callbacks(run_dir, use_early_stopping=False)
    names = [type(cb).__name__ for cb in callbacks]
    assert "EarlyStopping" not in names
    assert "ModelCheckpoint" in names


def test_compute_class_weights_uses_training_labels_only() -> None:
    records = (
        [{"target": "metal"}] * 6
        + [{"target": "organic"}] * 2
        + [{"target": "paper"}] * 3
        + [{"target": "plastic"}] * 1
    )
    weights = train.compute_class_weights(records, CLASS_ORDER)
    total, num_classes = 12, 4
    assert weights["metal"] == pytest.approx(total / (num_classes * 6))
    assert weights["plastic"] == pytest.approx(total / (num_classes * 1))
    assert weights["organic"] == pytest.approx(total / (num_classes * 2))

    with pytest.raises(ValueError, match="no training images"):
        train.compute_class_weights([{"target": "metal"}], ["metal", "glass"])


def test_class_weights_balanced_are_recorded_from_train_labels(tmp_path: Path) -> None:
    result = run_tiny_training(tmp_path, class_weights="balanced")
    metadata = result["run_metadata"]
    info = metadata["class_weights"]
    assert info["enabled"] is True
    assert info["passed_to_model_fit"] is True
    assert info["computed_from"] == "train.csv labels only (validation labels never used)"

    # Independently recompute from the training manifest and compare.
    with open(tmp_path / "data" / "metadata" / "train.csv", encoding="utf-8", newline="") as handle:
        train_targets = [row["target"] for row in csv.DictReader(handle)]
    counts = {name: train_targets.count(name) for name in CLASS_ORDER}
    total = len(train_targets)
    expected = {name: total / (len(CLASS_ORDER) * counts[name]) for name in CLASS_ORDER}
    for name in CLASS_ORDER:
        assert info["values"][name] == pytest.approx(expected[name])


def test_class_weights_off_by_default_in_metadata(tmp_path: Path) -> None:
    result = run_tiny_training(tmp_path)
    info = result["run_metadata"]["class_weights"]
    assert info["enabled"] is False
    assert info["mode"] == "off"
    assert info["values"] is None
    assert info["passed_to_model_fit"] is False


def test_cli_help_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert completed.returncode == 0
    # argparse wraps long descriptions, so match fragments that stay contiguous.
    assert "Baseline frozen-base MobileNetV2 training" in completed.stdout
    assert "test-set access" in completed.stdout


def test_main_prints_run_summary_without_training(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() must read keys that actually exist in run_metadata.

    Regression guard: the summary used to read training.epochs_completed
    (a configuration.* key), which crashed *after* training finished.
    """
    fake_metadata = {
        "configuration": {"epochs_completed": 3, "epochs_requested": 15},
        "training": {"best": {"epoch": 2, "val_loss": 0.5, "val_accuracy": 0.75}},
        "model": {"sha256": "abcdef0123456789" * 4},
        "class_weights": {"mode": "off"},
        "git": {"commit": "deadbeef", "uncommitted_changes": False},
    }
    monkeypatch.setattr(
        train,
        "run_baseline_training",
        lambda **kwargs: {
            "status": "ok",
            "run_dir": "/tmp/fake/baseline_20260101_000000",
            "run_metadata": fake_metadata,
            "exported_reports_to": "models/metadata/runs/baseline_20260101_000000",
        },
    )

    exit_code = train.main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Epochs completed: 3 (max 15)" in out
    assert "Best epoch: 2" in out
    assert "val_loss=0.5000" in out
    assert "val_accuracy=0.7500" in out
    assert "deadbeef" in out
    assert "No test-set evaluation" in out
