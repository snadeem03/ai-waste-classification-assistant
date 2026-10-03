"""Tests for src/finetune.py (controlled fine-tuning workflow).

Everything runs on tiny synthetic images in pytest's temporary directory
(outside tracked repo paths) with weights=None, so no network access and no
RealWaste download are required. These are *synthetic workflow checks* — not
pretrained fine-tuning, and no accuracy is claimed anywhere.

Helpers for configs/manifests are re-used from tests/test_train.py so the
baseline and fine-tune suites share one definition of the synthetic fixtures.
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

import data_pipeline  # noqa: E402
import finetune  # noqa: E402
import model as model_builder  # noqa: E402
from test_train import CLASS_ORDER, build_tiny_manifests, write_configs  # noqa: E402


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parent(
    tmp_path: Path,
    *,
    image_size: int = 32,
    with_sidecars: bool = True,
) -> Path:
    """A saved synthetic parent model (+ sibling reports), like the baseline bundle."""
    parent_dir = tmp_path / "parent_run"
    parent_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)
    model = model_builder.build_waste_classifier(weights=None, image_size=image_size)
    model_path = parent_dir / "best_model.keras"
    model.save(model_path)
    if with_sidecars:
        (parent_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "run_id": "baseline_synthetic",
                    "model": {"sha256": sha256_of(model_path)},
                    "class_order": CLASS_ORDER,
                    "training": {"best": {"epoch": 3, "val_loss": 0.5, "val_accuracy": 0.75}},
                    "git": {"commit": "f" * 40},
                }
            ),
            encoding="utf-8",
        )
        (parent_dir / "class_order.json").write_text(
            json.dumps({"class_order": CLASS_ORDER}), encoding="utf-8"
        )
    return model_path


def run_tiny_finetune(tmp_path: Path, **kwargs) -> dict:
    """Shared helper: one synthetic fine-tune run in a temp directory."""
    config_path = kwargs.pop("config_path", None) or write_configs(tmp_path)
    build_tiny_manifests(tmp_path)
    parent_model = kwargs.pop("parent_model", None) or build_parent(tmp_path)
    defaults = dict(
        parent_model=parent_model,
        config_path=config_path,
        metadata_dir=tmp_path / "data" / "metadata",
        path_root=tmp_path,
        output_dir=tmp_path / "runs",
        reports_root=tmp_path / "reports",
        epochs=1,
    )
    defaults.update(kwargs)
    return finetune.run_finetune(**defaults)


def build_variant_model(*, training_kwarg_present: bool, training_value=None) -> tf.keras.Model:
    """Small model whose MobileNetV2 backbone is called without training=False."""
    inputs = tf.keras.Input(shape=(32, 32, 3))
    base = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3), include_top=False, weights=None
    )
    base.trainable = False
    if training_kwarg_present:
        x = base(inputs, training=training_value)
    else:
        x = base(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs)


def build_plain_model() -> tf.keras.Model:
    """No backbone at all — the policy must refuse this clearly."""
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------
# Layer policy: block_13+, BatchNorm excluded, head kept
# ---------------------------------------------------------------------------

def test_policy_selects_block13_onward_and_freezes_batchnorm() -> None:
    net = model_builder.build_waste_classifier(weights=None, image_size=32)
    policy = finetune.apply_finetune_policy(net)

    backbone = finetune.find_backbone(net)
    assert policy["backbone"] == backbone.name
    assert policy["start_block"] == 13
    assert policy["start_layer"] == "block_13_expand"
    assert policy["backbone_call_training"] is False
    assert policy["backbone_layers"] == len(backbone.layers) == 154
    assert policy["batch_norm_layers"] == 52
    assert policy["batch_norm_layers_trainable"] == 0

    # Every BatchNorm layer stays frozen, everywhere in the backbone.
    assert all(
        not layer.trainable
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    )
    # Selected layers: non-BatchNorm from block_13 through the backbone end.
    assert backbone.get_layer("block_13_expand").trainable
    assert backbone.get_layer("block_13_project").trainable
    assert backbone.get_layer("Conv_1").trainable  # final conv = backbone end
    # Frozen: everything earlier, and every BN even inside block_13+.
    assert not backbone.get_layer("block_12_project").trainable
    assert not backbone.get_layer("block_13_expand_BN").trainable
    assert not backbone.get_layer("Conv_1_bn").trainable
    # The classification head keeps training.
    assert net.get_layer("predictions").trainable

    with_weights = policy["trainable_layers_with_weights"]
    assert "predictions" in with_weights
    assert "block_13_expand" in with_weights
    assert "Conv_1" in with_weights
    assert not any(name.endswith("_BN") or "_BN/" in name for name in with_weights)
    assert "block_12_project" not in with_weights


def test_policy_refuses_model_without_backbone() -> None:
    with pytest.raises(ValueError, match="MobileNetV2 backbone"):
        finetune.apply_finetune_policy(build_plain_model())


def test_policy_refuses_missing_start_block() -> None:
    net = model_builder.build_waste_classifier(weights=None, image_size=32)
    with pytest.raises(ValueError, match="block_99"):
        finetune.apply_finetune_policy(net, start_block=99)


def test_policy_refuses_backbone_called_with_training_true() -> None:
    model = build_variant_model(training_kwarg_present=True, training_value=True)
    assert finetune.backbone_training_flag(model) is True
    with pytest.raises(RuntimeError, match="training=True"):
        finetune.apply_finetune_policy(model)


def test_policy_refuses_unverifiable_training_flag() -> None:
    model = build_variant_model(training_kwarg_present=False)
    assert finetune.backbone_training_flag(model) is None
    with pytest.raises(RuntimeError, match="training=False"):
        finetune.apply_finetune_policy(model)


def test_policy_requires_predictions_head() -> None:
    inputs = tf.keras.Input(shape=(32, 32, 3))
    base = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3), include_top=False, weights=None
    )
    base.trainable = False
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs, outputs)
    with pytest.raises(ValueError, match="predictions"):
        finetune.apply_finetune_policy(model)


def test_backbone_training_flag_is_false_for_our_model() -> None:
    net = model_builder.build_waste_classifier(weights=None, image_size=32)
    assert finetune.backbone_training_flag(net) is False


# ---------------------------------------------------------------------------
# Fresh optimizer
# ---------------------------------------------------------------------------

def test_compile_finetune_creates_fresh_adam_at_fine_tune_rate() -> None:
    net = model_builder.build_waste_classifier(weights=None, image_size=32)
    original_optimizer = net.optimizer  # baseline default: Adam at 0.001
    assert float(original_optimizer.learning_rate) == pytest.approx(0.001)

    finetune.compile_finetune(net, 0.00001)

    assert isinstance(net.optimizer, tf.keras.optimizers.Adam)
    assert net.optimizer is not original_optimizer  # fresh instance, no old state
    assert float(net.optimizer.learning_rate) == pytest.approx(0.00001)
    assert isinstance(net.loss, tf.keras.losses.SparseCategoricalCrossentropy)
    compile_config = net.get_compile_config()
    assert compile_config["optimizer"]["class_name"] == "Adam"
    assert compile_config["loss"]["class_name"] == "SparseCategoricalCrossentropy"


def test_compile_finetune_rejects_non_positive_rate() -> None:
    net = model_builder.build_waste_classifier(weights=None, image_size=32)
    with pytest.raises(ValueError, match="learning_rate"):
        finetune.compile_finetune(net, 0.0)


# ---------------------------------------------------------------------------
# Fine-tune settings (config block + defaults)
# ---------------------------------------------------------------------------

def test_finetune_settings_defaults_without_config_block() -> None:
    settings = finetune.finetune_settings({"seed": 42})
    assert settings == finetune.FINETUNE_DEFAULTS
    assert settings["max_epochs"] == 10
    assert settings["learning_rate"] == 0.00001
    assert settings["start_block"] == 13


def test_finetune_settings_merge_config_block() -> None:
    settings = finetune.finetune_settings({"finetune": {"max_epochs": 5}})
    assert settings["max_epochs"] == 5
    assert settings["learning_rate"] == 0.00001  # untouched default
    assert settings["start_block"] == 13


def test_finetune_settings_reject_unknown_keys_and_bad_values() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        finetune.finetune_settings({"finetune": {"learning_rates": 1}})
    with pytest.raises(ValueError, match="max_epochs"):
        finetune.finetune_settings({"finetune": {"max_epochs": 0}})
    with pytest.raises(ValueError, match="learning_rate"):
        finetune.finetune_settings({"finetune": {"learning_rate": -1}})
    with pytest.raises(ValueError, match="start_block"):
        finetune.finetune_settings({"finetune": {"start_block": 0}})


# ---------------------------------------------------------------------------
# Full synthetic run + artifacts
# ---------------------------------------------------------------------------

def test_synthetic_finetune_run_writes_all_artifacts(tmp_path: Path) -> None:
    """SYNTHETIC check (weights=None) — workflow only, not real fine-tuning."""
    result = run_tiny_finetune(tmp_path)
    assert result["status"] == "ok"

    run_dir = Path(result["run_dir"])
    assert run_dir.name.startswith("finetune_")  # never collides with baseline_*
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

    with open(run_dir / "history.csv", encoding="utf-8", newline="") as handle:
        rows = handle.read().splitlines()
    assert len(rows) == 2  # header + 1 completed epoch

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "ok"
    assert metadata["profile"] == "finetune"

    parent = metadata["parent"]
    parent_path = tmp_path / "parent_run" / "best_model.keras"
    assert parent["sha256"] == sha256_of(parent_path)
    assert parent["checksum_verified"] is True
    assert parent["run_id"] == "baseline_synthetic"
    assert parent["baseline_best"] == {"epoch": 3, "val_loss": 0.5, "val_accuracy": 0.75}
    assert parent["unchanged_after_training"] is True
    assert parent["class_order_verified"] is True

    policy = metadata["fine_tuning_policy"]
    assert policy["start_block"] == 13
    assert policy["start_layer"] == "block_13_expand"
    assert policy["backbone_call_training"] is False
    assert policy["batch_norm_layers_trainable"] == 0
    assert "predictions" in policy["trainable_layers_with_weights"]

    configuration = metadata["configuration"]
    assert configuration["epochs_requested"] == 1
    assert configuration["epochs_completed"] == 1
    assert configuration["learning_rate"] == 0.00001
    assert configuration["batch_size"] == 2  # from the synthetic config
    assert configuration["image_size"] == 32  # derived from the parent model
    assert configuration["seed"] == 42
    assert configuration["start_block"] == 13
    assert configuration["finetune_defaults"] == finetune.FINETUNE_DEFAULTS
    assert configuration["early_stopping"]["patience"] == 3
    assert configuration["early_stopping"]["monitor"] == "val_loss"

    assert metadata["data"]["test_manifest_used"] is False
    assert set(metadata["data"]["manifests"]) == {"train.csv", "validation.csv"}
    assert metadata["data"]["train_images"] == 8
    assert metadata["data"]["validation_images"] == 4

    assert metadata["class_order"] == CLASS_ORDER
    assert metadata["class_weights"]["enabled"] is False
    assert metadata["seed_and_determinism"]["seed"] == 42
    assert metadata["environment"]["tensorflow"] == tf.__version__
    assert metadata["git"]["commit"] is not None or metadata["git"]["status"].startswith(
        "unavailable"
    )

    best = metadata["training"]["best"]
    assert isinstance(best["val_loss"], float) and best["val_loss"] >= 0.0
    assert 0.0 <= best["val_accuracy"] <= 1.0
    assert len(metadata["model"]["sha256"]) == 64
    assert metadata["model"]["reload_ok_in_this_environment"] is True
    counts = metadata["model"]["parameter_counts"]
    # Some parameters frozen (early blocks + BatchNorm), some trainable
    # (block_13+ includes the ~1.2M-weight Conv_1, so it outweighs the rest).
    assert counts["trainable"] > 0 and counts["non_trainable"] > 0
    assert counts["total"] == counts["trainable"] + counts["non_trainable"]

    # The parent file was not modified by training (independent re-check).
    assert sha256_of(parent_path) == parent["sha256"]

    # Small reports exported; the model binary was not.
    exported = Path(result["exported_reports_to"])
    assert (exported / "run_metadata.json").exists()
    assert (exported / "history.csv").exists()
    assert not (exported / "best_model.keras").exists()


def test_finetune_never_loads_test_manifest(
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
    result = run_tiny_finetune(tmp_path, config_path=config_path)

    assert result["status"] == "ok"
    assert captured["splits"] == ("train", "validation")
    assert (metadata / "test.csv").read_text(encoding="utf-8") == "not,a,valid,manifest\n"


def test_bn_stats_and_frozen_blocks_survive_the_run(tmp_path: Path) -> None:
    """BatchNorm running stats must equal the parent's; frozen blocks untouched."""
    result = run_tiny_finetune(tmp_path)
    parent = tf.keras.models.load_model(tmp_path / "parent_run" / "best_model.keras")
    tuned = tf.keras.models.load_model(Path(result["run_dir"]) / "best_model.keras")

    # Keras 2 get_layer() does not search nested models -> go via the backbone.
    parent_backbone = finetune.find_backbone(parent)
    tuned_backbone = finetune.find_backbone(tuned)

    parent_bn = parent_backbone.get_layer("block_13_expand_BN").get_weights()
    tuned_bn = tuned_backbone.get_layer("block_13_expand_BN").get_weights()
    assert all(np.array_equal(before, after) for before, after in zip(parent_bn, tuned_bn))

    parent_early = parent_backbone.get_layer("block_5_expand").get_weights()
    tuned_early = tuned_backbone.get_layer("block_5_expand").get_weights()
    assert all(np.array_equal(before, after) for before, after in zip(parent_early, tuned_early))

    # The unfrozen block_13 and the head must actually have been updated.
    parent_block13 = parent_backbone.get_layer("block_13_expand").get_weights()
    tuned_block13 = tuned_backbone.get_layer("block_13_expand").get_weights()
    assert not all(np.array_equal(a, b) for a, b in zip(parent_block13, tuned_block13))
    parent_head = parent.get_layer("predictions").get_weights()
    tuned_head = tuned.get_layer("predictions").get_weights()
    assert not all(np.array_equal(a, b) for a, b in zip(parent_head, tuned_head))


# ---------------------------------------------------------------------------
# Clear failures
# ---------------------------------------------------------------------------

def test_missing_parent_fails_clearly(tmp_path: Path) -> None:
    config_path = write_configs(tmp_path)  # class mapping must resolve first
    with pytest.raises(FileNotFoundError, match="Parent model not found"):
        finetune.run_finetune(
            parent_model=tmp_path / "nope.keras",
            config_path=config_path,
            path_root=tmp_path,
        )


def test_parent_checksum_mismatch_fails_clearly(tmp_path: Path) -> None:
    parent = build_parent(tmp_path)
    metadata_path = parent.parent / "run_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["model"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        finetune.inspect_parent(parent, sha256=sha256_of(parent), class_order=CLASS_ORDER)


def test_parent_class_order_mismatch_fails_clearly(tmp_path: Path) -> None:
    parent = build_parent(tmp_path)
    (parent.parent / "class_order.json").write_text(
        json.dumps({"class_order": list(reversed(CLASS_ORDER))}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="class order mismatch"):
        finetune.inspect_parent(parent, sha256=sha256_of(parent), class_order=CLASS_ORDER)


def test_image_size_mismatch_fails_clearly(tmp_path: Path) -> None:
    parent = build_parent(tmp_path, image_size=32)
    config_path = write_configs(tmp_path, image_size=64)  # disagrees with the parent
    build_tiny_manifests(tmp_path)
    with pytest.raises(ValueError, match="image_size"):
        finetune.run_finetune(
            parent_model=parent,
            config_path=config_path,
            metadata_dir=tmp_path / "data" / "metadata",
            path_root=tmp_path,
            output_dir=tmp_path / "runs",
        )


def test_output_count_mismatch_fails_clearly(tmp_path: Path) -> None:
    parent = build_parent(tmp_path, image_size=32, with_sidecars=False)
    config_path = write_configs(tmp_path)
    build_tiny_manifests(tmp_path)
    # Five classes configured, parent head outputs four.
    mapping_path = tmp_path / "configs" / "class_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["class_order"] = CLASS_ORDER + ["glass"]
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="outputs 4 classes"):
        finetune.run_finetune(
            parent_model=parent,
            config_path=config_path,
            metadata_dir=tmp_path / "data" / "metadata",
            path_root=tmp_path,
            output_dir=tmp_path / "runs",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_help_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src" / "finetune.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert completed.returncode == 0
    # argparse wraps help text across lines -> compare on normalized whitespace.
    flat = " ".join(completed.stdout.split())
    assert "Controlled fine-tuning" in flat
    assert "block_13" in flat
    assert "no test-set access" in flat


def test_main_reports_run_summary_without_training(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() must only read keys that exist in run metadata (regression-style
    guard, learned from the baseline milestone where a summary read a key from
    the wrong block and crashed after a successful run)."""
    fake_metadata = {
        "configuration": {"epochs_completed": 3, "epochs_requested": 10, "learning_rate": 0.00001},
        "training": {"best": {"epoch": 2, "val_loss": 0.4, "val_accuracy": 0.8}},
        "parent": {"model_file": "/x/best_model.keras", "sha256": "ab" * 32,
                   "unchanged_after_training": True},
        "fine_tuning_policy": {
            "start_layer": "block_13_expand",
            "backbone_trainable_layers": ["a"] * 25,
            "backbone_frozen_layers_count": 129,
            "batch_norm_layers": 52,
        },
        "model": {"sha256": "cd" * 32},
        "git": {"commit": "deadbeef", "uncommitted_changes": False},
    }
    monkeypatch.setattr(
        finetune,
        "run_finetune",
        lambda **kwargs: {
            "status": "ok",
            "run_dir": "/tmp/fake/finetune_20260101_000000",
            "run_metadata": fake_metadata,
            "exported_reports_to": "models/metadata/runs/finetune_20260101_000000",
        },
    )

    exit_code = finetune.main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Epochs        : 3 (max 10)" in out
    assert "unchanged: True" in out
    assert "val_loss=0.4000 val_accuracy=0.8000" in out
    assert "block_13_expand" in out
    assert "deadbeef" in out
    assert "compare_validation.py" in out


def test_main_reports_missing_parent_as_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = finetune.main(["--parent-model", r"C:\does\not\exist.keras"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[error]" in captured.err
    assert "Parent model not found" in captured.err
