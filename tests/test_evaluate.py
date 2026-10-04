"""Tests for src/evaluate.py (held-out test-set evaluation).

Synthetic checks only — tiny images, an untrained `weights=None` model, and
hand-computed expectations. Nothing here measures real accuracy; the tests
prove the *workflow*: metric math, prediction/label alignment, complete
sample coverage, checksum guard rails, and that evaluation never fits.
"""

from __future__ import annotations

import csv
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
import evaluate  # noqa: E402
import model as model_builder  # noqa: E402
from test_train import CLASS_ORDER, build_tiny_manifests, write_configs, write_jpeg  # noqa: E402

TEST_IMAGES_PER_CLASS = 2  # 4 classes -> 8 test images


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_test_manifest(tmp_path: Path, metadata: Path) -> Path:
    """Add data/metadata/test.csv (build_tiny_manifests creates none on purpose)."""
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    columns = ["path", "target", "class_index", "file_sha256", "decoded_sha256", "group_id"]
    rows = []
    for index, target in enumerate(CLASS_ORDER):
        for copy_index in range(TEST_IMAGES_PER_CLASS):
            relative = f"data/raw/tiny/{target}/test_{copy_index}.jpg"
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
    with open(metadata / "test.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return metadata / "test.csv"


def write_split_summary(metadata: Path) -> Path:
    """The committed facts evaluate.py insists on: test checksum + row count."""
    checksums = {}
    for name in ("train", "validation", "test"):
        manifest = metadata / f"{name}.csv"
        if manifest.is_file():
            checksums[f"{name}.csv"] = sha256_of(manifest)
    summary = {
        "manifest_checksums_sha256": checksums,
        "split_totals": {
            "train": 8, "validation": 4,
            "test": TEST_IMAGES_PER_CLASS * len(CLASS_ORDER),
        },
        "class_order": CLASS_ORDER,
    }
    path = metadata / "split_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def save_tiny_model(tmp_path: Path, seed: int = 7) -> Path:
    tf.keras.utils.set_random_seed(seed)
    model = model_builder.build_waste_classifier(weights=None, image_size=32)
    path = tmp_path / "selected" / "best_model.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def write_selection(tmp_path: Path, model_path: Path, *, sha256: str | None = None) -> Path:
    """A selection record like the real one, but pointing at the tiny model."""
    record = {
        "report_type": "selected_model_record",
        "created_at_utc": "2026-10-04T00:00:00+00:00",
        "created_before_test_evaluation": True,
        "selected": "finetuned",
        "selection_reason": "rule step 1: macro F1 improved 0.100000 -> 0.200000",
        "selection_rule": compare_validation.SELECTION_RULE,
        "selected_model": {
            "path": str(model_path),  # absolute: tmp trees are outside the repo
            "sha256": sha256 or sha256_of(model_path),
            "size_bytes": model_path.stat().st_size,
        },
        "run": {"run_id": "finetune_synthetic"},
        "class_order": CLASS_ORDER,
    }
    path = tmp_path / "selection" / "selected_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def build_fixture(tmp_path: Path, *, tamper_model_sha: str | None = None) -> dict:
    config_path = write_configs(tmp_path)
    metadata = build_tiny_manifests(tmp_path)
    build_test_manifest(tmp_path, metadata)
    write_split_summary(metadata)
    model_path = save_tiny_model(tmp_path)
    selection_path = write_selection(tmp_path, model_path, sha256=tamper_model_sha)
    return {
        "config_path": config_path,
        "metadata": metadata,
        "model_path": model_path,
        "selection_path": selection_path,
    }


def run_fixture_evaluation(tmp_path: Path, fixture: dict, **kwargs) -> dict:
    defaults = dict(
        selection_path=fixture["selection_path"],
        config_path=fixture["config_path"],
        metadata_dir=fixture["metadata"],
        path_root=tmp_path,
        output=tmp_path / "out" / "test_evaluation.json",
        grids_dir=None,
        batch_size=3,  # 8 rows -> batches of 3, 3, 2 (final partial batch kept)
    )
    defaults.update(kwargs)
    return evaluate.run_evaluation(**defaults)


# ---------------------------------------------------------------------------
# Metric helpers (hand-computed)
# ---------------------------------------------------------------------------

def test_row_normalized_confusion_hand_computed() -> None:
    raw = [[3, 1], [0, 2], [0, 0]]
    normalized = evaluate.row_normalized_confusion(raw)
    assert normalized[0] == pytest.approx([0.75, 0.25])
    assert normalized[1] == pytest.approx([0.0, 1.0])
    assert normalized[2] == [0.0, 0.0]  # empty support row stays all zeros
    for row in normalized:
        assert sum(row) == pytest.approx(1.0) or row == [0.0, 0.0]


def test_build_prediction_rows_aligns_paths_targets_and_scores() -> None:
    records = [
        {"path": "data/raw/a.jpg", "target": "metal", "class_index": 0},
        {"path": "data/raw/b.jpg", "target": "paper", "class_index": 2},
    ]
    probs = np.array([[0.6, 0.1, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]])
    rows = evaluate.build_prediction_rows(records, probs, CLASS_ORDER)

    assert [row["path"] for row in rows] == ["data/raw/a.jpg", "data/raw/b.jpg"]
    assert rows[0]["predicted"] == "metal"
    assert rows[0]["correct"] is True
    assert rows[1]["predicted"] == "plastic"  # argmax of last column
    assert rows[1]["correct"] is False
    assert rows[1]["target"] == "paper"
    for row in rows:
        assert len(row["scores"]) == 4
        assert sum(row["scores"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Coverage: every image exactly once
# ---------------------------------------------------------------------------

def test_coverage_accepts_complete_pass() -> None:
    records = [{"path": f"p{i}"} for i in range(8)]
    coverage = evaluate.assert_every_image_once(records, 8, 8)
    assert coverage["each_image_evaluated_exactly_once"] is True
    assert coverage["unique_image_paths"] == 8


@pytest.mark.parametrize(
    ("predictions", "expected_rows", "match"),
    [
        (7, 8, "predictions"),        # one image never scored
        (9, 8, "predictions"),        # one image scored twice
        (8, 9, "split_summary"),      # manifest and committed count disagree
    ],
)
def test_coverage_rejects_incomplete_or_excessive_pass(
    predictions: int, expected_rows: int, match: str
) -> None:
    records = [{"path": f"p{i}"} for i in range(8)]
    with pytest.raises(RuntimeError, match=match):
        evaluate.assert_every_image_once(records, predictions, expected_rows)


def test_coverage_rejects_duplicated_manifest_path() -> None:
    records = [{"path": "same.jpg"} for _ in range(8)]
    with pytest.raises(RuntimeError, match="duplicated"):
        evaluate.assert_every_image_once(records, 8, 8)


# ---------------------------------------------------------------------------
# Guard rails: checksums and class order
# ---------------------------------------------------------------------------

def test_verify_test_manifest_rejects_changed_manifest(tmp_path: Path) -> None:
    metadata = tmp_path / "data" / "metadata"
    build_test_manifest(tmp_path, metadata)
    summary = write_split_summary(metadata)
    manifest = metadata / "test.csv"

    assert evaluate.verify_test_manifest(manifest, summary, CLASS_ORDER)["sha256"]

    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write("data/raw/evil.jpg,metal,0,0,0,\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        evaluate.verify_test_manifest(manifest, summary, CLASS_ORDER)


def test_verify_test_manifest_rejects_class_order_change(tmp_path: Path) -> None:
    metadata = tmp_path / "data" / "metadata"
    build_test_manifest(tmp_path, metadata)
    summary = write_split_summary(metadata)
    with pytest.raises(RuntimeError, match="class order mismatch"):
        evaluate.verify_test_manifest(metadata / "test.csv", summary, ["a", "b", "c", "d"])


def test_verify_model_checksum_rejects_wrong_artifact(tmp_path: Path) -> None:
    model_path = save_tiny_model(tmp_path)
    real = sha256_of(model_path)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        evaluate.verify_model_checksum(model_path, "0" * 64)
    assert evaluate.verify_model_checksum(model_path, real) == real


def test_resolve_selected_model_requires_the_record_block(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selected_model"):
        evaluate.resolve_selected_model({"class_order": CLASS_ORDER}, tmp_path)


def test_run_evaluation_refuses_when_model_is_not_the_selected_one(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path, tamper_model_sha="0" * 64)
    with pytest.raises(RuntimeError, match="not the selected artifact"):
        run_fixture_evaluation(tmp_path, fixture)


def test_verify_class_order_sources_requires_agreement(tmp_path: Path) -> None:
    model_path = save_tiny_model(tmp_path)
    with pytest.raises(RuntimeError, match="disagrees"):
        evaluate.verify_class_order_sources(CLASS_ORDER, ["a", "b", "c", "d"], model_path)


# ---------------------------------------------------------------------------
# Alignment (shared deterministic pipeline)
# ---------------------------------------------------------------------------

def test_evaluation_detects_swapped_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path)

    def fake_make_dataset(records_arg, *, image_size, batch_size, training, seed, **kwargs):
        del training, seed, kwargs
        labels = [int(record["class_index"]) for record in records_arg][::-1]
        images = tf.zeros((len(records_arg), image_size, image_size, 3), dtype=tf.float32)
        return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)

    monkeypatch.setattr(data_pipeline, "make_dataset", fake_make_dataset)
    with pytest.raises(RuntimeError, match="alignment"):
        run_fixture_evaluation(tmp_path, fixture)


def test_evaluation_opens_only_the_test_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path)
    opened: list[str] = []
    original = data_pipeline.load_manifest

    def spy(manifest_path, *args, **kwargs):
        opened.append(Path(manifest_path).name)
        return original(manifest_path, *args, **kwargs)

    monkeypatch.setattr(data_pipeline, "load_manifest", spy)
    report = run_fixture_evaluation(tmp_path, fixture)
    assert report["status"] == "ok"
    assert opened == ["test.csv"]


# ---------------------------------------------------------------------------
# Full synthetic evaluation
# ---------------------------------------------------------------------------

def test_evaluation_writes_a_self_consistent_report(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    output = tmp_path / "out" / "test_evaluation.json"
    report = run_fixture_evaluation(tmp_path, fixture, output=output)

    assert report["status"] == "ok"
    assert report["split"] == "test"
    assert output.is_file()

    # Coverage: all 8 synthetic test images, each exactly once.
    assert report["coverage"]["expected_images"] == 8
    assert report["coverage"]["predictions_made"] == 8
    assert report["coverage"]["each_image_evaluated_exactly_once"] is True
    assert report["summary"]["images"] == 8

    # Provenance: model and manifest checksums are the verified ones.
    assert report["model"]["sha256_matches_selection_record"] is True
    assert report["model"]["sha256"] == sha256_of(fixture["model_path"])
    assert report["manifest"]["sha256_matches_split_summary"] is True
    assert report["manifest"]["sha256"] == sha256_of(fixture["metadata"] / "test.csv")
    assert report["class_order_check"]["verified"] is True
    assert report["class_order_check"]["class_order"] == CLASS_ORDER
    assert report["selection"]["selection_reason"].startswith("rule step 1")

    # Metrics: internally consistent (no hand-waved numbers).
    metrics = report["metrics"]
    assert metrics["images"] == 8
    confusion = metrics["confusion_matrix"]["rows_true_columns_predicted"]
    assert sum(sum(row) for row in confusion) == 8
    assert metrics["accuracy"] == pytest.approx(
        (8 - report["summary"]["misclassified"]) / 8
    )
    f1_values = [metrics["per_class"][name]["f1"] for name in CLASS_ORDER]
    assert metrics["macro_f1"] == pytest.approx(float(np.mean(f1_values)))
    assert sum(metrics["per_class"][name]["support"] for name in CLASS_ORDER) == 8

    normalized = report["confusion_matrix_row_normalized"]["rows_true_columns_predicted"]
    for row in normalized:
        assert sum(row) == pytest.approx(1.0)

    # Per-image predictions: relative paths + four class scores each.
    predictions = report["predictions"]
    assert len(predictions) == 8
    assert [row["path"] for row in predictions] == [
        record for record in sorted(row["path"] for row in predictions)
    ] or len({row["path"] for row in predictions}) == 8
    for row in predictions:
        assert not Path(row["path"]).is_absolute()
        assert len(row["scores"]) == 4
        assert sum(row["scores"]) == pytest.approx(1.0, abs=1e-4)
        assert row["target"] in CLASS_ORDER and row["predicted"] in CLASS_ORDER

    # Protocol recorded so a reader can see what was (not) done.
    protocol = report["protocol"]
    assert protocol["fitting"].startswith("none")
    assert protocol["threshold_tuning"].startswith("none")
    assert protocol["augmentation"] == "none (training=False)"
    assert protocol["final_partial_batch"].startswith("kept")
    assert report["misclassified_grid"]["tracked_in_git"] is False
    assert report["selection"]["recorded_before_test_evaluation"] is True


def test_evaluation_never_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = build_fixture(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("test evaluation must never call model.fit()")

    monkeypatch.setattr(tf.keras.Model, "fit", boom)
    report = run_fixture_evaluation(tmp_path, fixture)
    assert report["status"] == "ok"


def test_evaluation_is_deterministic(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    first = run_fixture_evaluation(tmp_path, fixture, output=tmp_path / "a.json")
    second = run_fixture_evaluation(
        tmp_path, fixture, output=tmp_path / "b.json",
        grids_dir=tmp_path / "grids",
    )
    assert first["metrics"] == second["metrics"]
    assert first["predictions"] == second["predictions"]


def test_misclassified_grid_is_written_and_stays_local(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    grids_dir = tmp_path / "models" / "runs" / "misclassified_grids"
    report = run_fixture_evaluation(tmp_path, fixture, grids_dir=grids_dir)

    if report["summary"]["misclassified"] == 0:
        assert report["misclassified_grid"]["path"] is None
        pytest.skip("untrained model happened to score 0 errors on 8 images")
    grid = Path(report["misclassified_grid"]["path"])
    assert grid.is_file()
    assert grid.name.startswith("sample_grid")
    assert grids_dir in grid.parents
    assert report["misclassified_grid"]["tracked_in_git"] is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_help_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src" / "evaluate.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
    )
    assert completed.returncode == 0
    # argparse wraps help text across lines -> compare normalized whitespace.
    flat = " ".join(completed.stdout.split())
    assert "SELECTED model" in flat
    assert "no fitting" in flat
    assert "held-out test split" in flat


def test_main_evaluates_and_prints_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = build_fixture(tmp_path)
    output = tmp_path / "out" / "cli_evaluation.json"
    exit_code = evaluate.main(
        [
            "--selection", str(fixture["selection_path"]),
            "--config", str(fixture["config_path"]),
            "--metadata-dir", str(fixture["metadata"]),
            "--output", str(output),
            "--no-grids",
            "--batch-size", "3",
        ],
        path_root=tmp_path,
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Split  : test (8 images" in captured.out
    assert "Macro F1:" in captured.out
    assert "each evaluated exactly once" in captured.out
    assert output.is_file()
    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert on_disk["report_type"] == "test_set_evaluation"
    assert len(on_disk["predictions"]) == 8
