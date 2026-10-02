"""Tests for src/data_pipeline.py using small synthetic images.

No RealWaste download required: everything runs on tiny JPEGs/PNGs created
in pytest's temporary directory (outside the tracked repo paths).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import data_pipeline  # noqa: E402

CLASS_ORDER = ["metal", "organic", "paper", "plastic"]


def write_jpeg(path: Path, color: tuple[int, int, int], size=(16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG", quality=95)


def write_png_rgba(path: Path, color: tuple[int, int, int, int], size=(16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path, format="PNG")


def write_manifest(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = columns or [
        "path", "target", "class_index", "file_sha256", "decoded_sha256", "group_id",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(col, "")) for col in fieldnames) + "\n")
    return path


def make_records(tmp_path: Path, count: int) -> list[dict]:
    """Create `count` images spread over the four classes + matching records."""
    records = []
    colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    for i in range(count):
        target = CLASS_ORDER[i % len(CLASS_ORDER)]
        relative = f"data/raw/synthetic/{target}/img_{i}.jpg"
        image_file = tmp_path / relative
        write_jpeg(image_file, colors[i % len(colors)])
        records.append(
            {
                "path": relative,
                "image_path": str(image_file),
                "target": target,
                "class_index": CLASS_ORDER.index(target),
                "file_sha256": f"{i:064d}",
                "decoded_sha256": f"{i:064d}"[::-1],
                "group_id": "",
            }
        )
    return records


# ---------------------------------------------------------------------------
# Config and class-order loading
# ---------------------------------------------------------------------------

def test_load_training_config_from_repo(tmp_path: Path) -> None:
    config = data_pipeline.load_training_config()
    assert config["seed"] == 42
    assert config["image_size"] == 224
    assert config["batch_size"] == 16
    assert config["learning_rate"] == 0.001
    assert config["baseline_max_epochs"] == 15
    assert config["dropout"] == 0.2
    # The class order must NOT be duplicated in training.json (single source).
    assert "class_order" not in config
    assert data_pipeline.load_class_order() == CLASS_ORDER

    bad = tmp_path / "training.json"
    bad.write_text(json.dumps({"seed": 42}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        data_pipeline.load_training_config(bad)

    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        data_pipeline.load_training_config(missing)


def test_load_class_order_matches_class_mapping(tmp_path: Path) -> None:
    mapping = tmp_path / "class_mapping.json"
    mapping.write_text(json.dumps({"class_order": CLASS_ORDER}), encoding="utf-8")
    assert data_pipeline.load_class_order(mapping) == CLASS_ORDER

    dupes = tmp_path / "dupes.json"
    dupes.write_text(json.dumps({"class_order": ["metal", "metal"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        data_pipeline.load_class_order(dupes)


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

def test_load_manifest_happy_path_resolves_paths(tmp_path: Path) -> None:
    records = make_records(tmp_path, 4)
    manifest = write_manifest(tmp_path / "data" / "metadata" / "train.csv", records)

    loaded = data_pipeline.load_manifest(manifest, CLASS_ORDER, path_root=tmp_path)
    assert len(loaded) == 4
    for record in loaded:
        assert "\\" not in record["path"]  # forward slashes preserved
        assert Path(record["image_path"]).is_absolute()
        assert Path(record["image_path"]).exists()
        assert record["target"] in CLASS_ORDER


def test_load_manifest_rejects_unknown_target(tmp_path: Path) -> None:
    rows = [{"path": "data/raw/x/a.jpg", "target": "glass", "class_index": 0}]
    manifest = write_manifest(tmp_path / "train.csv", rows)
    with pytest.raises(ValueError, match="unknown target"):
        data_pipeline.load_manifest(manifest, CLASS_ORDER, path_root=tmp_path)


def test_load_manifest_rejects_wrong_class_index(tmp_path: Path) -> None:
    rows = [{"path": "data/raw/x/a.jpg", "target": "metal", "class_index": 2}]
    manifest = write_manifest(tmp_path / "train.csv", rows)
    with pytest.raises(ValueError, match="class_index 2"):
        data_pipeline.load_manifest(manifest, CLASS_ORDER, path_root=tmp_path)


def test_load_manifest_rejects_missing_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "train.csv"
    manifest.write_text("path,label\ndata/raw/x/a.jpg,metal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        data_pipeline.load_manifest(manifest, CLASS_ORDER, path_root=tmp_path)


def test_load_manifest_rejects_empty_split(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path / "train.csv", [])
    with pytest.raises(ValueError, match="no rows"):
        data_pipeline.load_manifest(manifest, CLASS_ORDER, path_root=tmp_path)


def test_load_manifest_rejects_missing_image_file(tmp_path: Path) -> None:
    rows = [
        {"path": "data/raw/x/ghost.jpg", "target": "metal", "class_index": 0},
    ]
    manifest = write_manifest(tmp_path / "train.csv", rows)
    with pytest.raises(FileNotFoundError, match="ghost.jpg"):
        data_pipeline.load_manifest(manifest, CLASS_ORDER, path_root=tmp_path)


def test_load_splits_leaves_test_unopened(tmp_path: Path) -> None:
    """Default split loading must never touch test.csv (this milestone)."""
    records = make_records(tmp_path, 4)
    metadata = tmp_path / "data" / "metadata"
    write_manifest(metadata / "train.csv", records)
    write_manifest(metadata / "validation.csv", records)
    # A deliberately broken test manifest: would raise if it were opened.
    (metadata / "test.csv").write_text("path,label\nnot,a,valid,manifest\n", encoding="utf-8")

    loaded = data_pipeline.load_splits(
        metadata_dir=metadata, class_order=CLASS_ORDER, path_root=tmp_path
    )
    assert set(loaded) == {"train", "validation"}


# ---------------------------------------------------------------------------
# tf.data behaviour
# ---------------------------------------------------------------------------

def test_dataset_shape_rgb_float32_and_label_indices(tmp_path: Path) -> None:
    # RGBA PNG: alpha must be dropped; solid color proves RGB order is kept.
    rgba_path = tmp_path / "data/raw/synthetic/rgba.png"
    write_png_rgba(rgba_path, (200, 30, 30, 255))
    # Grayscale JPEG: must be promoted to 3 channels.
    gray_path = tmp_path / "data/raw/synthetic/gray.jpg"
    gray_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), 120).save(gray_path, format="JPEG", quality=95)

    records = [
        {
            "path": "data/raw/synthetic/rgba.png",
            "image_path": str(rgba_path),
            "target": "metal",
            "class_index": 0,
        },
        {
            "path": "data/raw/synthetic/gray.jpg",
            "image_path": str(gray_path),
            "target": "plastic",
            "class_index": 3,
        },
    ]
    dataset = data_pipeline.make_dataset(
        records, image_size=224, batch_size=2, training=False, seed=42
    )
    images, labels = next(iter(dataset))

    assert images.shape == (2, 224, 224, 3)
    assert images.dtype.name == "float32"
    # 0-255 scale, not 0-1: solid color survives resize at its original value.
    rgba_pixel = images[0, 112, 112].numpy()
    assert rgba_pixel[0] == pytest.approx(200, abs=1)
    assert rgba_pixel[1] == pytest.approx(30, abs=1)
    assert rgba_pixel[2] == pytest.approx(30, abs=1)
    gray_pixel = images[1, 112, 112].numpy()
    assert gray_pixel[0] == pytest.approx(gray_pixel[1], abs=0.5)
    assert gray_pixel[1] == pytest.approx(gray_pixel[2], abs=0.5)
    assert 0.0 <= float(images.numpy().min()) and float(images.numpy().max()) <= 255.0

    assert labels.dtype.name == "int32"
    label_values = labels.numpy().tolist()
    assert label_values == [0, 3]
    assert all(0 <= value < len(CLASS_ORDER) for value in label_values)


def test_validation_order_is_repeatable(tmp_path: Path) -> None:
    records = make_records(tmp_path, 7)

    def first_epoch_labels() -> list[int]:
        dataset = data_pipeline.make_dataset(
            records, image_size=32, batch_size=3, training=False, seed=42
        )
        labels: list[int] = []
        for _, batch_labels in dataset:
            labels.extend(batch_labels.numpy().tolist())
        return labels

    run_a = first_epoch_labels()
    run_b = first_epoch_labels()
    assert run_a == run_b
    # Deterministic order equals the manifest order (no shuffle).
    assert run_a == [record["class_index"] for record in records]


def test_training_shuffle_is_seed_deterministic(tmp_path: Path) -> None:
    records = make_records(tmp_path, 12)
    expected = [record["class_index"] for record in records]

    def shuffled(seed: int) -> list[int]:
        dataset = data_pipeline.make_dataset(
            records, image_size=32, batch_size=4, training=True, seed=seed
        )
        labels: list[int] = []
        for _, batch_labels in dataset:
            labels.extend(batch_labels.numpy().tolist())
        return labels

    run_a = shuffled(42)
    run_b = shuffled(42)
    run_other_seed = shuffled(43)

    assert run_a == run_b  # same seed -> same order
    assert sorted(run_a) == sorted(expected)  # no images lost or duplicated
    assert run_other_seed != run_a  # different seed -> different order


def test_final_partial_batch_is_retained(tmp_path: Path) -> None:
    records = make_records(tmp_path, 5)
    dataset = data_pipeline.make_dataset(
        records, image_size=32, batch_size=2, training=False, seed=42
    )
    batch_sizes = [int(images.shape[0]) for images, _ in dataset]
    assert batch_sizes == [2, 2, 1]


def test_make_dataset_rejects_empty_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty record list"):
        data_pipeline.make_dataset(
            [], image_size=32, batch_size=2, training=False, seed=42
        )
