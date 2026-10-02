"""Tests for src/prepare_data.py using small synthetic images.

These tests never download RealWaste. They build tiny JPEGs in a temporary
directory and exercise duplicate handling, conflict blocking, deterministic
splitting, and leakage checks.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import prepare_data  # noqa: E402


CLASS_ORDER = ["metal", "organic", "paper", "plastic"]
SOURCE_TO_TARGET = {
    "Metal": "metal",
    "Food Organics": "organic",
    "Paper": "paper",
    "Plastic": "plastic",
    "Vegetation": "organic",
}


def write_solid_jpeg(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    image.save(path, format="JPEG", quality=95)


def write_class_mapping(path: Path) -> Path:
    payload = {
        "dataset": "Synthetic",
        "source_to_target": SOURCE_TO_TARGET,
        "excluded_source_categories": ["Glass"],
        "class_order": CLASS_ORDER,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_tiny_dataset(root: Path) -> Path:
    """Create a miniature RealWaste-like tree with a few images per class."""
    dataset_root = root / "data" / "raw" / "synthetic" / "RealWaste"
    # metal
    write_solid_jpeg(dataset_root / "Metal" / "Metal_1.jpg", (200, 30, 30))
    write_solid_jpeg(dataset_root / "Metal" / "Metal_2.jpg", (180, 40, 40))
    write_solid_jpeg(dataset_root / "Metal" / "Metal_3.jpg", (160, 50, 50))
    write_solid_jpeg(dataset_root / "Metal" / "Metal_4.jpg", (140, 60, 60))
    write_solid_jpeg(dataset_root / "Metal" / "Metal_5.jpg", (120, 70, 70))
    # paper
    write_solid_jpeg(dataset_root / "Paper" / "Paper_1.jpg", (240, 240, 240))
    write_solid_jpeg(dataset_root / "Paper" / "Paper_2.jpg", (230, 230, 230))
    write_solid_jpeg(dataset_root / "Paper" / "Paper_3.jpg", (220, 220, 220))
    write_solid_jpeg(dataset_root / "Paper" / "Paper_4.jpg", (210, 210, 210))
    # plastic
    write_solid_jpeg(dataset_root / "Plastic" / "Plastic_1.jpg", (30, 30, 200))
    write_solid_jpeg(dataset_root / "Plastic" / "Plastic_2.jpg", (40, 40, 180))
    write_solid_jpeg(dataset_root / "Plastic" / "Plastic_3.jpg", (50, 50, 160))
    # organic via Food Organics + Vegetation
    write_solid_jpeg(dataset_root / "Food Organics" / "Food Organics_1.jpg", (30, 160, 30))
    write_solid_jpeg(dataset_root / "Food Organics" / "Food Organics_2.jpg", (40, 150, 40))
    write_solid_jpeg(dataset_root / "Vegetation" / "Vegetation_1.jpg", (20, 140, 20))
    write_solid_jpeg(dataset_root / "Vegetation" / "Vegetation_2.jpg", (25, 130, 25))
    # excluded class present on disk but must not enter splits
    write_solid_jpeg(dataset_root / "Glass" / "Glass_1.jpg", (200, 200, 255))
    return dataset_root


def test_same_label_duplicates_keep_one_representative(tmp_path: Path) -> None:
    dataset_root = build_tiny_dataset(tmp_path)
    mapping = prepare_data.load_class_mapping(write_class_mapping(tmp_path / "class_mapping.json"))
    records, invalid, folder_counts = prepare_data.collect_selected_records(
        dataset_root, mapping, repo_root=tmp_path
    )
    assert not invalid
    assert "Glass" not in folder_counts
    assert folder_counts["Metal"] == 5

    # Copy an existing metal image byte-for-byte under a new name
    src = dataset_root / "Metal" / "Metal_1.jpg"
    dup = dataset_root / "Metal" / "Metal_1_copy.jpg"
    dup.write_bytes(src.read_bytes())

    records, invalid, _ = prepare_data.collect_selected_records(
        dataset_root, mapping, repo_root=tmp_path
    )
    assert len(records) == 16 + 1  # original selected + 1 duplicate

    info = prepare_data.detect_duplicates(records)
    assert info["has_conflicts"] is False
    assert info["duplicate_excluded_count"] == 1
    retained_paths = {r["path"] for r in info["records_after_duplicate_policy"]}
    # Deterministic representative is the lexicographically smaller path
    assert any(p.endswith("Metal/Metal_1.jpg") for p in retained_paths)
    assert not any(p.endswith("Metal_1_copy.jpg") for p in retained_paths)
    group = info["same_label_duplicate_groups"][0]
    assert group["retained_path"].endswith("Metal/Metal_1.jpg")
    assert group["duplicate_paths"] == [prepare_data.rel_posix(dup, tmp_path)]


def test_conflicting_label_duplicates_block_splits(tmp_path: Path) -> None:
    dataset_root = build_tiny_dataset(tmp_path)
    mapping = prepare_data.load_class_mapping(write_class_mapping(tmp_path / "class_mapping.json"))

    # Same pixels, two labels
    same_pixels = tmp_path / "scratch_pixels.jpg"
    write_solid_jpeg(same_pixels, (10, 20, 30))
    plastic_copy = dataset_root / "Plastic" / "Plastic_conflict.jpg"
    metal_copy = dataset_root / "Metal" / "Metal_conflict.jpg"
    plastic_copy.write_bytes(same_pixels.read_bytes())
    metal_copy.write_bytes(same_pixels.read_bytes())

    records, invalid, _ = prepare_data.collect_selected_records(
        dataset_root, mapping, repo_root=tmp_path
    )
    info = prepare_data.detect_duplicates(records)
    assert info["has_conflicts"] is True
    assert len(info["conflicts"]) >= 1
    conflict_types = {c["conflict_type"] for c in info["conflicts"]}
    assert "decoded_content_label_mismatch" in conflict_types
    assert info["records_after_duplicate_policy"] == []

    metadata_dir = tmp_path / "data" / "metadata"
    result = prepare_data.run_prepare(
        raw_dir=tmp_path / "data" / "raw",
        class_mapping_path=tmp_path / "class_mapping.json",
        metadata_dir=metadata_dir,
        seed=42,
        repo_root=tmp_path,
    )
    assert result["status"] == "blocked"
    assert (metadata_dir / "duplicate_report.json").exists()
    assert (metadata_dir / "split_summary.json").exists()
    # Must not write split CSVs on conflict
    assert not (metadata_dir / "train.csv").exists()
    assert not (metadata_dir / "validation.csv").exists()
    assert not (metadata_dir / "test.csv").exists()


def test_stratified_split_is_deterministic_and_respects_class_order(tmp_path: Path) -> None:
    dataset_root = build_tiny_dataset(tmp_path)
    mapping = prepare_data.load_class_mapping(write_class_mapping(tmp_path / "class_mapping.json"))
    records, _, _ = prepare_data.collect_selected_records(dataset_root, mapping, repo_root=tmp_path)

    splits_a = prepare_data.stratified_two_stage_split(records, CLASS_ORDER, seed=42)
    splits_b = prepare_data.stratified_two_stage_split(records, CLASS_ORDER, seed=42)

    paths_a = {name: [r["path"] for r in recs] for name, recs in splits_a.items()}
    paths_b = {name: [r["path"] for r in recs] for name, recs in splits_b.items()}
    assert paths_a == paths_b

    # Different seed should (almost certainly) change at least one assignment
    splits_c = prepare_data.stratified_two_stage_split(records, CLASS_ORDER, seed=43)
    paths_c = {name: [r["path"] for r in recs] for name, recs in splits_c.items()}
    assert paths_c != paths_a

    # Every record appears exactly once
    all_paths = [r["path"] for recs in splits_a.values() for r in recs]
    assert len(all_paths) == len(records)
    assert len(set(all_paths)) == len(records)

    leakage = prepare_data.verify_no_leakage(splits_a)
    assert leakage["ok"] is True
    label_check = prepare_data.verify_label_config(splits_a, CLASS_ORDER)
    assert label_check["ok"] is True


def test_run_prepare_writes_manifests_and_checksums(tmp_path: Path) -> None:
    dataset_root = build_tiny_dataset(tmp_path)
    class_mapping = write_class_mapping(tmp_path / "class_mapping.json")
    raw_dir = tmp_path / "data" / "raw"
    metadata_dir = tmp_path / "data" / "metadata"

    result = prepare_data.run_prepare(
        raw_dir=raw_dir,
        class_mapping_path=class_mapping,
        metadata_dir=metadata_dir,
        seed=42,
        repo_root=tmp_path,
    )
    assert result["status"] == "ok"
    summary = result["summary"]
    # Metal 5 + Paper 4 + Plastic 3 + Food Organics 2 + Vegetation 2 = 16
    assert summary["retained_count"] == 16
    assert sum(summary["split_totals"].values()) == 16
    assert summary["leakage_check"]["ok"] is True
    assert set(summary["manifest_checksums_sha256"]) == {
        "train.csv",
        "validation.csv",
        "test.csv",
    }

    for name in ("train.csv", "validation.csv", "test.csv"):
        path = metadata_dir / name
        assert path.exists()
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            assert "\\" not in row["path"]
            assert row["path"].startswith("data/raw/")
            assert row["target"] in CLASS_ORDER
            assert row["class_index"] == str(CLASS_ORDER.index(row["target"]))
            assert len(row["file_sha256"]) == 64
            assert len(row["decoded_sha256"]) == 64

    # Re-run must produce byte-identical CSV manifests
    for name in ("train.csv", "validation.csv", "test.csv"):
        before = (metadata_dir / name).read_bytes()
        prepare_data.run_prepare(
            raw_dir=raw_dir,
            class_mapping_path=class_mapping,
            metadata_dir=metadata_dir,
            seed=42,
            repo_root=tmp_path,
        )
        after = (metadata_dir / name).read_bytes()
        assert before == after, f"{name} is not byte-identical across re-runs"


def test_group_metadata_keeps_groups_together(tmp_path: Path) -> None:
    dataset_root = build_tiny_dataset(tmp_path)
    class_mapping = write_class_mapping(tmp_path / "class_mapping.json")

    # Mark two paper photos as a known related pair
    p1 = prepare_data.rel_posix(dataset_root / "Paper" / "Paper_1.jpg", tmp_path)
    p2 = prepare_data.rel_posix(dataset_root / "Paper" / "Paper_2.jpg", tmp_path)
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(
        json.dumps({"path_to_group": {p1: "paper_pair", p2: "paper_pair"}}),
        encoding="utf-8",
    )

    result = prepare_data.run_prepare(
        raw_dir=tmp_path / "data" / "raw",
        class_mapping_path=class_mapping,
        metadata_dir=tmp_path / "data" / "metadata",
        seed=42,
        groups_path=groups_path,
        repo_root=tmp_path,
    )
    assert result["status"] == "ok"
    splits = result["splits"]
    owners = {}
    for split_name, records in splits.items():
        for record in records:
            if record.get("group_id") == "paper_pair":
                owners.setdefault(record["path"], set()).add(split_name)
    # Both members must land in the same split
    paper_pair_paths = {p1, p2}
    seen_splits = set()
    for path in paper_pair_paths:
        seen_splits |= owners[path]
    assert len(seen_splits) == 1

    leakage = prepare_data.verify_no_leakage(splits)
    assert leakage["ok"] is True


def test_verify_no_leakage_detects_overlap(tmp_path: Path) -> None:
    record = {
        "path": "data/raw/x/a.jpg",
        "target": "metal",
        "class_index": 0,
        "file_sha256": "a" * 64,
        "decoded_sha256": "b" * 64,
        "group_id": "",
    }
    other = dict(record)
    other["path"] = "data/raw/x/a_copy.jpg"
    splits = {"train": [record], "test": [other], "validation": []}
    leakage = prepare_data.verify_no_leakage(splits)
    assert leakage["ok"] is False
    assert any("file_sha256" in e or "decoded_sha256" in e for e in leakage["errors"])


def test_allocate_counts_rounding() -> None:
    assert prepare_data.allocate_counts(0) == (0, 0, 0)
    # Each call must sum to n and stay non-negative
    for n in [1, 2, 7, 10, 921]:
        train, val, test = prepare_data.allocate_counts(n)
        assert train + val + test == n
        assert min(train, val, test) >= 0
