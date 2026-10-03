"""Inspect extracted RealWaste images against configs/class_mapping.json.

Milestone 3 only: count, validate, and document the dataset on disk.
Do not split, augment, or train.

What this script does
---------------------
1. Loads class_mapping.json as the single source of truth for labels.
2. Finds category folders on disk (does not assume exact folder names).
3. Compares folder counts to the UCI card expectations.
4. Checks that each image opens with Pillow (readability, size, mode).
5. Writes JSON reports + a class-distribution chart under data/metadata/.
6. Builds a labeled sample grid under data/inspection/ (ignored by Git).

Why the sample grid is ignored
------------------------------
Dataset licensing is still ambiguous (UCI says CC BY 4.0; authors' pages say
CC BY-NC-SA 4.0). Aggregate counts and charts without photographs are fine to
track. Photos themselves stay out of Git.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# Use a non-interactive backend so charts work on headless/CI machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_MAPPING_PATH = REPO_ROOT / "configs" / "class_mapping.json"
RAW_DIR = REPO_ROOT / "data" / "raw"
METADATA_DIR = REPO_ROOT / "data" / "metadata"
INSPECTION_DIR = REPO_ROOT / "data" / "inspection"

SUMMARY_PATH = METADATA_DIR / "inspection_summary.json"
INVALID_PATH = METADATA_DIR / "invalid_images.json"
CHART_PATH = METADATA_DIR / "class_distribution.png"
SAMPLE_GRID_PATH = INSPECTION_DIR / "sample_grid.png"
# NOTE: SUMMARY_PATH/INVALID_PATH/CHART_PATH are the tracked defaults.
# output_paths() below lets a run redirect all three elsewhere (e.g. a
# Colab runtime directory) without touching the committed reports.

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Expected counts from the UCI RealWaste card (verified 2026-10-02).
# These are publisher-stated counts, not results we invented.
EXPECTED_SOURCE_COUNTS = {
    "Cardboard": 461,
    "Food Organics": 411,
    "Glass": 420,
    "Metal": 790,
    "Miscellaneous Trash": 495,
    "Paper": 500,
    "Plastic": 921,
    "Textile Trash": 318,
    "Vegetation": 436,
}
# Selected four-class subset after mapping (plastic/paper/metal/organic)
EXPECTED_SELECTED_TOTAL = 921 + 500 + 790 + (411 + 436)  # 3058
EXPECTED_MAPPED_COUNTS = {
    "plastic": 921,
    "paper": 500,
    "metal": 790,
    "organic": 411 + 436,  # Food Organics + Vegetation
}

SAMPLE_SEED = 42  # fixed seed so re-runs pick the same sample images
SAMPLES_PER_CLASS = 4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_class_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing class mapping: {path}")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("source_to_target", "excluded_source_categories", "class_order"):
        if key not in data:
            raise KeyError(f"class_mapping.json is missing required key: {key}")
    return data


def find_dataset_root(raw_dir: Path) -> Path | None:
    """Find the directory whose subfolders contain image files."""
    if not raw_dir.exists():
        return None

    def dir_has_images(path: Path) -> bool:
        if not path.is_dir():
            return False
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                return True
        return False

    candidates: list[Path] = []
    for path in [raw_dir, *raw_dir.rglob("*")]:
        if not path.is_dir():
            continue
        subdirs = [c for c in path.iterdir() if c.is_dir()]
        if subdirs and any(dir_has_images(sub) for sub in subdirs):
            candidates.append(path)

    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return candidates[0]


def list_image_files(folder: Path) -> list[Path]:
    files = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=lambda p: p.as_posix().lower())
    return files


def rel_posix(path: Path) -> str:
    """Path relative to repo root, with forward slashes (portable in reports)."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def output_paths(metadata_dir: Path) -> dict[str, Path]:
    """Where the three inspection reports are written for a given directory.

    The default (data/metadata) is tracked in Git. Colab runs should pass a
    runtime directory via --metadata-dir so regenerated timestamps never
    overwrite the committed reports.
    """
    return {
        "summary": metadata_dir / "inspection_summary.json",
        "invalid": metadata_dir / "invalid_images.json",
        "chart": metadata_dir / "class_distribution.png",
    }


def match_source_folder(folder_name: str, mapping: dict) -> str | None:
    """Map an on-disk folder name to a source category key.

    Matching is case-insensitive and tolerant of underscores vs spaces,
    e.g. 'food_organics' -> 'Food Organics'. Returns None if unmapped.
    """
    normalized = folder_name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    for source_name in mapping["source_to_target"]:
        source_norm = source_name.strip().lower().replace("_", " ").replace("-", " ")
        source_norm = " ".join(source_norm.split())
        if normalized == source_norm:
            return source_name
    return None


def discover_categories(dataset_root: Path, mapping: dict) -> dict:
    """Discover image-bearing folders and classify them via the mapping."""
    discovered = []
    for child in sorted(dataset_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        images = list_image_files(child)
        if not images:
            continue
        source_name = match_source_folder(child.name, mapping)
        target = mapping["source_to_target"].get(source_name) if source_name else None
        is_excluded = child.name in mapping["excluded_source_categories"] or (
            source_name in mapping["excluded_source_categories"]
        )
        # Also treat as excluded if folder name matches an excluded category case-insensitively
        if source_name is None:
            for excluded in mapping["excluded_source_categories"]:
                if match_source_folder(excluded, {"source_to_target": {excluded: ""}}) == child.name or (
                    child.name.strip().lower().replace("_", " ")
                    == excluded.strip().lower().replace("_", " ")
                ):
                    is_excluded = True
                    source_name = excluded
                    break

        discovered.append(
            {
                "folder_name": child.name,
                "folder_path": rel_posix(child),
                "image_count": len(images),
                "matched_source_category": source_name,
                "mapped_target_class": target,
                "is_excluded_source": bool(is_excluded) or source_name in mapping["excluded_source_categories"],
                "is_mapped_for_training": target is not None and target in mapping["class_order"],
            }
        )
    return {
        "dataset_root": rel_posix(dataset_root),
        "folders": discovered,
    }


def compare_counts(discovery: dict, mapping: dict) -> dict:
    """Compare discovered folder counts to UCI expectations."""
    mapped_expected = dict(EXPECTED_MAPPED_COUNTS)
    original_expected = dict(EXPECTED_SOURCE_COUNTS)

    actual_original: dict[str, int] = {}
    actual_mapped: dict[str, int] = {name: 0 for name in mapping["class_order"]}
    unexpected_folders = []
    missing_expected = []

    matched_source_names = set()
    for folder in discovery["folders"]:
        source = folder["matched_source_category"]
        count = folder["image_count"]
        if source is None:
            unexpected_folders.append(
                {
                    "folder_name": folder["folder_name"],
                    "folder_path": folder["folder_path"],
                    "image_count": count,
                    "reason": "folder name did not match any class_mapping.json source category",
                }
            )
            continue
        matched_source_names.add(source)
        actual_original[source] = actual_original.get(source, 0) + count
        target = folder["mapped_target_class"]
        if target is not None:
            actual_mapped[target] = actual_mapped.get(target, 0) + count

    # Expected source categories that never appeared on disk
    for source_name in original_expected:
        if source_name not in actual_original and source_name not in matched_source_names:
            # Only flag if we expected it to exist in the full RealWaste layout
            missing_expected.append(
                {
                    "source_category": source_name,
                    "expected_count": original_expected[source_name],
                    "note": "folder not discovered under dataset root",
                }
            )

    discrepancies = []

    # Original category comparison (full 9-class card vs disk)
    for source_name, expected in original_expected.items():
        actual = actual_original.get(source_name)
        if actual is None:
            discrepancies.append(
                {
                    "scope": "original_source_category",
                    "category": source_name,
                    "expected": expected,
                    "actual": None,
                    "delta": None,
                    "status": "missing_on_disk",
                }
            )
        elif actual != expected:
            discrepancies.append(
                {
                    "scope": "original_source_category",
                    "category": source_name,
                    "expected": expected,
                    "actual": actual,
                    "delta": actual - expected,
                    "status": "count_mismatch",
                }
            )

    # Mapped four-class comparison
    for target_name, expected in mapped_expected.items():
        actual = actual_mapped.get(target_name, 0)
        if actual != expected:
            discrepancies.append(
                {
                    "scope": "mapped_target_class",
                    "category": target_name,
                    "expected": expected,
                    "actual": actual,
                    "delta": actual - expected,
                    "status": "count_mismatch",
                }
            )

    actual_selected_total = sum(actual_mapped.get(name, 0) for name in mapping["class_order"])
    if actual_selected_total != EXPECTED_SELECTED_TOTAL:
        discrepancies.append(
            {
                "scope": "mapped_selected_total",
                "category": "(all training classes)",
                "expected": EXPECTED_SELECTED_TOTAL,
                "actual": actual_selected_total,
                "delta": actual_selected_total - EXPECTED_SELECTED_TOTAL,
                "status": "count_mismatch",
            }
        )

    return {
        "expected_source_counts_uci_card": original_expected,
        "expected_mapped_counts": mapped_expected,
        "expected_selected_total": EXPECTED_SELECTED_TOTAL,
        "actual_original_counts": actual_original,
        "actual_mapped_counts": actual_mapped,
        "actual_selected_total": actual_selected_total,
        "unexpected_folders": unexpected_folders,
        "missing_expected_source_folders": missing_expected,
        "discrepancies": discrepancies,
        "policy": (
            "Counts reflect files actually inspected on disk. "
            "Discrepancies are reported, never edited to force a match."
        ),
    }


def validate_images(discovery: dict) -> dict:
    """Open every image with Pillow; record problems without deleting files."""
    invalid = []
    mode_counter: Counter[str] = Counter()
    size_counter: Counter[tuple[int, int]] = Counter()
    valid_count = 0

    for folder in discovery["folders"]:
        folder_path = REPO_ROOT / folder["folder_path"]
        for image_path in list_image_files(folder_path):
            rel = rel_posix(image_path)
            try:
                if image_path.stat().st_size == 0:
                    invalid.append(
                        {
                            "path": rel,
                            "folder": folder["folder_name"],
                            "issue": "empty_file_zero_bytes",
                        }
                    )
                    continue
                with Image.open(image_path) as img:
                    img.verify()  # structural check
                # verify() can leave the file in a bad state; reopen for load
                with Image.open(image_path) as img:
                    img.load()
                    size_counter[(img.width, img.height)] += 1
                    mode_counter[img.mode] += 1
                valid_count += 1
            except Exception as exc:  # noqa: BLE001 - report any decoder failure
                invalid.append(
                    {
                        "path": rel,
                        "folder": folder["folder_name"],
                        "issue": "unreadable_or_corrupt",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return {
        "valid_image_count": valid_count,
        "invalid_image_count": len(invalid),
        "invalid_images": invalid,
        "mode_distribution": dict(mode_counter),
        "size_distribution": {
            f"{w}x{h}": count for (w, h), count in sorted(size_counter.items())
        },
        "note": "Invalid files are reported only. Nothing is deleted or silently skipped.",
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[report] Wrote {rel_posix(path)}")


def plot_class_distribution(counts: dict[str, int], class_order: list[str], path: Path) -> None:
    """Bar chart of mapped class counts (aggregate only — no photographs)."""
    labels = list(class_order)
    values = [counts.get(name, 0) for name in labels]
    colors = ["#4C78A8", "#54A24B", "#E45756", "#F58518"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_xlabel("Target class")
    ax.set_ylabel("Number of images")
    ax.set_title("RealWaste mapped class distribution (actual counts on disk)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[chart] Wrote {rel_posix(path)}")


def build_sample_grid(discovery: dict, mapping: dict, path: Path, seed: int = SAMPLE_SEED) -> dict:
    """Create a labeled sample grid with a fixed seed (local only; Git-ignored)."""
    rng = random.Random(seed)
    grid_rows: list[dict] = []

    # Collect valid image paths per mapped target class
    images_by_target: dict[str, list[Path]] = {name: [] for name in mapping["class_order"]}
    for folder in discovery["folders"]:
        target = folder["mapped_target_class"]
        if target is None:
            continue
        folder_path = REPO_ROOT / folder["folder_path"]
        for image_path in list_image_files(folder_path):
            images_by_target[target].append(image_path)

    selected: list[tuple[str, Path]] = []
    for target in mapping["class_order"]:
        pool = images_by_target.get(target, [])
        if not pool:
            continue
        pool_sorted = sorted(pool, key=lambda p: p.as_posix().lower())
        chosen = rng.sample(pool_sorted, k=min(SAMPLES_PER_CLASS, len(pool_sorted)))
        for image_path in chosen:
            selected.append((target, image_path))

    if not selected:
        return {
            "created": False,
            "reason": "no mapped images available for sampling",
            "path": None,
            "seed": seed,
        }

    cols = len(mapping["class_order"])
    rows = max(1, (len(selected) + cols - 1) // cols)
    thumb = 128
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    # Ensure axes is always 2D for easy indexing
    if rows == 1 and cols == 1:
        axes_list = [[axes]]
    elif rows == 1:
        axes_list = [list(axes)]
    elif cols == 1:
        axes_list = [[ax] for ax in axes]
    else:
        axes_list = axes.tolist()

    for r in range(rows):
        for c in range(cols):
            ax = axes_list[r][c]
            ax.axis("off")
            idx = r * cols + c
            if idx >= len(selected):
                continue
            target, image_path = selected[idx]
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((thumb, thumb))
                    ax.imshow(img)
            except Exception:
                ax.set_title("unreadable", fontsize=8)
                grid_rows.append(
                    {
                        "target_class": target,
                        "path": rel_posix(image_path),
                        "status": "unreadable_skipped_in_grid",
                    }
                )
                continue
            ax.set_title(target, fontsize=9)
            grid_rows.append(
                {
                    "target_class": target,
                    "path": rel_posix(image_path),
                    "status": "shown",
                }
            )

    fig.suptitle(
        f"RealWaste sample grid (seed={seed}) — LOCAL ONLY, not for Git",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)

    return {
        "created": True,
        "path": rel_posix(path),
        "seed": seed,
        "samples_per_class_limit": SAMPLES_PER_CLASS,
        "git_policy": (
            "This PNG contains dataset photographs. It stays under data/inspection/ "
            "and is ignored by Git because dataset licensing is unresolved."
        ),
        "samples": grid_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect RealWaste images against class_mapping.json (no training)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SAMPLE_SEED,
        help=f"Random seed for the sample grid (default: {SAMPLE_SEED}).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help=(
            "Where inspection_summary.json, invalid_images.json and "
            "class_distribution.png are written (default: data/metadata, "
            "tracked in Git). Pass a runtime directory outside the clone "
            "(e.g. on Drive in Colab) to avoid overwriting tracked reports."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = output_paths(args.metadata_dir)
    print("=== RealWaste inspection (milestone 3) ===")

    mapping = load_class_mapping(CLASS_MAPPING_PATH)
    print(f"[mapping] class_order = {mapping['class_order']}")
    print(f"[mapping] source_to_target = {mapping['source_to_target']}")
    print(f"[mapping] excluded = {mapping['excluded_source_categories']}")

    dataset_root = find_dataset_root(RAW_DIR)
    if dataset_root is None:
        print(
            "[error] No extracted dataset found under data/raw/. "
            "Run src/download_data.py first, or follow manual steps in data/README.md.",
        )
        return 1

    print(f"[discover] dataset root = {rel_posix(dataset_root)}")
    discovery = discover_categories(dataset_root, mapping)

    print("\n[discover] Category folders on disk:")
    for folder in discovery["folders"]:
        role = "mapped" if folder["is_mapped_for_training"] else (
            "excluded" if folder["is_excluded_source"] else "unmapped"
        )
        print(
            f"  - {folder['folder_name']!r}: {folder['image_count']} images, "
            f"source={folder['matched_source_category']!r}, "
            f"target={folder['mapped_target_class']!r}, role={role}"
        )

    counts = compare_counts(discovery, mapping)
    print("\n[counts] Actual mapped class counts:")
    for name in mapping["class_order"]:
        print(f"  - {name}: {counts['actual_mapped_counts'].get(name, 0)}")
    print(f"[counts] Selected total: {counts['actual_selected_total']} "
          f"(UCI-card expectation: {counts['expected_selected_total']})")

    if counts["discrepancies"]:
        print("\n[counts] Discrepancies vs UCI card (reported, not modified):")
        for item in counts["discrepancies"]:
            print(
                f"  - [{item['scope']}] {item['category']}: "
                f"expected={item['expected']}, actual={item['actual']}, "
                f"delta={item['delta']}, status={item['status']}"
            )
    else:
        print("[counts] No discrepancies vs UCI card expectations for discovered folders.")

    if counts["unexpected_folders"]:
        print("\n[counts] Unexpected folders:")
        for item in counts["unexpected_folders"]:
            print(f"  - {item['folder_name']}: {item['reason']}")

    print("\n[validate] Checking image readability...")
    validation = validate_images(discovery)
    print(
        f"[validate] valid={validation['valid_image_count']}, "
        f"invalid={validation['invalid_image_count']}"
    )
    print(f"[validate] modes={validation['mode_distribution']}")
    print(f"[validate] sizes={validation['size_distribution']}")
    if validation["invalid_images"]:
        print("[validate] Invalid image samples (first 10):")
        for item in validation["invalid_images"][:10]:
            print(f"  - {item['path']}: {item['issue']}")

    # Explicit excluded-category report
    excluded_report = []
    for folder in discovery["folders"]:
        if folder["is_excluded_source"] or folder["matched_source_category"] in mapping[
            "excluded_source_categories"
        ]:
            excluded_report.append(
                {
                    "folder_name": folder["folder_name"],
                    "source_category": folder["matched_source_category"],
                    "image_count": folder["image_count"],
                    "training_status": "excluded_from_initial_training",
                }
            )

    print("\n[excluded] Source categories excluded from initial training:")
    if excluded_report:
        for item in excluded_report:
            print(
                f"  - {item['folder_name']} ({item['source_category']}): "
                f"{item['image_count']} images — {item['training_status']}"
            )
    else:
        print("  (none discovered on disk)")

    plot_class_distribution(
        counts["actual_mapped_counts"], mapping["class_order"], outputs["chart"]
    )
    grid_info = build_sample_grid(discovery, mapping, SAMPLE_GRID_PATH, seed=args.seed)
    if grid_info["created"]:
        print(f"[grid] Wrote {grid_info['path']} (seed={grid_info['seed']}, Git-ignored)")
    else:
        print(f"[grid] Not created: {grid_info['reason']}")

    summary = {
        "generated_at_utc": utc_now_iso(),
        "class_mapping_path": rel_posix(CLASS_MAPPING_PATH),
        "class_order": mapping["class_order"],
        "source_to_target": mapping["source_to_target"],
        "excluded_source_categories": mapping["excluded_source_categories"],
        "discovery": discovery,
        "count_comparison": counts,
        "image_validation": validation,
        "excluded_category_report": excluded_report,
        "expected_selected_total": EXPECTED_SELECTED_TOTAL,
        "actual_selected_total": counts["actual_selected_total"],
        "chart_path": rel_posix(outputs["chart"]),
        "sample_grid": grid_info,
        "timestamp_note": (
            "generated_at_utc records when inspection ran on this machine. "
            "It is not evidence that the dataset content changed."
        ),
    }
    write_json(outputs["summary"], summary)
    write_json(
        outputs["invalid"],
        {
            "generated_at_utc": utc_now_iso(),
            "invalid_image_count": validation["invalid_image_count"],
            "invalid_images": validation["invalid_images"],
            "note": validation["note"],
        },
    )

    print("\n=== Inspection finished ===")
    print("Reports:")
    print(f"  - {rel_posix(outputs['summary'])}")
    print(f"  - {rel_posix(outputs['invalid'])}")
    print(f"  - {rel_posix(outputs['chart'])}")
    if grid_info["created"]:
        print(f"  - {grid_info['path']} (local only)")
    print("Next milestone (not started here): cleaning / splitting / training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
