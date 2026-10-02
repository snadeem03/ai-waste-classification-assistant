"""Duplicate checks and reproducible train/validation/test splits.

Milestone 4 only: detect duplicates, write split manifests, verify leakage.
Do not augment images or train a model. Do not copy images into split folders.

Why this script exists
----------------------
If the same photo (or a byte-identical copy) appears in both train and test,
the model can "cheat" by memorizing it — that is data leakage. This script:

1. Loads class_mapping.json (labels + fixed class order).
2. Collects only the four selected target classes.
3. Revalidates that each image opens with Pillow.
4. Hashes file bytes (SHA-256) AND decoded RGB pixels (with size).
5. Retains one deterministic representative per same-label duplicate group.
6. Stops split generation if identical content has conflicting labels.
7. Writes stratified 70/15/15 CSV manifests (paths only — no image copies).
8. Verifies no path/hash/group appears in more than one split.

Limitations (important)
-----------------------
Exact-duplicate checks do **not** find every near-duplicate, nor multiple
photos of the same physical object under different lighting/angles. Optional
group metadata can keep *known* related photos in one split; this script never
invents group identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_MAPPING_PATH = REPO_ROOT / "configs" / "class_mapping.json"
RAW_DIR = REPO_ROOT / "data" / "raw"
METADATA_DIR = REPO_ROOT / "data" / "metadata"

TRAIN_CSV = METADATA_DIR / "train.csv"
VALIDATION_CSV = METADATA_DIR / "validation.csv"
TEST_CSV = METADATA_DIR / "test.csv"
DUPLICATE_REPORT = METADATA_DIR / "duplicate_report.json"
SPLIT_SUMMARY = METADATA_DIR / "split_summary.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Aim for stratified 70% train, 15% validation, 15% test.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
DEFAULT_SEED = 42

MANIFEST_FIELDS = [
    "path",
    "target",
    "class_index",
    "file_sha256",
    "decoded_sha256",
    "group_id",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel_posix(path: Path, root: Path | None = None) -> str:
    """Relative path with forward slashes (works on Windows and Colab)."""
    base = (root or REPO_ROOT).resolve()
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def load_class_mapping(path: Path = CLASS_MAPPING_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing class mapping: {path}")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("source_to_target", "class_order"):
        if key not in data:
            raise KeyError(f"class_mapping.json is missing required key: {key}")
    class_order = data["class_order"]
    if len(class_order) != len(set(class_order)):
        raise ValueError("class_order contains duplicate class names")
    targets = set(data["source_to_target"].values())
    missing = targets - set(class_order)
    if missing:
        raise ValueError(f"source_to_target maps to classes not in class_order: {sorted(missing)}")
    return data


def find_dataset_root(raw_dir: Path) -> Path | None:
    """Same convention as inspect_data.py: folders whose subfolders hold images."""
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


def match_source_folder(folder_name: str, mapping: dict) -> str | None:
    """Case/underscore-tolerant folder name -> source category key."""
    normalized = folder_name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    for source_name in mapping["source_to_target"]:
        source_norm = source_name.strip().lower().replace("_", " ").replace("-", " ")
        source_norm = " ".join(source_norm.split())
        if normalized == source_norm:
            return source_name
    return None


def list_image_files(folder: Path) -> list[Path]:
    files = [
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=lambda p: p.as_posix().lower())
    return files


def hash_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_decoded_rgb(image_path: Path) -> tuple[str, int, int]:
    """Hash decoded RGB pixels plus dimensions.

    Catches identical pictures stored with different file metadata
    (re-saved JPEG, different compression, copied bytes vs re-encode).
    """
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        pixel_bytes = rgb.tobytes()
    digest = hashlib.sha256()
    digest.update(f"{width}x{height}:".encode("utf-8"))
    digest.update(pixel_bytes)
    return digest.hexdigest(), width, height


def validate_image(image_path: Path) -> tuple[bool, str | None]:
    """Return (ok, error_message). Never deletes the file."""
    try:
        if image_path.stat().st_size == 0:
            return False, "empty_file_zero_bytes"
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        return True, None
    except Exception as exc:  # noqa: BLE001 - any decoder failure is reported
        return False, f"{type(exc).__name__}: {exc}"


def collect_selected_records(
    dataset_root: Path,
    mapping: dict,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[dict], list[dict], dict]:
    """Build records for the four selected classes; report invalid images.

    Returns (valid_records, invalid_records, source_folder_counts).
    """
    class_order = mapping["class_order"]
    class_index = {name: idx for idx, name in enumerate(class_order)}
    source_to_target = mapping["source_to_target"]

    valid: list[dict] = []
    invalid: list[dict] = []
    folder_counts: dict[str, int] = {}

    for child in sorted(dataset_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        images = list_image_files(child)
        if not images:
            continue
        source_name = match_source_folder(child.name, mapping)
        if source_name is None:
            continue
        target = source_to_target.get(source_name)
        if target is None or target not in class_index:
            # Excluded / unmapped source folder — not part of the four-class set.
            continue
        folder_counts[source_name] = len(images)

        for image_path in images:
            rel = rel_posix(image_path, repo_root)
            ok, error = validate_image(image_path)
            if not ok:
                invalid.append(
                    {
                        "path": rel,
                        "source_category": source_name,
                        "target": target,
                        "error": error,
                    }
                )
                continue
            file_hash = hash_file_sha256(image_path)
            decoded_hash, width, height = hash_decoded_rgb(image_path)
            valid.append(
                {
                    "path": rel,
                    "source_category": source_name,
                    "target": target,
                    "class_index": class_index[target],
                    "file_sha256": file_hash,
                    "decoded_sha256": decoded_hash,
                    "width": width,
                    "height": height,
                    "group_id": "",
                }
            )

    # Stable order independent of filesystem enumeration
    valid.sort(key=lambda r: r["path"])
    return valid, invalid, folder_counts


def load_group_metadata(path: Path | None) -> dict[str, str]:
    """Optional known-related-photo groups: path -> group_id.

    Never invents groups. Empty mapping means all images are ungrouped.
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Group metadata file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    mapping: dict[str, str] = {}
    if isinstance(data, dict) and "path_to_group" in data:
        raw = data["path_to_group"]
        if not isinstance(raw, dict):
            raise ValueError("path_to_group must be an object mapping path -> group_id")
        for key, value in raw.items():
            mapping[str(key).replace("\\", "/")] = str(value)
    elif isinstance(data, dict):
        # Allow a bare {path: group_id} object
        for key, value in data.items():
            if key in {"note", "generated_at_utc"}:
                continue
            if not isinstance(value, str):
                raise ValueError(f"Group id for {key!r} must be a string")
            mapping[str(key).replace("\\", "/")] = value
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or "path" not in item or "group_id" not in item:
                raise ValueError("Each group list item needs 'path' and 'group_id'")
            mapping[str(item["path"]).replace("\\", "/")] = str(item["group_id"])
    else:
        raise ValueError("Group metadata must be an object or a list of {path, group_id}")
    return mapping


def attach_groups(records: list[dict], path_to_group: dict[str, str]) -> list[dict]:
    for record in records:
        record["group_id"] = path_to_group.get(record["path"], "")
    return records


def detect_duplicates(records: list[dict]) -> dict:
    """Find exact file and decoded-pixel duplicates; resolve same-label cases.

    Policy:
    - Same decoded content + same target -> keep one representative (lowest path).
    - Same content + different target -> conflict; caller must stop splitting.
    - Same file hash + different target -> conflict.
    - Source images are never deleted or relabeled.
    """
    by_decoded: dict[str, list[dict]] = defaultdict(list)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_decoded[record["decoded_sha256"]].append(record)
        by_file[record["file_sha256"]].append(record)

    same_label_duplicate_groups: list[dict] = []
    conflicts: list[dict] = []
    keep_paths: set[str] = set()
    excluded_duplicate_paths: list[dict] = []

    # --- decoded-pixel groups ---
    for decoded_hash, group in sorted(by_decoded.items(), key=lambda kv: kv[0]):
        targets = {r["target"] for r in group}
        if len(group) == 1:
            keep_paths.add(group[0]["path"])
            continue
        if len(targets) > 1:
            conflicts.append(
                {
                    "conflict_type": "decoded_content_label_mismatch",
                    "decoded_sha256": decoded_hash,
                    "targets": sorted(targets),
                    "paths": sorted(r["path"] for r in group),
                    "detail": (
                        "Identical decoded RGB pixels appear under more than one target label. "
                        "Split generation is blocked until this is resolved explicitly."
                    ),
                }
            )
            continue
        # Same label duplicates: retain deterministic representative (lowest path)
        ordered = sorted(group, key=lambda r: r["path"])
        representative = ordered[0]
        keep_paths.add(representative["path"])
        same_label_duplicate_groups.append(
            {
                "decoded_sha256": decoded_hash,
                "target": representative["target"],
                "retained_path": representative["path"],
                "duplicate_paths": [r["path"] for r in ordered[1:]],
                "file_sha256s": sorted({r["file_sha256"] for r in ordered}),
            }
        )
        for extra in ordered[1:]:
            excluded_duplicate_paths.append(
                {
                    "path": extra["path"],
                    "target": extra["target"],
                    "reason": "same_label_duplicate_of_retained_representative",
                    "retained_path": representative["path"],
                    "decoded_sha256": decoded_hash,
                }
            )

    # --- file-byte groups not already handled via decoded identity ---
    for file_hash, group in sorted(by_file.items(), key=lambda kv: kv[0]):
        if len(group) == 1:
            continue
        targets = {r["target"] for r in group}
        if len(targets) > 1:
            conflicts.append(
                {
                    "conflict_type": "file_hash_label_mismatch",
                    "file_sha256": file_hash,
                    "targets": sorted(targets),
                    "paths": sorted(r["path"] for r in group),
                    "detail": (
                        "Byte-identical files appear under more than one target label. "
                        "Split generation is blocked until this is resolved explicitly."
                    ),
                }
            )

    has_conflicts = bool(conflicts)
    if has_conflicts:
        retained = []  # do not produce splits on conflicts
    else:
        retained = [r for r in records if r["path"] in keep_paths]

    excluded_count = len(excluded_duplicate_paths) + len(
        [c for c in conflicts for _ in c.get("paths", [])]
    )
    # Recount excluded carefully for report clarity
    duplicate_excluded_count = len(excluded_duplicate_paths)

    return {
        "has_conflicts": has_conflicts,
        "conflicts": conflicts,
        "same_label_duplicate_groups": same_label_duplicate_groups,
        "excluded_duplicate_paths": excluded_duplicate_paths,
        "duplicate_excluded_count": duplicate_excluded_count,
        "records_after_duplicate_policy": retained,
        "scanned_record_count": len(records),
        "unique_decoded_hashes": len(by_decoded),
        "unique_file_hashes": len(by_file),
        "limitations": [
            "Exact SHA-256 file hashes only catch byte-identical files.",
            "Decoded RGB hashes catch identical pixel content even if JPEG metadata differs.",
            "Neither check finds near-duplicates, rephotographs of the same object, or heavily edited copies.",
            "Optional group metadata can keep known related photos in one split; groups are never invented.",
        ],
    }


def allocate_counts(n: int) -> tuple[int, int, int]:
    """Split n items into train/val/test close to 70/15/15."""
    if n <= 0:
        return 0, 0, 0
    n_train = int(round(n * TRAIN_FRACTION))
    n_val = int(round(n * VAL_FRACTION))
    n_test = n - n_train - n_val
    # Fix rounding so parts stay non-negative and sum to n
    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = 0
    if n_train + n_val + n_test != n:
        n_test = n - n_train - n_val
    return n_train, n_val, n_test


def stratified_two_stage_split(
    records: list[dict],
    class_order: list[str],
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict]]:
    """Reproducible two-stage stratified split for ordinary ungrouped data.

    Stage 1: each class -> train (~70%) + holdout (~30%).
    Stage 2: holdout -> validation (~50%) + test (~50%) => ~15% / ~15% overall.

    Records are sorted by path first so filesystem order cannot affect results.
    """
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}

    for class_name in class_order:
        class_records = sorted(
            [r for r in records if r["target"] == class_name],
            key=lambda r: r["path"],
        )
        if not class_records:
            continue
        rng.shuffle(class_records)
        n = len(class_records)
        n_train, n_val, n_test = allocate_counts(n)
        if n_train + n_val + n_test != n:
            raise RuntimeError(
                f"Split allocation failed for class {class_name!r}: "
                f"n={n}, parts=({n_train},{n_val},{n_test})"
            )
        splits["train"].extend(class_records[:n_train])
        splits["validation"].extend(class_records[n_train : n_train + n_val])
        splits["test"].extend(class_records[n_train + n_val :])

    for name in splits:
        splits[name].sort(key=lambda r: r["path"])
    return splits


def group_stratified_split(
    records: list[dict],
    class_order: list[str],
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Keep each known group inside a single split; report proportion deviations.

    Groups must not mix target classes. Ungrouped records are treated as
    single-member groups. Allocation is greedy by shuffled group order within
    each class, targeting 70/15/15 by image count.
    """
    rng = random.Random(seed)

    # Build groups per class
    groups_by_class: dict[str, list[list[dict]]] = {c: [] for c in class_order}
    conflict_notes: list[dict] = []

    by_group: dict[str, list[dict]] = defaultdict(list)
    ungrouped: list[dict] = []
    for record in sorted(records, key=lambda r: r["path"]):
        gid = record.get("group_id") or ""
        if gid:
            by_group[gid].append(record)
        else:
            ungrouped.append(record)

    for gid, members in sorted(by_group.items()):
        targets = {m["target"] for m in members}
        if len(targets) > 1:
            conflict_notes.append(
                {
                    "group_id": gid,
                    "targets": sorted(targets),
                    "paths": [m["path"] for m in members],
                    "detail": "Group mixes multiple target classes; group integrity cannot be preserved.",
                }
            )
            continue
        target = next(iter(targets))
        if target in groups_by_class:
            groups_by_class[target].append(sorted(members, key=lambda m: m["path"]))

    for record in ungrouped:
        groups_by_class[record["target"]].append([record])

    if conflict_notes:
        raise RuntimeError(
            "Group metadata mixes labels within a group_id. "
            f"Conflicts: {json.dumps(conflict_notes, indent=2)}"
        )

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    deviation_notes: list[dict] = []

    for class_name in class_order:
        groups = groups_by_class.get(class_name, [])
        if not groups:
            continue
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        n_train_target, n_val_target, n_test_target = allocate_counts(total)

        current = {"train": 0, "validation": 0, "test": 0}
        targets = {
            "train": n_train_target,
            "validation": n_val_target,
            "test": n_test_target,
        }
        # Greedy: put each group into the split farthest below its target count
        for group in groups:
            size = len(group)
            order = sorted(
                targets.keys(),
                key=lambda name: (targets[name] - current[name], name),
                reverse=True,
            )
            chosen = order[0]
            splits[chosen].extend(group)
            current[chosen] += size

        if current != targets:
            deviation_notes.append(
                {
                    "class": class_name,
                    "target_counts": targets,
                    "actual_counts": current,
                    "reason": "group sizes cannot hit exact 70/15/15 proportions",
                }
            )

    for name in splits:
        splits[name].sort(key=lambda r: r["path"])
    return splits, deviation_notes


def verify_no_leakage(splits: dict[str, list[dict]]) -> dict:
    """Assert paths, hashes, and groups do not cross splits."""
    errors: list[str] = []
    path_owner: dict[str, str] = {}
    file_hash_owner: dict[str, str] = {}
    decoded_hash_owner: dict[str, str] = {}
    group_owner: dict[str, str] = {}
    all_paths: list[str] = []

    for split_name, records in splits.items():
        for record in records:
            path = record["path"]
            all_paths.append(path)
            if path in path_owner:
                errors.append(
                    f"Path {path} appears in both {path_owner[path]} and {split_name}"
                )
            else:
                path_owner[path] = split_name

            for key, owner in (
                ("file_sha256", file_hash_owner),
                ("decoded_sha256", decoded_hash_owner),
            ):
                value = record[key]
                if value in owner and owner[value] != split_name:
                    # Same hash in two splits is leakage even across different paths
                    errors.append(
                        f"{key} {value} appears in both {owner[value]} and {split_name} "
                        f"(paths involved in leakage check)"
                    )
                else:
                    owner.setdefault(value, split_name)

            gid = record.get("group_id") or ""
            if gid:
                if gid in group_owner and group_owner[gid] != split_name:
                    errors.append(
                        f"Group {gid} appears in both {group_owner[gid]} and {split_name}"
                    )
                else:
                    group_owner.setdefault(gid, split_name)

    unique_paths = set(all_paths)
    if len(unique_paths) != len(all_paths):
        errors.append("Duplicate paths detected across the combined split manifests")

    return {
        "ok": not errors,
        "errors": errors,
        "path_count": len(all_paths),
        "unique_path_count": len(unique_paths),
        "unique_file_hashes": len(file_hash_owner),
        "unique_decoded_hashes": len(decoded_hash_owner),
        "grouped_path_count": sum(1 for s in splits.values() for r in s if r.get("group_id")),
    }


def verify_label_config(
    splits: dict[str, list[dict]],
    class_order: list[str],
) -> dict:
    """Every manifest label/index must match configs/class_mapping.json."""
    errors: list[str] = []
    class_index = {name: idx for idx, name in enumerate(class_order)}
    seen = 0
    for split_name, records in splits.items():
        for record in records:
            seen += 1
            target = record["target"]
            if target not in class_index:
                errors.append(f"{split_name}: unknown target {target!r}")
                continue
            if record["class_index"] != class_index[target]:
                errors.append(
                    f"{split_name}: class_index {record['class_index']} != "
                    f"configured index {class_index[target]} for {target}"
                )
    return {"ok": not errors, "errors": errors, "records_checked": seen}


def write_csv(path: Path, records: list[dict]) -> str:
    """Write a manifest CSV; return SHA-256 of the file bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in MANIFEST_FIELDS})
    return hash_file_sha256(path)


def per_class_counts(records: list[dict], class_order: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in class_order}
    for record in records:
        counts[record["target"]] = counts.get(record["target"], 0) + 1
    return counts


def run_prepare(
    raw_dir: Path = RAW_DIR,
    class_mapping_path: Path = CLASS_MAPPING_PATH,
    metadata_dir: Path = METADATA_DIR,
    seed: int = DEFAULT_SEED,
    groups_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Full pipeline. Returns a result dict; raises on blocking conflicts."""
    mapping = load_class_mapping(class_mapping_path)
    class_order = mapping["class_order"]

    dataset_root = find_dataset_root(raw_dir)
    if dataset_root is None:
        raise FileNotFoundError(
            f"No extracted dataset found under {raw_dir}. "
            "Run src/download_data.py first."
        )

    records, invalid, folder_counts = collect_selected_records(
        dataset_root, mapping, repo_root=repo_root
    )
    path_to_group = load_group_metadata(groups_path)
    records = attach_groups(records, path_to_group)

    duplicate_info = detect_duplicates(records)

    report_payload = {
        "generated_at_utc": utc_now_iso(),
        "dataset_root": rel_posix(dataset_root, repo_root),
        "class_mapping_path": rel_posix(class_mapping_path, repo_root),
        "class_order": class_order,
        "selected_source_counts": folder_counts,
        "selected_record_count": len(records),
        "invalid_image_count": len(invalid),
        "invalid_images": invalid,
        "duplicate_summary": {
            "has_conflicts": duplicate_info["has_conflicts"],
            "same_label_duplicate_group_count": len(
                duplicate_info["same_label_duplicate_groups"]
            ),
            "duplicate_excluded_count": duplicate_info["duplicate_excluded_count"],
            "unique_decoded_hashes": duplicate_info["unique_decoded_hashes"],
            "unique_file_hashes": duplicate_info["unique_file_hashes"],
        },
        "same_label_duplicate_groups": duplicate_info["same_label_duplicate_groups"],
        "excluded_duplicate_paths": duplicate_info["excluded_duplicate_paths"],
        "conflicts": duplicate_info["conflicts"],
        "limitations": duplicate_info["limitations"],
        "group_metadata_path": rel_posix(groups_path, repo_root) if groups_path else None,
        "grouped_record_count": sum(1 for r in records if r.get("group_id")),
        "policy": (
            "Source images are never deleted or relabeled. "
            "Same-label exact duplicates keep one deterministic representative. "
            "Conflicting labels block split generation."
        ),
    }

    metadata_dir.mkdir(parents=True, exist_ok=True)
    dup_path = metadata_dir / "duplicate_report.json"
    with open(dup_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)
        handle.write("\n")

    if duplicate_info["has_conflicts"]:
        summary = {
            "generated_at_utc": utc_now_iso(),
            "status": "blocked_duplicate_label_conflicts",
            "seed": seed,
            "class_order": class_order,
            "conflict_count": len(duplicate_info["conflicts"]),
            "duplicate_report": rel_posix(dup_path, repo_root),
            "message": (
                "Split manifests were NOT written because identical content "
                "has conflicting target labels. Resolve conflicts explicitly, "
                "then re-run prepare_data.py."
            ),
        }
        sum_path = metadata_dir / "split_summary.json"
        with open(sum_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        return {
            "status": "blocked",
            "duplicate_report_path": rel_posix(dup_path, repo_root),
            "summary_path": rel_posix(sum_path, repo_root),
            "duplicate_info": duplicate_info,
            "invalid": invalid,
        }

    retained = duplicate_info["records_after_duplicate_policy"]

    if path_to_group:
        splits, deviation_notes = group_stratified_split(retained, class_order, seed=seed)
        split_method = "stratified_group_aware_greedy_70_15_15"
    else:
        splits = stratified_two_stage_split(retained, class_order, seed=seed)
        deviation_notes = []
        split_method = "stratified_two_stage_70_15_15"

    leakage = verify_no_leakage(splits)
    label_check = verify_label_config(splits, class_order)
    if not leakage["ok"]:
        raise RuntimeError("Leakage check failed:\n" + "\n".join(leakage["errors"]))
    if not label_check["ok"]:
        raise RuntimeError("Label config check failed:\n" + "\n".join(label_check["errors"]))

    train_csv = metadata_dir / "train.csv"
    val_csv = metadata_dir / "validation.csv"
    test_csv = metadata_dir / "test.csv"
    checksums = {
        "train.csv": write_csv(train_csv, splits["train"]),
        "validation.csv": write_csv(val_csv, splits["validation"]),
        "test.csv": write_csv(test_csv, splits["test"]),
    }

    split_counts = {
        name: per_class_counts(records, class_order) for name, records in splits.items()
    }
    split_totals = {name: len(records) for name, records in splits.items()}

    summary = {
        "generated_at_utc": utc_now_iso(),
        "status": "ok",
        "seed": seed,
        "split_method": split_method,
        "target_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "class_order": class_order,
        "class_index": {name: idx for idx, name in enumerate(class_order)},
        "original_selected_counts": folder_counts,
        "original_selected_total": sum(folder_counts.values()),
        "invalid_excluded_count": len(invalid),
        "duplicate_excluded_count": duplicate_info["duplicate_excluded_count"],
        "retained_count": len(retained),
        "retained_per_class": per_class_counts(retained, class_order),
        "split_counts": split_counts,
        "split_totals": split_totals,
        "group_proportion_deviations": deviation_notes,
        "leakage_check": leakage,
        "label_config_check": label_check,
        "manifest_checksums_sha256": checksums,
        "manifest_paths": {
            "train": rel_posix(train_csv, repo_root),
            "validation": rel_posix(val_csv, repo_root),
            "test": rel_posix(test_csv, repo_root),
        },
        "duplicate_report": rel_posix(dup_path, repo_root),
        "why_split_before_augmentation": (
            "Augmentation must run only on training images. If augmented copies "
            "of a test photo were created before splitting, the model could see "
            "test content during training (leakage) and report inflated accuracy."
        ),
        "timestamp_note": (
            "generated_at_utc records when this script ran. CSV manifests are "
            "deterministic and contain no timestamps so re-runs can be compared byte-for-byte."
        ),
    }
    sum_path = metadata_dir / "split_summary.json"
    with open(sum_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return {
        "status": "ok",
        "summary": summary,
        "splits": splits,
        "duplicate_info": duplicate_info,
        "invalid": invalid,
        "summary_path": rel_posix(sum_path, repo_root),
        "duplicate_report_path": rel_posix(dup_path, repo_root),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect duplicates and write stratified split manifests (no training)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for splitting (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help=(
            "Optional JSON with known related photographs "
            "(path_to_group mapping or list of {path, group_id})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Dataset preparation: duplicates + splits (milestone 4) ===")
    print(f"Seed: {args.seed}")
    print(f"Groups file: {args.groups}")

    try:
        result = run_prepare(
            seed=args.seed,
            groups_path=args.groups,
        )
    except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[error] Preparation did not finish successfully. "
            "See data/metadata/duplicate_report.json if it was written.",
            file=sys.stderr,
        )
        return 1

    if result["status"] == "blocked":
        print("[error] BLOCKED: duplicate label conflicts prevent split generation.")
        print(f"[error] Report: {result['duplicate_report_path']}")
        print(f"[error] Summary: {result['summary_path']}")
        return 2

    summary = result["summary"]
    print("\n[duplicates] Same-label duplicate groups:",
          summary["duplicate_excluded_count"], "extra copies excluded from splits")
    print("[duplicates] Conflicts:", summary.get("conflict_count", 0) or len(
        result["duplicate_info"]["conflicts"]
    ))
    print("[counts] Original selected total:", summary["original_selected_total"])
    print("[counts] Invalid excluded:", summary["invalid_excluded_count"])
    print("[counts] Retained:", summary["retained_count"])
    print("[counts] Retained per class:", summary["retained_per_class"])
    print("[splits] Totals:", summary["split_totals"])
    print("[splits] Per class:")
    for split_name, counts in summary["split_counts"].items():
        print(f"  - {split_name}: {counts}")
    print("[verify] Leakage check ok:", summary["leakage_check"]["ok"])
    print("[verify] Label config ok:", summary["label_config_check"]["ok"])
    print("[verify] Manifest SHA-256:")
    for name, digest in summary["manifest_checksums_sha256"].items():
        print(f"  - {name}: {digest}")
    print("\nManifests written under data/metadata/ (images not copied).")
    print("No augmentation or training in this milestone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
