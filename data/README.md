# `data/` directory

## Intended contents

| Path | Tracked in Git? | Purpose |
|---|---|---|
| `data/README.md` | **Yes** | This file — explains layout and rules |
| `data/metadata/` | **Yes** (small files only) | Download/inspection/duplicate/split reports and CSV manifests (no photos) |
| `data/raw/realwaste.zip` | **No** | Official UCI archive (~656 MB) |
| `data/raw/<extracted folders>/` | **No** | Extracted RealWaste category images |
| `data/inspection/sample_grid.png` | **No** | Labeled sample grid (contains dataset photos; licensing unresolved) |
| `data/processed/...` | **No** | Resized / normalized copies (future milestone) |

## Why large files are excluded from Git

- Repositories stay small and clone quickly on college laptops.
- GitHub has file-size limits (~100 MB per file); a 656 MB dataset zip will fail or bloat history.
- Datasets are better restored from the official source using the commands below.
- Binary images do not diff meaningfully in code review.
- **Do not redistribute dataset images in Git** — license notes are in [`docs/dataset.md`](../docs/dataset.md).

## Official source

| Field | Value |
|---|---|
| Dataset page | https://archive.ics.uci.edu/dataset/908/realwaste |
| Download URL (verified) | https://archive.ics.uci.edu/static/public/908/realwaste.zip |
| Citation | Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5SS4G |

## Exact Windows commands (PowerShell, from the repo root)

Always use the project virtual environment — never the system Python.

### 1. Download + extract + checksum + metadata

```powershell
.venv\Scripts\python.exe src\download_data.py
```

Useful flags:

```powershell
# Force a fresh download even if data\raw\realwaste.zip already exists
.venv\Scripts\python.exe src\download_data.py --force

# Download only (skip extraction)
.venv\Scripts\python.exe src\download_data.py --skip-extract
```

What this writes:

- `data/raw/realwaste.zip` (ignored)
- extracted images under `data/raw/` (ignored)
- `data/metadata/download_metadata.json` (tracked) with:
  - source URL + retrieval timestamp
  - **locally computed** SHA-256
  - `publisher_checksum: null` (UCI does not publish a zip checksum — we do not invent one)

Re-running **reuses** a valid existing archive unless you pass `--force`.

### 2. Inspect counts, images, mapping, and charts

```powershell
.venv\Scripts\python.exe src\inspect_data.py
```

Useful flags:

```powershell
# Fixed seed is the default (42); override only if you need a different sample
.venv\Scripts\python.exe src\inspect_data.py --seed 42
```

What this writes:

- `data/metadata/inspection_summary.json` (tracked) — category counts, mapping check vs UCI card
- `data/metadata/invalid_images.json` (tracked) — corrupt/unreadable files (reported, not deleted)
- `data/metadata/class_distribution.png` (tracked) — aggregate bar chart, no photographs
- `data/inspection/sample_grid.png` (**ignored**) — labeled photos for local viewing only

### 3. Duplicate checks + stratified train/val/test splits

```powershell
.venv\Scripts\python.exe src\prepare_data.py
```

Useful flags:

```powershell
# Explicit seed (default is already 42)
.venv\Scripts\python.exe src\prepare_data.py --seed 42

# Optional known-related-photo groups (keeps each group inside one split).
# The groups file is supplied by you; no example file is committed because
# group identities must come from real known relationships, never invented.
.venv\Scripts\python.exe src\prepare_data.py --groups path\to\my_groups.json
```

The groups file may be either `{"path_to_group": {"data/raw/.../a.jpg": "pair_1", ...}}`
or a list of `{"path": ..., "group_id": ...}` entries. Paths use forward slashes.

What this writes under `data/metadata/` (tracked, small text only):

| File | Purpose |
|---|---|
| `train.csv` | Training manifest (paths + labels + hashes) |
| `validation.csv` | Validation manifest |
| `test.csv` | Test manifest |
| `duplicate_report.json` | Duplicate groups, exclusions, any label conflicts |
| `split_summary.json` | Counts, seed, method, per-class split sizes, manifest SHA-256s |

Manifest columns: `path`, `target`, `class_index`, `file_sha256`, `decoded_sha256`, `group_id`.

**Why split before augmentation**

Augmentation (flips, crops, color jitter) must run **only on training images**. If augmented copies of a test photo were created *before* splitting, the model could see near-identical test content during training. That is **data leakage** — validation/test accuracy looks better than it really is. So this milestone writes path manifests first; a later training milestone will augment train rows only.

**What this script does not do**

- Does not copy images into `train/` / `val/` / `test/` folders
- Does not delete or relabel source photos
- Does not augment or train
- Exact-duplicate checks do **not** catch every near-duplicate or rephoto of the same object

### 4. Run the unit tests (synthetic images only — no RealWaste download)

```powershell
.venv\Scripts\python.exe -m pytest tests\test_prepare_data.py -v
```

### 5. Confirm Git ignore behavior (optional)

```powershell
git check-ignore -v data/raw/realwaste.zip
git check-ignore -v data/inspection/sample_grid.png
git check-ignore -v data/metadata/train.csv
git check-ignore -v data/metadata/split_summary.json
```

Expected: the first two paths are ignored; small files under `data/metadata/` are **not** ignored.

## Expected vs actual counts (four-class subset)

From the UCI card (not invented):

| Mapped class | UCI sources | Expected count |
|---|---|---:|
| plastic | Plastic | 921 |
| paper | Paper | 500 |
| metal | Metal | 790 |
| organic | Food Organics + Vegetation | 411 + 436 = **847** |
| **Selected total** | | **3,058** |

`src/inspect_data.py` compares these to files actually on disk and **reports discrepancies** — it never edits counts to force a match.

## Current status

- Scripts: `src/download_data.py`, `src/inspect_data.py`, `src/prepare_data.py`
- Dataset download: see `data/metadata/download_metadata.json`
- Inspection: see `data/metadata/inspection_summary.json`
- Duplicates + splits: see `data/metadata/duplicate_report.json` and `data/metadata/split_summary.json`
- Unit tests: `tests/test_prepare_data.py` (synthetic images; no download required)
- **No augmentation and no training in this milestone**

## Expected class mapping

Target classes (fixed order): `metal`, `organic`, `paper`, `plastic`  
Full mapping and exclusions: [`docs/dataset.md`](../docs/dataset.md) and [`configs/class_mapping.json`](../configs/class_mapping.json)
