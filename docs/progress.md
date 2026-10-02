# Project progress log

Honest milestone tracker for **AI Waste Classification Assistant**.
Update this file after every milestone. Never invent results.

---

## Milestone 0 — Environment setup and workflow

**Date:** 2026-10-02  
**Status:** Complete (setup only; no model training)

### What was done

1. Verified existing Git repo (branch `main`, no prior commits) and Python 3.11.9.
2. Verified existing `.venv` created with `py -3.11 -m venv .venv`. Did **not** re-initialize Git or recreate the venv.
3. Added Git remote `origin` → `https://github.com/snadeem03/ai-waste-classification-assistant.git`.
4. Installed dependencies into `.venv` using `.venv\Scripts\python.exe -m pip`:
   - tensorflow 2.15.1 (Keras 2, MobileNetV2 transfer learning)
   - numpy 1.26.4 (`numpy<2` constraint for TF 2.15)
   - streamlit 1.60.0
   - pandas 3.0.6
   - matplotlib 3.11.2
   - scikit-learn 1.9.1
   - pillow 12.3.0
   - pytest 9.1.1
5. Created:
   - `requirements.txt` (direct dependencies)
   - `requirements.lock.txt` (`pip freeze` of working local versions)
   - `.gitignore` (venv, caches, secrets, datasets, model binaries)
   - `README.md` (honest project description; model not trained yet)
   - `AGENTS.md` (rules for future milestones)
   - `docs/progress.md` (this file)

### Verification

| Check | Result |
|---|---|
| `git --version` | 2.55.0.windows.3 |
| `py -3.11 --version` | Python 3.11.9 |
| `.venv\Scripts\python.exe --version` | Python 3.11.9 |
| Git branch | `main` |
| Git remote | `origin` set to the GitHub URL above |
| Import tensorflow / streamlit / numpy / pandas / matplotlib / sklearn / PIL / pytest / keras | OK |
| `keras.applications.mobilenet_v2.MobileNetV2` importable | True |
| `ALL_IMPORTS_OK` printed | Yes |

Working TF stack note: TensorFlow 2.15.1 reports `keras 2.15.0`. Access via `import keras` or `tf.keras` (attribute `tf.keras.__version__` is not reliable on this TF build; use `keras.__version__` instead).

### Blockers

- None for setup. Push to `origin/main` depends on GitHub authentication being available in this environment; if push fails, the local commit is kept and the error is reported.

### Next steps (future milestones — not done yet)

1. Choose/prepare a waste dataset with classes: plastic, paper, metal, organic.
2. Build a train/validation split manifest (small CSV/JSON stays in Git; images stay out).
3. Implement MobileNetV2 transfer-learning training script under `src/`.
4. Train, evaluate, and write metrics to an evaluation report (only real numbers).
5. Build the Streamlit upload → predict → display app.
6. Add pytest coverage for preprocessing and prediction helpers.

---

## Milestone 2 — Project scope and dataset mapping

**Date:** 2026-10-02  
**Status:** Complete (documentation and config only; **no dataset download, no training**)

### What was done

1. Inspected existing repo state (clean `main`, prior commit `18fd01b`, venv and lock file preserved).
2. Re-read `AGENTS.md` and `.gitignore`.
3. Updated `.gitignore` so large files stay ignored while small metadata stays trackable:
   - Still ignored: virtualenvs, caches, secrets, `data/raw/`, `data/processed/`, archives, model binaries (`*.keras`, `*.h5`, etc.), `models/*` and `data/*` bulk content.
   - Now trackable via negation rules: `data/README.md`, `models/README.md`, and future `data/metadata/**` + `models/metadata/**` (split manifests, class maps, eval notes).
4. Verified RealWaste on the official UCI page (https://archive.ics.uci.edu/dataset/908/realwaste):
   - 4,752 images, 9 categories, 524×524 release size
   - UCI license text: **CC BY 4.0** with attribution
   - Creators: Sam Single, Saeid Iranmanesh, Raad Raad; DOI 10.24432/C5SS4G
   - Noted honest discrepancy: authors’ GitHub/IEEE mention **CC BY-NC-SA 4.0** (documented in `docs/dataset.md`)
5. Created/updated:
   - `docs/project_scope.md` — problem, objectives, competencies, prerequisites, I/O, stack, scope, exclusions, acceptance criteria, limitations (no accuracy promises)
   - `docs/dataset.md` — source, attribution, license notes, original categories + counts, exact mapping, exclusions, TrashNet organic gap, TACO deferral, planned duplicate checks and stratified splits
   - `configs/class_mapping.json` — source→target map, excluded categories, `class_order`: `["metal", "organic", "paper", "plastic"]`
   - `data/README.md` and `models/README.md` — intended contents; large files excluded; dataset/model **not available yet**
   - `README.md` — links to new docs + RealWaste snapshot
   - `docs/progress.md` — this entry
6. Mapping locked for later code:
   - Plastic → `plastic`
   - Paper → `paper`
   - Metal → `metal`
   - Food Organics + Vegetation → `organic`
   - Excluded: Cardboard, Glass, Miscellaneous Trash, Textile Trash

### Verification

| Check | Result |
|---|---|
| JSON syntax of `configs/class_mapping.json` | Valid (parsed with Python `json.load`) |
| `class_order` value | `["metal", "organic", "paper", "plastic"]` |
| Mapping targets ⊆ class_order | Yes |
| Excluded categories not in class_order | Yes |
| `git check-ignore` on sample image/zip/venv/model paths | Ignored as expected |
| `git check-ignore` on `data/README.md`, `models/README.md` | **Not ignored** (trackable) |
| Environment / `requirements.lock.txt` | Unchanged, still present |
| Dataset downloaded? | **No** (stopped before download, as required) |
| Model trained? | **No** |

### Blockers

- None. License discrepancy is documented as a usage caution, not a blocker for this documentation milestone.

### Next milestone (not started)

- **Milestone 3 — Data download and mapping prep:** download RealWaste from UCI into `data/raw/` (ignored), verify category folders against UCI counts, apply the four-class mapping, run duplicate checks, write stratified split manifest under `data/metadata/` (tracked), and stop before training if the milestone says so.

---

## Milestone 3 — Reproducible dataset download and inspection

**Date:** 2026-10-02  
**Status:** Complete (download + inspection only; **no cleaning, splitting, augmentation, or training**)

### What was done

1. Preserved `.venv/`, `requirements.txt`, and `requirements.lock.txt` (unchanged).
2. Verified official UCI download URL:
   - Page: https://archive.ics.uci.edu/dataset/908/realwaste
   - Archive: https://archive.ics.uci.edu/static/public/908/realwaste.zip
   - Method: ranged GET returned HTTP 200 + ZIP magic bytes `PK\x03\x04`
3. Created `src/download_data.py`:
   - Streams to `data/raw/realwaste.zip.part`, then renames (partial files discarded on error)
   - Computes **locally computed** SHA-256; `publisher_checksum` remains `null` with an explicit note (UCI does not publish one)
   - Safe zip extraction (rejects path escape / zip-slip); 0 rejected paths on this run
   - Reuses valid archive + extraction unless `--force`
   - Writes `data/metadata/download_metadata.json`
4. Created `src/inspect_data.py`:
   - Reads `configs/class_mapping.json` as source of truth
   - Discovers category folders on disk (does not assume names; actual root: `data/raw/realwaste-main/RealWaste`)
   - Counts original vs mapped classes; reports excluded categories explicitly
   - Validates every image with Pillow (readability, size, mode)
   - Writes aggregate chart + JSON reports; sample grid is local-only (Git-ignored)
5. Updated `.gitignore`:
   - Still ignores archives, `data/raw/`, `data/processed/`, `data/inspection/`, `**/sample_grid*`
   - Still tracks `data/README.md` and `data/metadata/**`
6. Updated `data/README.md` with exact Windows commands using `.venv\Scripts\python.exe`.

### Actual results (files inspected on disk)

**Download**

| Field | Value |
|---|---|
| Archive path | `data/raw/realwaste.zip` |
| Archive size | 688,545,323 bytes |
| Locally computed SHA-256 | `1ede08b32358ee62065bcc1c8cb47a2ece04e8dff5b1fb53342bb8750895b2f3` |
| Publisher checksum | `null` (not published by UCI — not invented) |
| Extracted files | 4,753 (4,752 images + directory entries) |
| Unsafe paths rejected | 0 |
| Dataset root discovered | `data/raw/realwaste-main/RealWaste` |

**Category counts (actual = UCI card — no discrepancies)**

| Source folder | Actual images | UCI card | Mapped class |
|---|---:|---:|---|
| Plastic | 921 | 921 | plastic |
| Paper | 500 | 500 | paper |
| Metal | 790 | 790 | metal |
| Food Organics | 411 | 411 | organic |
| Vegetation | 436 | 436 | organic |
| **Selected total** | **3,058** | **3,058** | — |
| Cardboard (excluded) | 461 | 461 | — |
| Glass (excluded) | 420 | 420 | — |
| Miscellaneous Trash (excluded) | 495 | 495 | — |
| Textile Trash (excluded) | 318 | 318 | — |
| **Full dataset total** | **4,752** | **4,752** | — |

Mapped counts: metal=790, organic=847 (411+436), paper=500, plastic=921.

**Image validation**

| Check | Result |
|---|---|
| Valid images | 4,752 / 4,752 |
| Invalid / unreadable | **0** |
| Modes | all `RGB` |
| Dimensions | all `524x524` |

**Invalid-image findings:** none. Report still written to `data/metadata/invalid_images.json` (empty list) so the check is auditable.

### Verification

| Check | Result |
|---|---|
| Official URL verified before download | Yes (HTTP 200 + ZIP magic) |
| `src/download_data.py` run | Success |
| `src/inspect_data.py` run | Success |
| Re-run without `--force` | Reuses archive; does not re-download |
| Re-run extraction logic | Reuses existing extraction when dataset root already present |
| `git check-ignore` `data/raw/realwaste.zip` | Ignored |
| `git check-ignore` extracted image path | Ignored |
| `git check-ignore` `data/inspection/sample_grid.png` | Ignored |
| `git check-ignore` `data/metadata/*.json` / chart | **Not ignored** (trackable) |
| Counts vs UCI card | Match exactly (0 discrepancies) |
| Requirements files | Unchanged |
| Dataset cleaning / split / train | **Not done** (out of scope for this milestone) |

### Trackable artifacts committed

- `src/download_data.py`
- `src/inspect_data.py`
- `data/README.md`
- `data/metadata/download_metadata.json`
- `data/metadata/inspection_summary.json`
- `data/metadata/invalid_images.json`
- `data/metadata/class_distribution.png`
- `.gitignore` update
- this progress entry

### Not committed (ignored on purpose)

- `data/raw/realwaste.zip` and extracted images
- `data/inspection/sample_grid.png` (contains dataset photographs; licensing unresolved)

### Blockers

- None for this milestone. License discrepancy (UCI CC BY 4.0 vs authors’ CC BY-NC-SA 4.0) remains a documentation caution, not a download blocker.

### Next milestone (not started)

- **Milestone 4 — Data cleaning and stratified splitting:** optional quarantine of any future invalid files, build `data/metadata/split_manifest.json` (paths/labels only) with stratified train/val/test split, **no augmentation yet**, stop before training if the milestone says so.

---

## Milestone 4 — Duplicate checks and reproducible dataset splitting

**Date:** 2026-10-02  
**Status:** Complete (duplicates + split manifests only; **no augmentation, no training**)

### What was done

1. Preserved `.venv/`, `requirements.txt`, `requirements.lock.txt`, and all raw dataset images.
2. Created `src/prepare_data.py`:
   - Discovers dataset root with the same conventions as `inspect_data.py`
   - Reads labels + `class_order` from `configs/class_mapping.json`
   - Includes only the four selected target classes
   - Revalidates image readability (Pillow); invalid files reported, never deleted
   - Hashes file bytes (SHA-256) and decoded RGB pixels (size + pixel bytes)
   - Same-label exact duplicates → keep one deterministic representative (lowest path)
   - Conflicting labels on identical content → write conflict report and **block** split generation
   - Optional group metadata (`--groups`) keeps known related photos in one split; groups never invented
   - Stratified two-stage 70/15/15 split, seed **42**, records sorted by path before shuffle
   - Writes `train.csv`, `validation.csv`, `test.csv`, `duplicate_report.json`, `split_summary.json`
   - Manifests use forward-slash relative paths; **no image copies** into split folders
   - Verifies no path/hash/group leakage; re-runs produce byte-identical CSVs
3. Created `tests/test_prepare_data.py` (7 tests, synthetic images in temp dirs — no RealWaste download).
4. Updated `data/README.md` with Windows preparation commands and why splitting happens before augmentation.

### Actual duplicate / conflict findings (real dataset)

| Check | Result |
|---|---|
| Selected records scanned | 3,058 |
| Invalid / unreadable images | **0** |
| Unique file SHA-256 hashes | 3,058 |
| Unique decoded RGB hashes | 3,058 |
| Same-label duplicate groups | **0** |
| Duplicate copies excluded | **0** |
| Label conflicts | **0** |
| Group metadata used | None (all images ungrouped) |

**Honest note:** these exact-hash checks found **no** byte-identical or pixel-identical duplicates among the four selected classes. That does **not** prove the dataset has zero near-duplicates or rephotos of the same object — those are out of scope for exact hashing (documented in `duplicate_report.json` limitations).

### Retained counts and split sizes

| Class | Retained | Train (~70%) | Validation (~15%) | Test (~15%) |
|---|---:|---:|---:|---:|
| metal | 790 | 553 | 118 | 119 |
| organic | 847 | 593 | 127 | 127 |
| paper | 500 | 350 | 75 | 75 |
| plastic | 921 | 645 | 138 | 138 |
| **Total** | **3,058** | **2,141** | **458** | **459** |

Method: `stratified_two_stage_70_15_15`, seed `42`.  
Split totals: train=2141 (70.0%), validation=458 (15.0%), test=459 (15.0%) — rounding absorbed in test by 1 image.

### Manifest checksums (SHA-256 of CSV files)

| Manifest | SHA-256 |
|---|---|
| `train.csv` | `f76d789b7bc6a1ac8fa168ee824ab975e44bfa29df19df8327aa9f7f9fa3abbc` |
| `validation.csv` | `77423bfc11f37df0bb836210f4866543857b56bfe73572054438902ecae6997d` |
| `test.csv` | `74494c13f6d9be7f1fc3a00124a7138035c96d866abe96cfa2eefc84d7530c2d` |

### Verification

| Check | Result |
|---|---|
| `pytest tests/test_prepare_data.py -v` | **7 passed** |
| Coverage | same-label duplicates, conflicting-label block, deterministic split, group integrity, leakage detection, manifest checksums, count allocation |
| Real-data `prepare_data.py` | Success, status `ok` |
| Leakage check (paths / file hashes / decoded hashes / groups) | **ok=True**, 3,058 unique paths |
| Label config vs `class_order` | **ok=True**, 3,058 records checked |
| Paths use forward slashes | Yes (0 backslashes in CSVs) |
| Determinism re-run (seed 42) | **Byte-identical** CSVs for train/validation/test |
| `git check-ignore` raw images / sample grid | Ignored |
| `git check-ignore` split CSVs + JSON reports | **Not ignored** (trackable) |
| Images copied into split folders? | **No** (manifests only) |
| Source images deleted/relabelled? | **No** |
| Augmentation / training | **Not done** |

### Limitations

- Exact file + decoded-pixel hashes do not detect near-duplicates or multiple photos of the same physical item.
- Optional group metadata was not supplied for RealWaste; if known related photos are identified later, re-run with `--groups` so each group stays in one split.
- Small per-class rounding differences vs a perfect 70/15/15 split are expected and reported in `split_summary.json`.
- Stratified split does not balance lighting, viewpoint, or hardness — only class proportions.

### Blockers

- None. No label conflicts were found; split manifests were written.

### Next milestone (not started)

- **Milestone 5 — Training pipeline (MobileNetV2 transfer learning):** load split CSVs, augment **train only**, freeze MobileNetV2 base, train classification head, save model under `models/` (Git-ignored), write real evaluation metrics under `models/metadata/`. Stop before the Streamlit app if that is a separate milestone.

---

*Append new milestones below this line.*
