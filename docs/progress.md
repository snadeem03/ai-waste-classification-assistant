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

## Milestone 5 — Preprocessing, input pipeline, and MobileNetV2 construction

**Date:** 2026-10-02  
**Status:** Complete (preprocessing + model construction + forward-pass verification only; **no `model.fit`, no fine-tuning, no test-set evaluation, no Streamlit**)

### What was done

1. Preserved `.venv/`, `requirements.txt`, `requirements.lock.txt`, all dataset images, and the milestone-4 split manifests (unchanged).
2. Created `configs/training.json`: seed `42`, image size `224`, batch size `16`, learning rate `0.001`, planned maximum baseline epochs `15`, dropout `0.2`. It stores only the **path** to `configs/class_mapping.json` — the class order is not copied, so the two files can never disagree.
3. Created `src/data_pipeline.py`:
   - Loads `train.csv` / `validation.csv` with explicit `path_root` resolution (repo root on Windows, wherever `data/` sits on Colab); relative forward-slash paths stay portable.
   - Validates required columns, nonempty splits, unknown targets, `class_index` ↔ `class_order` consistency, and file existence (clear errors, no silent skips).
   - Decodes to RGB, resizes to 224×224, emits float32 on the **0–255** scale and integer labels for sparse cross-entropy.
   - Training data shuffled with seed 42; validation order fixed (manifest order); final partial batch kept (`drop_remainder=False`); bounded `prefetch(2)`; no dataset-wide RAM cache unless `cache=True` is passed explicitly.
   - Defaults open only train + validation — **test images stay unopened this milestone** (covered by a test using a deliberately broken `test.csv`).
4. Created `src/model.py`:
   - Input `(224, 224, 3)`; built-in `RandomFlip` + `RandomRotation(0.05)` inside the model, active only when `training=True`.
   - MobileNetV2 preprocessing applied **exactly once** inside the graph (0–255 → [-1, 1]); exposed as `preprocess_mobilenet_v2()` so tests can prove 0 → -1 and 255 → 1.
   - `MobileNetV2(include_top=False, weights="imagenet")`, whole base frozen, called with `training=False` to keep frozen BatchNorm in inference mode; GAP → dropout → 4-class softmax.
   - Compiled with Adam + `SparseCategoricalCrossentropy` + accuracy.
   - `weights=None` is allowed only as an explicit argument; an ImageNet load failure raises `RuntimeError` and is never silently replaced with random weights (tested with a simulated failure).
5. Created 25 new tests: `tests/test_data_pipeline.py` (14), `tests/test_model.py` (10), `tests/test_smoke_real_data.py` (1, skips without the dataset).
6. Updated `README.md` (status, layout, test commands, milestone-5 explanations) and this file.

### Test results (actual run, `.venv\Scripts\python.exe -m pytest -v`)

| Suite | Result |
|---|---|
| `tests/test_data_pipeline.py` | **14 passed** |
| `tests/test_model.py` | **10 passed** |
| `tests/test_smoke_real_data.py` | **1 passed** (real data + ImageNet weights) |
| `tests/test_prepare_data.py` (milestone 4) | **7 passed** |
| `tests/test_train.py` (pre-existing, untracked) | **6 passed** |
| **Total** | **38 passed**, 0 failed (80.7 s) |

Covered: input shape/RGB/float32/label indices, invalid labels and missing files failing clearly, repeatable validation order, seed-deterministic training shuffle, partial-batch retention, `(batch, 4)` output, finite softmax rows summing to 1, preprocessing applied exactly once (0 → -1, 255 → 1), frozen base, stable repeated inference with `training=False`, plain `load_model` round trip with no custom objects, and explicit-weights policy.

### Real-data smoke check (actual output, no fitting)

Command: `.venv\Scripts\python.exe -m pytest tests\test_smoke_real_data.py -v -s`

| Check | Result |
|---|---|
| Train batch | `(16, 224, 224, 3)` float32, pixels `0.0 .. 255.0` |
| Train labels | `[0, 3, 2, 1, 1, 2, 0, 0, 0, 3, 0, 0, 1, 1, 2, 0]` — range `0..3` |
| Validation batch | `(16, 224, 224, 3)` float32, labels range `1..1` |
| ImageNet weights | **Loaded** from the local Keras cache (`~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5`) |
| Model output shape | `(16, 4)` |
| Output row sums | `min=1.000000 max=1.000000` |
| Parameters | trainable `5,124` / non-trainable `2,257,984` / total `2,263,108` |
| Base frozen | `True` |
| `model.fit` called | **No** |

Honest notes on the smoke output: the validation batch shows `1..1` because validation runs in deterministic manifest order (path-sorted, so the first rows are all `organic`) — full-epoch evaluation is unaffected, and only training is shuffled. The head still has random weights: a successful forward pass proves the graph works, **not** that anything is trained.

### Key concepts documented (README + code comments)

- Resizing (shape 224×224) vs normalization (0–255 → [-1, 1] once, inside the model).
- Augmentation is training-only so validation/inference stay deterministic.
- The base is frozen to protect pretrained ImageNet features from a small dataset.
- Class order lives only in `configs/class_mapping.json` to prevent silent label renames.
- A forward pass only proves the graph runs; training happens next milestone.

### Previously untracked files (status)

- `src/train.py` and `tests/test_train.py` — **left untracked and unchanged**. Their content is the *training loop* (`model.fit`, early stopping, validation/test evaluation, model saving), which milestone 5 explicitly excludes ("stop before `model.fit`"). Their useful ideas (manifest checks, RGB decode, frozen base) were re-implemented within this milestone's scope in `src/data_pipeline.py` and `src/model.py`. The 6 local tests in `tests/test_train.py` pass, but the file duplicates logic now split across the new modules; it should be refactored to import them when the training milestone lands.

### Limitations

- No training, no metrics, no evaluation claims in this milestone — by design.
- Unit tests build the model with `weights=None` (offline). The pretrained check ran locally because ImageNet weights were already cached; on a machine with no cache and no network the smoke test **skips and reports pending** rather than passing with random weights.
- Data loading keeps paths in host memory and prefetches 2 batches; it does not cache decoded images in RAM (by design).
- Augmentation randomness is drawn per call from TensorFlow's RNGs; the training milestone should call `tf.keras.utils.set_random_seed(seed)` before `model.fit` for end-to-end reproducibility.

### Blockers

- None.

### Next milestone (not started)

- **Milestone 6 — Colab baseline training:** refactor `src/train.py` to reuse `src/data_pipeline.py` + `src/model.py`, run `model.fit` on the train split with validation monitoring (frozen base, batch 16, lr 0.001, max 15 epochs, seed 42), save the `.keras` model (Git-ignored) and write honest metrics under `models/metadata/`. Stop before fine-tuning, test-set evaluation, or the Streamlit app if those are separate milestones.

---

## Milestone 6 — Reproducible baseline training workflow (Colab)

**Date:** 2026-10-02  
**Status:** Complete as a *workflow* — prepared and verified locally; **no training executed in Colab and no accuracy numbers** (the notebook has not been run in Google Colab)

### What was done

1. Preserved `.venv/`, `requirements.txt`, `requirements.lock.txt`, all dataset images, and all earlier milestone files (only the untracked `src/train.py` / `tests/test_train.py` from before were rewritten as this milestone's deliverable).
2. Refactored `src/train.py` into the baseline training CLI:
   - **Reuses** `src/data_pipeline.py` (config/class-order/manifest loading, `make_dataset`) and `src/model.py` (`build_waste_classifier`) instead of duplicating their logic — proven by tests with call-count spies.
   - Opens **only** `train.csv` + `validation.csv` (`splits=("train", "validation")`); `test.csv` is never opened (tests: spy on `load_splits`, and a deliberately broken/absent `test.csv` case).
   - Per-run output directory `models/runs/baseline_<timestamp>/` (overridable with `--output-dir`, which Colab points at Drive).
   - Callbacks: `HistoryCsvCallback` (appends + fsyncs each epoch so completed epochs survive interruption), `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)` → `best_model.keras`.
   - Writes honest per-run artifacts: `run_metadata.json` (git commit + dirty flag, environment incl. `pip freeze`, seed/determinism record with explicit limits, manifest SHA-256s, parameter counts, `data.test_manifest_used: false`, honesty notes), `class_order.json`, `history.csv`, `plots/*.png`, `environment_freeze.txt`.
   - `--export-reports RUN_DIR` copies only the small report files to `models/metadata/runs/<run_id>/` — never a model binary.
   - Clear failure modes: missing config keys → `ValueError` listing them; missing manifest → `FileNotFoundError` naming the path; ImageNet weight failure → `RuntimeError` (no silent random-weight fallback).
   - CLI flags: `--config --epochs --batch-size --learning-rate --image-size --dropout --seed --class-weights {off,balanced} --weights {imagenet,none} --output-dir --no-early-stopping --no-export-reports --export-reports`.
3. Created `notebooks/train_colab.ipynb` (29 cells): environment inspection **before** any install (never installs `requirements.lock.txt` — Windows-only pins; repo-tested `tensorflow==2.15.1` pin on Python 3.11, otherwise newest TF with the difference recorded), guarded runtime restart, Drive mount, pinned `REVISION`, idempotent clone, dataset download/inspection via the existing scripts, **manifest validation** against `split_summary.json` (checksums + row counts; regeneration explicitly refused), accelerator check, training subprocess with `PYTHONHASHSEED`, curve/best-metric display read from the run's own files, report+model zips to Drive, restore/verify commands, reproducibility limits, and explicit honesty statements.
4. Tests: rewrote `tests/test_train.py` (14 synthetic tests, `weights=None`, tiny images) and created `tests/test_notebook.py` (5 structural tests: JSON validity, `ast.parse` of every code cell, required topics, no lock-file install in code, no accuracy claims).
5. **Bug found and fixed during verification:** `main()`'s final summary read `metadata['training']['epochs_completed']`, but that key lives under `metadata['configuration']` — a *successful* training run would have crashed after saving everything. Fixed to read `configuration.*` and guarded by the new regression test `test_main_prints_run_summary_without_training` (mocked run, asserts exit 0 + summary lines).
6. Updated `README.md` (status, layout, test commands, step-by-step Colab section, restore + pending verification commands), `models/README.md` (runs/ vs metadata/runs/ layout), and this file.

### Verification (actual runs)

| Check | Result |
|---|---|
| `.venv\Scripts\python.exe -m pytest -v` (full suite) | **51 passed**, 0 failed (164.6 s) |
| `tests/test_train.py` | **14 passed** (synthetic workflow, `weights=None` — not accuracy tests) |
| `tests/test_notebook.py` | **5 passed** (valid JSON, all code cells parse, required topics, no lock install, no accuracy claims) |
| Prior suites unchanged | data_pipeline 14, model 10, prepare_data 7, smoke_real_data 1 — all pass |
| `src/train.py --help` | exit 0 |
| Broken config (missing keys) | `ValueError: training.json is missing required keys: [...]` |
| Missing manifest | `FileNotFoundError: Manifest not found: ...train.csv` |
| `main()` run summary (mocked run) | exit 0; prints epochs/best/git lines (regression test for the fixed bug) |
| Notebook executed in Google Colab | **No — not done in this milestone** |
| Model trained / accuracy numbers | **None — deliberately not claimed** |

Note on test noise: the synthetic `model.fit` loops emit `DeprecationWarning: non-integer arguments to randrange()` from Python's `random` module via TensorFlow 2.15 internals — pre-existing TF/Python 3.11 behaviour, not caused by this milestone's code.

### Trackable artifacts committed

- `src/train.py` (refactored baseline training CLI)
- `tests/test_train.py` (rewritten), `tests/test_notebook.py` (new)
- `notebooks/train_colab.ipynb` (new)
- `README.md`, `models/README.md`, `docs/progress.md` (this entry)

### Limitations / honest limits

- The Colab notebook has **not** been executed in Google Colab; local verification covers JSON validity, Python syntax of every cell, and topic presence — not real execution.
- No `model.fit` on real data was run in this milestone; the synthetic runs (tiny images, `weights=None`) verify the *workflow*, never model quality.
- Reproducibility across machines is bounded by hardware/library differences; each run records its own environment and explicit determinism limits instead of promising bit-identical results.
- Class weights default to `off` (config not changed); `--class-weights balanced` is available and tested.

### Blockers

- None locally. The first real Colab run requires the student's Google account and a GPU runtime — that is the next step, not a blocker for this milestone.

### Next milestone (not started)

- **Milestone 7 — First real baseline run and honest metrics:** execute `notebooks/train_colab.ipynb` in Colab (or `src/train.py` locally), bring back `reports_*.zip` + `model_*.zip`, run the pending local load verification (`.venv\Scripts\python.exe -c "import tensorflow as tf; m = tf.keras.models.load_model(r'models\best_model.keras'); print(m.output_shape)"`), record the run's real validation metrics under `models/metadata/runs/<run_id>/`, and only then consider test-set evaluation / the Streamlit app as separate milestones.

---

## Follow-up fix — Colab repository path normalization

**Date:** 2026-10-03  
**Status:** Fix committed and verified locally; **the notebook was NOT re-executed in Google Colab** (no Colab run was performed in this session)

### Reported error (from a Colab run)

```text
TypeError: unsupported operand type(s) for /: 'str' and 'str'
at: if not (REPO_DIR / ".git").exists():     # Step 6 clone cell
```

### Diagnosis

- Inspected **every** assignment to `REPO_DIR` in `notebooks/train_colab.ipynb`: only the Step 5 configuration cell assigns it (`REPO_DIR = Path("/content") / REPO_NAME` — already a `Path`), and no later cell overwrites it.
- The Step 6 clone cell was the **first** place `REPO_DIR` was used in a path operation (`REPO_DIR / ".git"`) but performed no conversion itself, so if `REPO_DIR` reached that cell as a plain `str` (different runtime state, re-ordered or edited cell), it raised exactly the reported `TypeError`.
- `REPO_URL` is a string constant and must stay one (it is a `git clone` argument).

### What changed

1. `notebooks/train_colab.ipynb`, Step 6 clone cell (id `c15`):
   - imports `pathlib.Path` in the cell itself and runs `REPO_DIR = Path(REPO_DIR)` **before any path operation**, with a comment naming the exact error it prevents;
   - the `git()` helper now passes `cwd=str(cwd or REPO_DIR)` — external commands receive strings.
2. Other external-command sites now pass strings explicitly: `cwd=str(REPO_DIR)` in the download/inspect cell (`c18`), the training cell (`c24`), and the export/zip cell (`c28`).
3. `REPO_URL` unchanged (plain string); the Step 5 cell unchanged (already constructs a `Path`).
4. `tests/test_notebook.py` — three focused regression checks:
   - `test_repo_dir_is_always_built_with_path` (AST: every `REPO_DIR` assignment must be rooted in a `Path(...)` call),
   - `test_repo_dir_string_input_is_normalized_before_path_ops` (extracts the notebook's actual `REPO_DIR = Path(REPO_DIR)` statement, executes it with `REPO_DIR` supplied as a **string**, and asserts `REPO_DIR / ".git"` then works),
   - `test_repo_url_stays_string_and_cwd_is_passed_as_str`.

### Verification

| Check | Result |
|---|---|
| `pytest tests\test_notebook.py -v` | **8 passed** (5 existing + 3 new regression checks) |
| Regression check vs the **pre-fix** notebook (`git show HEAD:...`) | Fails as intended: no `Path(REPO_DIR)` normalization found |
| Reported error reproduced locally (`"/content/..." / ".git"`) | `TypeError: unsupported operand type(s) for /: 'str' and 'str'` — exact match |
| Notebook JSON + Python syntax of all code cells | Valid (existing syntax test passes) |
| Full suite `.venv\Scripts\python.exe -m pytest -q` | **54 passed** (51 prior + 3 new), 0 failed |
| Notebook executed in Google Colab after the fix | **No — not performed; not claimed** |

### Blockers

- None locally. Confirming the fix end-to-end requires re-running the notebook in Colab (student action).

### Next steps

- In Colab, re-run **Step 5 (configuration)** and then **Step 6 (clone)** — the cell that failed — before continuing; the rest of the notebook is unchanged. Then continue with the pending first training run (milestone 7).

---

## Milestone 7 — Baseline artifact import and local compatibility verification

**Date:** 2026-10-03  
**Status:** Artifact import + verification complete; **local model loading FAILED** (Keras 3 vs Keras 2) — a separate inference environment is proposed but **not yet built or verified**. No retraining, no fine-tuning, no test-set evaluation, no Streamlit.

### Actual Colab execution (reported, then verified from the extracted artifacts)

| Field | Value |
|---|---|
| Run ID | `baseline_20261003_172906` |
| Code commit | `199f2539c821ee1b70d673cec8050ed86cf62e2e` |
| Environment | Google Colab — Python 3.13.15, TensorFlow 2.20.0, Keras 3.13.2, Tesla T4 |
| Epochs | 15 completed (max 15) |
| Best epoch | 14 (lowest `val_loss`) |
| **Validation** loss | **0.2097** (actual `0.20965705811977386`) |
| **Validation** accuracy | **0.9323** (actual `0.932314395904541`) |
| Test manifest used | `False` |
| Checkout at training time | `uncommitted_changes: true` (investigated below) |

All values are **validation metrics** from that run's own `run_metadata.json` /
`history.csv`. No test-set metrics exist.

### What was done

1. **Located** `reports_baseline_20261003_172906.zip` and
   `model_baseline_20261003_172906.zip` in the repo root (both present).
   Integrity: `ZipFile.testzip()` OK; members checked for absolute paths /
   `..` traversal — none found.
2. **Extracted** without overwriting anything:
   - reports → `models/metadata/runs/baseline_20261003_172906/`
     (`run_metadata.json`, `class_order.json`, `history.csv`,
     `environment_freeze.txt`, `plots/*.png` — 7 members);
   - model bundle → `models/baseline_20261003_172906/`
     (`best_model.keras`, `class_order.json`, `run_metadata.json` — 3 members;
     directory is Git-ignored).
   Both targets were empty/absent beforehand; no other run directories exist
   or were touched. **Originals preserved byte-for-byte — never modified.**
3. **Verified the reported facts against the actual files — 37/37 checks
   passed** (read-only script): run id, commit + dirty flag, Python/TF/Keras
   versions, GPU count, 15 epochs, best epoch 14, both metric values rounded
   as reported (and matching `history.csv` argmin exactly), class order equal
   across metadata / `class_order.json` / committed `configs/class_mapping.json`,
   `test_manifest_used: false`, model SHA-256 equal to the hash of the
   extracted `best_model.keras` (`c25f275cba8b5520…`, 9,691,268 bytes),
   recorded manifest checksums equal to the local committed `train.csv` /
   `validation.csv`, and train/validation image counts (2141/458).
4. **Attempted local model load** with the existing `.venv`
   (Python 3.11.9, TF 2.15.1, Keras 2.15) using `compile=False` —
   **FAILED** (full captured log:
   [`docs/load_failure_baseline_20261003_172906.txt`](load_failure_baseline_20261003_172906.txt)).
   The `.venv` environment was **not changed**; no packages installed.
5. **Investigated the dirty Colab checkout** (see below): mechanism identified
   from code, cause documented as **unconfirmed**, recorded flag left untouched.
6. **Prevention for future runs:** added an explicit runtime output directory
   option to both scripts (below) and wired the notebook to use it; committed
   manifests untouched.
7. Updated `README.md`, `models/README.md`, and this entry.

### Verification (actual local runs)

| Check | Result |
|---|---|
| ZIP integrity + safe members | OK / none unsafe |
| Artifact verification script | **37/37 passed** |
| `tf.keras.models.load_model(..., compile=False)` in `.venv` | **FAILED** — `TypeError: Could not deserialize class 'Functional' because its parent module keras.src.models.functional cannot be imported` |
| Inference on synthetic 0–255 RGB batch | **Not reached** (load failed first) |
| `pytest tests\test_inspection_cli.py -v` | **5 passed** (new) |
| `pytest` full suite | **59 passed** (54 prior + 5 new), 0 failed |
| `inspect_data.py --help` / `download_data.py --help` | exit 0, `--metadata-dir` listed |
| Colab re-execution after these changes | **Not performed; not claimed** |

### Local compatibility status: **INCOMPATIBLE (as saved)**

- The artifact is a **Keras 3.13.2** `.keras` file (Keras-3 module paths such
  as `keras.src.models.functional`, `DTypePolicy`). The project `.venv` is
  **Keras 2.15**, whose module layout differs → deserialization fails even
  with `compile=False`. The file itself is intact (SHA-256 matches metadata).
- **Portability is NOT claimed.** Proposals (neither built nor verified yet):
  1. **Separate inference environment** (recommended): a second venv, e.g.
     `.venv-infer`, with Python 3.11 + `tensorflow==2.20.0`. Checked on
     2026-10-03: PyPI publishes `tensorflow-2.20.0-cp311-cp311-win_amd64.whl`,
     so it can coexist with `.venv` without modifying it. Must be created and
     verified with a real load + inference test before any portability claim.
  2. **Conversion in a Keras 3 environment**: re-export (SavedModel or legacy
     `.h5`) from a Keras 3 runtime, then verify a load + inference test in
     TF 2.15. Unverified — do not assume it works.

### Dirty-checkout investigation (unconfirmed)

If the Colab runtime is still available, run inside the clone to confirm:

```bash
cd /content/ai-waste-classification-assistant
git status --short
git diff --stat
git diff -- data/metadata/          # inspect affected tracked files
```

Mechanism found by code inspection (**likely cause; not confirmed without the
runtime output above**):

- `src/download_data.py` rewrites tracked `data/metadata/download_metadata.json`
  on **every** run with a fresh `retrieved_at_utc` (even when the archive is
  reused) — Step 7 runs it before training.
- `src/inspect_data.py` rewrites tracked `data/metadata/inspection_summary.json`
  (fresh `generated_at_utc`), `invalid_images.json`, and
  `class_distribution.png` on every run — also Step 7, before training.
- Training then collects `git status --porcelain` → non-empty → records
  `uncommitted_changes: true`.

Other candidate paths (`data/raw/`, `data/inspection/`, `__pycache__`,
notebook checkpoints) are Git-ignored, and the report export into
`models/metadata/runs/` happens *after* git info is collected — neither can
explain the flag. **The recorded dirty flag was not altered and no unexplained
changes were discarded.**

**Prevention added:** both scripts now accept `--metadata-dir` (default:
tracked `data/metadata`, so local workflows are unchanged) and the notebook
passes a Drive runtime directory (`<outputs>/runtime_metadata`), so future
Colab inspection runs no longer overwrite tracked metadata. The committed
manifests (`train.csv`, `validation.csv`, `test.csv`, `split_summary.json`)
are never written by these scripts and were preserved.

### Trackable artifacts committed

- `models/metadata/runs/baseline_20261003_172906/` — the 7 extracted small
  reports (originals, unmodified)
- `src/download_data.py`, `src/inspect_data.py` — `--metadata-dir` support
- `notebooks/train_colab.ipynb` — Step 5/7 wired to the runtime directory
- `tests/test_inspection_cli.py` — 5 regression tests
- `docs/load_failure_baseline_20261003_172906.txt` — captured load error
- `README.md`, `models/README.md`, `docs/progress.md`

**Not committed (ignored):** both ZIPs (`*.zip`), the model bundle under
`models/baseline_20261003_172906/` (`models/*` rule).

### Blockers

- Local inference is blocked until a Keras-3-compatible environment is built
  (proposal 1) or a conversion path is verified (proposal 2). Both require an
  explicit decision to add a *separate* environment — the existing `.venv`
  stays untouched.

### Next steps (not started)

1. Build + verify the separate inference environment (or a verified
   conversion), then run the synthetic-batch inference check (shape, finite,
   softmax sums) and only then claim local compatibility.
2. Later milestones: fine-tuning, test-set evaluation, Streamlit app — each
   scoped separately.

---

## Milestone 8 — Compatible baseline inference environment and inference verification

**Date:** 2026-10-04  
**Status:** `.venv-infer` created and verified; baseline model **loads and executes** locally — post-training compatibility verification **19/19 passed**. No model conversion, no retraining, no fine-tuning, no test-image access, no Streamlit. `.venv`, `requirements.txt`, `requirements.lock.txt` left unchanged.

### Environment roles (which interpreter for which task)

| Task | Interpreter | Versions |
|---|---|---|
| Tests, development, data/training scripts | `.venv\Scripts\python.exe` | TF 2.15.1 / Keras 2.15 (unchanged) |
| Loading + running the baseline model | `.venv-infer\Scripts\python.exe` | TF 2.20.0 / Keras 3.13.2 (matches Colab run) |

The Keras 2 main environment still cannot deserialize the artifact (that
failure stays documented in
[`docs/load_failure_baseline_20261003_172906.txt`](load_failure_baseline_20261003_172906.txt));
this milestone resolved compatibility by **adding a separate environment**,
never by converting the model or touching the original one.

### What was done

1. **Confirmed `.venv-infer/` was Git-ignored *before* creating it** — added
   `.venv-infer/` to `.gitignore` (line 3), then
   `git check-ignore -v .venv-infer/` matched `.gitignore:3`. Verified again
   after creation (`.venv-infer/pyvenv.cfg` → ignored).
2. **Created the environment** (preserving existing work):
   ```powershell
   py -3.11 -m venv .venv-infer
   .venv-infer\Scripts\python.exe -m pip install tensorflow==2.20.0 keras==3.13.2
   ```
   pip resolved the rest; **both target versions installed together, no
   substitution needed**. Resulting key versions: Python 3.11.9,
   TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.4.6 (Colab ran NumPy 2.1.3 —
   recorded as-is, not forced down).
3. **Dependency checks:** `.venv-infer\Scripts\python.exe -m pip check` →
   `No broken requirements found.` (exit 0); import check printed
   `2.20.0 3.13.2 2.4.6`. CPU-only device visible
   (`/physical_device:CPU:0`) — no GPU on this machine.
4. **Wrote the inference dependency files** (separate from the main ones):
   - `requirements-infer.txt` — direct pins (`tensorflow==2.20.0`,
     `keras==3.13.2`) + environment-role notes;
   - `requirements-infer.lock.txt` — full `pip freeze` for this Windows
     environment (36 packages, header records date/platform and that NumPy
     resolved to 2.4.6).
5. **Wrote `src/verify_baseline_inference.py`** — the post-training
   compatibility verification CLI (run with `.venv-infer` only). It checks,
   in order: model SHA-256 against the original `run_metadata.json`; safe
   load (`compile=False`, `safe_mode=True`, unsafe deserialization never
   enabled); input/output shapes; class order across all three sources;
   synthetic batch numerics; embedded-preprocessing probe; and a real-image
   sample through the shared pipeline. Report:
   `models/metadata/verification/post_training_compatibility_baseline_20261003_172906.json`
   (`report_type: post_training_compatibility_verification`).
6. **Fixed two weaknesses found during the first runs** (both re-verified):
   - the initial synthetic `arange % 256` batch made both images *identical*
     (150528 = 588×256), so the batch now applies a per-image offset → the
     two images produce genuinely different outputs;
   - the first real-image sample was all one class (`train.csv` is
     folder-sorted), so the sample is now chosen round-robin across the four
     classes — which is exactly what the label-order check needs.
7. Updated `README.md`, `models/README.md`, and this entry.

### Verification (actual runs)

Command used for rows 1–6:

```powershell
.venv-infer\Scripts\python.exe src\verify_baseline_inference.py
```

| Check | Result |
|---|---|
| `pip check` in `.venv-infer` | `No broken requirements found.` (exit 0) |
| Import check | `tensorflow 2.20.0, keras 3.13.2, numpy 2.4.6` |
| Model SHA-256 vs original `run_metadata.json` | **match** — `c25f275cba8b5520…32b524f` |
| Load `compile=False, safe_mode=True` | **success**, 0 warnings, `unsafe_deserialization_used: false` |
| Input / output shape | `(None, 224, 224, 3)` → `(None, 4)` |
| Class order (config / run metadata / model bundle) | identical: `metal, organic, paper, plastic` |
| Synthetic batch `(2, 224, 224, 3)` float32 0–255 | shape/dtype/range OK; two **distinct** images |
| Predictions | shape `(2, 4)`; all finite; scores in [0, 1] |
| Softmax sums | `[1.0, 1.0]`, max abs deviation `6.0e-08` |
| Repeated inference (`training=False`, 3 runs) | max abs diff **0.0** (stable) |
| Preprocessing probe (0→−1, 127.5→0, 255→+1) | **exact**; embedded in graph, applied once |
| Normalization outside frozen backbone | none (stray layers: `[]`) |
| Real-image sample (6 training images, shared `data_pipeline`) | executed; `(6, 4)` finite, sums within 1.2e-07; round-robin over all 4 classes; `test_manifest_opened: false` |
| **Overall verification status** | **`pass` — 19/19 checks, 0 errors, exit 0** |
| Optional `--save-predictions` / `--compare-predictions` round-trip | works (self-comparison `max_abs_diff: 0.0`, 20/20 with that check) |
| Full test suite in `.venv` (unchanged env) | **59 passed**, 0 failed |
| Model conversion / retraining / fine-tuning | **not performed** (out of scope) |
| Test images opened | **no** (only `train.csv`; recorded in the report) |
| Numerical parity with Colab | **not claimed** — no Colab output file exists |

### What the report does and does not say

- **Does:** environment versions; model checksum (expected vs actual); load
  options; input/output shapes; numerical checks (finite, 4 scores/image,
  softmax sums, stability); the preprocessing probe evidence; the explicitly
  recorded real-image sample rows (path, target, predicted label, scores);
  warnings/errors (none occurred); `verification_status: pass`.
- **Does not:** claim accuracy of any kind (synthetic outputs are labeled
  execution-only; no metric is computed from the 6-image training sample);
  claim Colab parity; imply the model is good — only that it loads, runs,
  and behaves consistently with how it was trained.

### Blockers

- **None for local inference.** Remaining limitations (not blockers, just
  recorded honestly):
  - NumPy differs from the Colab run (2.4.6 local vs 2.1.3 recorded) — TF/Keras
    match; not pinned down because pip resolved it and no check failed.
  - No GPU on this machine (CPU-only device); inference speed differs from
    Colab's T4 — irrelevant to correctness.
  - Numerical parity with Colab unverified until someone runs the optional
    `--save-predictions` flow in a live Colab runtime.

### Next steps (not started; each a separate milestone)

1. Optional: run `--save-predictions` in Colab + `--compare-predictions`
   locally to establish cross-machine parity (commands in `README.md`).
2. Later milestones: test-set evaluation, fine-tuning, Streamlit app — each
   scoped separately; the compatibility verification above does **not**
   unlock them automatically.

---

*Append new milestones below this line.*

---

## Milestone 9 — Controlled fine-tuning and validation comparison

**Date:** 2026-10-04  
**Status:** Code + tests + Colab notebook prepared and verified locally. **No real fine-tuning run exists yet** — the notebook has not been executed in Google Colab, so there are **no fine-tuning metrics and no comparison decision** in this milestone. No test-set evaluation, no Streamlit.

### What was done

1. **`src/finetune.py`** (818 lines, 10 functions) — controlled fine-tuning CLI/workflow that starts from the **verified baseline artifact** (`models/baseline_20261003_172906/best_model.keras`):
   - **Selective unfreeze** (`apply_finetune_policy`): `backbone.trainable = True` first, then per-layer flags — only non-BatchNorm layers from `block_13` onward train (index-based start, because MobileNetV2's first block is `expanded_conv`); every BatchNorm layer stays frozen; the `predictions` head stays trainable. Self-checks refuse to continue if a BN layer became trainable or a block-13+ layer stayed frozen.
   - **`training=False` guard:** the backbone call's recorded `training` kwarg is read from the graph (Keras 3: `node.arguments.kwargs`, Keras 2: `node.call_kwargs`). If it is `True` or cannot be verified, fine-tuning **refuses to start** (`RuntimeError`) — BN running statistics must never be updated from small batches.
   - **Fresh optimizer** (`compile_finetune`): new `Adam(learning_rate=0.00001)` + `SparseCategoricalCrossentropy`, because reusing baseline optimizer state at a lower LR would make the first steps wrong.
   - **Parent integrity:** parent SHA-256 recorded *before* training; sidecar `run_metadata.json`/`class_order.json` are cross-checked (checksum mismatch or class-order mismatch → refuse); SHA-256 re-checked *after* training and the run fails if the baseline bytes changed (`parent.unchanged_after_training: true`).
   - **Data:** only `train.csv` + `validation.csv` are opened (`test_manifest_used: false` recorded); same manifests, seed 42, and class-weight policy as the baseline.
   - **Run directory:** separate `models/runs/finetune_<timestamp>/` (`RUN_PROFILE = "finetune"`) — never overwrites the baseline run; `best_model.keras`, `history.csv`, `run_metadata.json` (policy, configuration, honesty notes), `plots/`, `environment_freeze.txt`; optional export of small reports to `models/metadata/runs/finetune_<timestamp>/`.
   - Defaults: max 10 epochs, lr 0.00001, start block 13 (overridable via an optional `"finetune"` block in `configs/training.json`; unknown keys fail loudly). EarlyStopping + ModelCheckpoint on `val_loss`, patience 3 (shared `train.build_callbacks`).
   - Helpers reused from `src/train.py` — its three private helpers were made public for this: `_resolve_config_path` → `resolve_config_path`, `_new_run_dir` → `new_run_dir`, `_history_rows` → `history_to_rows` (rename-only diff; tests green).
2. **`src/compare_validation.py`** (503 lines, 8 functions) — validation-only comparison and model selection:
   - Evaluates baseline and fine-tuned model on the **identical** deterministic validation pass (manifest order, `training=False`, no shuffle, no augmentation, full split including the final partial batch); test manifest never opened.
   - Metrics in **pure NumPy** (no scikit-learn): loss, accuracy, macro F1, per-class precision/recall/F1/support, confusion matrix. Zero-division → `0.0`; `argmax` ties → lowest class index.
   - **Alignment and output guards:** row alignment is checked against the manifest (mismatch → `RuntimeError`, never a silently wrong report); non-finite outputs and rows that don't sum to 1 also raise.
   - **Selection rule fixed in code before any run** (`SELECTION_RULE`, travels inside the report): (1) fine-tuned wins iff macro F1 strictly higher; (2) F1 tie → fine-tuned only if accuracy strictly higher; (3) otherwise keep the baseline. Within one run the best checkpoint is still lowest `val_loss` — this rule never compares checkpoints inside a run.
   - **Honest pending state:** if `--finetune-model` is missing/omitted, the report is written with `status: "pending"` and **no metrics** — nothing is invented. Default output: `models/metadata/comparison/validation_comparison.json`.
3. **`notebooks/finetune_colab.ipynb`** (33 cells: 17 code, 16 markdown) — the real fine-tuning workflow for Google Colab: verifies/installs TF 2.20.0 + Keras 3.13.2 (restart guard), Drive mount, repo clone at a pinned `REVISION`, baseline zip download with **double SHA-256 check** (against the zip's own `run_metadata.json` *and* the committed run metadata), dataset download via `--metadata-dir` (checkout stays clean), manifest checksum validation, `nvidia-smi`, subprocess runs of `src/finetune.py` then `src/compare_validation.py` (`PYTHONHASHSEED`, `cwd=` the repo), and zips of reports/model/comparison back to Drive. States honestly that it "has not been executed in Google Colab" and reports **validation** metrics only.
4. **Tests** — 4 new files + 1 compatibility fix (all synthetic or manifest-level; no RealWaste download):
   - `tests/test_finetune.py` — **23 tests**: layer policy (block_13+, BN frozen, head trainable), fresh optimizer, parent checksum/class-order refusal, config validation, `run_tiny_finetune` smoke (1 epoch, 32 px, batch 2), CLI help/errors.
   - `tests/test_compare_validation.py` — **17 tests**: metric math, zero-division and tie policies, alignment/softmax guards, all three selection-rule branches, pending report (no metrics, exit 0, checksum recorded), CLI.
   - `tests/test_finetune_notebook.py` — **9 tests**: cell structure/syntax, required topics, selection-rule text appears *before* the compare cell, no lock-file install in code, no accuracy claims, `REPO_DIR` normalized to a `Path`.
   - `tests/test_finetune_keras3.py` — **4 tests, Keras 3 only** (`pytest.mark.skipif` on `keras.__version__`): real baseline policy + recompile (1,668,484 trainable / 594,624 non-trainable params, `trainable > 0`), tiny-fine-tune save/load round trip with **parent SHA unchanged**, `run_tiny_finetune` smoke, 4-image real-validation alignment through `collect_predictions` (no metrics).
   - `tests/test_model.py` — made Keras 2/3 tolerant (preprocessing probe via `_inbound_nodes[0].input_tensors`; head-weight assertion accepts both naming schemes). Existing tests otherwise untouched.
5. **Inference env test deps** — new `requirements-infer-test.txt` (`pytest==9.1.1`, `matplotlib==3.11.2`); `requirements-infer.lock.txt` regenerated (47 packages; 3 header lines + 11 new test deps; LF + UTF-8 BOM preserved); `pip check` → no broken requirements. `requirements-infer.txt` points to the new file.
6. Updated `README.md`, `models/README.md`, and this entry.

### Verification (actual runs)

| Check | Result |
|---|---|
| `.venv` full suite (`pytest -q`) | **108 passed, 4 skipped** (the 4 Keras 3-only tests skip on Keras 2), 0 failed |
| `.venv-infer` full suite (`pytest -q`) | **112 passed**, 0 failed |
| New test counts | `test_finetune.py` 23, `test_compare_validation.py` 17, `test_finetune_notebook.py` 9, `test_finetune_keras3.py` 4 |
| `test_finetune_keras3.py` in `.venv-infer` | 4 passed (real baseline: policy, recompile, training round trip, validation alignment) |
| `test_model.py` in both envs | 10 passed each (Keras 2 **and** Keras 3) |
| Baseline artifact SHA-256 vs `run_metadata.json` | still matches (`c25f275cba8b5520…`); parent file never modified by tests |
| `.venv-infer` `pip check` | `No broken requirements found.` (exit 0) |
| `src/finetune.py --help`, `src/compare_validation.py --help` | exit 0 in `.venv` |
| Comparison without a fine-tuned model | writes `status: pending`, **no metrics**, exit 0 |
| **Real fine-tuning run** | **not performed** (Colab notebook prepared but not executed) |
| **Comparison decision (baseline vs fine-tuned)** | **not performed** — no fine-tuned model exists; only the pending path was exercised |
| Test-set access | **none** (`test_manifest_used: false` in both workflows) |

### What this milestone does and does not say

- **Does:** reusable, test-covered code for controlled fine-tuning and for a fair validation-only comparison with a rule fixed *before* results; a Colab notebook ready to run; honest `pending` state when the fine-tuned model does not exist.
- **Does not:** claim any fine-tuning metric, improvement, or model-selection decision — no run has happened; claim test-set results (still untouched); start Streamlit work.

### Blockers

- **None for the code.** The remaining work is execution, not development: run `notebooks/finetune_colab.ipynb` in Google Colab (T4 GPU) to produce the fine-tuned model and the comparison report, then bring the zips back as in milestone 6.

### Next steps (not started; each a separate milestone)

1. Execute `notebooks/finetune_colab.ipynb` on Colab; record the real fine-tune run + comparison decision in `docs/progress.md` (never invent numbers).
2. Test-set evaluation of whichever model the selection rule picked (held out until now).
3. Streamlit app serving the selected model.

---

## Milestone 10 — Selected-model import and held-out test evaluation

**Date:** 2026-10-04  
**Status:** Complete — fine-tuned artifacts imported + verified, selection recorded **before** the test split was opened, `src/evaluate.py` written and tested, test set evaluated **once** (459/459 images). No further training, no Streamlit.

### What was done

1. **Import (24/24 checks passed).** Located the three Colab ZIPs in the repo root (`reports_finetune_20261004_133620.zip`, `model_finetune_20261004_133620.zip`, `validation_comparison.zip`); `ZipFile.testzip()` OK, no absolute/`..` members, targets did not pre-exist, extraction byte-identical, source ZIPs never modified. Extracted to `models/metadata/runs/finetune_20261004_133620/` (6 report files, tracked), `models/finetune_20261004_133620/` (model bundle, Git-ignored), `models/metadata/comparison/validation_comparison.json` (tracked). Evidence: [`models/metadata/verification/import_finetune_20261004_133620.json`](../models/metadata/verification/import_finetune_20261004_133620.json) — model SHA `39f7b78b…` consistent across run metadata, comparison report, and the extracted file; class order and manifest checksums agree with the committed configs.
2. **Local compatibility of the fine-tuned artifact (19/19 checks passed).** `src/verify_baseline_inference.py` gained generic *subject* naming (baseline vs fine-tuned, derived from `run_metadata["profile"]`) — no check logic changed. Run in `.venv-infer` against the fine-tuned model: SHA match, load with `compile=False, safe_mode=True` (no warnings), `(None, 224, 224, 3) → (None, 4)`, softmax sums dev ≤ 5.96e-08, repeated inference max abs diff 0.0, preprocessing once inside the graph. Evidence: `models/metadata/verification/post_training_compatibility_finetune_20261004_133620.json`.
3. **Selection record frozen before test access.** `models/metadata/selection/selected_model.json` written from the comparison report + run metadata + model file (script refuses to overwrite an existing record): `selected: finetuned`, reason `rule step 1: macro F1 improved 0.932254 -> 0.937250`, model SHA, class order, verification evidence, and the **git checkout flags as found**. The comparison report's `uncommitted_changes: true` (created 13:42:01Z) vs the fine-tune run's `false` (finished 13:38:42Z) is recorded verbatim with `confirmation_status: not confirmed by runtime output` — the code path (`collect_git_info` = `git status --porcelain`, non-empty because the notebook exported run reports **inside the clone** before `run_metadata.json` was written) is documented as a *mechanism*, not a proven cause. **The flag was not rewritten and no cause was invented.**
4. **`src/evaluate.py`** (new) — held-out test evaluation with guard rails: the model path comes from the selection record only (no `--model` flag, so the choice cannot be reopened); model SHA re-checked against the record; `test.csv` SHA-256 + row count re-checked against committed `split_summary.json`; class order must agree across selection record, config, and model bundle; coverage assertion requires **every image exactly once** (row count, prediction count, no duplicate path). Metrics reuse `compare_validation.metrics_from_predictions` (same code as validation). No fitting, no tuning, argmax predictions, row-normalized confusion matrix, per-image score table, optional misclassification grid into the **Git-ignored** `models/runs/misclassified_grids/`. Default output: `models/metadata/evaluation/test_evaluation.json`.
5. **`tests/test_evaluate.py`** (new, 21 tests, synthetic fixtures in `tmp_path`): hand-computed confusion normalization, prediction/label row alignment, coverage refusals (missing/extra predictions, duplicate path, manifest≠summary count), manifest and model checksum refusals, class-order disagreement, swapped-label detection via a monkeypatched dataset, a spy proving **only `test.csv` is opened**, self-consistent report invariants, `model.fit` never called, determinism across runs, grid locality, CLI help/exit codes. A local fixture helper builds `test.csv` + `split_summary.json` because `build_tiny_manifests` deliberately creates none.
6. **Test-set evaluation executed once** with `.venv-infer\Scripts\python.exe src\evaluate.py` → report written, exit 0 (results below).
7. **Docs:** created [`docs/model_card.md`](model_card.md); updated `README.md` (status, layout, test commands, milestone-9 status now *executed*, new milestone-10 section), `models/README.md` (imported fine-tune bundle + selection/evaluation locations), and this entry.

### Actual measured test results (from the run's own report)

Command: `.venv-infer\Scripts\python.exe src\evaluate.py` →
[`models/metadata/evaluation/test_evaluation.json`](../models/metadata/evaluation/test_evaluation.json)
(created 2026-10-04T14:48:16Z; **one execution**, no re-run for better numbers).

| Metric | Value |
|---|---:|
| Images | 459 (430 correct, 29 misclassified; each evaluated exactly once) |
| Loss | 0.17239077061109537 |
| Accuracy | 0.9368191721132898 |
| Macro F1 | 0.934730625209095 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| metal | 0.8837209302325582 | 0.957983193277311 | 0.9193548387096775 | 119 |
| organic | 1.0 | 1.0 | 1.0 | 127 |
| paper | 0.9078947368421053 | 0.92 | 0.913907284768212 | 75 |
| plastic | 0.9448818897637795 | 0.8695652173913043 | 0.9056603773584906 | 138 |

Confusion matrix (rows true, columns predicted; order metal, organic, paper, plastic):

|  | metal | organic | paper | plastic |
|---|---:|---:|---:|---:|
| **metal** | 114 | 0 | 1 | 4 |
| **organic** | 0 | 127 | 0 | 0 |
| **paper** | 3 | 0 | 69 | 3 |
| **plastic** | 12 | 0 | 6 | 120 |

Also held in the report: protocol (no fitting / no tuning / augmentation inert / final partial batch kept), coverage block, both checksum verifications, per-image softmax scores (459 rows), 5 honesty notes, environment (Python 3.11.9, TF 2.20.0, Keras 3.13.2) and the git block. The grid (29 wrong images) was written to `models/runs/misclassified_grids/sample_grid_test_misclassified.png` and confirmed **ignored** by `.gitignore:57 models/*`.

### Verification (actual runs)

| Check | Result |
|---|---|
| Import verification | **24/24 passed**, evidence JSON written |
| Fine-tuned artifact compatibility (`.venv-infer`) | **19/19 passed** |
| `pytest` full suite in `.venv` | **129 passed, 4 skipped** (108 prior + 21 new), 0 failed |
| `pytest` full suite in `.venv-infer` | **133 passed**, 0 failed |
| `src/evaluate.py --help` | exit 0 |
| Real evaluation run (`.venv-infer`) | exit 0; coverage `459/459`, `each_image_evaluated_exactly_once: true` |
| Model SHA vs selection record | match (`39f7b78befd182c6…`) |
| `test.csv` SHA vs `split_summary.json` | match (`74494c13f6d9be7f…`) |
| Class order (selection / config / model bundle) | identical, verified |
| `git check-ignore` on the misclassification grid | ignored (`.gitignore:57`) |
| Test images opened before the selection record | **no** — training/fine-tune/compare recorded `test_manifest_used: false`; compatibility runs recorded `test_manifest_opened: false` |
| Baseline artifact unchanged | SHA still `c25f275cba8b5520…` (never modified) |
| Model choice influenced by test results | **no** — the selection record predates `src/evaluate.py` and this run |

Honest note on the evaluation run's own git block: it records
`uncommitted_changes: true` because the milestone-10 files were not yet
committed when the script ran (HEAD was `7db03a6`, the import/selection
commit); nothing in the report or the model was changed afterwards.

### Blockers

- None for this milestone. Remaining limitations are documented, not
  blocking: closed set of four classes (no unknown rejection), single
  dataset/single seed, small selection gain, and unconfirmed dirty-checkout
  cause on the Colab comparison run (recorded in the selection record).

### Next milestone (not started)

- **Milestone 11 — Streamlit app:** serve the selected model
  (`models/finetune_20261004_133620/best_model.keras`, Git-ignored) with
  upload → predict → display, class order from `configs/class_mapping.json`,
  honest confidence wording, and tests. Do not retrain or re-evaluate.

---

*Append new milestones below this line.*
