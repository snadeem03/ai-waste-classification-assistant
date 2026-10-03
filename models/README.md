# `models/` directory

## Intended contents (after the training milestone)

| Path | Tracked in Git? | Purpose |
|---|---|---|
| `models/README.md` | **Yes** | This file — explains layout and rules |
| `models/metadata/` | **Yes** (small files only) | Training run notes, label order copy, evaluation summary (JSON/CSV/MD) — **no weight binaries** |
| `models/metadata/runs/<run_id>/` | **Yes** | Small per-run reports exported by `src/train.py --export-reports`: `run_metadata.json`, `class_order.json`, `history.csv`, `environment_freeze.txt`, `plots/*.png` |
| `models/metadata/verification/` | **Yes** | Post-training compatibility verification reports (JSON) produced by `src/verify_baseline_inference.py` |
| `models/runs/baseline_<timestamp>/` | **No** | Full local run directory (default `--output-dir`): `best_model.keras`, logs, everything — ignored by `models/*` |
| `models/best_model.keras` (or similar) | **No** | Saved Keras model restored from a Colab run zip |
| Checkpoints / SavedModel folders | **No** | Intermediate training artifacts |
| TensorBoard logs | **No** | Training curves (local only) |

## Why large model files are excluded from Git

- Model weights are megabytes to hundreds of megabytes — bad for clones and PRs.
- Models are reproducible from code + dataset + training script (when those exist).
- `.gitignore` blocks `*.keras`, `*.h5`, `*.tflite`, SavedModel folders, and `models/*` except this README and `models/metadata/`.

## Current status

**First baseline run exists** — `baseline_20261003_172906` (Google Colab,
2026-10-03, commit `199f253`):

| Artifact | Location | Tracked? |
|---|---|---|
| Small run reports | `models/metadata/runs/baseline_20261003_172906/` | **Yes** |
| `best_model.keras` (9.7 MB, SHA-256 `c25f275cba8b5520…`) | `models/baseline_20261003_172906/best_model.keras` | **No** (ignored) |

Measured **validation** results (from that run's own `run_metadata.json`,
verified locally): best epoch 14 of 15, `val_loss` 0.2097, `val_accuracy`
0.9323, class order `["metal", "organic", "paper", "plastic"]`, test set
never used. No test-set metrics exist yet.

**Local loading status: VERIFIED (2026-10-04).** The model does **not** load
in the project `.venv` (TF 2.15.1 / Keras 2.15 — it was saved by Keras 3.13.2
and fails with `TypeError: Could not deserialize class 'Functional'…`; full
log in [`docs/load_failure_baseline_20261003_172906.txt`](../../docs/load_failure_baseline_20261003_172906.txt)).
It **does** load and execute in the separate inference environment
`.venv-infer` (TF 2.20.0 / Keras 3.13.2, matching the Colab run), loaded with
`compile=False, safe_mode=True` (no unsafe deserialization).

Post-training compatibility verification — **19/19 checks passed**:
[`models/metadata/verification/post_training_compatibility_baseline_20261003_172906.json`](metadata/verification/post_training_compatibility_baseline_20261003_172906.json)
(SHA-256 match, shapes, class order, synthetic numerical checks, embedded
MobileNetV2 preprocessing, real-image sample execution). Reproduce with:

```powershell
.venv-infer\Scripts\python.exe src\verify_baseline_inference.py
```

The report contains **execution checks only — not accuracy**. Test-set
evaluation has not been run, and numerical parity with Colab is not claimed
(no Colab output file exists).

## When a model is trained later

1. Run the Colab notebook (or `src/train.py` locally) and keep the run
   directory (`best_model.keras` lives there, ignored by Git).
2. Export the small reports into `models/metadata/runs/<run_id>/` (tracked)
   and record honest metrics there — copied from that run's own
   `run_metadata.json`, never invented.
3. Document class order identical to [`configs/class_mapping.json`](../../configs/class_mapping.json): `["metal", "organic", "paper", "plastic"]`.
4. Never invent or copy accuracy figures from other projects.
