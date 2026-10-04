# `models/` directory

## Intended contents (after the training milestone)

| Path | Tracked in Git? | Purpose |
|---|---|---|
| `models/README.md` | **Yes** | This file — explains layout and rules |
| `models/metadata/` | **Yes** (small files only) | Training run notes, label order copy, evaluation summary (JSON/CSV/MD) — **no weight binaries** |
| `models/metadata/runs/<run_id>/` | **Yes** | Small per-run reports exported by `src/train.py --export-reports` (and `src/finetune.py`): `run_metadata.json`, `class_order.json`, `history.csv`, `environment_freeze.txt`, `plots/*.png` |
| `models/metadata/verification/` | **Yes** | Post-training compatibility verification reports (JSON) produced by `src/verify_baseline_inference.py` |
| `models/metadata/comparison/` | **Yes** | Model-selection report from `src/compare_validation.py` (`validation_comparison.json`) |
| `models/metadata/selection/` | **Yes** | `selected_model.json` — which model was chosen and why, frozen **before** the test split was opened |
| `models/metadata/evaluation/` | **Yes** | Held-out test report from `src/evaluate.py` (`test_evaluation.json`) |
| `models/finetune_<run_id>/` | **No** | Fine-tuned model bundle (`best_model.keras` + sidecars) imported from Colab |
| `models/runs/baseline_<timestamp>/` | **No** | Full local run directory (default `--output-dir`): `best_model.keras`, logs, everything — ignored by `models/*` |
| `models/runs/finetune_<timestamp>/` | **No** | Full fine-tuning run directory from `src/finetune.py` (parent checksum + policy recorded in its `run_metadata.json`) |
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
never used — and the baseline was **not** evaluated on the test split later
either (only the selected fine-tuned model was).

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

The report contains **execution checks only — not accuracy**, and numerical
parity with Colab is not claimed (no Colab output file exists). The held-out
**test** metrics live in `models/metadata/evaluation/test_evaluation.json`
(see [`docs/model_card.md`](../docs/model_card.md)).

## Fine-tuning and model selection (milestone 9)

`src/finetune.py` fine-tunes the baseline into a **separate**
`models/runs/finetune_<timestamp>/` directory — the baseline bundle above is
never modified (its SHA-256 is checked before and after training, recorded as
`parent.unchanged_after_training`). `src/compare_validation.py` then writes
the selection report to `models/metadata/comparison/validation_comparison.json`.

**Executed and imported (2026-10-04).** The Colab run produced fine-tune run
`finetune_20261004_133620` and the comparison selected the **fine-tuned**
model (`rule step 1: macro F1 improved 0.932254 -> 0.937250`):

| Artifact | Location | Tracked? |
|---|---|---|
| Fine-tune run reports | `models/metadata/runs/finetune_20261004_133620/` | **Yes** |
| Fine-tuned `best_model.keras` (23.0 MB, SHA-256 `39f7b78befd182c6…`) | `models/finetune_20261004_133620/best_model.keras` | **No** (ignored) |
| Validation comparison | `models/metadata/comparison/validation_comparison.json` | **Yes** |
| Selection record (frozen before test access) | `models/metadata/selection/selected_model.json` | **Yes** |
| Held-out test report (459 images, one pass) | `models/metadata/evaluation/test_evaluation.json` | **Yes** |

Measured numbers for that model — validation **and** held-out test — are
documented in [`docs/model_card.md`](../docs/model_card.md); never retype
them from memory. Reproduce the evaluation with
`.venv-infer\Scripts\python.exe src\evaluate.py`.

## When a model is trained later

1. Run the Colab notebook (or `src/train.py` locally) and keep the run
   directory (`best_model.keras` lives there, ignored by Git).
2. Export the small reports into `models/metadata/runs/<run_id>/` (tracked)
   and record honest metrics there — copied from that run's own
   `run_metadata.json`, never invented.
3. Document class order identical to [`configs/class_mapping.json`](../../configs/class_mapping.json): `["metal", "organic", "paper", "plastic"]`.
4. Never invent or copy accuracy figures from other projects.
