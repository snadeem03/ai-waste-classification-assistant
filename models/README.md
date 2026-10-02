# `models/` directory

## Intended contents (after the training milestone)

| Path | Tracked in Git? | Purpose |
|---|---|---|
| `models/README.md` | **Yes** | This file — explains layout and rules |
| `models/metadata/` | **Yes** (small files only) | Training run notes, label order copy, evaluation summary (JSON/CSV/MD) — **no weight binaries** |
| `models/metadata/runs/<run_id>/` | **Yes** | Small per-run reports exported by `src/train.py --export-reports`: `run_metadata.json`, `class_order.json`, `history.csv`, `environment_freeze.txt`, `plots/*.png` |
| `models/runs/baseline_<timestamp>/` | **No** | Full local run directory (default `--output-dir`): `best_model.keras`, logs, everything — ignored by `models/*` |
| `models/best_model.keras` (or similar) | **No** | Saved Keras model restored from a Colab run zip |
| Checkpoints / SavedModel folders | **No** | Intermediate training artifacts |
| TensorBoard logs | **No** | Training curves (local only) |

## Why large model files are excluded from Git

- Model weights are megabytes to hundreds of megabytes — bad for clones and PRs.
- Models are reproducible from code + dataset + training script (when those exist).
- `.gitignore` blocks `*.keras`, `*.h5`, `*.tflite`, SavedModel folders, and `models/*` except this README and `models/metadata/`.

## Current status

**No trained model exists yet.** The baseline training workflow
(`src/train.py` + `notebooks/train_colab.ipynb`) was prepared in milestone 6
and verified with synthetic images and structural tests, but it has **not**
been executed in Google Colab — there are no weights and no accuracy numbers.
See [`docs/progress.md`](../../docs/progress.md).

## When a model is trained later

1. Run the Colab notebook (or `src/train.py` locally) and keep the run
   directory (`best_model.keras` lives there, ignored by Git).
2. Export the small reports into `models/metadata/runs/<run_id>/` (tracked)
   and record honest metrics there — copied from that run's own
   `run_metadata.json`, never invented.
3. Document class order identical to [`configs/class_mapping.json`](../../configs/class_mapping.json): `["metal", "organic", "paper", "plastic"]`.
4. Never invent or copy accuracy figures from other projects.
