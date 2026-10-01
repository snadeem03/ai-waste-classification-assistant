# `models/` directory

## Intended contents (after the training milestone)

| Path | Tracked in Git? | Purpose |
|---|---|---|
| `models/README.md` | **Yes** | This file — explains layout and rules |
| `models/metadata/` | **Yes** (small files only) | Training run notes, label order copy, evaluation summary (JSON/CSV/MD) — **no weight binaries** |
| `models/mobilenetv2_waste.keras` (or similar) | **No** | Saved Keras model for Streamlit inference |
| Checkpoints / SavedModel folders | **No** | Intermediate training artifacts |
| TensorBoard logs | **No** | Training curves (local only) |

## Why large model files are excluded from Git

- Model weights are megabytes to hundreds of megabytes — bad for clones and PRs.
- Models are reproducible from code + dataset + training script (when those exist).
- `.gitignore` blocks `*.keras`, `*.h5`, `*.tflite`, SavedModel folders, and `models/*` except this README and `models/metadata/`.

## Current status

**No trained model exists yet.** There are no weights, no accuracy numbers, and no Streamlit demo that loads a real classifier. Training is a later milestone per [`docs/progress.md`](../../docs/progress.md).

## When a model is trained later

1. Save the binary under `models/` (ignored by Git).
2. Record honest metrics under `models/metadata/` (tracked).
3. Document class order identical to [`configs/class_mapping.json`](../../configs/class_mapping.json): `["metal", "organic", "paper", "plastic"]`.
4. Never invent or copy accuracy figures from other projects.
