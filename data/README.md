# `data/` directory

## Intended contents (when data prep runs)

| Path | Tracked in Git? | Purpose |
|---|---|---|
| `data/README.md` | **Yes** | This file — explains layout and rules |
| `data/metadata/` | **Yes** (small files only) | Split manifests, class maps, image count tables, hash reports (CSV/JSON/MD) |
| `data/raw/RealWaste/...` | **No** | Original downloaded images, one folder per source category |
| `data/processed/...` | **No** | Resized / normalized copies used for training |
| Downloaded archives (`*.zip`, etc.) | **No** | e.g. RealWaste zip from UCI |

## Why large files are excluded from Git

- Repositories stay small and clone quickly on college laptops.
- GitHub has file-size limits (~100 MB per file); a 656 MB dataset zip will fail or bloat history.
- Datasets are better restored from the official source using documented commands.
- Binary images do not diff meaningfully in code review.

## Current status

**The dataset is not available yet.** Nothing has been downloaded. No images exist under `data/raw/` or `data/processed/`. Do not claim training-ready data until the data-prep milestone fills those folders and writes metadata.

## Expected class mapping (preview)

Target classes: `metal`, `organic`, `paper`, `plastic`  
Full mapping and exclusions: [`docs/dataset.md`](../../docs/dataset.md) and [`configs/class_mapping.json`](../../configs/class_mapping.json)
