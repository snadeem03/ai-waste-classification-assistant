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

*Append new milestones below this line.*
