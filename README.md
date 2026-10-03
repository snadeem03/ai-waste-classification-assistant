# AI Waste Classification Assistant

College AIML Project-Based Learning (PBL) — **status: dataset downloaded, splits created, model built, first Colab baseline run completed (validation metrics recorded); baseline model verified to load + execute locally in the isolated `.venv-infer` environment (TensorFlow 2.20 / Keras 3).**

## What this project will do

Upload an image of waste and get a predicted category:

| Class | Meaning |
|---|---|
| plastic | Plastic bottles, containers, packaging |
| paper | Paper, cardboard, notebooks |
| metal | Cans, foil, metal scraps |
| organic | Food waste, leaves, biodegradable material |

The planned approach:

1. Train **MobileNetV2** (ImageNet weights) with **transfer learning** on a labeled waste dataset.
2. Freeze the base convolutional backbone and train a small classification head for the four classes.
3. Save the trained model as a binary (kept **out of Git**; `.gitignore` blocks it).
4. Serve predictions through a **Streamlit** web app: the user uploads an image, the app loads the model, and displays the predicted class with a confidence score.

## Important honesty note

**Honest status.** Milestones 1–4 defined scope, downloaded RealWaste, and wrote duplicate-checked 70/15/15 split manifests. Milestone 5 added the input pipeline and model *construction*. Milestone 6 added the training workflow (`src/train.py`) and the Colab notebook. **The first real baseline run has now executed in Google Colab** (run `baseline_20261003_172906`, recorded below and in `docs/progress.md`): its **validation** metrics are real, read from that run's own `run_metadata.json`/`history.csv`, and verified against the extracted artifacts. **The saved model now loads and executes locally** in a separate inference environment (`.venv-infer`, TensorFlow 2.20 / Keras 3 — post-training compatibility verification passed, see *Local compatibility status*). No test-set evaluation, no fine-tuning, and no demo app exist yet.

## Project documentation

| Document | What it covers |
|---|---|
| [`docs/project_scope.md`](docs/project_scope.md) | Problem, objectives, stack, scope, acceptance criteria, limitations |
| [`docs/dataset.md`](docs/dataset.md) | RealWaste source, license, categories, mapping, exclusions, split plan |
| [`configs/class_mapping.json`](configs/class_mapping.json) | Machine-readable source→target map + fixed class order |
| [`data/README.md`](data/README.md) | What belongs under `data/` and why images stay out of Git |
| [`models/README.md`](models/README.md) | What belongs under `models/` and why weights stay out of Git |
| [`docs/progress.md`](docs/progress.md) | Milestone log (what changed, verification, next steps) |
| [`AGENTS.md`](AGENTS.md) | Rules for every future milestone |

### Dataset snapshot (RealWaste)

- Source: [UCI RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) (4,752 images, 9 original categories)
- Training mapping: Plastic→`plastic`, Paper→`paper`, Metal→`metal`, Food Organics + Vegetation→`organic`
- Excluded for now: Cardboard, Glass, Miscellaneous Trash, Textile Trash
- Fixed class order: `["metal", "organic", "paper", "plastic"]`
- Details and license notes: [`docs/dataset.md`](docs/dataset.md)

## Project layout

```
ai-waste-classification-assistant/
├── .gitignore                 # ignores venvs, data images, model binaries, secrets
├── README.md                  # this file
├── AGENTS.md                  # rules for future work on this repo
├── requirements.txt           # direct dependencies (main .venv)
├── requirements.lock.txt      # exact working local versions (main .venv)
├── requirements-infer.txt     # inference env direct deps (.venv-infer)
├── requirements-infer.lock.txt# exact Windows versions (.venv-infer)
├── configs/
│   ├── class_mapping.json     # source→target map + class order (single source of truth)
│   └── training.json          # seed, image/batch size, lr, epochs, dropout
├── docs/
│   ├── project_scope.md       # scope and acceptance criteria
│   ├── dataset.md             # RealWaste mapping and license notes
│   ├── load_failure_baseline_20261003_172906.txt  # Keras 2 load error (captured)
│   └── progress.md            # milestone log
├── data/
│   ├── README.md              # tracked
│   ├── metadata/              # tracked manifests + reports (train/validation/test CSVs)
│   └── raw|processed/         # ignored images
├── models/
│   ├── README.md              # tracked
│   ├── metadata/              # tracked: run reports + compatibility verification
│   └── *.keras etc.           # ignored weights
├── src/
│   ├── download_data.py       # UCI download + safe extraction
│   ├── inspect_data.py        # counts, mapping check, image validation
│   ├── prepare_data.py        # duplicate checks + stratified split manifests
│   ├── data_pipeline.py       # manifest loading + tf.data input pipeline
│   ├── model.py               # MobileNetV2 model construction
│   ├── train.py               # frozen-base baseline training CLI (Colab)
│   └── verify_baseline_inference.py  # post-training compatibility verification
│                              #   (run with .venv-infer, NOT .venv)
├── app/                       # Streamlit UI (future)
├── notebooks/
│   └── train_colab.ipynb      # Colab baseline training (first run 2026-10-03)
└── tests/                     # pytest checks (synthetic images; no download)
```

## Environment setup (Windows, Python 3.11)

**Two environments, two jobs — keep them separate:**

| Environment | Versions | Used for |
|---|---|---|
| `.venv` | TensorFlow 2.15.1 / Keras 2.15 | Tests, development (`pytest`, pipeline, training code) |
| `.venv-infer` | TensorFlow 2.20.0 / Keras 3.13.2 | Loading + running the baseline model (matches the Colab training run) |

The baseline artifact is a **Keras 3** file; the main `.venv` (Keras 2) cannot
deserialize it — that failure is captured in
[`docs/load_failure_baseline_20261003_172906.txt`](docs/load_failure_baseline_20261003_172906.txt).
`.venv-infer` exists purely so the model can be loaded without touching the
original environment. Both are Git-ignored.

### Main environment (`.venv`) — tests and development

A virtual environment was already created in this repo:

```powershell
py -3.11 -m venv .venv
```

Install dependencies with the venv interpreter (do not use the system Python):

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Or reproduce the exact working set:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

Verify imports:

```powershell
.venv\Scripts\python.exe -c "import tensorflow, streamlit, numpy, pandas, matplotlib, sklearn, PIL, pytest; print(tensorflow.__version__)"
```

### Inference environment (`.venv-infer`) — loading the baseline model

```powershell
# create (Git-ignored; .gitignore lists .venv-infer/)
py -3.11 -m venv .venv-infer

# install the exact Colab-run versions; pip resolves the rest
.venv-infer\Scripts\python.exe -m pip install -r requirements-infer.txt

# exact Windows versions captured here:
.venv-infer\Scripts\python.exe -m pip install -r requirements-infer.lock.txt

# dependency + import check
.venv-infer\Scripts\python.exe -m pip check
.venv-infer\Scripts\python.exe -c "import tensorflow as tf, keras, numpy; print(tf.__version__, keras.__version__, numpy.__version__)"
# -> 2.20.0 3.13.2 2.4.6   (verified 2026-10-04; pip check: no broken requirements)

# post-training compatibility verification of the baseline artifact
.venv-infer\Scripts\python.exe src\verify_baseline_inference.py
```

Use `.venv-infer\Scripts\python.exe` for anything that loads
`models/*/best_model.keras`. Use `.venv\Scripts\python.exe` for `pytest` and
the data/training scripts — they do not load the artifact.

### Optional: cross-machine synthetic comparison (Colab vs Windows)

The verification script can export its fixed synthetic batch outputs and
compare them against a file produced on another machine. **Only claim
numerical parity when both files exist and the comparison passes:**

```powershell
# Windows (or Colab): write this machine's outputs for the fixed batch
.venv-infer\Scripts\python.exe src\verify_baseline_inference.py --save-predictions synthetic_windows.json

# Colab (TF 2.20 runtime, repo cloned): same command with a different flag
#   !python src/verify_baseline_inference.py --save-predictions synthetic_colab.json

# Windows: compare against the Colab file (max abs diff must be <= tolerance)
.venv-infer\Scripts\python.exe src\verify_baseline_inference.py --compare-predictions synthetic_colab.json
```

No Colab output file has been produced yet — **parity with Colab is not
claimed.**

## Running tests

```powershell
# All tests (synthetic images only — no RealWaste download needed)
.venv\Scripts\python.exe -m pytest

# Only the milestone-5 preprocessing/model tests
.venv\Scripts\python.exe -m pytest tests\test_data_pipeline.py tests\test_model.py -v

# Milestone-6 training workflow + Colab notebook structure
.venv\Scripts\python.exe -m pytest tests\test_train.py tests\test_notebook.py -v

# Real-data smoke check with printed batch/model shapes (needs data\raw\ + manifests)
.venv\Scripts\python.exe -m pytest tests\test_smoke_real_data.py -v -s
```

## Preprocessing and model (milestone 5)

Key files:

| File | Purpose |
|---|---|
| [`configs/training.json`](configs/training.json) | seed 42, 224×224, batch 16, lr 0.001, 15 max epochs, dropout 0.2 |
| [`src/data_pipeline.py`](src/data_pipeline.py) | manifest validation + `tf.data` batches (RGB, 224×224, float32 0–255) |
| [`src/model.py`](src/model.py) | frozen MobileNetV2 + GAP + dropout + 4-class softmax head |

Five things worth understanding:

- **Resizing vs. normalization.** *Resizing* changes the picture's
  dimensions (any photo → 224×224) so every batch has one shape.
  *Normalization* changes the numbers' scale. The pipeline only resizes and
  keeps pixels as float32 on 0–255; MobileNetV2's preprocessing then maps
  0–255 to [-1, 1] **once, inside the model**, so a saved model always
  preprocesses correctly on its own.
- **Why augmentation runs only during training.** Flips/rotations teach the
  model invariance, but if validation images were also flipped, evaluation
  would measure the wrong thing — and repeated inference would become
  random. Keras augmentation layers therefore act only when the model is
  called with `training=True`; validation and inference stay deterministic.
- **Why the pretrained base starts frozen.** ImageNet already gave the
  backbone useful edge/texture filters. With ~3k images and no GPU budget,
  training the 2M+ backbone weights from day one would overwrite them with
  noise. We train only the small new head first (fine-tuning comes later).
- **Why class order must stay consistent.** `class_index` in the manifests
  is just a position in `class_order`. If training, evaluation, and the
  Streamlit app ever disagreed about the order, every prediction would be
  silently renamed (e.g. "metal" reported as "paper"). The order lives in
  one file — `configs/class_mapping.json` — and `training.json` only
  references it by path so copies can never drift apart.
- **Why a forward pass ≠ a trained model.** The smoke check only proves the
  graph runs: shapes line up, probabilities sum to 1, weights load. The
  classification head still has random weights, so its predictions are
  meaningless until `model.fit` runs in the next milestone.

## Baseline training on Google Colab (milestone 6)

| File | Purpose |
|---|---|
| [`src/train.py`](src/train.py) | Frozen-base baseline training CLI (reuses `data_pipeline` + `model`; never opens `test.csv`) |
| [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) | Step-by-step Colab run: environment → clone → data → validate manifests → train → package |
| [`tests/test_train.py`](tests/test_train.py) | Synthetic workflow tests (tiny images, `weights=None` — **not** accuracy tests) |
| [`tests/test_notebook.py`](tests/test_notebook.py) | Notebook JSON + cell syntax + required-topic checks |

### How to run it (Google Colab)

1. Upload `notebooks/train_colab.ipynb` to Colab (GitHub tab also works).
2. `Runtime → Change runtime type → T4 GPU`.
3. Run the cells in order. They inspect the runtime first and install
   TensorFlow **only if needed** (never `requirements.lock.txt` — that file has
   Windows-only pins); if anything was installed, the guarded cell restarts
   the runtime for you, after which you re-run Step 1–2 (they then say
   "nothing to install") and continue.
4. The notebook mounts Drive, clones this repo (pin `REVISION` to a commit SHA
   for a reproducible run), downloads/inspects RealWaste with the existing
   scripts, and **validates** the committed manifests against
   `split_summary.json` (it never regenerates them). Download/inspection
   reports are written to a runtime directory on Drive via `--metadata-dir`
   so regenerated timestamps never dirty the checkout.
5. Training runs `src/train.py` with the `configs/training.json` settings
   (seed 42, batch 16, lr 0.001, max 15 epochs, early stopping on
   `val_loss`) and writes everything to
   **`Drive/MyDrive/ai-waste-classification-assistant-outputs/runs/baseline_<timestamp>/`**
   (`best_model.keras`, `history.csv`, `run_metadata.json`, `plots/`, logs).
6. The notebook exports the small reports into the clone, then zips both
   bundles into the same Drive folder: `reports_<run>.zip` (Git-trackable)
   and `model_<run>.zip` (stays out of Git).

### Bringing results back to this machine

Download both zips from Drive, then in PowerShell:

```powershell
# restore Git-trackable reports
Expand-Archive .\reports_baseline_<stamp>.zip -DestinationPath .\models\metadata\runs\baseline_<stamp>

# keep the weights next to the other ignored binaries (stays out of Git)
Expand-Archive .\model_baseline_<stamp>.zip -DestinationPath .\models
```

### First baseline run (actual, Google Colab)

| Field | Value |
|---|---|
| Run ID | `baseline_20261003_172906` |
| Date | 2026-10-03 |
| Code commit | `199f2539c821ee1b70d673cec8050ed86cf62e2e` |
| Runtime | Google Colab — Python 3.13.15, TensorFlow 2.20.0, Keras 3.13.2, Tesla T4 |
| Epochs | 15 completed (max 15; early stopping on `val_loss`, patience 3) |
| Best epoch | 14 (lowest `val_loss`) |
| **Validation** loss | **0.2097** (full value `0.20965705811977386`) |
| **Validation** accuracy | **0.9323** (full value `0.932314395904541`) |
| Test set | **not used** (`data.test_manifest_used: false`) |
| Class order | `["metal", "organic", "paper", "plastic"]` |
| Model SHA-256 | `c25f275cba8b5520…` (verified against the extracted file) |
| Checkout at training time | dirty (`uncommitted_changes: true` — cause investigated, unconfirmed) |

Every row was verified against `models/metadata/runs/baseline_20261003_172906/`
(`run_metadata.json` + `history.csv`): **37/37 checks passed** — details in
[`docs/progress.md`](docs/progress.md). These are **validation** metrics from
this one run; no test-set evaluation has been performed.

### Local compatibility status (verified 2026-10-04, `.venv-infer`: TF 2.20.0 / Keras 3.13.2)

**The baseline model now loads and executes on this machine.** The first
attempt in the main `.venv` (Keras 2.15) failed with:

```text
TypeError: Could not deserialize class 'Functional' because its parent
module keras.src.models.functional cannot be imported.
```

(full log: [`docs/load_failure_baseline_20261003_172906.txt`](docs/load_failure_baseline_20261003_172906.txt)).
Cause: the artifact was saved by **Keras 3.13.2**; the main environment is
**Keras 2.15**, whose module layout differs — a serialization-format mismatch,
not a corrupt file (the file's SHA-256 matches the run metadata).

Resolution: a **separate** environment, not a conversion — `.venv-infer`
(Python 3.11, TensorFlow 2.20.0, Keras 3.13.2, matching the Colab run).
The original `.venv`, `requirements.txt`, and `requirements.lock.txt` were
left untouched.

**Verification** (`.venv-infer\Scripts\python.exe src\verify_baseline_inference.py`)
— **19/19 checks passed**, report:
[`models/metadata/verification/post_training_compatibility_baseline_20261003_172906.json`](models/metadata/verification/post_training_compatibility_baseline_20261003_172906.json)

| Check | Result |
|---|---|
| Model SHA-256 vs original `run_metadata.json` | match (`c25f275cba8b5520…`) |
| Load with `compile=False, safe_mode=True` | success, no warnings, unsafe deserialization **not** used |
| Input / output shape | `(None, 224, 224, 3)` → `(None, 4)` |
| Class order (3 sources: config / run metadata / bundle) | identical: metal, organic, paper, plastic |
| Synthetic batch `(2, 224, 224, 3)` float32 0–255 | finite, 4 scores/image, softmax sums 1.0 (dev ≤ 6e-8) |
| Repeated inference, `training=False` | stable: max abs diff **0.0** over 3 runs |
| MobileNetV2 preprocessing probe (0→−1, 127.5→0, 255→+1) | embedded in graph, applied exactly once; no normalization outside the frozen backbone |
| Real-image sample (6 training images, shared pipeline) | executes; index→label mapping consistent (test set never opened) |

**These numbers are execution checks, not accuracy.** Synthetic predictions
prove only that the graph runs; the real-image sample is recorded to show
label-order handling, and no accuracy metric is computed from it.
**Numerical parity with Colab is not claimed** — no Colab output file exists
(use the optional `--save-predictions` / `--compare-predictions` flow above to
produce one).

### What the workflow guarantees (and does not)

- **No test-set access:** training opens only `train.csv` + `validation.csv`;
  `run_metadata.json` records `data.test_manifest_used: false`.
- **Honest artifacts:** metrics, environment (`pip freeze`), git commit, seed
  settings, and manifest checksums are written by the run itself — nothing is
  typed by hand, and an interrupted run is marked as such.
- **Reports ≠ model:** the exported `models/metadata/runs/<run_id>/` folder
  never contains weights; `.gitignore` blocks `*.keras` and `*.zip`.
- **Validation only, for now:** the recorded numbers are validation metrics
  from the actual Colab run. The test split has not been evaluated and the
  model has not been fine-tuned. Local loading now works — in `.venv-infer`
  only (see *Local compatibility status*) — and the verification report
  deliberately contains execution checks, not accuracy claims.

## Git workflow

- Branch: `main`
- Remote: `origin` → `https://github.com/snadeem03/ai-waste-classification-assistant.git`
- Commit style: `chore: ...`, `feat: ...`, `fix: ...`, `docs: ...`
- Never commit virtual environments, datasets, trained model binaries, or secrets.
- See `AGENTS.md` and `docs/progress.md` for milestone rules and status.

## References

- TensorFlow Keras MobileNetV2: https://keras.io/api/applications/mobilenet/
- Streamlit: https://docs.streamlit.io/
- RealWaste (UCI): https://archive.ics.uci.edu/dataset/908/realwaste
