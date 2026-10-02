# AI Waste Classification Assistant

College AIML Project-Based Learning (PBL) — **status: dataset downloaded, splits created, preprocessing + MobileNetV2 model built, Colab baseline training workflow prepared; model not trained yet.**

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

**This repository has not trained a model yet.** Milestones 1–4 defined scope, downloaded RealWaste, and wrote duplicate-checked 70/15/15 split manifests. Milestone 5 added the image input pipeline and the MobileNetV2 model *construction* (verified with a forward pass only). Milestone 6 added the frozen-base training workflow (`src/train.py`) and a Google Colab notebook (`notebooks/train_colab.ipynb`) — the workflow is verified with synthetic images and syntax/structure checks, but the notebook has **not** been executed in Colab. Accuracy figures, a trained model, and a working demo app will be added in later milestones — they are not claimed here.

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
├── .gitignore                 # ignores venv, data images, model binaries, secrets
├── README.md                  # this file
├── AGENTS.md                  # rules for future work on this repo
├── requirements.txt           # direct dependencies
├── requirements.lock.txt      # exact working local versions
├── configs/
│   ├── class_mapping.json     # source→target map + class order (single source of truth)
│   └── training.json          # seed, image/batch size, lr, epochs, dropout
├── docs/
│   ├── project_scope.md       # scope and acceptance criteria
│   ├── dataset.md             # RealWaste mapping and license notes
│   └── progress.md            # milestone log
├── data/
│   ├── README.md              # tracked
│   ├── metadata/              # tracked manifests + reports (train/validation/test CSVs)
│   └── raw|processed/         # ignored images
├── models/
│   ├── README.md              # tracked
│   ├── metadata/              # tracked small eval notes (future)
│   └── *.keras etc.           # ignored weights (not created yet)
├── src/
│   ├── download_data.py       # UCI download + safe extraction
│   ├── inspect_data.py        # counts, mapping check, image validation
│   ├── prepare_data.py        # duplicate checks + stratified split manifests
│   ├── data_pipeline.py       # manifest loading + tf.data input pipeline
│   ├── model.py               # MobileNetV2 model construction
│   └── train.py               # frozen-base baseline training CLI (Colab)
├── app/                       # Streamlit UI (future)
├── notebooks/
│   └── train_colab.ipynb      # Colab baseline training (not yet executed in Colab)
└── tests/                     # pytest checks (synthetic images; no download)
```

## Environment setup (Windows, Python 3.11)

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
   `split_summary.json` (it never regenerates them).
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

**Pending verification — run before claiming the model loads locally** (not
yet run: no `.keras` file exists in this repo until a real Colab run):

```powershell
.venv\Scripts\python.exe -c "import tensorflow as tf; m = tf.keras.models.load_model(r'models\best_model.keras'); print(m.output_shape)"
```

### What the workflow guarantees (and does not)

- **No test-set access:** training opens only `train.csv` + `validation.csv`;
  `run_metadata.json` records `data.test_manifest_used: false`.
- **Honest artifacts:** metrics, environment (`pip freeze`), git commit, seed
  settings, and manifest checksums are written by the run itself — nothing is
  typed by hand, and an interrupted run is marked as such.
- **Reports ≠ model:** the exported `models/metadata/runs/<run_id>/` folder
  never contains weights; `.gitignore` blocks `*.keras` and `*.zip`.
- **Not yet executed in Colab:** the notebook is validated locally (JSON,
  cell syntax, topic tests) but no training has been run in this milestone,
  so **no accuracy numbers exist yet**.

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
