# AI Waste Classification Assistant

College AIML Project-Based Learning (PBL) — **status: scope and dataset mapping defined; dataset not downloaded; model not trained yet.**

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

**This repository has not downloaded a dataset or trained a model yet.** Milestone 2 only defined project scope, RealWaste category mapping, and documentation. Training results, accuracy figures, and a working demo app will be added in later milestones — they are not claimed here.

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

## Planned project layout (not all files exist yet)

```
ai-waste-classification-assistant/
├── .gitignore                 # ignores venv, data images, model binaries, secrets
├── README.md                  # this file
├── AGENTS.md                  # rules for future work on this repo
├── requirements.txt           # direct dependencies
├── requirements.lock.txt      # exact working local versions
├── configs/
│   └── class_mapping.json     # source→target map + class order
├── docs/
│   ├── project_scope.md       # scope and acceptance criteria
│   ├── dataset.md             # RealWaste mapping and license notes
│   └── progress.md            # milestone log
├── data/
│   ├── README.md              # tracked
│   ├── metadata/              # tracked small manifests (future)
│   └── raw|processed/         # ignored images (not downloaded yet)
├── models/
│   ├── README.md              # tracked
│   ├── metadata/              # tracked small eval notes (future)
│   └── *.keras etc.           # ignored weights (do not exist yet)
├── src/                       # training + inference code (future)
├── app/                       # Streamlit UI (future)
├── notebooks/                 # optional exploration (future)
└── tests/                     # pytest checks (future)
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

## Running tests (when they exist)

```powershell
.venv\Scripts\python.exe -m pytest
```

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
