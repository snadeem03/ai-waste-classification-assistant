# AI Waste Classification Assistant

College AIML Project-Based Learning (PBL) — **status: setup complete, model not trained yet.**

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

**This repository has not trained a model yet.** The current milestone only set up the development environment, dependency pins, documentation, and workflow rules. Training results, accuracy figures, and a working demo app will be added in later milestones — they are not claimed here.

## Planned project layout (not all files exist yet)

```
ai-waste-classification-assistant/
├── .gitignore                 # ignores venv, data, models, secrets
├── README.md                  # this file
├── AGENTS.md                  # rules for future work on this repo
├── requirements.txt           # direct dependencies
├── requirements.lock.txt      # exact working local versions
├── docs/
│   └── progress.md            # milestone log
├── src/                       # training + inference code (future)
├── app/                       # Streamlit UI (future)
├── notebooks/                 # optional exploration (future)
├── data/                      # datasets (ignored by Git)
├── models/                    # trained weights (ignored by Git)
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
