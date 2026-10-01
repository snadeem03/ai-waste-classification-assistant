# Project scope

**Project:** AI Waste Classification Assistant  
**Milestone:** 2 — project scope and dataset mapping  
**Status:** Scope defined; dataset not downloaded; model not trained.

---

## 1. Problem statement

Waste sorting is slow and error-prone when done only by hand. Recycling facilities and campus waste programs need a simple way to tell common material types apart from a photo. This project builds an educational image classifier that reads a photo of one waste item and predicts whether it is **plastic**, **paper**, **metal**, or **organic**.

This is a college AIML Project-Based Learning (PBL) build. It demonstrates transfer learning and a small Streamlit app. It is **not** a production recycling plant system.

---

## 2. Objectives

1. Map the official RealWaste dataset categories onto four beginner-friendly target classes.
2. Train a **MobileNetV2** classifier with transfer learning (ImageNet weights, frozen base, new classification head).
3. Save the trained model locally (never commit the binary to Git).
4. Serve predictions through **Streamlit**: upload an image → show predicted category + confidence score.
5. Keep the workflow honest: real evaluation numbers only, after real training runs.

---

## 3. Expected competencies (what you practice)

- Creating and using a Python virtual environment
- Reading dataset cards and writing class mappings
- Transfer learning with TensorFlow/Keras and MobileNetV2
- Stratified train / validation / test splitting
- Building a small web UI with Streamlit
- Git hygiene: ignore rules, meaningful commits, no secrets or big binaries

---

## 4. Prerequisites

| Item | Requirement |
|---|---|
| OS | Windows (commands in this repo use `.venv\Scripts\...`) |
| Python | 3.11 (already set up in `.venv/`) |
| Packages | Listed in `requirements.txt` / `requirements.lock.txt` |
| Git | Initialized repo on branch `main` with remote `origin` |
| Dataset | RealWaste from UCI (not downloaded yet — future milestone) |
| Hardware | CPU is enough to train a small head on MobileNetV2; GPU optional |

---

## 5. Input

- A single color image file (JPG/PNG).
- The image should contain **one prominent waste item** centered or clearly visible.
- Typical sources: phone camera, landfill / bin photos, dataset samples.

**Not in scope for input:** multi-object cluttered scenes where several materials overlap heavily (harder detection problem).

---

## 6. Output

- **Predicted category:** one of `metal`, `organic`, `paper`, `plastic` (fixed order in `configs/class_mapping.json`).
- **Confidence score:** model softmax probability for the predicted class (0.0–1.0), shown in the UI.

No accuracy percentage is promised in this document. Any number in future reports must come from an actual evaluation run on a held-out test split.

---

## 7. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | College standard; matches installed venv |
| Deep learning | TensorFlow / Keras 2.15 | Stable transfer-learning path |
| Backbone | MobileNetV2 | Lightweight, ImageNet-pretrained, good for small datasets |
| Classical ML utils | scikit-learn | Stratified splits, metrics, confusion matrices |
| UI | Streamlit | Fast image-upload demo without web-framework boilerplate |
| Config | JSON (`configs/class_mapping.json`) | One source of truth for class order and mapping |

---

## 8. In scope (this PBL)

- Four-class image classification: plastic, paper, metal, organic
- RealWaste as the primary dataset (with documented category mapping)
- MobileNetV2 transfer learning + frozen base + custom head
- Stratified train/val/test split before any augmentation
- Local model saving and loading for Streamlit inference
- Documentation of mapping, scope, and progress
- Small tracked metadata: class mapping JSON, split manifests (paths/labels only)

---

## 9. Out of scope / exclusions

- **No dataset download or training in milestone 2.** Those are later milestones.
- Other RealWaste categories are **excluded** from initial training:
  - Cardboard
  - Glass
  - Miscellaneous Trash
  - Textile Trash
- Not building object detection / bounding boxes (TACO deferred — see `docs/dataset.md`)
- Not promising a specific accuracy value
- Not deploying to cloud production infrastructure
- Not committing raw images, zip archives, or trained model weights to Git

---

## 10. Functional acceptance criteria

The project is accepted for this scope when all of the following are true:

1. `configs/class_mapping.json` is valid JSON and stores:
   - source→target mapping for Plastic, Paper, Metal, Food Organics, Vegetation
   - explicit `class_order`: `["metal", "organic", "paper", "plastic"]`
   - excluded source categories listed so they cannot silently become supported classes
2. `docs/dataset.md` documents RealWaste source, attribution, license, original categories, mapping, and exclusions.
3. A user can run the Streamlit app locally, upload one image, and see a predicted class + confidence score (after the training milestone exists).
4. Training code (when written) uses only the four target classes and the fixed class order.
5. Git never contains venv files, raw/processed image datasets, archives, secrets, or trained model binaries.
6. Progress log (`docs/progress.md`) stays factual — no invented metrics.

---

## 11. Limitations

- **Small merged organic class:** Food Organics + Vegetation become one class; fine-grained food vs leaf separation is not attempted.
- **Excluded materials:** A photo of glass or cardboard may be forced into one of the four classes incorrectly; the model has no “unknown” class in v1.
- **Single-item assumption:** Cluttered scenes with mixed materials are not reliably handled.
- **RealWaste context:** Images are landfill-reception photos at 524×524; domain shift to clean product photos is possible.
- **License terms:** UCI lists CC BY 4.0; the authors’ GitHub/IEEE notes mention CC BY-NC-SA 4.0. Credit the authors either way; treat use as non-commercial for safety until clarified (see `docs/dataset.md`).
- **No accuracy guarantee:** Performance depends on data quality, split hygiene, and training choices. Numbers will only be reported after real runs.
- **Educational scope:** Not a certified recycling sorter.

---

## 12. Related documents

- Dataset details and mapping: [`docs/dataset.md`](dataset.md)
- Class mapping config: [`configs/class_mapping.json`](../configs/class_mapping.json)
- Milestone log: [`progress.md`](progress.md)
- Repo rules: [`../AGENTS.md`](../AGENTS.md)
