# Demo results — observations from running the app by hand

This file records what **actually happened** during manual demos of the
Streamlit app (`README.md` → "Streamlit app (milestone 11)"). Photographs and
screenshots stay **local only** under `demo/` (ignored by Git); only these
written observations are committed.

---

## Demo 1 — coffee sachet predicted as metal (2026-10-05)

### Observation

| Field | Value |
|---|---|
| When | 2026-10-05, screenshot timestamp 20:21:54 (local) |
| Item | Nescafé Classic soluble coffee sachet (red foil-laminated ₹2 sachet) |
| Photo (local, not in Git) | `demo\photos\WhatsApp Image 2026-10-05 at 20.20.31.jpeg` — sha256 `84a5c32c77523e8a…c99ad3`, 4032×3024 JPEG, 813,765 bytes |
| Screenshot (local, not in Git) | `demo\screenshots\Screenshot 2026-10-05 202154.png` — sha256 `f12f73d6b0468b89…f84fef`, 10,122 bytes |
| Displayed in the app | **Predicted category: metal**, **Model confidence 97.7%**, bar chart ≈ metal 0.98 / organic 0.02 / paper ≈ 0 / plastic ≈ 0 |
| **Expected label** | **plastic — user-provided, pending material confirmation.** The sachet's exact material composition has **not** been independently verified, so "plastic" is recorded as the user's expectation, not as confirmed ground truth. |

### CLI comparison on the exact image

```powershell
.venv-infer\Scripts\python.exe src\predict.py "demo\photos\WhatsApp Image 2026-10-05 at 20.20.31.jpeg"
```

Exact CLI output (2026-10-05, exit 0):

| Field | CLI value |
|---|---|
| Model | `models\finetune_20261004_133620\best_model.keras` (sha256 `39f7b78befd182c6…`) |
| Image | 4032×3024 RGB → resized to 224×224, float32 0–255 |
| Predicted | **metal** (top score **0.9769**) |
| metal | 0.9769 |
| organic | 0.0209 |
| paper | 0.0022 |
| plastic | 0.0001 |

**Comparison with the screenshot:**

- The CLI top score `0.9769` formats to exactly the displayed **97.7%**
  (`f"{0.9769:.1%}"`), same category (**metal**).
- The screenshot only shows one decimal place, so agreement is verified at
  the precision it displays; its bar heights match the CLI scores
  (metal ≈ 0.98, organic ≈ 0.02, paper/plastic ≈ 0).
- This agreement is expected by construction: the app's button calls the
  **same** `predict.predict_image` function the CLI uses (`app/app.py`), and
  both run in `.venv-infer` through the same decode → resize → model path.

### Model identity, class order, and shared preprocessing (re-verified today)

| Check | Result |
|---|---|
| Selected-model identity | File SHA-256 `39f7b78befd182c6a65cc94db3a2f3a09d15018ab5ae34145ed4a48f28e1cb4a` = selection record value; size 23,001,516 bytes = record. The CLI re-verifies this checksum on **every** load and exited 0. |
| Class order | Identical across all three sources: selection record, `configs/class_mapping.json`, and the model bundle's `class_order.json` → `["metal", "organic", "paper", "plastic"]` |
| Shared preprocessing | `src/predict.py` and `src/data_pipeline.py` both call `resize_to_model_input` (bilinear `tf.image.resize`, float32, 0–255, no second normalization). Focused test run: **8 passed** (`tests/test_predict.py -k "resize_matches or real_selected or class_order or double or scale"`), including the app-vs-evaluation resize consistency test (max abs diff ≤ 1e-5). |

### Conclusion: model classification failure, not a UI bug

The CLI and the UI **agree** (metal ≈ 0.977 on the same file, same model,
same code path). The app faithfully displayed what the model computed, so
this is recorded as a **model classification failure on this image** — not a
display, mapping, or preprocessing bug.

Supporting context from the **recorded** held-out test evaluation
(`models/metadata/evaluation/test_evaluation.json`, milestone 10 — read-only,
unchanged):

- The confusion matrix's `plastic` row is `[12, 0, 6, 120]`: **12 plastic
  test images were predicted metal** — the largest single error type among
  the 29 total errors.
- `metal` has predicted_count 129 vs support 119, i.e. the model already
  over-predicts `metal` on this dataset.

A shiny red/silver laminate sachet being read as metal is consistent with
that recorded pattern. (Consistency is context, not proof of cause.)

### Same physical item, second view (grouped variant)

The second photo of the **same sachet**
(`demo\photos\WhatsApp Image 2026-10-05 at 20.20.32.jpeg`, sha256
`22599fad03b47962…5daa110`, 3024×4032, angle view) gives, via the same CLI:

| Predicted | metal 0.0276 · organic 0.2493 · paper 0.0143 · **plastic 0.7089** |

So two views of **one physical item** disagree with each other
(metal 97.7% vs plastic 70.9%). No UI screenshot exists for this second
view — it is a CLI-only observation. This is recorded as **one item with
two observations**, not as two separate demo cases.

### Comparison checklist (follow-up captures — one physical item)

All rows below refer to the **same coffee sachet**; keep them grouped as one
item when recording results:

- [ ] **A. Original image** — already captured (front view on lined paper,
      `…20.20.31.jpeg`). Recorded above: UI metal 97.7% = CLI metal 0.9769.
- [ ] **B. Tighter crop** — crop/photo of just the sachet filling the frame
      (same lighting), so background paper lines are minimal. Run through
      the UI **and** the CLI; record photo SHA-256, full CLI scores, and a
      screenshot of the displayed result.
- [ ] **C. New photograph on a plain background** — the same sachet on a
      plain white or solid-color surface, evenly lit, no notebook lines or
      strong shadows. Same recording procedure as B.

For every capture: store the file under `demo\photos\` (local, ignored),
note its SHA-256, run `src\predict.py`, and save the UI screenshot under
`demo\screenshots\`. **Never edit scores or images to make them agree.**

### Deliberately not done

- No prediction correction was hardcoded; no confidence score was altered.
- No retraining, no threshold tuning, no changes to the selected model or
  the selection record.
- Recorded test results (`test_evaluation.json`) were read for context only
  and remain byte-for-byte unchanged.
- The test split was not opened.

### Remaining uncertainty

- The sachet's real material composition is **unconfirmed**; "plastic" is
  the user-provided expectation only (metallized laminate films are common
  for coffee sachets, but we did not verify this one).
- Screenshot precision is one decimal place; CLI/UI equality is asserted at
  that precision and by shared code path, not to full float precision.
- One item, two photos — no conclusion about other real-world objects; this
  is a single demo observation, consistent with (but not proof of) the
  recorded plastic→metal confusion pattern.
