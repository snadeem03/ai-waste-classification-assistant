# Model card — selected waste classifier (`finetune_20261004_133620`)

Honest description of the one model this project currently serves. Every
number below is copied from a machine-written report on disk; nothing is
estimated or rounded up. When a fact was not measured, it says so.

| Field | Value |
|---|---|
| Selected model | `models/finetune_20261004_133620/best_model.keras` (**Git-ignored**, 23,001,516 bytes) |
| SHA-256 | `39f7b78befd182c6a65cc94db3a2f3a09d15018ab5ae34145ed4a48f28e1cb4a` |
| Run ID | `finetune_20261004_133620` (controlled fine-tune of the baseline, Google Colab, 2026-10-04) |
| Trained at code commit | `7a7fdd9` (milestone 9 code; checkout recorded clean) |
| Runtime of training | Google Colab — Python 3.13, TensorFlow 2.20.0, Keras 3.13.2, Tesla T4 |
| Runtime of test evaluation | Local Windows, Python 3.11.9, TensorFlow 2.20.0, Keras 3.13.2 (`.venv-infer`), CPU |
| Selection record | [`models/metadata/selection/selected_model.json`](../models/metadata/selection/selected_model.json) — written **before** the test split was opened |
| Test report | [`models/metadata/evaluation/test_evaluation.json`](../models/metadata/evaluation/test_evaluation.json) — one pass, 459/459 images, 2026-10-04T14:48:16Z |

## Intended use

Upload a photo of waste and get one of four labels with a confidence score:

`metal` · `organic` · `paper` · `plastic` (fixed class order, from
[`configs/class_mapping.json`](../configs/class_mapping.json)).

**Out of scope:** rejecting unknown objects, images outside these four
classes, anything the dataset never showed (rocks, electronics, liquids),
and any safety-critical sorting decision. The model always emits one of the
four labels — it has no "unknown" option.

## Architecture

- **MobileNetV2** (ImageNet weights) as the convolutional backbone,
  global-average-pooling → dropout (0.2) → 4-class softmax head
  ([`src/model.py`](../src/model.py)).
- Input `224×224` RGB; MobileNetV2 preprocessing runs **once inside the
  model** (0–255 float32 → [-1, 1]), verified by the compatibility report.
- Fine-tuning policy (milestone 9): backbone unfrozen from `block_13`
  onward with **every BatchNorm layer kept frozen** (running statistics
  never updated), fresh Adam at `lr=0.00001`, max 10 epochs, early stopping
  on `val_loss` (patience 3), seed 42
  ([`src/finetune.py`](../src/finetune.py)).

## Training data

RealWaste (UCI, 4,752 source images) mapped to the four classes; duplicate
checks found no exact duplicates; stratified 70/15/15 split, seed 42
([`docs/dataset.md`](dataset.md)):

| Split | Images | Used for |
|---|---:|---|
| train | 2,141 | head training + fine-tuning |
| validation | 458 | early stopping, baseline-vs-fine-tuned comparison, model **selection** |
| test | 459 | **held out** until the evaluation described below |

Manifest checksums are committed in `data/metadata/split_summary.json`; the
test manifest SHA-256 (`74494c13f6d9be7f…`) is re-checked before evaluation.

## How this model was selected (before the test set was opened)

1. Baseline run `baseline_20261003_172906` (frozen backbone) vs fine-tuned
   run `finetune_20261004_133620`, both scored on the **identical**
   deterministic validation pass ([`src/compare_validation.py`](../src/compare_validation.py)).
2. Rule fixed in code before any run: fine-tuned wins iff validation
   **macro F1 is strictly higher** (ties broken by accuracy), otherwise the
   baseline stays.

| Validation metric | Baseline | Fine-tuned |
|---|---:|---:|
| macro F1 | 0.9322544605989562 | **0.9372495337333778** |
| accuracy | 0.9323144104803494 | **0.9366812227074236** |

→ **Fine-tuned selected** (`rule step 1: macro F1 improved 0.932254 ->
0.937250`), recorded in the selection record
(`created_at_utc 2026-10-04T13:38Z`, model SHA locked). The record existed
on disk **before** `src/evaluate.py` was written and before any test image
was read, so test results could not influence the choice.

## Held-out test results (measured once, 2026-10-04)

Command: `.venv-infer\Scripts\python.exe src\evaluate.py` → report
[`models/metadata/evaluation/test_evaluation.json`](../models/metadata/evaluation/test_evaluation.json).

- **459 images, 430 correct, 29 misclassified**
- **Accuracy: 0.9368** (`0.9368191721132898`)
- **Macro F1: 0.9347** (`0.934730625209095`)
- **Loss: 0.1724** (`0.17239077061109537`)

| Class | Precision | Recall | F1 | Support (test images) |
|---|---:|---:|---:|---:|
| metal | 0.8837 | 0.9580 | 0.9194 | 119 |
| organic | 1.0000 | 1.0000 | 1.0000 | 127 |
| paper | 0.9079 | 0.9200 | 0.9139 | 75 |
| plastic | 0.9449 | 0.8696 | 0.9057 | 138 |

Confusion matrix (rows = true class, columns = predicted; order
`metal, organic, paper, plastic`):

| true \ predicted | metal | organic | paper | plastic |
|---|---:|---:|---:|---:|
| metal | 114 | 0 | 1 | 4 |
| organic | 0 | 127 | 0 | 0 |
| paper | 3 | 0 | 69 | 3 |
| plastic | 12 | 0 | 6 | 120 |

Reading: the model never confuses organic with anything else on this split;
its main weakness is **plastic misread as metal** (12 images — metal items
sort inside the metal row at 114/119 recall, but precision drops because 12
plastic images land there).

### What the test protocol guarantees

- One pass, manifest order, `training=False`, no shuffle, no augmentation,
  final partial batch kept → all 459 images scored exactly once
  (`coverage.each_image_evaluated_exactly_once: true`).
- **No fitting, no tuning, no threshold or checkpoint changes** after seeing
  test results — the checkpoint came from the pre-existing selection record
  (model SHA re-verified against it before loading).
- Metrics come from the same pure-NumPy code as the validation comparison
  (`compare_validation.metrics_from_predictions`), so validation and test
  numbers are computed the same way.
- Full prediction list (path, truth, prediction, four softmax scores) is in
  the report; a misclassification grid is written to the Git-ignored
  `models/runs/misclassified_grids/sample_grid_test_misclassified.png`
  (dataset photographs are never committed).

## Limitations

1. **Four closed-set classes only.** Anything outside
   metal/organic/paper/plastic is forced into one of them; no
   unknown/rejection threshold exists or is calibrated.
2. **Same source, same conditions.** Train/validation/test come from one
   dataset (RealWaste) with the same camera characteristics. The test split
   measures held-out images from that dataset, **not** performance on new
   sources, phone photos, or other lighting/backgrounds — real-world
   accuracy is unknown and not claimed.
3. **Class imbalance in errors.** plastic→metal (12) and metal→plastic (4)
   dominate the 29 errors; per-class recall ranges 0.87–1.00.
4. **Small validation gap to baseline.** The selection gain (macro F1
   0.9323 → 0.9373 on 458 validation images) is small; test macro F1
   (0.9347) does not prove a large real-world improvement over the baseline
   (which was never evaluated on the test set — by design, only the
   selected model was).
5. **Single run, single seed.** One training run at seed 42; no repeated-run
   variance, no confidence intervals.
6. **Not calibrated.** Softmax scores are raw outputs; treat them as
   ranking, not as probabilities of correctness.

## Reproduce / verify

```powershell
# model hash + load/compatibility checks (write to a separate report so the
# baseline report is never overwritten)
.venv-infer\Scripts\python.exe src\verify_baseline_inference.py `
    --model models\finetune_20261004_133620\best_model.keras `
    --run-metadata models\finetune_20261004_133620\run_metadata.json `
    --model-bundle-dir models\finetune_20261004_133620 `
    --report models\metadata\verification\post_training_compatibility_finetune_20261004_133620.json

# the held-out evaluation (writes the report; do not "re-run for better numbers")
.venv-infer\Scripts\python.exe src\evaluate.py
```

Supporting evidence on disk:

| Evidence | Path |
|---|---|
| Import + artifact verification (24/24 checks) | `models/metadata/verification/import_finetune_20261004_133620.json` |
| Local load/compatibility verification (19/19 checks) | `models/metadata/verification/post_training_compatibility_finetune_20261004_133620.json` |
| Validation comparison + selection rule | `models/metadata/comparison/validation_comparison.json` |
| Selection record (frozen before test) | `models/metadata/selection/selected_model.json` |
| Held-out test evaluation | `models/metadata/evaluation/test_evaluation.json` |
