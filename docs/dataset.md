# Dataset plan — RealWaste

**Status:** Documentation only. Dataset is **not downloaded yet**.  
**Primary source:** UCI Machine Learning Repository — RealWaste (id 908)

---

## 1. Source and attribution

| Field | Value |
|---|---|
| Dataset name | RealWaste |
| Official URL | https://archive.ics.uci.edu/dataset/908/realwaste |
| UCI dataset id | 908 |
| DOI | https://doi.org/10.24432/C5SS4G |
| Download size (UCI page) | ~656.6 MB zip |
| Instances | 4,752 color images |
| Image resolution (released) | 524 × 524 |
| Collection context | Whyte's Gully Waste and Resource Recovery facility, Wollongong, NSW, Australia |
| Creators | Sam Single, Saeid Iranmanesh, Raad Raad |
| Introductory paper | [RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning](https://www.mdpi.com/2078-2489/14/12/633) (Information, 2023) |

### Citation (required when the dataset is used)

> Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G

Also credit the paper if results are reported in a report or demo.

---

## 2. License / terms (verified on UCI page)

**UCI Machine Learning Repository page states:**

> This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.  
> This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

**Honest discrepancy note:** The authors’ GitHub repository and IEEE DataPort entry describe the dataset as **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike). The UCI listing we were asked to use says **CC BY 4.0**.

**Practical rule for this college project:**

1. Always give attribution (UCI citation above + paper if used).
2. Prefer **non-commercial educational use** until the license conflict is resolved with the authors.
3. Do not redistribute the raw image archive inside this Git repository (already blocked by `.gitignore`).
4. If a stricter license applies, it does not change our four-class mapping plan — only how the images may be shared.

**Verification method:** Fetched https://archive.ics.uci.edu/dataset/908/realwaste on 2026-10-02 and read the License section on that page.

---

## 3. Original categories (all 9, from UCI card)

| Source category | Image count (UCI) | In initial training? |
|---|---:|---|
| Cardboard | 461 | **No — excluded** |
| Food Organics | 411 | **Yes → organic** |
| Glass | 420 | **No — excluded** |
| Metal | 790 | **Yes → metal** |
| Miscellaneous Trash | 495 | **No — excluded** |
| Paper | 500 | **Yes → paper** |
| Plastic | 921 | **Yes → plastic** |
| Textile Trash | 318 | **No — excluded** |
| Vegetation | 436 | **Yes → organic** |
| **Total** | **4,752** | |

Counts above are **from the UCI dataset card**, not from a local download. They must be re-checked after unzip in the data-prep milestone.

---

## 4. Exact mapping (initial training)

Machine-readable copy: [`configs/class_mapping.json`](../configs/class_mapping.json)

| Source category (RealWaste) | Target class | Rationale |
|---|---|---|
| Plastic | `plastic` | Direct 1:1 material match |
| Paper | `paper` | Direct 1:1 material match |
| Metal | `metal` | Direct 1:1 material match |
| Food Organics | `organic` | Food waste is biodegradable / compostable |
| Vegetation | `organic` | Leaves and plant matter are organic waste |

### Explicit class order (fixed)

```json
["metal", "organic", "paper", "plastic"]
```

This order is used for:

- integer label encoding in training
- confusion-matrix axis order
- Streamlit display order

Training code must **assert** that every target label is in this list and that excluded source names never appear as targets.

---

## 5. Exclusions (not in the initial four-class model)

| Excluded source category | Why excluded now |
|---|---|
| Cardboard | Often labeled near paper; would blur the paper class for a beginner 4-class task. Can be a later extension. |
| Glass | Visually distinct and hazardous handling context; outside the simple plastic/paper/metal/organic story. |
| Miscellaneous Trash | Catch-all label — poor semantics for a clean 4-class head. |
| Textile Trash | Different material family; deferred to keep scope tight. |

**Guarantee against silent leakage:** excluded names live in `configs/class_mapping.json` under `excluded_source_categories`. Any prep script must filter the dataset to mapped source folders only and raise an error if an excluded folder is selected for training.

---

## 6. Why not TrashNet? (no organic class)

[TrashNet](https://github.com/garythung/trashnet) is a classic educational waste dataset, but its labels are:

- glass, paper, cardboard, plastic, metal, trash

There is **no dedicated organic / food-waste class**. Our required output includes `organic`, so TrashNet cannot supply that class without inventing labels or merging unrelated materials. RealWaste explicitly provides **Food Organics** and **Vegetation**, which map cleanly to `organic`. TrashNet may be used later for supplementary experiments, not as the primary source for this four-class goal.

---

## 7. Why TACO is deferred (future object-detection extension)

[TACO (Trash Annotations in Context)](https://github.com/pedropro/TACO) is an **object-detection** dataset: images with bounding boxes and open-set category annotations (bottle caps, wrappers, etc.).

It is deferred because:

1. Our current milestone and app are **image-level classification**, not detection.
2. TACO’s open-world labels need heavy curation to form the same four clean classes.
3. Detection changes the model head, loss, UI (draw boxes), and evaluation metrics — a separate project extension.
4. For this PBL, one prominent item + MobileNetV2 + Streamlit is the intended learning path.

If the project is extended later, TACO is a good candidate for “find all waste items in a messy photo” rather than replacing RealWaste classification.

---

## 8. Planned data hygiene (before training)

These steps run in a **future data-prep milestone** — not now.

### 8.1 Duplicate and integrity checks

- Record file size + SHA-256 for every image under `data/raw/RealWaste/...`.
- Flag exact duplicate hashes (same file content).
- Optionally flag near-duplicates (perceptual hash) before splitting.
- Remove or quarantine unreadable / zero-byte images.
- Write a small report to `data/metadata/` (CSV/JSON — tracked in Git). Images themselves stay ignored.

### 8.2 Stratified splitting (before any augmentation)

Split **by original class label** (the four target classes after mapping), using stratification so each class keeps a similar proportion in train / val / test.

Suggested starting ratios (to be confirmed when data exists):

| Split | Role | Augmentation? |
|---|---|---|
| Train | Fit MobileNetV2 head | Yes (only here) |
| Validation | Tune / early stop | No |
| Test | Final honest metrics | No |

Rules:

- Split **before** augmentation so the same original photo never appears in both train and test.
- Persist split membership as path lists in `data/metadata/split_manifest.json` (or CSV) — **tracked**; image bytes stay **untracked**.
- Keep class order identical to `configs/class_mapping.json`.

---

## 9. What is intentionally not done yet

- No download of `realwaste.zip` or the 656.6 MB archive
- No unzip into `data/raw/`
- No training, no accuracy numbers, no model binary
- No split manifest file yet (will be created after the dataset is on disk)

---

## 10. References

- UCI RealWaste page: https://archive.ics.uci.edu/dataset/908/realwaste
- Dataset DOI: https://doi.org/10.24432/C5SS4G
- Paper: https://www.mdpi.com/2078-2489/14/12/633
- Authors’ GitHub (license note): https://github.com/sam-single/realwaste
- TrashNet: https://github.com/garythung/trashnet
- TACO: https://github.com/pedropro/TACO
- Project scope: [`project_scope.md`](project_scope.md)
