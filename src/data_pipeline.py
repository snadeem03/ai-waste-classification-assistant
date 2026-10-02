"""Training input pipeline: split manifests -> batched tf.data datasets.

Milestone 5 scope
-----------------
- Load `train.csv` / `validation.csv` written by `src/prepare_data.py`.
- Validate required columns, label/index consistency, file existence, and
  that each split is nonempty.
- Decode images to RGB, resize to the configured size, and emit float32
  pixels on the **0-255** scale (see "Why 0-255" below).
- Emit integer labels for `SparseCategoricalCrossentropy`.
- Shuffle training data with the fixed seed; keep validation order
  deterministic; keep the final partial batch.
- The **test** split is deliberately not opened in this milestone.

Important choices
-----------------
Why 0-255 floats instead of 0-1: MobileNetV2's `preprocess_input` expects
0-255 input and maps it to [-1, 1]. Scaling happens exactly once, inside
the model (`src/model.py`), so the saved model always preprocesses
whatever it is given — including Streamlit uploads in a later milestone.

Why no full-dataset cache by default: caching the decoded 3k-image dataset
in RAM is unnecessary for this dataset size and hides I/O problems. A
bounded `prefetch(2)` keeps the next couple of batches ready without
growing memory.

Portable paths: manifests store relative forward-slash paths
(`data/raw/.../Metal_1.jpg`) that work on Windows and Colab alike.
`path_root` is the directory those relative paths resolve against — the
repo root on Windows, or wherever `data/` lives on Colab.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "training.json"
DEFAULT_CLASS_MAPPING = REPO_ROOT / "configs" / "class_mapping.json"
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "metadata"

# Columns every manifest from src/prepare_data.py must have.
REQUIRED_MANIFEST_COLUMNS = ("path", "target", "class_index")

REQUIRED_CONFIG_KEYS = (
    "seed",
    "image_size",
    "batch_size",
    "learning_rate",
    "baseline_max_epochs",
    "dropout",
    "class_mapping_path",
)

# Bounded prefetch: at most this many batches are prepared ahead of training.
PREFETCH_SIZE = 2


def load_training_config(path: Path = DEFAULT_TRAINING_CONFIG) -> dict:
    """Load configs/training.json and check it has every key we rely on."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing training config: {path}")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path.name} is missing required keys: {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError("training config: seed must be an integer")
    if not isinstance(config["image_size"], int) or config["image_size"] <= 0:
        raise ValueError("training config: image_size must be a positive integer")
    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("training config: batch_size must be a positive integer")
    if not isinstance(config["learning_rate"], (int, float)) or config["learning_rate"] <= 0:
        raise ValueError("training config: learning_rate must be > 0")
    if not 0.0 <= float(config["dropout"]) < 1.0:
        raise ValueError("training config: dropout must be in [0, 1)")
    return config


def load_class_order(class_mapping_path: Path = DEFAULT_CLASS_MAPPING) -> list[str]:
    """Read the fixed class order from configs/class_mapping.json.

    The training config stores only the *path* to the class mapping so the
    class order lives in exactly one file and can never drift.
    """
    class_mapping_path = Path(class_mapping_path)
    if not class_mapping_path.exists():
        raise FileNotFoundError(f"Missing class mapping: {class_mapping_path}")
    with open(class_mapping_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    class_order = payload.get("class_order")
    if not isinstance(class_order, list) or not class_order:
        raise ValueError(f"{class_mapping_path.name}: class_order must be a non-empty list")
    if len(class_order) != len(set(class_order)):
        raise ValueError(f"{class_mapping_path.name}: class_order contains duplicates")
    return list(class_order)


def resolve_image_path(relative_path: str, path_root: Path) -> Path:
    """Resolve a manifest path (forward slashes) against an explicit root."""
    normalized = str(relative_path).replace("\\", "/").lstrip("/")
    return (Path(path_root) / normalized).resolve()


def load_manifest(
    manifest_path: Path,
    class_order: list[str],
    *,
    path_root: Path,
    check_files: bool = True,
) -> list[dict]:
    """Load one split CSV and validate it before any image is decoded.

    Checks, in order:
    1. Manifest file exists.
    2. Required columns are present.
    3. Split has at least one row.
    4. Every target is in class_order and every class_index matches it.
    5. Every referenced image file exists (unless check_files=False).

    Returns a list of records with `image_path` resolved to an absolute
    path; the original relative `path` is kept untouched.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not class_order:
        raise ValueError("class_order must not be empty")

    with open(manifest_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(
                f"{manifest_path.name} is missing required columns: {missing} "
                f"(found: {fieldnames})"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{manifest_path.name} has no rows; split is empty")

    records: list[dict] = []
    for line_number, row in enumerate(rows, start=2):  # start=2: row 1 is the header
        target = row["target"]
        if target not in class_order:
            raise ValueError(
                f"{manifest_path.name} line {line_number}: unknown target {target!r}; "
                f"expected one of {class_order}"
            )
        try:
            class_index = int(row["class_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{manifest_path.name} line {line_number}: "
                f"class_index {row['class_index']!r} is not an integer"
            ) from exc
        expected_index = class_order.index(target)
        if class_index != expected_index:
            raise ValueError(
                f"{manifest_path.name} line {line_number}: class_index {class_index} "
                f"does not match target {target!r} (expected {expected_index} "
                f"from class_order {class_order})"
            )
        if not (0 <= class_index < len(class_order)):
            raise ValueError(
                f"{manifest_path.name} line {line_number}: class_index {class_index} "
                f"is outside class_order range"
            )

        relative_path = row["path"]
        records.append(
            {
                "path": relative_path,
                "image_path": str(resolve_image_path(relative_path, path_root)),
                "target": target,
                "class_index": class_index,
                "file_sha256": row.get("file_sha256", ""),
                "decoded_sha256": row.get("decoded_sha256", ""),
                "group_id": row.get("group_id", ""),
            }
        )

    if check_files:
        missing_files = [
            record["path"] for record in records if not Path(record["image_path"]).exists()
        ]
        if missing_files:
            shown = ", ".join(missing_files[:5])
            more = "" if len(missing_files) <= 5 else f" (+{len(missing_files) - 5} more)"
            raise FileNotFoundError(
                f"{manifest_path.name}: {len(missing_files)} image file(s) not found "
                f"under {path_root}: {shown}{more}. "
                "Check path_root or re-run src/download_data.py."
            )

    return records


def load_splits(
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    class_order: list[str] | None = None,
    *,
    path_root: Path = REPO_ROOT,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, list[dict]]:
    """Load the requested splits. Defaults to train + validation only.

    The test split is intentionally excluded from the defaults for this
    milestone: test images must stay unopened until evaluation time.
    """
    if class_order is None:
        class_order = load_class_order()
    metadata_dir = Path(metadata_dir)
    loaded: dict[str, list[dict]] = {}
    for split in splits:
        loaded[split] = load_manifest(
            metadata_dir / f"{split}.csv", class_order, path_root=path_root
        )
    return loaded


def _decode_to_rgb(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Decode any supported image to RGB, resize, float32 on the 0-255 scale."""
    raw = tf.io.read_file(path)
    # channels=3 converts grayscale/RGBA to RGB; expand_animations skips GIF frames.
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])  # decode_image has static shape unknown
    image = tf.image.resize(image, [image_size, image_size])
    # resize returns float32 while keeping the original 0-255 value scale.
    return tf.cast(image, tf.float32)


def make_dataset(
    records: list[dict],
    *,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
    cache: bool = False,
    prefetch_size: int = PREFETCH_SIZE,
) -> tf.data.Dataset:
    """Build a batched (images, labels) dataset from validated manifest records.

    - training=True  -> shuffled with `seed`, reshuffled each epoch.
    - training=False -> fixed manifest order, so validation is repeatable.
    - drop_remainder is always False, so the final partial batch is kept.
    - prefetch is bounded (default 2 batches) and no full-dataset cache
      is created unless cache=True is passed explicitly.
    """
    if not records:
        raise ValueError("Cannot build a dataset from an empty record list")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    paths = [record["image_path"] for record in records]
    labels = [int(record["class_index"]) for record in records]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: (_decode_to_rgb(path, image_size), tf.cast(label, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        # Opt-in only: never cache the whole decoded dataset in RAM by default.
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(prefetch_size)
    return dataset
