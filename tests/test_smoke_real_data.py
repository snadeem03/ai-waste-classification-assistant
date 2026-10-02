"""Real-data smoke check for milestone 5 (no fitting, no test-split access).

Skips cleanly when the RealWaste dataset is not downloaded or when the
ImageNet weights cannot be loaded, so this test never fakes a pretrained
result with random weights.

Run with output visible:

    .venv\\Scripts\\python.exe -m pytest tests\\test_smoke_real_data.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import data_pipeline  # noqa: E402
import model as waste_model  # noqa: E402

METADATA_DIR = ROOT / "data" / "metadata"
REQUIRED_MANIFESTS = ("train.csv", "validation.csv")


def real_data_available() -> bool:
    if not all((METADATA_DIR / name).exists() for name in REQUIRED_MANIFESTS):
        return False
    first_row = (METADATA_DIR / "train.csv").read_text(encoding="utf-8").splitlines()
    if len(first_row) < 2:
        return False
    relative_path = first_row[1].split(",")[0]
    return (ROOT / relative_path).exists()


@pytest.mark.skipif(
    not real_data_available(),
    reason="RealWaste images/manifests not present (run src/download_data.py first)",
)
def test_real_data_smoke_check() -> None:
    config = data_pipeline.load_training_config()
    class_order = data_pipeline.load_class_order()

    # Only train + validation are opened; the test split stays unopened.
    splits = data_pipeline.load_splits(
        metadata_dir=METADATA_DIR,
        class_order=class_order,
        path_root=ROOT,
        splits=("train", "validation"),
    )
    assert splits["train"] and splits["validation"]

    train_dataset = data_pipeline.make_dataset(
        splits["train"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        training=True,
        seed=config["seed"],
    )
    validation_dataset = data_pipeline.make_dataset(
        splits["validation"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        training=False,
        seed=config["seed"],
    )

    train_images, train_labels = next(iter(train_dataset))
    validation_images, validation_labels = next(iter(validation_dataset))

    print("\n=== milestone 5 real-data smoke check ===")
    print(f"train batch images:      {train_images.shape} {train_images.dtype.name}")
    train_pixels = train_images.numpy()
    print(
        f"train pixel range:       {float(train_pixels.min()):.1f} .. "
        f"{float(train_pixels.max()):.1f} (0-255 scale)"
    )
    print(
        f"train labels:            {train_labels.numpy().tolist()} "
        f"(range {int(train_labels.numpy().min())}..{int(train_labels.numpy().max())})"
    )
    print(f"validation batch images: {validation_images.shape} {validation_images.dtype.name}")
    print(
        f"validation labels:       {validation_labels.numpy().tolist()} "
        f"(range {int(validation_labels.numpy().min())}.."
        f"{int(validation_labels.numpy().max())})"
    )

    assert tuple(train_images.shape) == (
        config["batch_size"], config["image_size"], config["image_size"], 3
    )
    assert tuple(validation_images.shape) == tuple(train_images.shape)
    all_labels = list(train_labels.numpy()) + list(validation_labels.numpy())
    assert all(0 <= int(label) < len(class_order) for label in all_labels)

    # ImageNet-backed build: skip (report pending) if weights are unavailable.
    try:
        classifier = waste_model.build_waste_classifier(
            num_classes=len(class_order),
            image_size=config["image_size"],
            weights="imagenet",
            dropout_rate=config["dropout"],
            learning_rate=config["learning_rate"],
        )
    except RuntimeError as exc:
        pytest.skip(f"PRETRAINED SMOKE CHECK PENDING — ImageNet weights unavailable: {exc}")

    outputs = classifier.predict(train_images, verbose=0)
    counts = waste_model.parameter_counts(classifier)

    print(f"model output shape:      {outputs.shape}")
    print(
        f"output row sums:         min={float(outputs.sum(axis=1).min()):.6f} "
        f"max={float(outputs.sum(axis=1).max()):.6f}"
    )
    print(
        f"parameters:              trainable={counts['trainable']:,} "
        f"non-trainable={counts['non_trainable']:,} total={counts['total']:,}"
    )
    print(f"base frozen:             {waste_model.base_model_is_frozen(classifier)}")
    print("forward pass only — no model.fit() was called")

    assert outputs.shape == (config["batch_size"], len(class_order))
    assert np.all(np.isfinite(outputs))
    assert np.allclose(outputs.sum(axis=1), 1.0, atol=1e-5)
    assert waste_model.base_model_is_frozen(classifier)
    assert counts["non_trainable"] > counts["trainable"] > 0
