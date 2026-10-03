"""Tests for src/model.py (weights=None — no network access required).

Temporary model files are written to pytest's tmp_path, which lives outside
the tracked repository paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model as waste_model  # noqa: E402

CLASS_ORDER = ["metal", "organic", "paper", "plastic"]


def random_images(batch_size: int, image_size: int = 224) -> np.ndarray:
    """Random inputs on the 0-255 scale, as the pipeline produces them."""
    return np.random.default_rng(42).uniform(
        0, 255, size=(batch_size, image_size, image_size, 3)
    ).astype("float32")


def test_output_shape_is_batch_by_four() -> None:
    net = waste_model.build_waste_classifier(weights=None, image_size=224)
    images = random_images(3)
    outputs = net.predict(images, verbose=0)
    assert outputs.shape == (3, 4)
    assert net.output_shape == (None, 4)


def test_outputs_are_finite_and_softmax_rows_sum_to_one() -> None:
    net = waste_model.build_waste_classifier(weights=None, image_size=224)
    outputs = net.predict(random_images(4), verbose=0)
    assert np.all(np.isfinite(outputs))
    assert np.all(outputs >= 0.0)
    assert np.allclose(outputs.sum(axis=1), 1.0, atol=1e-5)


def test_preprocessing_maps_0_to_minus1_and_255_to_1_exactly_once() -> None:
    """Probe the tensor that actually feeds MobileNetV2 inside the model.

    If preprocessing were applied twice, 0 would map to -1.0078..., not -1.
    The probe ends at the base's inbound-node input, which is the
    preprocessed tensor on both Keras 2 (preprocess listed as a layer) and
    Keras 3 (preprocess only recorded in the operation graph).
    """
    net = waste_model.build_waste_classifier(weights=None, image_size=224)
    base = next(
        layer
        for layer in net.layers
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower()
    )
    feed = base._inbound_nodes[0].input_tensors
    if isinstance(feed, (list, tuple)):  # Keras 3 wraps it in a list
        feed = feed[0]
    probe = tf.keras.Model(net.input, feed)

    zeros = np.zeros((1, 224, 224, 3), dtype="float32")
    full = np.full((1, 224, 224, 3), 255.0, dtype="float32")
    assert np.allclose(probe(zeros, training=False).numpy(), -1.0, atol=1e-6)
    assert np.allclose(probe(full, training=False).numpy(), 1.0, atol=1e-6)


def test_pretrained_base_is_frozen_and_head_is_trainable() -> None:
    net = waste_model.build_waste_classifier(weights=None, image_size=224)
    assert waste_model.base_model_is_frozen(net)

    # The head's weights must be the ONLY trainable weights. Names differ by
    # Keras major version (Keras 2 prefixes the layer and adds ':0').
    head = net.get_layer("predictions")
    assert len(net.trainable_weights) == len(head.weights) == 2
    assert all(
        any(weight is head_weight for head_weight in head.weights)
        for weight in net.trainable_weights
    )
    trainable_names = sorted(weight.name for weight in net.trainable_weights)
    assert trainable_names in (
        ["predictions/bias:0", "predictions/kernel:0"],  # Keras 2
        ["bias", "kernel"],  # Keras 3
    )

    counts = waste_model.parameter_counts(net)
    assert counts["trainable"] > 0
    assert counts["non_trainable"] > counts["trainable"]  # frozen backbone dominates
    assert counts["total"] == counts["trainable"] + counts["non_trainable"]


def test_repeated_inference_is_stable() -> None:
    """training=False must ignore augmentation and give identical outputs."""
    net = waste_model.build_waste_classifier(weights=None, image_size=224)
    images = random_images(2)
    first = net.predict(images, verbose=0)
    second = net.predict(images, verbose=0)
    assert np.array_equal(first, second)

    tensor_first = net(tf.constant(images), training=False).numpy()
    assert np.allclose(first, tensor_first, atol=1e-6)


def test_compile_uses_adam_sparse_ce_and_accuracy() -> None:
    net = waste_model.build_waste_classifier(weights=None, image_size=224)
    assert isinstance(net.optimizer, tf.keras.optimizers.Adam)
    assert isinstance(net.loss, tf.keras.losses.SparseCategoricalCrossentropy)
    # model.metrics stays empty until the first train step, so ask Keras for
    # the compile configuration directly.
    compile_config = net.get_compile_config()
    assert compile_config["optimizer"]["class_name"] == "Adam"
    assert compile_config["loss"]["class_name"] == "SparseCategoricalCrossentropy"
    assert "accuracy" in compile_config["metrics"]


def test_save_load_roundtrip_preserves_inference(tmp_path: Path) -> None:
    """Plain load_model (safe mode, no custom objects) must reproduce outputs."""
    net = waste_model.build_waste_classifier(weights=None, image_size=224)
    images = random_images(2)
    expected = net.predict(images, verbose=0)

    model_path = tmp_path / "roundtrip_model.keras"
    net.save(model_path)
    reloaded = tf.keras.models.load_model(model_path)  # no custom_objects, safe mode

    actual = reloaded.predict(images, verbose=0)
    assert actual.shape == (2, 4)
    assert np.allclose(expected, actual, atol=1e-5)


def test_weights_argument_must_be_explicit() -> None:
    with pytest.raises(ValueError, match="weights must be"):
        waste_model.build_waste_classifier(weights="random")


def test_imagenet_failure_is_not_silently_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ImageNet weights fail to load, we must get an error — not random weights."""

    def failing_mobilenet_v2(**kwargs):  # noqa: ANN003 - mirrors Keras signature
        raise OSError("simulated offline download failure")

    monkeypatch.setattr(tf.keras.applications, "MobileNetV2", failing_mobilenet_v2)
    with pytest.raises(RuntimeError, match="weights=None"):
        waste_model.build_waste_classifier(weights="imagenet")


def test_augmentation_layers_exist_and_are_built_in() -> None:
    net = waste_model.build_waste_classifier(weights=None, image_size=224)
    layer_names = [layer.name for layer in net.layers]
    assert "augmentation" in layer_names
    augmentation = net.get_layer("augmentation")
    inner_names = [layer.name for layer in augmentation.layers]
    assert inner_names == ["random_flip", "random_rotation"]
    # Built-in serializable layer classes only — no Lambda.
    assert not any(isinstance(layer, tf.keras.layers.Lambda) for layer in net.layers)
