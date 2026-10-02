"""MobileNetV2 model construction for the four-class waste classifier.

Milestone 5 scope: build and compile the baseline model. No `model.fit`,
no fine-tuning, no evaluation here.

Important choices
-----------------
- **Input (224, 224, 3)** is MobileNetV2's native ImageNet resolution.
- **Augmentation (RandomFlip + RandomRotation) lives inside the model** but
  runs only when the model is called with `training=True` (Keras passes the
  training flag to layers that accept it). Validation and inference are
  therefore deterministic, and augmentation can never touch validation or
  test images.
- **Preprocessing happens exactly once, inside the model**: MobileNetV2's
  `preprocess_input` maps 0-255 floats to [-1, 1]. Bundling it in the graph
  means the saved `.keras` file always preprocesses correctly on its own.
- **The pretrained base is frozen** for the baseline: with only ~3k images
  and no GPU budget, training 2M+ backbone weights would overfit and destroy
  the useful ImageNet features. Fine-tuning is a later milestone.
- **`base(x, training=False)`** keeps BatchNormalization layers in the
  frozen backbone in inference mode, so their running statistics are not
  updated by the small training batches.
- Built-in Keras layers only (no custom `Lambda`), so `load_model` works
  with safe serialization — no `safe_mode=False` or custom objects needed.

Weights policy: `weights="imagenet"` is the default. If the weights cannot
be loaded (no network, no cache), this raises an error instead of quietly
falling back to random weights — a silently random "pretrained" model would
make later accuracy claims meaningless. Tests pass `weights=None` explicitly
to stay offline.
"""

from __future__ import annotations

import tensorflow as tf

DEFAULT_IMAGE_SIZE = 224
DEFAULT_NUM_CLASSES = 4
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 0.001

# Modest augmentation: horizontal flip plus a small rotation (±18 degrees).
ROTATION_FACTOR = 0.05


def build_augmentation() -> tf.keras.Sequential:
    """Light geometric augmentation; active only when training=True."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", name="random_flip"),
            tf.keras.layers.RandomRotation(ROTATION_FACTOR, name="random_rotation"),
        ],
        name="augmentation",
    )


def preprocess_mobilenet_v2(inputs: tf.Tensor) -> tf.Tensor:
    """MobileNetV2 preprocessing: scale 0-255 floats to [-1, 1] exactly once.

    Exposed as a named function so tests can prove the mapping (0 -> -1,
    255 -> 1) and confirm it is applied a single time inside the graph.
    """
    return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)


def build_waste_classifier(
    num_classes: int = DEFAULT_NUM_CLASSES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    *,
    weights: str | None = "imagenet",
    dropout_rate: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    """Build MobileNetV2 (frozen) -> GAP -> Dropout -> softmax head.

    `weights` must be either "imagenet" or None. Passing None is an explicit,
    deliberate choice for offline tests; ImageNet failures are never silently
    replaced with random weights.
    """
    if weights not in ("imagenet", None):
        raise ValueError(
            f"weights must be 'imagenet' or None, got {weights!r}. "
            "Random initialization must be requested explicitly via weights=None."
        )
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs
    if use_augmentation:
        x = build_augmentation()(x)
    # Single preprocessing call: 0-255 in, [-1, 1] out (see module docstring).
    x = preprocess_mobilenet_v2(x)

    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=weights,
        )
    except Exception as exc:  # noqa: BLE001 - network/IO errors vary by platform
        if weights == "imagenet":
            raise RuntimeError(
                "MobileNetV2 ImageNet weights could not be loaded "
                f"({type(exc).__name__}: {exc}). "
                "Check your network connection or the local Keras cache "
                "(~/.keras/models). To build with random weights on purpose, "
                "pass weights=None explicitly — this function will not do it for you."
            ) from exc
        raise

    # Freeze the whole pretrained base for the baseline run.
    base.trainable = False
    # training=False keeps frozen BatchNorm layers in inference mode.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="waste_classifier_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def parameter_counts(model: tf.keras.Model) -> dict[str, int]:
    """Trainable vs non-trainable parameter counts (for smoke reports)."""
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    return {
        "trainable": int(trainable),
        "non_trainable": int(non_trainable),
        "total": int(trainable + non_trainable),
    }


def base_model_is_frozen(model: tf.keras.Model) -> bool:
    """True when every layer of the nested MobileNetV2 base is frozen.

    The augmentation block is also a Sequential (a Model subclass), so the
    base is identified by its MobileNetV2 name rather than by type alone.
    """
    base_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower()
    ]
    if not base_layers:
        return False
    return all(not layer.trainable for layer in base_layers)
