"""Reusable inference for the SELECTED model (milestone 11).

Scope
-----
- Loads exactly one checkpoint: the model named in the committed selection
  record (`models/metadata/selection/selected_model.json`) — the same record
  the held-out evaluation used. The resolution, SHA-256 and class-order
  checks are the functions of `src/evaluate.py`, so the app can never serve
  a different artifact than the recorded results describe.
- Loads with `compile=False, safe_mode=True` (unsafe deserialization is
  never enabled) and refuses to continue when the file, checksum, class
  order, input shape or output units are wrong.
- Accepts uploaded image **bytes**, an already-decoded PIL image, or a
  NumPy array; validates by actually decoding (the file name is never
  trusted), applies EXIF orientation, converts to RGB, and composites
  transparency onto white explicitly.
- Resize + value scale come from `data_pipeline.resize_to_model_input` —
  the same helper training/validation/evaluation use: bilinear
  `tf.image.resize` to the configured size, float32 on the **0-255** scale.
- **No extra normalization.** MobileNetV2 preprocessing (0-255 -> [-1, 1])
  is embedded in the saved graph (proved by the compatibility reports under
  models/metadata/verification/), so this module must not scale again.
- Inference runs with `training=False`; outputs must be finite, four class
  scores in [0, 1] summing to ~1, otherwise the prediction is rejected.

Deliberate difference from the evaluation pipeline (documented, tested):
- Evaluation reads manifest files with `tf.io.decode_image`; uploads are
  decoded with **Pillow** because only Pillow applies EXIF orientation and
  composites transparency (TensorFlow's decoder ignores EXIF, so phone
  photos would arrive sideways). Both paths then share the same
  `tf.image.resize` helper, so size and value scale can never drift.
- Transparent pixels are composited onto **white** for uploads; evaluation
  images are opaque RGB JPEGs, where alpha never arises.

CLI example (run with the inference environment — the artifact is Keras 3):

    .venv-infer\\Scripts\\python.exe src\\predict.py path\\to\\image.jpg
    .venv-infer\\Scripts\\python.exe src\\predict.py image.png --json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_pipeline  # noqa: E402
import evaluate  # noqa: E402  (shared selection/checksum/class-order checks)

DEFAULT_SELECTION = evaluate.DEFAULT_SELECTION

# Formats the app accepts. Checked on the *decoded* file, never on the name.
ALLOWED_FORMATS = ("JPEG", "PNG")
# Refuse images larger than this before any resize/decode work (pixels = w*h).
MAX_IMAGE_PIXELS = 40_000_000
# Softmax rows must sum to 1 within this tolerance (float32 headroom).
SOFTMAX_SUM_ATOL = 1e-3


class InvalidImageError(ValueError):
    """Uploaded/decoded image is unusable (corrupt, unsupported, too large)."""


# ---------------------------------------------------------------------------
# Selection record -> verified model
# ---------------------------------------------------------------------------

def _placement_instructions(selection: dict, path_root: Path) -> str:
    """Actionable text for a missing model file (the binary is Git-ignored)."""
    relative = str((selection.get("selected_model") or {}).get("path", ""))
    run_id = str((selection.get("run") or {}).get("run_id", "<run_id>"))
    expected = Path(path_root) / relative if relative else Path(path_root)
    return (
        f"Selected model file not found: {expected}\n"
        f"The model binary is intentionally Git-ignored, so a fresh clone does "
        f"not contain it. Restore it from the Colab outputs: extract "
        f"model_{run_id}.zip into the models\\ folder so the file lands at "
        f"{relative}, then start again."
    )


def read_selection_record(selection_path: Path = DEFAULT_SELECTION) -> dict:
    """Read the committed selection record (the app uses this to key its cache)."""
    return evaluate.read_json(Path(selection_path))


def resolve_selection(
    selection_path: Path = DEFAULT_SELECTION,
    *,
    path_root: Path = REPO_ROOT,
) -> dict:
    """Read the selection record and resolve the chosen artifact.

    Returns the record plus `path` (absolute), `sha256`, `class_order`,
    `run_id` and `selection_reason` — exactly the fields `evaluate.py`
    evaluates, resolved by the same function.
    """
    selection = read_selection_record(selection_path)
    try:
        chosen = evaluate.resolve_selected_model(selection, path_root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(_placement_instructions(selection, path_root)) from exc
    chosen["selection_path"] = str(selection_path)
    chosen["selection_record"] = selection
    return chosen


def load_classifier(
    selection_path: Path = DEFAULT_SELECTION,
    *,
    path_root: Path = REPO_ROOT,
    config_path: Path | None = None,
) -> dict:
    """Resolve, verify and load the selected model.

    Checks, in order:
    1. Selection record exists and names a model file.
    2. Model file SHA-256 matches the record (any other artifact is refused).
    3. Class order agrees across the record, configs/class_mapping.json and
       the model bundle's class_order.json (when present).
    4. The file loads with compile=False, safe_mode=True.
    5. Input size matches configs/training.json; output units match the
       class count.

    Returns a dict: model, class_order, class_index, model_path,
    model_sha256, run_id, selection_reason, image_size.
    """
    chosen = resolve_selection(selection_path, path_root=path_root)
    class_order = list(chosen["class_order"])

    # 1) checksum: never serve bytes that are not the selected artifact
    model_sha = evaluate.verify_model_checksum(chosen["path"], chosen["sha256"])

    # 2) class order across every source that names the classes
    config_file = Path(config_path) if config_path else Path(path_root) / "configs" / "training.json"
    config = data_pipeline.load_training_config(config_file)
    configured_mapping = Path(config["class_mapping_path"])
    mapping = (
        configured_mapping
        if configured_mapping.is_absolute()
        else Path(path_root) / configured_mapping
    )
    config_order = data_pipeline.load_class_order(mapping)
    order_check = evaluate.verify_class_order_sources(
        class_order, config_order, chosen["path"]
    )
    class_order = order_check["class_order"]

    # 3) load (same options as every other verification in this project)
    try:
        model = tf.keras.models.load_model(
            str(chosen["path"]), compile=False, safe_mode=True
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
        raise RuntimeError(
            f"The selected model at {chosen['path']} could not be loaded with "
            f"tensorflow {tf.__version__}. The artifact was saved by Keras 3, "
            f"so run this from the inference environment: "
            f".venv-infer\\Scripts\\python.exe (see README.md). "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    # 4) shape agreement with the pipeline and the class order
    image_size = int(config["image_size"])
    input_shape = tuple(model.input_shape)
    if not (len(input_shape) == 4 and int(input_shape[1]) == image_size):
        raise RuntimeError(
            f"selected model expects input {input_shape!r} but "
            f"configs/training.json uses image_size={image_size} — the "
            "artifact and the config do not belong together"
        )
    output_units = int(model.output_shape[-1])
    if output_units != len(class_order):
        raise RuntimeError(
            f"selected model outputs {output_units} classes but the class "
            f"order has {len(class_order)}: {class_order}"
        )

    return {
        "model": model,
        "class_order": class_order,
        "class_index": {name: index for index, name in enumerate(class_order)},
        "model_path": str(chosen["path"]),
        "model_sha256": model_sha,
        "run_id": chosen["run_id"],
        "selection_reason": chosen["selection_reason"],
        "image_size": image_size,
        "selection_path": chosen["selection_path"],
        "order_check": order_check,
    }


# ---------------------------------------------------------------------------
# Image decoding (uploads) -> shared resize helper
# ---------------------------------------------------------------------------

def _check_pixel_count(image: Image.Image, max_pixels: int) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise InvalidImageError(f"Image has invalid dimensions {image.size}.")
    if width * height > max_pixels:
        raise InvalidImageError(
            f"Image is too large to process: {width}x{height} = "
            f"{width * height:,} pixels exceeds the {max_pixels:,}-pixel "
            "limit. Downscale or re-export it first."
        )


def prepare_image(image: Image.Image, *, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    """EXIF orientation, then RGB with transparency composited onto white.

    This is the single place where pixel conventions are decided for the app:
    orientation is corrected first (so downstream code never sees sideways
    photos), transparency is handled explicitly (white background), and
    grayscale/palette/CMYK inputs are converted to plain RGB.
    """
    _check_pixel_count(image, max_pixels)
    transposed = ImageOps.exif_transpose(image)
    if transposed is not None:
        image = transposed

    has_alpha = image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        # Explicit choice: composite onto WHITE. (Evaluation images are
        # opaque RGB JPEGs, so alpha never arises on that path.)
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if image.mode != "RGB":
        # Grayscale (L) -> replicate channels, exactly like tf.io.decode_image
        # with channels=3 does on the evaluation path.
        return image.convert("RGB")
    return image


def decode_image_bytes(
    data: bytes | bytearray | memoryview,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    allowed_formats: tuple[str, ...] = ALLOWED_FORMATS,
) -> Image.Image:
    """Validate upload bytes by decoding them, then normalize to RGB.

    The file name is ignored entirely: format, decodability and dimensions
    are read from the actual bytes.
    """
    raw = bytes(data)
    if not raw:
        raise InvalidImageError(
            "The uploaded file is empty (0 bytes). Choose a JPG or PNG image."
        )

    try:
        probe = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError as exc:
        raise InvalidImageError(
            "The image declares dangerously large dimensions and was refused "
            "before decoding. Downscale it first."
        ) from exc
    except Image.UnidentifiedImageError as exc:
        raise InvalidImageError(
            "This file is not a recognizable image, so it cannot be "
            "classified. The app accepts JPG/JPEG/PNG images — re-export or "
            "re-download the photo."
        ) from exc
    except OSError as exc:
        raise InvalidImageError(
            f"The image header could not be read ({exc}). The file may be "
            "corrupted; re-export or re-download it."
        ) from exc

    with probe:
        image_format = (probe.format or "").upper()
        if image_format not in allowed_formats:
            raise InvalidImageError(
                f"Unsupported image format: {image_format or 'unknown'}. "
                "This app accepts JPG/JPEG/PNG only."
            )
        # Header dimensions are known now — refuse before decoding pixels.
        if probe.size[0] * probe.size[1] > max_pixels:
            raise InvalidImageError(
                f"Image is too large to process: {probe.size[0]}x{probe.size[1]} = "
                f"{probe.size[0] * probe.size[1]:,} pixels exceeds the "
                f"{max_pixels:,}-pixel limit. Downscale or re-export it first."
            )
        try:
            probe.load()  # force full decode; truncated/corrupt files raise here
        except Exception as exc:  # noqa: BLE001 - any decode failure == unusable
            raise InvalidImageError(
                "The image data could not be decoded — the file appears to be "
                "truncated or corrupted. Re-export or re-download it."
            ) from exc
        decoded = probe.copy()

    return prepare_image(decoded, max_pixels=max_pixels)


def image_from_array(
    array: Any, *, max_pixels: int = MAX_IMAGE_PIXELS
) -> Image.Image:
    """Wrap a NumPy array (H,W), (H,W,3) or (H,W,4) as an RGB image.

    Values are interpreted on the 0-255 scale (the model's convention);
    floating-point inputs are clipped to that range. Non-finite values are
    refused rather than silently replaced.
    """
    try:
        arr = np.asarray(array)
    except Exception as exc:  # noqa: BLE001 - give the caller a clear message
        raise InvalidImageError(f"Could not read the image array: {exc}") from exc

    if arr.ndim not in (2, 3):
        raise InvalidImageError(
            f"Expected a 2-D grayscale or 3-D image array, got shape {arr.shape}."
        )
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise InvalidImageError(
            f"Expected 1, 3 or 4 channels, got shape {arr.shape}."
        )
    if arr.dtype == np.uint8:
        pixels = arr
    else:
        if not np.issubdtype(arr.dtype, np.number):
            raise InvalidImageError(f"Unsupported pixel dtype {arr.dtype}.")
        arr_float = arr.astype(np.float64)
        if not np.isfinite(arr_float).all():
            raise InvalidImageError(
                "The image array contains NaN/Inf values; expected pixels on "
                "the 0-255 scale."
            )
        pixels = np.clip(arr_float, 0.0, 255.0).astype(np.uint8)

    if pixels.ndim == 2:
        image = Image.fromarray(pixels, mode="L")
    elif pixels.shape[2] == 1:
        image = Image.fromarray(pixels[:, :, 0], mode="L")
    else:
        # fromarray picks RGB or RGBA from the channel count
        image = Image.fromarray(pixels)
    return prepare_image(image, max_pixels=max_pixels)


def prepare_input(
    image: Any, *, max_pixels: int = MAX_IMAGE_PIXELS
) -> Image.Image:
    """Accept upload bytes, a PIL image or an array -> EXIF-corrected RGB."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return decode_image_bytes(image, max_pixels=max_pixels)
    if isinstance(image, Image.Image):
        return prepare_image(image, max_pixels=max_pixels)
    if isinstance(image, np.ndarray):
        return image_from_array(image, max_pixels=max_pixels)
    raise TypeError(
        f"Expected upload bytes, a PIL.Image.Image or a NumPy array, got "
        f"{type(image).__name__}."
    )


def image_to_input(image: Image.Image, image_size: int) -> np.ndarray:
    """RGB image -> (image_size, image_size, 3) float32 array on 0-255.

    Uses the shared `data_pipeline.resize_to_model_input` so this is the
    exact resize/scale the training and evaluation pipelines use. No
    normalization happens here — the model does that itself.
    """
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise InvalidImageError(
            f"Expected an RGB image with 3 channels, got shape {array.shape}."
        )
    resized = data_pipeline.resize_to_model_input(
        tf.convert_to_tensor(array), image_size
    )
    return np.asarray(resized, dtype=np.float32)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _validate_scores(outputs: Any, class_order: list[str]) -> np.ndarray:
    """Reject anything that is not one finite softmax row for these classes."""
    scores = np.asarray(outputs, dtype=np.float32)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    expected = (len(class_order),)
    if scores.shape != expected:
        raise RuntimeError(
            f"model returned shape {scores.shape}, expected {expected} for "
            f"class_order {class_order} — refusing to guess a label"
        )
    if not np.isfinite(scores).all():
        raise RuntimeError(
            "model returned non-finite scores (NaN/Inf); refusing to report "
            "a prediction"
        )
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise RuntimeError(
            f"model scores are outside [0, 1] (min={float(scores.min()):.6f}, "
            f"max={float(scores.max()):.6f}); the output does not look like a "
            "softmax over the four classes"
        )
    total = float(scores.sum())
    if abs(total - 1.0) > SOFTMAX_SUM_ATOL:
        raise RuntimeError(
            f"model scores sum to {total:.6f} (expected 1 within "
            f"{SOFTMAX_SUM_ATOL}); refusing to report a prediction"
        )
    return scores


def predict_image(classifier: dict, image: Any) -> dict:
    """Run one prediction on upload bytes, a PIL image or an array.

    `classifier` is the dict returned by `load_classifier` (tests may build
    an equivalent one around a stub model). Returns the predicted category,
    the top score and every class score.
    """
    class_order = list(classifier["class_order"])
    image_size = int(classifier["image_size"])

    prepared = prepare_input(image)
    width, height = prepared.size
    model_input = np.expand_dims(image_to_input(prepared, image_size), axis=0)

    outputs = classifier["model"](model_input, training=False)
    scores = _validate_scores(outputs, class_order)

    top_index = int(np.argmax(scores))
    return {
        "category": class_order[top_index],
        "class_index": top_index,
        "confidence": float(scores[top_index]),
        "scores": {
            name: float(value) for name, value in zip(class_order, scores)
        },
        "class_order": class_order,
        "decoded_size": {"width": width, "height": height},
        "image_size": image_size,
        "model": {
            "path": classifier.get("model_path"),
            "sha256": classifier.get("model_sha256"),
            "run_id": classifier.get("run_id"),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify one image with the previously SELECTED model "
            "(checksum + class order verified at load time)."
        )
    )
    parser.add_argument("image", type=Path, help="Path to a JPG or PNG image.")
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="Selection record naming the model (default: the committed one).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the result as JSON."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.json:
        # Human-readable banner only in text mode: --json output stays pure
        # JSON on stdout so scripts can pipe it straight into a parser.
        print("=== Waste classification prediction (selected model only) ===")
    try:
        classifier = load_classifier(args.selection)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    try:
        image_bytes = args.image.read_bytes()
    except OSError as exc:
        print(f"[error] Could not read the image: {exc}", file=sys.stderr)
        return 1

    try:
        result = predict_image(classifier, image_bytes)
    except (InvalidImageError, RuntimeError, TypeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    size = result["decoded_size"]
    print(f"model     : {result['model']['path']} "
          f"(sha256 {str(result['model']['sha256'])[:16]}...)")
    print(f"image     : {args.image} ({size['width']}x{size['height']} RGB, "
          f"resized to {result['image_size']}x{result['image_size']}, "
          f"float32 0-255)")
    print(f"predicted : {result['category']} "
          f"(top score {result['confidence']:.4f})")
    print("scores    :")
    for name in result["class_order"]:
        print(f"  {name:<8} {result['scores'][name]:.4f}")
    print("note      : the top score is the model's confidence, not a "
          "guarantee that the label is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
