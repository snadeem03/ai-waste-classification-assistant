"""Tests for src/predict.py — reusable inference for the selected model.

Synthetic images and tiny untrained models only: these tests prove the
*workflow* (decoding, orientation, resize/scale, verification, output
validation), never accuracy. The real selected artifact is touched read-only
by the tests marked `requires_real_model` (Keras 3 environment only), and
nothing here opens the test split.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import data_pipeline  # noqa: E402
import predict  # noqa: E402
from test_train import CLASS_ORDER, write_configs  # noqa: E402

import keras  # noqa: E402

KERAS3 = int(keras.__version__.split(".")[0]) >= 3
SELECTED_MODEL = ROOT / "models" / "finetune_20261004_133620" / "best_model.keras"
SELECTED_RECORD = ROOT / "models" / "metadata" / "selection" / "selected_model.json"

requires_real_model = pytest.mark.skipif(
    not KERAS3 or not SELECTED_MODEL.is_file() or not SELECTED_RECORD.is_file(),
    reason=(
        "requires Keras 3 (.venv-infer) and the restored selected artifact "
        "models/finetune_20261004_133620/best_model.keras"
    ),
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def png_bytes(image: Image.Image, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **save_kwargs)
    return buffer.getvalue()


def save_tiny_model(tmp_path: Path, *, image_size: int = 32, units: int = 4) -> Path:
    """A minimal, untrained classifier with the right input/output shapes.

    Deliberately NOT MobileNetV2: these fixtures only prove the verification
    and inference plumbing, and a tiny head saves/loads instantly in both
    Keras versions (.venv = Keras 2, .venv-infer = Keras 3).
    """
    tf.keras.utils.set_random_seed(7)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units, activation="softmax"),
        ]
    )
    path = tmp_path / "bundle" / "best_model.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def write_selection(
    tmp_path: Path,
    model_path: Path,
    *,
    sha256: str | None = None,
    class_order: list[str] | None = None,
    run_id: str = "finetune_fixture",
) -> Path:
    import hashlib

    if model_path.is_file():
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    else:
        # Model intentionally absent (missing-artifact test): a placeholder
        # digest is fine because loading must fail before checksumming.
        digest = "f" * 64
    record = {
        "report_type": "selected_model_record",
        "created_at_utc": "2026-10-04T00:00:00+00:00",
        "selected": "finetuned",
        "selection_reason": "fixture record for tests",
        "selected_model": {
            "path": str(model_path),
            "sha256": sha256 if sha256 is not None else digest,
        },
        "run": {"run_id": run_id, "profile": "finetune"},
        "class_order": class_order if class_order is not None else list(CLASS_ORDER),
    }
    path = tmp_path / "selection" / "selected_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def build_fixture(
    tmp_path: Path,
    *,
    image_size: int = 32,
    units: int = 4,
    sha256: str | None = None,
    class_order: list[str] | None = None,
    run_id: str = "finetune_fixture",
    model_on_disk: bool = True,
) -> dict:
    config_path = write_configs(tmp_path, image_size=image_size)
    model_path = tmp_path / "bundle" / "best_model.keras"
    if model_on_disk:
        model_path = save_tiny_model(tmp_path, image_size=image_size, units=units)
    selection_path = write_selection(
        tmp_path, model_path, sha256=sha256, class_order=class_order, run_id=run_id
    )
    return {
        "config_path": config_path,
        "model_path": model_path,
        "selection_path": selection_path,
        "path_root": tmp_path,
    }


class StubModel:
    """Fake model recording its call arguments and returning fixed scores."""

    def __init__(self, scores, *, image_size: int = 32):
        self.scores = np.asarray([scores], dtype=np.float32)
        self.input_shape = (None, image_size, image_size, 3)
        self.output_shape = (None, len(scores))
        self.calls: list[dict] = []

    def __call__(self, batch, *, training):
        self.calls.append(
            {
                "training": training,
                "dtype": batch.dtype,
                "shape": batch.shape,
                "min": float(batch.min()),
                "max": float(batch.max()),
            }
        )
        return self.scores


def stub_classifier(stub: StubModel) -> dict:
    return {
        "model": stub,
        "class_order": list(CLASS_ORDER),
        "class_index": {name: i for i, name in enumerate(CLASS_ORDER)},
        "model_path": "models/fixture/best_model.keras",
        "model_sha256": "0" * 64,
        "run_id": "finetune_fixture",
        "image_size": 32,
    }


# ---------------------------------------------------------------------------
# Decoding: validation, grayscale, transparency, EXIF
# ---------------------------------------------------------------------------

def test_decode_rejects_empty_bytes() -> None:
    with pytest.raises(predict.InvalidImageError, match="empty"):
        predict.decode_image_bytes(b"")


def test_decode_rejects_unrecognizable_bytes() -> None:
    with pytest.raises(predict.InvalidImageError, match="not a recognizable image"):
        predict.decode_image_bytes(b"this is definitely not an image")


def test_filename_is_not_trusted() -> None:
    # A .png name with garbage inside must fail on the bytes, not the name.
    with pytest.raises(predict.InvalidImageError):
        predict.decode_image_bytes(b"<html><body>not a png</body></html>")


def test_decode_rejects_unsupported_format(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format="GIF")
    with pytest.raises(predict.InvalidImageError, match="JPG/JPEG/PNG only"):
        predict.decode_image_bytes(buffer.getvalue())


def test_decode_rejects_truncated_png() -> None:
    complete = png_bytes(Image.new("RGB", (40, 40), (10, 20, 30)))
    truncated = complete[:60]
    with pytest.raises(predict.InvalidImageError, match="truncated or corrupted"):
        predict.decode_image_bytes(truncated)


def test_decode_rejects_oversized_images() -> None:
    payload = png_bytes(Image.new("RGB", (40, 30), (5, 5, 5)))
    with pytest.raises(predict.InvalidImageError, match="pixel limit"):
        predict.decode_image_bytes(payload, max_pixels=100)


def test_grayscale_is_converted_to_rgb() -> None:
    decoded = predict.decode_image_bytes(png_bytes(Image.new("L", (30, 12), 128)))
    assert decoded.mode == "RGB"
    array = np.asarray(decoded)
    assert array.shape == (12, 30, 3)
    assert (array[:, :, 0] == array[:, :, 1]).all()
    assert (array[:, :, 1] == array[:, :, 2]).all()
    assert int(array[0, 0, 0]) == 128  # value preserved, not rescaled


def test_transparent_pixels_composite_onto_white() -> None:
    transparent = Image.new("RGBA", (20, 20), (0, 128, 255, 0))
    decoded = predict.decode_image_bytes(png_bytes(transparent))
    assert decoded.mode == "RGB"
    array = np.asarray(decoded)
    assert tuple(int(v) for v in array[3, 3]) == (255, 255, 255)


def test_partial_alpha_is_composited_not_dropped() -> None:
    # Half-transparent blue over white -> blended channel (0+255)/2 style mix.
    image = Image.new("RGBA", (10, 10), (0, 0, 255, 128))
    array = np.asarray(predict.decode_image_bytes(png_bytes(image)))
    blue_channel = int(array[0, 0, 2])
    red_channel = int(array[0, 0, 0])
    assert red_channel > 100 and blue_channel > 200  # blended toward white/blue


def test_exif_orientation_is_applied() -> None:
    # 40x20 image, left half red / right half blue, EXIF orientation 6
    # ("rotate 90 CW" to display) -> displayed as 20x40 with red on top.
    source = Image.new("RGB", (40, 20))
    pixels = source.load()
    for x in range(40):
        for y in range(20):
            pixels[x, y] = (255, 0, 0) if x < 20 else (0, 0, 255)
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=100, exif=exif)

    decoded = predict.decode_image_bytes(buffer.getvalue())
    assert decoded.size == (20, 40)  # dimensions swapped by the rotation
    array = np.asarray(decoded)
    assert tuple(int(v) for v in array[2, 10])[0] > 150  # top row is red-ish
    assert tuple(int(v) for v in array[37, 10])[2] > 150  # bottom row blue-ish


# ---------------------------------------------------------------------------
# Resize / value scale: same helper as the evaluation pipeline
# ---------------------------------------------------------------------------

def test_resize_matches_the_evaluation_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (37, 23), (10, 200, 90)).save(path)

    via_app = predict.image_to_input(Image.open(path), 224)
    via_pipeline = np.asarray(
        data_pipeline._decode_to_rgb(tf.convert_to_tensor(str(path)), 224),
        dtype=np.float32,
    )
    assert via_app.shape == (224, 224, 3)
    assert via_app.dtype == np.float32
    np.testing.assert_allclose(via_app, via_pipeline, atol=1e-5)


def test_model_input_stays_on_the_0_255_scale() -> None:
    white = predict.image_to_input(Image.new("RGB", (16, 16), (255, 255, 255)), 32)
    black = predict.image_to_input(Image.new("RGB", (16, 16), (0, 0, 0)), 32)
    assert white.shape == (32, 32, 3) and white.dtype == np.float32
    # 255 must arrive as 255: the model does 0-255 -> [-1,1] itself, so any
    # scaling here would be a DOUBLE normalization bug.
    assert float(white.max()) == pytest.approx(255.0, abs=1e-4)
    assert float(black.max()) == pytest.approx(0.0, abs=1e-4)
    assert float(white.min()) > 1.0  # not already normalized to [-1, 1]


def test_prepare_input_accepts_bytes_pil_and_array() -> None:
    stub = StubModel([0.05, 0.10, 0.15, 0.70])
    classifier = stub_classifier(stub)
    payload = png_bytes(Image.new("RGB", (24, 18), (30, 90, 160)))

    from_bytes = predict.predict_image(classifier, payload)
    from_pil = predict.predict_image(classifier, Image.open(io.BytesIO(payload)))
    array = np.asarray(Image.open(io.BytesIO(payload)), dtype=np.uint8)
    from_array = predict.predict_image(classifier, array)

    assert from_bytes["scores"] == from_pil["scores"] == from_array["scores"]
    assert len(stub.calls) == 3


def test_image_from_array_rejects_nonfinite_values() -> None:
    array = np.full((4, 4, 3), np.nan, dtype=np.float32)
    with pytest.raises(predict.InvalidImageError, match="NaN/Inf"):
        predict.image_from_array(array)


def test_image_from_array_handles_grayscale_and_rgba_shapes() -> None:
    gray = predict.image_from_array(np.full((6, 8), 77, dtype=np.uint8))
    assert gray.mode == "RGB" and gray.size == (8, 6)
    rgba = predict.image_from_array(
        np.dstack(
            [
                np.zeros((6, 8), dtype=np.uint8),
                np.zeros((6, 8), dtype=np.uint8),
                np.zeros((6, 8), dtype=np.uint8),
                np.zeros((6, 8), dtype=np.uint8),
            ]
        )
    )
    assert rgba.mode == "RGB"
    assert tuple(int(v) for v in np.asarray(rgba)[0, 0]) == (255, 255, 255)


# ---------------------------------------------------------------------------
# Output validation and score mapping
# ---------------------------------------------------------------------------

def test_prediction_maps_scores_onto_the_class_order() -> None:
    stub = StubModel([0.05, 0.10, 0.15, 0.70])
    result = predict.predict_image(
        stub_classifier(stub), png_bytes(Image.new("RGB", (16, 16), (20, 20, 20)))
    )
    assert result["category"] == "plastic"  # argmax of the 4th slot
    assert result["class_index"] == 3
    assert result["confidence"] == pytest.approx(0.70)
    assert list(result["scores"]) == CLASS_ORDER  # mapping is by name, not order
    assert result["scores"]["metal"] == pytest.approx(0.05)
    assert result["scores"]["paper"] == pytest.approx(0.15)
    assert sum(result["scores"].values()) == pytest.approx(1.0)
    assert result["model"]["run_id"] == "finetune_fixture"


def test_prediction_runs_with_training_false_and_unscaled_input() -> None:
    stub = StubModel([0.25, 0.25, 0.25, 0.25], image_size=32)
    predict.predict_image(
        stub_classifier(stub), png_bytes(Image.new("RGB", (20, 20), (255, 128, 0)))
    )
    call = stub.calls[0]
    assert call["training"] is False
    assert call["dtype"] == np.float32
    assert call["shape"] == (1, 32, 32, 3)
    assert call["max"] > 1.5  # 0-255 scale reached the model, not [-1, 1]
    assert call["max"] <= 255.0


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        ([0.1, 0.2, 0.3, float("nan")], "non-finite"),
        ([0.1, 0.2, 0.3], r"expected \(4,\)"),
        ([0.5, 0.5, 0.5, 0.5], "sum"),
        ([1.5, -0.2, 0.4, 0.3], "outside"),
    ],
)
def test_prediction_rejects_invalid_model_outputs(scores, match) -> None:
    stub = StubModel(scores)
    with pytest.raises(RuntimeError, match=match):
        predict.predict_image(
            stub_classifier(stub), png_bytes(Image.new("RGB", (8, 8), (9, 9, 9)))
        )


# ---------------------------------------------------------------------------
# Selection record: checksum, class order, missing/incompatible artifacts
# ---------------------------------------------------------------------------

def test_load_classifier_returns_verified_fields(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    classifier = predict.load_classifier(
        fixture["selection_path"],
        path_root=fixture["path_root"],
        config_path=fixture["config_path"],
    )
    assert classifier["class_order"] == CLASS_ORDER
    assert classifier["image_size"] == 32
    assert classifier["model"] is not None
    assert Path(classifier["model_path"]) == fixture["model_path"]
    import hashlib

    expected_sha = hashlib.sha256(fixture["model_path"].read_bytes()).hexdigest()
    assert classifier["model_sha256"] == expected_sha


def test_load_classifier_refuses_checksum_mismatch(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path, sha256="0" * 64)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        predict.load_classifier(
            fixture["selection_path"],
            path_root=fixture["path_root"],
            config_path=fixture["config_path"],
        )


def test_load_classifier_refuses_class_order_mismatch(tmp_path: Path) -> None:
    fixture = build_fixture(
        tmp_path, class_order=["glass", "organic", "paper", "plastic"]
    )
    with pytest.raises(RuntimeError, match="class order disagrees"):
        predict.load_classifier(
            fixture["selection_path"],
            path_root=fixture["path_root"],
            config_path=fixture["config_path"],
        )


def test_missing_model_reports_exact_placement(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path, model_on_disk=False, run_id="finetune_missing")
    with pytest.raises(FileNotFoundError) as excinfo:
        predict.load_classifier(
            fixture["selection_path"],
            path_root=fixture["path_root"],
            config_path=fixture["config_path"],
        )
    message = str(excinfo.value)
    assert str(fixture["model_path"]) in message
    assert "model_finetune_missing.zip" in message
    assert "Git-ignored" in message


def test_missing_selection_record_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Selection record not found"):
        predict.read_selection_record(tmp_path / "nope.json")


def test_load_classifier_refuses_input_shape_mismatch(tmp_path: Path) -> None:
    # config demands 224px, the fixture model was built for 32px
    fixture = build_fixture(tmp_path, image_size=32)
    write_configs(tmp_path, image_size=224)
    with pytest.raises(RuntimeError, match="expects input"):
        predict.load_classifier(
            fixture["selection_path"],
            path_root=fixture["path_root"],
            config_path=fixture["config_path"],
        )


def test_load_classifier_refuses_output_units_mismatch(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path, units=5)
    with pytest.raises(RuntimeError, match="outputs 5 classes"):
        predict.load_classifier(
            fixture["selection_path"],
            path_root=fixture["path_root"],
            config_path=fixture["config_path"],
        )


def test_cli_help_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(SRC / "predict.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
    )
    assert completed.returncode == 0
    flat = " ".join(completed.stdout.split())
    assert "SELECTED model" in flat
    assert "--json" in flat


# ---------------------------------------------------------------------------
# The REAL selected artifact (read-only; Keras 3 environment only)
# ---------------------------------------------------------------------------

@requires_real_model
def test_real_selected_model_loads_and_predicts(monkeypatch) -> None:
    # The application must never open a dataset manifest (test set included).
    def explode(*args, **kwargs):
        raise AssertionError("src/predict.py must never open a dataset manifest")

    monkeypatch.setattr(data_pipeline, "load_manifest", explode)

    classifier = predict.load_classifier()
    record = json.loads(SELECTED_RECORD.read_text(encoding="utf-8"))
    assert classifier["model_sha256"] == record["selected_model"]["sha256"]
    assert classifier["class_order"] == record["class_order"]
    assert classifier["image_size"] == 224

    gradient = np.linspace(0, 255, 224 * 224 * 3, dtype=np.float32).reshape(
        224, 224, 3
    )
    first = predict.predict_image(classifier, gradient)
    second = predict.predict_image(classifier, gradient)

    assert list(first["scores"]) == CLASS_ORDER
    assert first["category"] in CLASS_ORDER
    assert sum(first["scores"].values()) == pytest.approx(1.0, abs=1e-4)
    assert first["confidence"] == pytest.approx(max(first["scores"].values()))
    assert first["scores"] == second["scores"]  # deterministic, training=False


@requires_real_model
def test_cli_predicts_with_the_real_model(tmp_path: Path) -> None:
    image_path = tmp_path / "synthetic.png"
    Image.new("RGB", (64, 48), (40, 120, 200)).save(image_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SRC / "predict.py"),
            str(image_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["category"] in CLASS_ORDER
    assert set(payload["scores"]) == set(CLASS_ORDER)
    assert payload["model"]["sha256"].startswith("39f7b78b")
    assert "non-finite" not in completed.stderr
