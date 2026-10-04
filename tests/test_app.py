"""Streamlit app tests (milestone 11): upload flow, guards, verification.

Two layers:
- fixture tests point the app at a tiny untrained model through the
  WASTE_SELECTION_RECORD override, so they run in BOTH environments and never
  touch the real artifact;
- `requires_real_model` tests exercise the committed selection record and run
  only where Keras 3 (.venv-infer) and the model file exist.

Predictions of synthetic/fixture images are checked for *structure* only —
no accuracy is claimed anywhere in this file.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import keras  # noqa: E402
from test_predict import build_fixture  # noqa: E402
from test_train import CLASS_ORDER  # noqa: E402

KERAS3 = int(keras.__version__.split(".")[0]) >= 3
APP_PATH = ROOT / "app" / "app.py"
SELECTED_RECORD = ROOT / "models" / "metadata" / "selection" / "selected_model.json"
SELECTED_MODEL = ROOT / "models" / "finetune_20261004_133620" / "best_model.keras"
EVALUATION_REPORT = ROOT / "models" / "metadata" / "evaluation" / "test_evaluation.json"

requires_real_model = pytest.mark.skipif(
    not KERAS3 or not SELECTED_MODEL.is_file() or not SELECTED_RECORD.is_file(),
    reason=(
        "requires Keras 3 (.venv-infer) and the restored selected artifact "
        "models/finetune_20261004_133620/best_model.keras"
    ),
)


def jpeg_bytes(color, size=(120, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# The app always resolves configs/training.json against the repo root, so the
# fixture model must match the COMMITTED image_size (224) to pass load checks.
REPO_IMAGE_SIZE = int(
    json.loads((ROOT / "configs" / "training.json").read_text(encoding="utf-8"))[
        "image_size"
    ]
)


@pytest.fixture(scope="session")
def shared_selection(tmp_path_factory) -> Path:
    """One tiny untrained 224px model + selection record for all UI tests.

    Built once per session (session-scoped tmp dir), reused by every test:
    the app's `st.cache_resource` then loads it exactly once as well.
    """
    tmp = tmp_path_factory.mktemp("app_selection")
    fixture = build_fixture(
        tmp, image_size=REPO_IMAGE_SIZE, run_id="finetune_fixture"
    )
    return fixture["selection_path"]


def open_app(timeout: int = 180) -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=timeout)


def session_get(at: AppTest, key: str):
    """SafeSessionState has no .get(); probe missing keys with try/except."""
    try:
        return at.session_state[key]
    except KeyError:
        return None


def error_text(at: AppTest) -> str:
    return "\n".join(element.value for element in at.error)


def assert_clean(at: AppTest) -> None:
    assert not [e for e in at.exception], [str(e.value) for e in at.exception]


# ---------------------------------------------------------------------------
# Startup and model-availability paths (tiny fixture model)
# ---------------------------------------------------------------------------

def test_app_starts_with_the_required_elements(
    shared_selection, monkeypatch
) -> None:
    monkeypatch.setenv("WASTE_SELECTION_RECORD", str(shared_selection))

    at = open_app()
    at.run()
    assert_clean(at)
    assert not at.error

    assert at.title[0].value == "AI Waste Classification Assistant"
    assert len(at.file_uploader) == 1
    assert "Upload an image" in at.file_uploader[0].label
    # all four categories are explained on the page
    markdown = " ".join(element.value for element in at.markdown)
    for name in CLASS_ORDER:
        assert f"**{name}**" in markdown
    # classify button must not exist before anything is uploaded
    assert len(at.button) == 0
    # footer identifies the loaded artifact
    captions = " ".join(element.value for element in at.caption)
    assert "finetune_fixture" in captions
    assert "metal, organic, paper, plastic" in captions


def test_app_shows_placement_instructions_when_model_missing(
    monkeypatch, tmp_path: Path
) -> None:
    fixture = build_fixture(
        tmp_path,
        image_size=REPO_IMAGE_SIZE,
        model_on_disk=False,
        run_id="finetune_missing",
    )
    monkeypatch.setenv("WASTE_SELECTION_RECORD", str(fixture["selection_path"]))

    at = open_app()
    at.run()
    assert_clean(at)

    message = error_text(at)
    assert str(fixture["model_path"]) in message
    assert "model_finetune_missing.zip" in message
    assert "Git-ignored" in message
    # inference is disabled: no uploader, no classify button
    assert len(at.file_uploader) == 0
    assert len(at.button) == 0


def test_app_reports_a_missing_selection_record(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "WASTE_SELECTION_RECORD", str(tmp_path / "nowhere" / "selected_model.json")
    )

    at = open_app()
    at.run()
    assert_clean(at)
    assert "Selection record not found" in error_text(at)
    assert len(at.button) == 0


def test_app_hides_performance_numbers_of_a_different_model(
    shared_selection, monkeypatch
) -> None:
    """Tiny fixture model: recorded test metrics belong to another sha256."""
    monkeypatch.setenv("WASTE_SELECTION_RECORD", str(shared_selection))

    at = open_app()
    at.run()
    assert_clean(at)

    assert len(at.expander) == 1
    assert (
        "Model performance details (RealWaste held-out test results)"
        in at.expander[0].label
    )
    captions = " ".join(element.value for element in at.caption)
    assert "different model" in captions
    # no fabricated numbers: no test metrics are rendered for this model
    assert [m.label for m in at.metric] == []


# ---------------------------------------------------------------------------
# Upload -> classify flow
# ---------------------------------------------------------------------------

def test_classify_flow_shows_result_for_the_uploaded_image(
    shared_selection, monkeypatch
) -> None:
    monkeypatch.setenv("WASTE_SELECTION_RECORD", str(shared_selection))

    at = open_app()
    at.run()

    at.file_uploader[0].set_value(
        ("sample.jpg", jpeg_bytes((30, 120, 200)), "image/jpeg")
    )
    at.run()
    assert_clean(at)
    assert len(at.button) == 1 and at.button[0].label == "Classify waste"
    assert len(at.image) >= 1  # preview of the decoded upload
    assert session_get(at, "prediction") is None  # nothing classified yet

    at.button[0].click().run()
    assert_clean(at)
    assert not at.error

    prediction = session_get(at, "prediction")
    assert prediction is not None
    assert prediction["category"] in CLASS_ORDER
    assert set(prediction["scores"]) == set(CLASS_ORDER)
    assert sum(prediction["scores"].values()) == pytest.approx(1.0, abs=1e-4)

    assert len(at.success) == 1
    assert "Predicted category" in at.success[0].value
    assert prediction["category"] in at.success[0].value
    metric_labels = [metric.label for metric in at.metric]
    assert "Model confidence" in metric_labels
    confidence = next(m for m in at.metric if m.label == "Model confidence")
    assert confidence.value.endswith("%")


def test_new_upload_clears_the_stale_prediction(
    shared_selection, monkeypatch
) -> None:
    monkeypatch.setenv("WASTE_SELECTION_RECORD", str(shared_selection))

    at = open_app()
    at.run()
    at.file_uploader[0].set_value(
        ("first.jpg", jpeg_bytes((30, 120, 200), size=(120, 80)), "image/jpeg")
    )
    at.run()
    at.button[0].click().run()
    first = session_get(at, "prediction")
    assert first is not None and first["decoded_size"] == {"width": 120, "height": 80}

    # a DIFFERENT file arrives: the old result must disappear immediately
    at.file_uploader[0].set_value(
        ("second.jpg", jpeg_bytes((200, 30, 30), size=(60, 60)), "image/jpeg")
    )
    at.run()
    assert_clean(at)
    assert session_get(at, "prediction") is None
    assert len(at.success) == 0
    assert [m.label for m in at.metric] == []

    # classifying the new image produces a result for THAT image
    at.button[0].click().run()
    second = session_get(at, "prediction")
    assert second is not None
    assert second["decoded_size"] == {"width": 60, "height": 60}
    assert len(at.success) == 1


def test_rejects_corrupt_upload_instead_of_classifying_it(
    shared_selection, monkeypatch
) -> None:
    monkeypatch.setenv("WASTE_SELECTION_RECORD", str(shared_selection))

    at = open_app()
    at.run()
    # file name says .png, the bytes are HTML garbage: bytes win over name
    at.file_uploader[0].set_value(
        ("page.png", b"<html><body>not an image</body></html>", "image/png")
    )
    at.run()
    assert_clean(at)

    message = error_text(at)
    assert "That file cannot be used" in message
    assert "not a recognizable image" in message
    assert session_get(at, "prediction") is None
    assert len(at.button) == 0  # nothing to classify


def test_rejects_upload_above_the_size_limit(
    shared_selection, monkeypatch
) -> None:
    monkeypatch.setenv("WASTE_SELECTION_RECORD", str(shared_selection))

    at = open_app()
    at.run()
    oversized = b"x" * (10 * 1024 * 1024 + 1234)
    at.file_uploader[0].set_value(("big.png", oversized, "image/png"))
    at.run()
    assert_clean(at)

    message = error_text(at)
    assert "the limit is 10 MB" in message
    assert "smaller image" in message
    assert len(at.button) == 0


def test_exif_rotated_upload_is_upright_in_the_result(
    shared_selection, monkeypatch
) -> None:
    """Orientation 6 photo: the preview/result use the rotated dimensions."""
    monkeypatch.setenv("WASTE_SELECTION_RECORD", str(shared_selection))

    source = Image.new("RGB", (40, 20), (10, 160, 60))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=95, exif=exif)

    at = open_app()
    at.run()
    at.file_uploader[0].set_value(("phone.jpg", buffer.getvalue(), "image/jpeg"))
    at.run()
    assert_clean(at)
    assert len(at.button) == 1

    at.button[0].click().run()
    assert_clean(at)
    prediction = session_get(at, "prediction")
    assert prediction is not None
    assert prediction["decoded_size"] == {"width": 20, "height": 40}  # swapped


# ---------------------------------------------------------------------------
# The committed selection record end to end (Keras 3 environment only)
# ---------------------------------------------------------------------------

@requires_real_model
def test_real_model_app_end_to_end(monkeypatch) -> None:
    monkeypatch.delenv("WASTE_SELECTION_RECORD", raising=False)

    at = open_app(timeout=300)
    at.run()
    assert_clean(at)
    assert not at.error

    # footer identifies the committed artifact
    captions = " ".join(element.value for element in at.caption)
    assert "finetune_20261004_133620" in captions

    # recorded test metrics are shown ONLY because sha256 matches
    assert len(at.expander) == 1
    report = json.loads(EVALUATION_REPORT.read_text(encoding="utf-8"))
    labels = [metric.label for metric in at.metric]
    assert "Test accuracy" in labels and "Test macro F1" in labels
    accuracy = next(m for m in at.metric if m.label == "Test accuracy")
    assert accuracy.value == f"{report['metrics']['accuracy']:.4f}"
    macro_f1 = next(m for m in at.metric if m.label == "Test macro F1")
    assert macro_f1.value == f"{report['metrics']['macro_f1']:.4f}"

    # upload -> classify with the real selected model
    at.file_uploader[0].set_value(
        ("waste.jpg", jpeg_bytes((90, 60, 30), size=(160, 120)), "image/jpeg")
    )
    at.run()
    assert_clean(at)
    assert len(at.button) == 1

    at.button[0].click().run()
    assert_clean(at)
    assert not at.error
    prediction = session_get(at, "prediction")
    assert prediction is not None
    assert prediction["category"] in CLASS_ORDER
    assert set(prediction["scores"]) == set(CLASS_ORDER)
    assert sum(prediction["scores"].values()) == pytest.approx(1.0, abs=1e-4)
    assert prediction["model"]["sha256"] == json.loads(
        SELECTED_RECORD.read_text(encoding="utf-8")
    )["selected_model"]["sha256"]
    assert len(at.success) == 1
