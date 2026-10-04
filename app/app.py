"""Streamlit UI for the AI Waste Classification Assistant (milestone 11).

Windows launch command (run from the repo root):

    .\\.venv-infer\\Scripts\\python.exe -m streamlit run app/app.py

The app serves ONLY the model named in the committed selection record
(`models/metadata/selection/selected_model.json`), loaded through
`src/predict.py`, which re-verifies the checksum and class order at load
time. Model loading is cached with `st.cache_resource` and keyed on the
selected artifact's identity (sha256 + size + mtime), so replacing the
selected model invalidates the cache automatically.

Privacy rules enforced here:
- uploads stay in memory for this browser session only: never written to
  disk, never passed to a cache decorator, never sent anywhere;
- a new upload clears the previous result, so a stale prediction can never
  be shown for a different image;
- there is no "unknown" class and no confidence threshold — the model
  always reports one of the four categories it was trained on.

Environment override (testing only): WASTE_SELECTION_RECORD points the app
at a different selection record; the default is always the committed one.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import predict  # noqa: E402

DEFAULT_SELECTION = REPO_ROOT / "models" / "metadata" / "selection" / "selected_model.json"
SELECTION_PATH = Path(os.environ.get("WASTE_SELECTION_RECORD") or DEFAULT_SELECTION)
EVALUATION_REPORT = (
    REPO_ROOT / "models" / "metadata" / "evaluation" / "test_evaluation.json"
)

# Upload guard rails (enforced twice: the widget rejects oversize files and
# the byte check below refuses anything that still slips through).
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_TYPES = ["jpg", "jpeg", "png"]

CATEGORY_HINTS = {
    "metal": "Cans, foil, metal scraps",
    "organic": "Food waste, leaves, biodegradable material",
    "paper": "Paper, cardboard, notebooks",
    "plastic": "Plastic bottles, containers, packaging",
}


# ---------------------------------------------------------------------------
# Cached data: selection record, model, recorded test metrics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_selection_record(path_str: str) -> dict:
    """The committed selection record (small JSON, safe to cache)."""
    return predict.read_selection_record(Path(path_str))


def selected_identity(selection: dict) -> str:
    """Identity of the selected artifact, used as part of the cache key.

    Includes the record's sha256 plus the model file's size and mtime, so
    both "the record changed" and "the file was replaced" trigger a reload.
    """
    chosen = selection.get("selected_model") or {}
    identity = f"{chosen.get('sha256', 'missing')}:{selection.get('created_at_utc', '')}"
    model_path = REPO_ROOT / str(chosen.get("path", ""))
    try:
        stat = model_path.stat()
    except OSError:
        return identity + ":file-missing"
    return f"{identity}:{stat.st_size}:{stat.st_mtime_ns}"


@st.cache_resource(show_spinner="Loading the selected model (first load takes a few seconds) ...")
def load_cached_classifier(path_str: str, identity: str) -> dict:
    """Load + verify the selected model once per process.

    `identity` participates in the cache key only: when the selected
    artifact changes (different sha256/size/mtime in the selection record),
    Streamlit discards this entry and `predict.load_classifier` runs again,
    checksum and class order included.
    """
    del identity  # used exclusively for cache keying
    return predict.load_classifier(Path(path_str))


@st.cache_data(show_spinner=False)
def read_test_metrics(path_str: str, expected_sha: str) -> dict:
    """Recorded held-out test metrics — only if they belong to THIS model."""
    path = Path(path_str)
    if not path.is_file():
        return {"available": False, "reason": f"no recorded report at {path}"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"could not read {path.name}: {exc}"}

    recorded_sha = report.get("model", {}).get("sha256")
    if recorded_sha != expected_sha:
        return {
            "available": False,
            "reason": "the recorded evaluation belongs to a different model",
        }
    metrics = report.get("metrics") or {}
    if "accuracy" not in metrics or "macro_f1" not in metrics:
        return {"available": False, "reason": "the recorded report has no metrics"}
    return {
        "available": True,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "images": int(metrics.get("images", 0)),
        "created_at_utc": report.get("created_at_utc", ""),
        "run_id": report.get("model", {}).get("run_id"),
        "report": str(path.relative_to(REPO_ROOT).as_posix())
        if path.is_relative_to(REPO_ROOT)
        else str(path),
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def clear_results() -> None:
    """Drop any prediction tied to a previous image (or no image at all)."""
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_error", None)
    st.session_state.pop("upload_fingerprint", None)


def render_category_hints() -> None:
    columns = st.columns(len(CATEGORY_HINTS))
    for column, (name, hint) in zip(columns, CATEGORY_HINTS.items()):
        with column:
            st.markdown(f"**{name}**")
            st.caption(hint)


def render_result(result: dict | None) -> None:
    if not result:
        return
    st.subheader("Result")
    st.success(f"Predicted category: **{result['category']}**")
    st.metric(label="Model confidence", value=f"{result['confidence']:.1%}")

    chart_data = pd.DataFrame(
        {"class score": [result["scores"][name] for name in result["class_order"]]},
        index=result["class_order"],
    )
    st.bar_chart(chart_data, height=220, width="stretch")

    st.caption(
        "Confidence is the model's highest class score. It does **not** "
        "guarantee the prediction is correct, and an object outside these "
        "four categories still receives one of these four labels — the app "
        "has no reliable unknown-object detection."
    )


def render_performance_details(expected_sha: str) -> None:
    with st.expander("Model performance details (RealWaste held-out test results)"):
        metrics = read_test_metrics(str(EVALUATION_REPORT), expected_sha)
        if not metrics.get("available"):
            st.caption(f"Recorded performance not shown: {metrics.get('reason')}.")
            return
        st.write(
            f"Measured once on the **RealWaste held-out test set** "
            f"({metrics['images']} images, {metrics['created_at_utc'][:10]}), "
            f"using the shared evaluation code — separate from the validation "
            f"numbers that selected this model."
        )
        left, right = st.columns(2)
        left.metric("Test accuracy", f"{metrics['accuracy']:.4f}")
        right.metric("Test macro F1", f"{metrics['macro_f1']:.4f}")
        st.caption(
            f"Source: {metrics['report']}. These describe the RealWaste test "
            "split only — not the image you uploaded, and not real-world "
            "sorting accuracy."
        )


def render_footer(classifier: dict) -> None:
    st.divider()
    st.caption(
        f"Model: {classifier['run_id']} · sha256 "
        f"{str(classifier['model_sha256'])[:16]}... · selected by "
        f"\"{classifier['selection_reason']}\" · classes: "
        f"{', '.join(classifier['class_order'])}"
    )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AI Waste Classification Assistant", layout="centered")
    st.title("AI Waste Classification Assistant")
    st.write(
        "Upload a photo of a waste item and the selected model predicts which "
        "of the four supported categories it belongs to."
    )
    render_category_hints()
    st.write(
        "**Upload one prominent waste item** — fill the frame with a single "
        "object on a simple background. JPG/JPEG/PNG, up to "
        f"{MAX_UPLOAD_MB} MB. The image stays in this browser session and is "
        "never saved to disk."
    )

    # --- model availability: clear instructions, then stop gracefully -----
    try:
        selection = read_selection_record(str(SELECTION_PATH))
        expected_sha = str((selection.get("selected_model") or {}).get("sha256", ""))
        classifier = load_cached_classifier(
            str(SELECTION_PATH), selected_identity(selection)
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Inference is disabled until the model file is in place. Nothing "
            "was changed on disk, and no other part of the project is "
            "affected — see README.md (Milestone 11) for the exact command "
            "and model placement."
        )
        st.stop()
    except (ValueError, KeyError, RuntimeError) as exc:
        st.error(f"The selected model could not be loaded:\n\n{exc}")
        st.stop()

    # --- upload ----------------------------------------------------------
    uploaded = st.file_uploader(
        "Upload an image of one waste item",
        type=UPLOAD_TYPES,
        accept_multiple_files=False,
        max_upload_size=MAX_UPLOAD_MB,
        help=(
            f"JPG/JPEG/PNG, up to {MAX_UPLOAD_MB} MB. Decoding, EXIF "
            "orientation and transparency are handled automatically."
        ),
    )

    if uploaded is None:
        clear_results()
        st.caption("No image selected yet — upload a JPG or PNG to begin.")
        render_performance_details(expected_sha)
        render_footer(classifier)
        return

    try:
        data = uploaded.getvalue()
    except Exception as exc:  # noqa: BLE001 - surface, do not crash the page
        clear_results()
        st.error(f"Could not read the uploaded file: {exc}")
        render_footer(classifier)
        return

    if len(data) > MAX_UPLOAD_BYTES:
        clear_results()
        st.error(
            f"That file is {len(data) / (1024 * 1024):.1f} MB — the limit is "
            f"{MAX_UPLOAD_MB} MB. Please upload a smaller image."
        )
        render_footer(classifier)
        return

    # Validate by ACTUALLY decoding the bytes (never by file name): this
    # rejects corrupt/unsupported files up front with a readable message and
    # lets the preview show the same orientation-corrected RGB the model sees.
    try:
        preview_image = predict.decode_image_bytes(data)
    except predict.InvalidImageError as exc:
        clear_results()
        st.error(f"That file cannot be used: {exc}")
        render_performance_details(expected_sha)
        render_footer(classifier)
        return

    # A different file (or a cleared/re-added upload) invalidates the old
    # result BEFORE anything is displayed, so a stale prediction can never
    # appear next to a new image.
    fingerprint = hashlib.sha256(data).hexdigest()
    if st.session_state.get("upload_fingerprint") != fingerprint:
        st.session_state["upload_fingerprint"] = fingerprint
        st.session_state.pop("prediction", None)
        st.session_state.pop("prediction_error", None)

    st.image(preview_image, caption=uploaded.name or "Uploaded image",
             width="stretch")

    if st.button("Classify waste", type="primary", width=True):
        try:
            result = predict.predict_image(classifier, data)
        except (predict.InvalidImageError, RuntimeError, TypeError, ValueError) as exc:
            st.session_state.pop("prediction", None)
            st.session_state["prediction_error"] = str(exc)
        else:
            st.session_state.pop("prediction_error", None)
            st.session_state["prediction"] = result

    if st.session_state.get("prediction_error"):
        st.error(
            "Could not classify that image: "
            f"{st.session_state['prediction_error']}"
        )

    render_result(st.session_state.get("prediction"))
    render_performance_details(expected_sha)
    render_footer(classifier)


if __name__ == "__main__":
    main()
