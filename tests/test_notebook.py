"""Structural validation of notebooks/train_colab.ipynb.

The notebook can only *really* run inside Google Colab, so what we verify
locally is everything that can break without a GPU:

- valid notebook JSON (nbformat 4.x, unique cell ids, empty outputs),
- every code cell is syntactically valid Python (Colab magic/shell lines
  stripped before parsing),
- required topics are present (Drive persistence, pinned revision, the
  lock-file prohibition, manifest validation, accelerator check, training
  invocation, report packaging, honesty statements),
- forbidden content is absent (nothing in a code cell ever installs
  requirements.lock.txt).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "train_colab.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cell_sources(nb: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in nb["cells"]]


def code_cells(nb: dict) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    ]


def stripped_python(source: str) -> str:
    """Drop Colab shell (!) and magic (%) lines, then de-indent is unnecessary
    (notebook cells are already top-level)."""
    kept = [
        line
        for line in source.splitlines()
        if not line.lstrip().startswith(("!", "%"))
    ]
    return "\n".join(kept)


def test_notebook_exists_and_is_valid_json() -> None:
    assert NOTEBOOK.is_file(), "notebooks/train_colab.ipynb is missing"
    nb = load_notebook()
    assert nb["nbformat"] == 4
    assert isinstance(nb["nbformat_minor"], int)
    assert len(nb["cells"]) >= 20, "notebook looks too short for the full workflow"

    ids = [cell.get("id") for cell in nb["cells"]]
    assert all(isinstance(cell_id, str) and cell_id for cell_id in ids)
    assert len(ids) == len(set(ids)), "cell ids must be unique"

    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        assert all(isinstance(line, str) for line in cell["source"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "committed notebooks must not carry outputs"
            assert cell["execution_count"] is None


def test_every_code_cell_parses_as_python() -> None:
    nb = load_notebook()
    for index, source in enumerate(code_cells(nb)):
        try:
            ast.parse(stripped_python(source))
        except SyntaxError as exc:
            raise AssertionError(f"code cell {index} has a syntax error: {exc}") from exc


def test_required_topics_are_present() -> None:
    nb = load_notebook()
    everything = "\n".join(cell_sources(nb))
    code = "\n".join(code_cells(nb))

    required_in_notebook = {
        "Drive mount": "drive.mount",
        "revision pinning": "REVISION",
        "lock-file warning": "requirements.lock.txt",
        "clone step": "git",
        "download script": "download_data.py",
        "inspect script": "inspect_data.py",
        "checksum source": "split_summary.json",
        "shared loader": "load_splits",
        "train/validation only": 'splits=("train", "validation")',
        "accelerator check": "nvidia-smi",
        "training script": "train.py",
        "hash seed": "PYTHONHASHSEED",
        "history file": "history.csv",
        "run metadata": "run_metadata.json",
        "report export": "--export-reports",
        "model file": "best_model.keras",
        "not-run honesty": "not been executed in Google Colab",
        "validation-only honesty": "validation** metrics only",
        "local verification command": "tf.keras.models.load_model",
    }
    for label, needle in required_in_notebook.items():
        assert needle in everything, f"missing required topic: {label!r} ({needle!r})"

    required_in_code = {
        "Drive mount call": "drive.mount",
        "manifest validation call": "load_splits",
        "training subprocess": "train.py",
        "report export call": "--export-reports",
    }
    for label, needle in required_in_code.items():
        assert needle in code, f"missing required code step: {label!r}"


def test_code_cells_never_install_the_windows_lock_file() -> None:
    for source in code_cells(load_notebook()):
        assert "requirements.lock.txt" not in source, (
            "a code cell references requirements.lock.txt — the lock file has "
            "Windows-only pins and must never be installed in Colab"
        )


def test_notebook_contains_no_accuracy_claims() -> None:
    """Guard against invented metrics sneaking into the notebook text."""
    everything = "\n".join(cell_sources(load_notebook()))
    # Flag things like "accuracy of 0.87" / "87% accuracy" phrasings.
    suspicious = re.findall(
        r"(?:accuracy|accurate)[^\n]{0,30}?\b\d{1,2}(?:\.\d+)?\s*%|"
        r"\b\d\.\d{2}\s+(?:accuracy|acc\b)",
        everything,
        flags=re.IGNORECASE,
    )
    assert suspicious == [], f"notebook appears to state accuracy numbers: {suspicious}"
