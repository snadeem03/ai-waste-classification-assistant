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
  requirements.lock.txt),
- path safety: `REPO_DIR` is always a `pathlib.Path` (regression guard for
  the Colab `TypeError: unsupported operand type(s) for /: 'str' and 'str'`),
  `REPO_URL` stays a plain string, and subprocess `cwd=` values are strings.
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


# ---------------------------------------------------------------------------
# Regression: REPO_DIR must never be a plain string (Colab TypeError)
# ---------------------------------------------------------------------------


def _path_root_of(value: ast.AST) -> ast.AST:
    """Unwrap a chain of `/` operations down to its leftmost operand."""
    while isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div):
        value = value.left
    return value


def assignments_to(nb: dict, name: str) -> list[tuple[str, ast.AST]]:
    """All (cell_id, value) assignments to `name` across the code cells."""
    found: list[tuple[str, ast.AST]] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        found.append((cell["id"], node.value))
    return found


def test_repo_dir_is_always_built_with_path() -> None:
    """Colab failed with `TypeError: unsupported operand type(s) for /:
    'str' and 'str'` at `REPO_DIR / ".git"`.

    Every assignment to REPO_DIR must be rooted in a `Path(...)` call, so no
    cell can (re)define it as a plain string.
    """
    assignments = assignments_to(load_notebook(), "REPO_DIR")
    assert assignments, "notebook must define REPO_DIR somewhere"
    for cell_id, value in assignments:
        root = _path_root_of(value)
        assert (
            isinstance(root, ast.Call)
            and isinstance(root.func, ast.Name)
            and root.func.id == "Path"
        ), f"cell {cell_id}: REPO_DIR is not rooted in Path(...): {ast.dump(root)[:80]}"


def test_repo_dir_string_input_is_normalized_before_path_ops() -> None:
    """The clone cell must re-normalize with `Path(REPO_DIR)` *before* its
    first path operation, so a string left by an earlier/edited cell works."""
    nb = load_notebook()
    normalization = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            is_path_of_self = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "Path"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id == "REPO_DIR"
            )
            if not is_path_of_self:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REPO_DIR":
                    normalization = node
    assert normalization is not None, (
        "no `REPO_DIR = Path(REPO_DIR)` normalization found in the notebook"
    )

    # Simulate the reported failure: REPO_DIR supplied as a plain string.
    namespace = {"Path": Path, "REPO_DIR": "/content/ai-waste-classification-assistant"}
    exec(
        compile(ast.Module(body=[normalization], type_ignores=[]), "<cell>", "exec"),
        namespace,
    )
    result = namespace["REPO_DIR"]
    assert isinstance(result, Path)
    # The exact operation that raised in Colab:
    assert isinstance(result / ".git", Path)


def test_repo_url_stays_string_and_cwd_is_passed_as_str() -> None:
    """REPO_URL remains a plain string (git argument) and external commands
    receive `str` working directories, never Path objects."""
    nb = load_notebook()
    url_assignments = assignments_to(nb, "REPO_URL")
    assert url_assignments, "notebook must define REPO_URL somewhere"
    for cell_id, value in url_assignments:
        assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
            f"cell {cell_id}: REPO_URL must stay a plain string constant"
        )

    cwd_checked = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "cwd":
                    continue
                cwd_checked += 1
                is_none = isinstance(kw.value, ast.Constant) and kw.value.value is None
                is_str_call = (
                    isinstance(kw.value, ast.Call)
                    and isinstance(kw.value.func, ast.Name)
                    and kw.value.func.id == "str"
                )
                assert is_none or is_str_call, (
                    f"{cell['id']}: cwd must be None (default) or str(...), "
                    f"got {ast.dump(kw.value)[:80]}"
                )
    assert cwd_checked >= 4, f"expected several cwd= call sites, saw {cwd_checked}"
