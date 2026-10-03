"""Tests for the --metadata-dir runtime output option (download + inspect).

Motivation: every Colab run used to rewrite tracked data/metadata/* reports
(regenerated timestamps), which made the checkout dirty and set
`uncommitted_changes=True` in run metadata. These tests lock the new behavior:

- the default output location is still the tracked data/metadata directory
  (existing local workflows are unchanged),
- --metadata-dir redirects all reports to a caller-chosen directory,
- writing metadata outside the repo does not crash on path printing.

Everything runs on temp directories; nothing here touches the network, the
dataset, or the committed manifests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import download_data  # noqa: E402
import inspect_data  # noqa: E402


def test_inspect_default_output_is_the_tracked_directory() -> None:
    args = inspect_data.parse_args([])
    assert args.metadata_dir == inspect_data.METADATA_DIR

    outputs = inspect_data.output_paths(args.metadata_dir)
    # The helper must agree with the documented tracked defaults.
    assert outputs["summary"] == inspect_data.SUMMARY_PATH
    assert outputs["invalid"] == inspect_data.INVALID_PATH
    assert outputs["chart"] == inspect_data.CHART_PATH
    for path in outputs.values():
        assert path.parent == ROOT / "data" / "metadata"


def test_inspect_metadata_dir_flag_redirects_all_reports(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime_metadata"
    args = inspect_data.parse_args(["--metadata-dir", str(runtime_dir)])
    assert args.metadata_dir == runtime_dir

    outputs = inspect_data.output_paths(args.metadata_dir)
    assert set(outputs) == {"summary", "invalid", "chart"}
    assert outputs["summary"].name == "inspection_summary.json"
    assert outputs["invalid"].name == "invalid_images.json"
    assert outputs["chart"].name == "class_distribution.png"
    for path in outputs.values():
        assert path.parent == runtime_dir
        # Redirected reports must leave the tracked directory untouched.
        assert not str(path).startswith(str(ROOT / "data" / "metadata"))


def test_download_default_and_flag() -> None:
    args = download_data.parse_args([])
    assert args.metadata_dir == download_data.METADATA_DIR

    runtime_dir = Path("/content/drive/MyDrive/outputs/runtime_metadata")
    args = download_data.parse_args(["--metadata-dir", str(runtime_dir)])
    assert args.metadata_dir == runtime_dir
    # The filename still comes from the tracked default constant.
    assert (args.metadata_dir / download_data.DOWNLOAD_META_PATH.name).name == (
        "download_metadata.json"
    )


def test_write_download_metadata_outside_repo_does_not_crash(tmp_path: Path) -> None:
    """Regression: the final print used path.relative_to(REPO_ROOT), which
    raised ValueError when --metadata-dir pointed outside the repo (Colab)."""
    outside = tmp_path / "runtime_metadata" / "download_metadata.json"
    assert ROOT not in outside.parents  # sanity: truly outside the repo

    download_data.write_download_metadata(
        path=outside,
        url="https://example.invalid/realwaste.zip",
        verification={"checked": True},
        download_result={
            "archive_path": "data/raw/realwaste.zip",
            "bytes": 123,
            "action": "reused_existing_archive",
            "downloaded_now": False,
        },
        sha256="0" * 64,
        extract_result={"action": "reused_existing_extraction"},
        dataset_root=None,
    )

    assert outside.is_file()
    payload = json.loads(outside.read_text(encoding="utf-8"))
    assert payload["dataset"] == "RealWaste"
    assert payload["retrieved_at_utc"]
    assert payload["checksum"]["locally_computed_sha256"] == "0" * 64


def test_both_script_help_exits_zero() -> None:
    for script in ("inspect_data.py", "download_data.py"):
        completed = subprocess.run(
            [sys.executable, str(SRC / script), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=ROOT,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--metadata-dir" in completed.stdout
