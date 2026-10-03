"""Download and safely extract the RealWaste dataset from UCI.

Milestone 3 only: fetch the archive, record checksums, extract images.
Do not split, augment, or train.

Why this script exists
----------------------
Hand-downloading a 656 MB zip is slow and easy to get wrong. This script:
1. Checks the official UCI download URL is reachable.
2. Streams the file to a temporary path, then renames it (safe against partial files).
3. Computes a local SHA-256 so you can re-check integrity later.
4. Extracts only paths that stay inside the destination folder (blocks zip-slip).
5. Reuses an existing valid download unless you pass --force.

Publisher checksum note
-----------------------
UCI's RealWaste page does not publish a SHA-256 for the zip. We therefore record
a LOCALLY computed checksum only, and set publisher_checksum to null with an
explicit note. We never invent a publisher value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Official UCI static download path for dataset id 908 (RealWaste).
# Verified on 2026-10-02 against https://archive.ics.uci.edu/dataset/908/realwaste
# (page lists "Download (656.6 MB)") and by a ranged GET that returned ZIP magic bytes.
OFFICIAL_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
DATASET_PAGE_URL = "https://archive.ics.uci.edu/dataset/908/realwaste"
DATASET_DOI = "https://doi.org/10.24432/C5SS4G"

# Project paths (relative to repository root)
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "realwaste.zip"
TEMP_ARCHIVE_PATH = RAW_DIR / "realwaste.zip.part"
METADATA_DIR = REPO_ROOT / "data" / "metadata"
DOWNLOAD_META_PATH = METADATA_DIR / "download_metadata.json"

# Network settings: stream in chunks; fail clearly if the server stalls.
CHUNK_SIZE = 1024 * 1024  # 1 MiB
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 60
USER_AGENT = "ai-waste-classification-assistant/1.0 (college AIML PBL; educational use)"


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_request(url: str, method: str = "GET", headers: dict | None = None) -> urllib.request.Request:
    """Create a urllib request with a stable User-Agent."""
    all_headers = {"User-Agent": USER_AGENT}
    if headers:
        all_headers.update(headers)
    return urllib.request.Request(url, method=method, headers=all_headers)


def verify_download_url(url: str) -> dict:
    """Confirm the UCI archive URL responds and looks like a zip file.

    Uses a 1-byte ranged GET instead of trusting a guessed path alone.
    Returns a small report dict for metadata.
    """
    print(f"[verify] Checking official download URL: {url}")
    request = build_request(url, method="GET", headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(request, timeout=CONNECT_TIMEOUT_SECONDS) as response:
            status = response.status
            final_url = response.geturl()
            # ZIP files start with the local file header signature PK\x03\x04
            first_bytes = response.read(4)
            content_type = response.headers.get("Content-Type")
            content_disposition = response.headers.get("Content-Disposition")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP error {exc.code} while verifying {url}. "
            "The UCI path may have changed — check the dataset page."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error while verifying {url}: {exc.reason}. "
            "Check your internet connection or campus firewall."
        ) from exc

    looks_like_zip = first_bytes == b"PK\x03\x04"
    report = {
        "url": url,
        "http_status": status,
        "final_url": final_url,
        "content_type": content_type,
        "content_disposition": content_disposition,
        "zip_magic_ok": looks_like_zip,
        "checked_at_utc": utc_now_iso(),
    }
    print(f"[verify] HTTP {status}, final URL={final_url}, zip_magic_ok={looks_like_zip}")
    if status != 200:
        raise RuntimeError(f"Unexpected HTTP status {status} for {url}")
    if not looks_like_zip:
        raise RuntimeError(
            f"Response for {url} did not start with ZIP magic bytes. "
            "Refusing to download a non-zip file."
        )
    return report


def download_archive(url: str, dest: Path, temp_dest: Path, force: bool = False) -> dict:
    """Stream the archive to a temporary file, then rename it into place.

    Partial downloads stay as *.part and are removed on failure so a later
    run does not treat a truncated file as complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        size = dest.stat().st_size
        print(f"[download] Reusing existing archive: {dest} ({size:,} bytes). Use --force to re-download.")
        return {
            "action": "reused_existing_archive",
            "archive_path": str(dest.relative_to(REPO_ROOT).as_posix()),
            "bytes": size,
            "downloaded_now": False,
        }

    if temp_dest.exists():
        print(f"[download] Removing leftover partial file: {temp_dest}")
        temp_dest.unlink()

    print(f"[download] Streaming {url}")
    print(f"[download] Temporary file: {temp_dest}")
    request = build_request(url, method="GET")
    bytes_written = 0
    try:
        with urllib.request.urlopen(request, timeout=CONNECT_TIMEOUT_SECONDS) as response, open(
            temp_dest, "wb"
        ) as handle:
            # Some servers omit Content-Length; still stream until EOF.
            total_header = response.headers.get("Content-Length")
            expected_total = int(total_header) if total_header and total_header.isdigit() else None
            if expected_total:
                print(f"[download] Expected size: {expected_total:,} bytes")
            else:
                print("[download] Server did not send Content-Length; streaming until done.")

            while True:
                # read() can block; urllib timeout applies to socket operations.
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
                if expected_total:
                    percent = 100.0 * bytes_written / expected_total
                    print(
                        f"\r[download] {bytes_written:,} / {expected_total:,} bytes ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r[download] {bytes_written:,} bytes", end="", flush=True)
            print()  # newline after progress
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        if temp_dest.exists():
            temp_dest.unlink()
        raise RuntimeError(
            f"Download failed after {bytes_written:,} bytes: {exc}. "
            f"Partial file removed. Re-run the script to try again."
        ) from exc

    if bytes_written == 0:
        temp_dest.unlink(missing_ok=True)
        raise RuntimeError("Download finished with 0 bytes. Re-run the script to try again.")

    # Atomic-ish publish: only rename after a complete stream.
    if dest.exists():
        dest.unlink()
    temp_dest.rename(dest)
    print(f"[download] Saved archive: {dest} ({bytes_written:,} bytes)")
    return {
        "action": "downloaded",
        "archive_path": str(dest.relative_to(REPO_ROOT).as_posix()),
        "bytes": bytes_written,
        "downloaded_now": True,
    }


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file by streaming (memory-safe for large archives)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_safe_zip_member(member_name: str, dest_dir: Path) -> bool:
    """Return True if extracting member_name would stay inside dest_dir.

    Blocks absolute paths and '..' traversal (zip-slip).
    """
    # Normalize zip member separators
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return False  # absolute path
    candidate = (dest_dir / normalized).resolve()
    dest_resolved = dest_dir.resolve()
    try:
        candidate.relative_to(dest_resolved)
        return True
    except ValueError:
        return False


def safe_extract(archive_path: Path, dest_dir: Path) -> dict:
    """Extract zip members only if each path stays under dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    rejected: list[str] = []
    extracted_count = 0

    print(f"[extract] Extracting {archive_path} -> {dest_dir}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        # Test archive integrity first (bad CRCs show up here when possible)
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(
                f"Archive failed integrity check at member: {bad_member}. "
                "Delete the zip and re-run with --force to download again."
            )

        for info in archive.infolist():
            name = info.filename
            if name.endswith("/"):
                # Directory entry — create only if the path is safe
                if not is_safe_zip_member(name, dest_dir):
                    rejected.append(name)
                    continue
                target_dir = dest_dir / name.replace("\\", "/")
                target_dir.mkdir(parents=True, exist_ok=True)
                continue

            if not is_safe_zip_member(name, dest_dir):
                rejected.append(name)
                continue

            target_path = dest_dir / name.replace("\\", "/")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, open(target_path, "wb") as target:
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    target.write(chunk)
            extracted_count += 1

    print(f"[extract] Extracted {extracted_count} files; rejected {len(rejected)} unsafe path(s)")
    if rejected:
        preview = ", ".join(rejected[:5])
        print(f"[extract] Rejected examples: {preview}")
    return {
        "extracted_files": extracted_count,
        "rejected_paths": rejected,
        "rejected_count": len(rejected),
    }


def find_dataset_root(extract_dir: Path) -> Path | None:
    """Locate the folder that contains category subfolders with images.

    Does not assume a single hard-coded layout. Looks for any directory that
    directly contains subdirectories holding image files.
    """
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def dir_has_images(path: Path) -> bool:
        if not path.is_dir():
            return False
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in image_suffixes:
                return True
        return False

    # Prefer a directory named RealWaste if present (matches UCI card examples)
    candidates: list[Path] = []
    for path in [extract_dir, *extract_dir.rglob("*")]:
        if not path.is_dir():
            continue
        # Category folders are subdirectories of the dataset root
        subdirs = [c for c in path.iterdir() if c.is_dir()]
        if not subdirs:
            continue
        if any(dir_has_images(sub) for sub in subdirs):
            candidates.append(path)

    if not candidates:
        return None

    # Prefer shallowest path that has image-bearing category folders
    candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return candidates[0]


def extraction_looks_valid(extract_dir: Path) -> bool:
    """True if a dataset root with category folders already exists under extract_dir."""
    return find_dataset_root(extract_dir) is not None


def write_download_metadata(
    path: Path,
    url: str,
    verification: dict,
    download_result: dict,
    sha256: str,
    extract_result: dict,
    dataset_root: Path | None,
) -> None:
    """Persist small reproducibility metadata (tracked in Git)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "RealWaste",
        "source_page_url": DATASET_PAGE_URL,
        "source_download_url": url,
        "source_doi": DATASET_DOI,
        "retrieved_at_utc": utc_now_iso(),
        "retrieval_timestamp_note": (
            "Timestamp records when THIS machine retrieved the archive. "
            "It does not prove the dataset content changed."
        ),
        "archive": {
            "path": download_result.get("archive_path"),
            "bytes": download_result.get("bytes"),
            "action": download_result.get("action"),
            "downloaded_now": download_result.get("downloaded_now"),
        },
        "checksum": {
            "algorithm": "SHA-256",
            "locally_computed_sha256": sha256,
            "publisher_checksum": None,
            "publisher_checksum_note": (
                "UCI's RealWaste page did not publish a checksum for the zip "
                "when this metadata was written. publisher_checksum is null "
                "on purpose — do not invent one."
            ),
        },
        "url_verification": verification,
        "extraction": extract_result,
        "dataset_root": (
            str(dataset_root.relative_to(REPO_ROOT).as_posix()) if dataset_root else None
        ),
        "git_policy": (
            "Archives and extracted images stay ignored by Git. "
            "Only this small metadata file is tracked."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        shown = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # --metadata-dir may point outside the repo (e.g. a Colab Drive
        # runtime directory); show the full path instead of crashing.
        shown = path.as_posix()
    print(f"[metadata] Wrote {shown}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract RealWaste into data/raw/ (no training)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if a valid archive/folder already exists.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download/reuse archive only; do not extract (advanced/debug).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help=(
            "Where download_metadata.json is written (default: data/metadata, "
            "tracked in Git). Pass a runtime directory outside the clone "
            "(e.g. on Drive in Colab) to avoid overwriting tracked metadata."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== RealWaste download (milestone 3) ===")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Official URL: {OFFICIAL_DOWNLOAD_URL}")

    try:
        verification = verify_download_url(OFFICIAL_DOWNLOAD_URL)
        download_result = download_archive(
            OFFICIAL_DOWNLOAD_URL,
            dest=ARCHIVE_PATH,
            temp_dest=TEMP_ARCHIVE_PATH,
            force=args.force,
        )

        if not ARCHIVE_PATH.exists():
            print("[error] Archive missing after download step.", file=sys.stderr)
            return 1

        print("[checksum] Computing local SHA-256 (streaming)...")
        sha256 = compute_sha256(ARCHIVE_PATH)
        print(f"[checksum] locally_computed_sha256 = {sha256}")
        print("[checksum] publisher_checksum = null (UCI page did not publish one)")

        if args.skip_extract:
            extract_result = {"skipped": True, "extracted_files": 0, "rejected_paths": []}
            dataset_root = None
        elif extraction_looks_valid(RAW_DIR) and not args.force:
            # Reuse existing extraction unless the user asked for a fresh run.
            dataset_root = find_dataset_root(RAW_DIR)
            rel = dataset_root.relative_to(REPO_ROOT).as_posix() if dataset_root else None
            print(f"[extract] Reusing existing extraction under data/raw/ (root={rel}). Use --force to re-extract.")
            extract_result = {
                "action": "reused_existing_extraction",
                "extracted_files": 0,
                "rejected_paths": [],
                "rejected_count": 0,
            }
            if dataset_root is not None:
                try:
                    names = sorted(c.name for c in dataset_root.iterdir() if c.is_dir())
                    print(f"[extract] Category-like folders found ({len(names)}): {names}")
                except OSError as exc:
                    print(f"[extract] Could not list folders: {exc}", file=sys.stderr)
        else:
            extract_result = safe_extract(ARCHIVE_PATH, RAW_DIR)
            dataset_root = find_dataset_root(RAW_DIR)
            if dataset_root is None:
                print(
                    "[extract] WARNING: could not locate category folders with images. "
                    "Inspect data/raw/ manually.",
                    file=sys.stderr,
                )
            else:
                rel = dataset_root.relative_to(REPO_ROOT).as_posix()
                print(f"[extract] Detected dataset root: {rel}")
                # Show top-level category folder names without assuming them
                try:
                    names = sorted(
                        c.name for c in dataset_root.iterdir() if c.is_dir()
                    )
                    print(f"[extract] Category-like folders found ({len(names)}): {names}")
                except OSError as exc:
                    print(f"[extract] Could not list folders: {exc}", file=sys.stderr)

        write_download_metadata(
            path=args.metadata_dir / DOWNLOAD_META_PATH.name,
            url=OFFICIAL_DOWNLOAD_URL,
            verification=verification,
            download_result=download_result,
            sha256=sha256,
            extract_result=extract_result,
            dataset_root=dataset_root,
        )

        print("=== Download step finished ===")
        print("Next: run src/inspect_data.py to count and validate images.")
        return 0
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[error] Download/extract failed. The local commit is unaffected. "
            "Fix the issue or follow manual steps in data/README.md.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
