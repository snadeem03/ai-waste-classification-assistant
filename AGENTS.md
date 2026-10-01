# AGENTS.md — rules for AI Waste Classification Assistant

These rules apply to **every** future milestone in this repository. Follow them exactly.

## Milestone discipline

- Complete **only** the requested milestone. Do not jump ahead, skip steps, or bundle unrelated work into one commit.
- Inspect existing files **before** editing them. Never overwrite work that is already correct.
- Preserve existing work. Do not re-run `git init`, recreate the virtual environment, or replace a working setup.

## Code and communication

- Keep code **beginner-friendly**: clear names, small functions, straightforward structure.
- Explain the **important choices** (why MobileNetV2, why a class is frozen, why NumPy is pinned, etc.) in comments or docs — but do not dump noise comments everywhere.
- Match the style of surrounding code when editing.

## Honesty and verification

- Verify functionality with appropriate checks (imports, pytest, a Streamlit smoke run, etc.) before claiming success.
- **Never invent training results**, accuracy numbers, or demo screenshots.
- **Never mark unfinished work complete.** If something is planned but not done, say so.
- If verification fails: fix the problem, or report the blocker clearly. Do **not** claim success.

## Progress tracking

- After each milestone, update `docs/progress.md` with:
  - what changed
  - how it was verified
  - blockers (if any)
  - next steps
- Keep progress notes factual. No aspirational "done" language for unverified work.

## Git hygiene

- Review `git status` and `git diff` **before staging**.
- Stage **only** milestone-related files. Do not stage `.venv/`, datasets, model binaries, caches, or secrets.
- Create a **meaningful commit message** after successful verification, then push to `origin/main`.
- **Never** force-push.
- **Never** delete unrelated work or rewrite history that others depend on.
- **Never** commit secrets, raw/processed datasets, downloaded archives, or trained model binaries (`.keras`, `.h5`, `.tflite`, SavedModel folders, etc.).
- If pushing fails: **preserve the local commit** and report the exact error. Do not abandon or rewrite the commit to "make push work."
- If Git identity or authentication is missing, explain what the student must configure. Do not invent identity details.

## Platform notes (this project)

- Windows + Python 3.11 virtual environment at `.venv/`.
- Always use `.venv\Scripts\python.exe` and `.venv\Scripts\python.exe -m pip` for Python and pip.
- `requirements.txt` holds direct dependencies; `requirements.lock.txt` holds the exact working local versions from `pip freeze`.
