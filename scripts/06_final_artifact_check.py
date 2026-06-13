from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from _pipeline_common import ROOT, manifest_paths
from src.artifacts import write_manifest

FORBIDDEN_TRACKED = (
    ".codex/",
    ".env",
    "project_cap_vol_ts.xlsx",
    "cap_curves_2025-06-30.xlsx",
    "ref_rates.xlsx",
)
FORBIDDEN_TEXT_PATTERNS = (
    re.compile(r"/Users/"),
    re.compile(r"api[_-]?key", re.I),
    re.compile(r"secret", re.I),
    re.compile(r"password", re.I),
    re.compile(r"NotebookLM", re.I),
    re.compile(r"Options_Portfolio_Model"),
)
TEXT_EXTS = {".py", ".md", ".tex", ".toml", ".txt", ".ini"}
CHECKER_FILES = {"tests/test_public_hygiene.py", "scripts/06_final_artifact_check.py"}


def git_ls_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.splitlines()


def main() -> int:
    errors: list[str] = []
    tracked = git_ls_files()
    for path in tracked:
        if any(path == item or path.startswith(item) for item in FORBIDDEN_TRACKED):
            errors.append(f"forbidden tracked public artifact: {path}")
        full = ROOT / path
        if path in CHECKER_FILES:
            continue
        if full.suffix in TEXT_EXTS and full.exists():
            text = full.read_text(encoding="utf-8", errors="ignore")
            for pattern in FORBIDDEN_TEXT_PATTERNS:
                if pattern.search(text):
                    errors.append(f"forbidden public text pattern {pattern.pattern!r} in {path}")

    required_dirs = [ROOT / "tables", ROOT / "figures", ROOT / "paper" / "figures", ROOT / "paper" / "generated"]
    for directory in required_dirs:
        if not directory.exists():
            errors.append(f"missing artifact directory: {directory.relative_to(ROOT)}")

    artifact_paths = manifest_paths(ROOT)
    if not artifact_paths:
        errors.append("no artifact paths discovered for manifest")
    manifest = write_manifest(artifact_paths, ROOT, ROOT / "artifact_manifest.json")

    print("Final artifact check:")
    print(f"  tracked files scanned: {len(tracked)}")
    print(f"  artifacts hashed: {len(artifact_paths)}")
    print(f"  manifest: {manifest.relative_to(ROOT)}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Final artifact check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
