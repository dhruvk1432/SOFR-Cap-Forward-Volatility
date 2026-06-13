import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTS = {".py", ".md", ".tex", ".toml", ".txt", ".ini"}
SKIP_PARTS = {".git", ".codex", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"}
CHECKER_FILES = {
    Path("tests/test_public_hygiene.py"),
    Path("scripts/06_final_artifact_check.py"),
}


def test_no_private_or_raw_files_are_tracked():
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=True)
    tracked = result.stdout.splitlines()
    forbidden = [
        ".codex/",
        ".env",
        "project_cap_vol_ts.xlsx",
        "cap_curves_2025-06-30.xlsx",
        "ref_rates.xlsx",
    ]
    for path in tracked:
        assert not any(path == item or path.startswith(item) for item in forbidden)


def test_public_text_has_no_local_private_paths_or_sensitive_tokens():
    patterns = [
        re.compile(r"/Users/"),
        re.compile(r"api[_-]?key", re.I),
        re.compile(r"secret", re.I),
        re.compile(r"password", re.I),
        re.compile(r"NotebookLM", re.I),
        re.compile(r"Options_Portfolio_Model"),
    ]
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.is_dir() or path.suffix not in TEXT_EXTS or relative in CHECKER_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            assert not pattern.search(text), f"{pattern.pattern} found in {relative}"
