from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT / "paper" / "SOFR_Cap_Forward_Volatility.tex"

INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
REF_PATTERN = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}")


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        cleaned = []
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            cleaned.append(ch)
        out.append("".join(cleaned))
    return "\n".join(out)


def resolve(path_token: str, base_dir: Path, suffix: str | None = None) -> Path:
    rel = path_token.strip()
    candidate = base_dir / rel
    if suffix and not candidate.suffix:
        candidate = candidate.with_suffix(suffix)
    return candidate.resolve()


def main() -> int:
    errors: list[str] = []
    if not MAIN_TEX.exists():
        errors.append(f"missing main TeX file: {MAIN_TEX.relative_to(ROOT)}")
    text = strip_comments(MAIN_TEX.read_text(encoding="utf-8")) if MAIN_TEX.exists() else ""
    base = MAIN_TEX.parent

    labels = LABEL_PATTERN.findall(text)
    refs = [part.strip() for raw in REF_PATTERN.findall(text) for part in raw.split(",") if part.strip()]
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        if count > 1:
            errors.append(f"duplicate label {label}")
    for ref in sorted(set(refs) - set(labels)):
        errors.append(f"missing label for reference {ref}")

    input_paths = [resolve(token, base, ".tex") for token in INPUT_PATTERN.findall(text)]
    for path in input_paths:
        if not path.exists():
            errors.append(f"missing input file: {path.relative_to(ROOT)}")

    graphics_paths = [resolve(token, base, None) for token in GRAPHICS_PATTERN.findall(text)]
    for path in graphics_paths:
        if not path.exists():
            errors.append(f"missing figure asset: {path.relative_to(ROOT)}")

    referenced_figures = {p.resolve() for p in graphics_paths}
    final_figures = set((ROOT / "paper" / "figures").glob("*.pdf"))
    unused = sorted(p for p in final_figures if p.resolve() not in referenced_figures)
    for path in unused:
        errors.append(f"unused paper figure: {path.relative_to(ROOT)}")

    print("TeX integrity summary:")
    print(f"  labels: {len(labels)}")
    print(f"  refs: {len(refs)}")
    print(f"  generated inputs: {len(input_paths)}")
    print(f"  figure refs: {len(graphics_paths)}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("TeX integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
