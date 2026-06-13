from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_table_artifacts(
    frame: pd.DataFrame,
    stem: str,
    tables_dir: str | Path,
    generated_dir: str | Path | None = None,
    *,
    index: bool = True,
    float_format: str = "%.4f",
) -> dict[str, Path]:
    tables_dir = ensure_dir(tables_dir)
    generated_dir = ensure_dir(generated_dir) if generated_dir else None
    out: dict[str, Path] = {}
    csv_path = tables_dir / f"{stem}.csv"
    frame.to_csv(csv_path, index=index)
    out["csv"] = csv_path

    latex = frame.to_latex(index=index, escape=True, float_format=float_format)
    tex_path = tables_dir / f"{stem}.tex"
    tex_path.write_text(latex, encoding="utf-8")
    out["tex"] = tex_path
    if generated_dir:
        generated_tex = generated_dir / f"{stem}.tex"
        generated_tex.write_text(latex, encoding="utf-8")
        out["generated_tex"] = generated_tex
    return out


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(paths: Iterable[str | Path], root: str | Path, target: str | Path) -> Path:
    root = Path(root)
    manifest = {
        "root": str(root),
        "artifacts": {
            str(Path(path).relative_to(root)): sha256_file(path)
            for path in sorted({Path(p) for p in paths})
            if Path(path).exists() and Path(path).is_file()
        },
    }
    target = Path(target)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
