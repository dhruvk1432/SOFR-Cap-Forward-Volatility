from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


FULL_STEPS = [
    "scripts/01_build_curves.py",
    "scripts/02_analysis_tables.py",
    "scripts/03_generate_figures.py",
    "scripts/04_compile_paper_inputs.py",
    "scripts/05_tex_integrity_check.py",
    "scripts/06_final_artifact_check.py",
]
QUICK_STEPS = [
    "scripts/02_analysis_tables.py",
    "scripts/03_generate_figures.py",
    "scripts/04_compile_paper_inputs.py",
    "scripts/05_tex_integrity_check.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SOFR cap forward-vol research pipeline.")
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    args = parser.parse_args()
    steps = FULL_STEPS if args.mode == "full" else QUICK_STEPS
    print("=" * 72)
    print(f"SOFR cap forward-volatility pipeline: {args.mode}")
    print("=" * 72)
    for step in steps:
        print(f"\n---> {step} <---")
        result = subprocess.run([sys.executable, step], cwd=ROOT)
        if result.returncode != 0:
            print(f"pipeline failed at {step}")
            return result.returncode
    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
