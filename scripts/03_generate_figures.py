from __future__ import annotations

from _pipeline_common import ROOT, load_research_state, write_all_figures


def main() -> None:
    state = load_research_state(ROOT)
    paths = write_all_figures(state, ROOT)
    print(f"wrote {len(paths)} figure artifacts")


if __name__ == "__main__":
    main()
