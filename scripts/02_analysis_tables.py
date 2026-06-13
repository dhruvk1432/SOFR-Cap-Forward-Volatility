from __future__ import annotations

from _pipeline_common import ROOT, load_research_state, write_all_tables


def main() -> None:
    state = load_research_state(ROOT)
    paths = write_all_tables(state, ROOT)
    print(f"wrote {len(paths)} table artifacts")


if __name__ == "__main__":
    main()
