from __future__ import annotations

from _pipeline_common import ROOT, load_research_state, write_processed_data


def main() -> None:
    state = load_research_state(ROOT)
    paths = write_processed_data(state, ROOT)
    print(f"wrote {len(paths)} processed research data files")


if __name__ == "__main__":
    main()
