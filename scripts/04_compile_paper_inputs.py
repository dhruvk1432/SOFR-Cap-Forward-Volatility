from __future__ import annotations

from _pipeline_common import ROOT, load_research_state, write_all_tables, write_paper


def main() -> None:
    state = load_research_state(ROOT)
    write_all_tables(state, ROOT)
    path = write_paper(state, ROOT)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
