from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig: plt.Figure, stem: str, figures_dir: str | Path, paper_figures_dir: str | Path) -> dict[str, Path]:
    figures_dir = Path(figures_dir)
    paper_figures_dir = Path(paper_figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    pdf_path = paper_figures_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def draw_regime_spans(ax: plt.Axes, regimes: pd.Series) -> None:
    colors = {"Hiking": "tab:red", "Pause": "tab:blue", "Easing": "tab:green"}
    current = None
    start = None
    for date, regime in regimes.items():
        if regime != current:
            if current in colors and start is not None:
                ax.axvspan(start, date, color=colors[current], alpha=0.08, linewidth=0)
            current = regime
            start = date
    if current in colors and start is not None:
        ax.axvspan(start, regimes.index[-1], color=colors[current], alpha=0.08, linewidth=0)


def plot_forward_vol_surface(normal_panel: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = (normal_panel.astype(float) * 10000).T
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels([f"{x:g}Y" for x in data.index])
    tick_idx = np.linspace(0, len(data.columns) - 1, min(8, len(data.columns))).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([data.columns[i].strftime("%Y-%m") for i in tick_idx], rotation=30, ha="right")
    ax.set_title("SOFR cap stripped forward normal-volatility surface")
    ax.set_xlabel("Quote week")
    ax.set_ylabel("Forward-vol tenor")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normal vol, bp")
    fig.tight_layout()
    return fig
