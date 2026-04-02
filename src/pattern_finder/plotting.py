from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_plot(
    sample: dict,
    output_path: str | Path,
    *,
    figsize: tuple[float, float],
    dpi: int,
    show_annotations: bool,
    show_axes: bool,
    show_title: bool,
    line_width: float,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    series = sample["series"]
    labels = sample["point_labels"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series, color="#1f4b99", linewidth=line_width)

    if show_annotations:
        in_region = False
        region_start = 0
        for idx, label in enumerate(labels + [0]):
            if label == 1 and not in_region:
                in_region = True
                region_start = idx
            elif label == 0 and in_region:
                in_region = False
                ax.axvspan(region_start, idx, color="#e55b4b", alpha=0.2, linewidth=0)

    if show_title:
        ax.set_title(sample["sample_id"])

    if show_axes:
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.2)
        fig.tight_layout()
    else:
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="white")
    plt.close(fig)


def save_sample_plot(sample: dict, output_path: str | Path) -> None:
    _save_plot(
        sample,
        output_path,
        figsize=(10, 2.8),
        dpi=160,
        show_annotations=True,
        show_axes=True,
        show_title=True,
        line_width=1.5,
    )


def save_plain_sample_plot(
    sample: dict,
    output_path: str | Path,
    *,
    width: int = 768,
    height: int = 384,
) -> None:
    _save_plot(
        sample,
        output_path,
        figsize=(width / 100.0, height / 100.0),
        dpi=100,
        show_annotations=False,
        show_axes=False,
        show_title=False,
        line_width=2.0,
    )
