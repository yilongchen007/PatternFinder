from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_sample_plot(sample: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    series = sample["series"]
    labels = sample["point_labels"]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(series, color="#1f4b99", linewidth=1.5)

    in_region = False
    region_start = 0
    for idx, label in enumerate(labels + [0]):
        if label == 1 and not in_region:
            in_region = True
            region_start = idx
        elif label == 0 and in_region:
            in_region = False
            ax.axvspan(region_start, idx, color="#e55b4b", alpha=0.2, linewidth=0)

    ax.set_title(sample["sample_id"])
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
