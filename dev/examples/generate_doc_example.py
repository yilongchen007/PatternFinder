from __future__ import annotations

import json
from pathlib import Path

from pattern_finder.generator import generate_sample
from pattern_finder.plotting import save_sample_plot


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    docs_dir = root / "docs"
    assets_dir = docs_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    sample = generate_sample(
        sample_id="sample_000001",
        anomaly_type="trend",
        series_length=128,
        seed=24,
    )

    json_path = assets_dir / "example_sample.json"
    json_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    image_path = assets_dir / "example_sample.png"
    save_sample_plot(sample, image_path)


if __name__ == "__main__":
    main()
