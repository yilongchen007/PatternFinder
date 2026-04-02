from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pattern_finder import SyntheticGenerator
from pattern_finder.plotting import save_plain_sample_plot


ANOMALY_TYPES = ("point", "freq", "trend", "range")
DEFAULT_SPLITS = {
    "train": {"num_samples": 8000, "seed": 101},
    "val": {"num_samples": 1000, "seed": 202},
    "test": {"num_samples": 1000, "seed": 303},
}
DEFAULT_OUTPUT_DIR = Path("datasets/anomaly_db_v1")
IMAGE_DIR_NAME = "images_plain_768x384"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 384
CHECKSUM_FILENAME = "checksums.sha256"


def build_type_plan(num_samples: int) -> list[str]:
    if num_samples % len(ANOMALY_TYPES) != 0:
        raise ValueError("num_samples must be divisible by 4 for balanced generation")
    per_type = num_samples // len(ANOMALY_TYPES)
    return [anomaly_type for anomaly_type in ANOMALY_TYPES for _ in range(per_type)]


def generate_split(
    split_name: str,
    *,
    num_samples: int,
    seed: int,
) -> list[dict]:
    generator = SyntheticGenerator(seed=seed)
    samples: list[dict] = []
    for index, anomaly_type in enumerate(build_type_plan(num_samples), start=1):
        sample = generator.generate_sample(
            sample_id=f"{split_name}_{index:06d}",
            anomaly_type=anomaly_type,
            series_length=256,
            max_number_of_intervals=1,
            background_noise=None,
            variant="default" if anomaly_type == "trend" else None,
        )
        samples.append(sample)
    return samples


def build_image_path(split_name: str, sample_id: str) -> str:
    return f"{IMAGE_DIR_NAME}/{split_name}/{sample_id}.png"


def render_and_attach_images(
    output_dir: Path,
    split_name: str,
    samples: list[dict],
) -> None:
    split_dir = output_dir / IMAGE_DIR_NAME / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        sample["image_path"] = build_image_path(split_name, sample["sample_id"])
        save_plain_sample_plot(
            sample,
            output_dir / sample["image_path"],
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
        )


def summarize_split(samples: list[dict]) -> dict:
    type_counts = Counter(sample["parameters"]["anomaly_type"] for sample in samples)
    noise_counts = Counter(sample["context"]["background_noise"] for sample in samples)
    return {
        "num_samples": len(samples),
        "anomaly_type_counts": dict(sorted(type_counts.items())),
        "background_noise_counts": dict(sorted(noise_counts.items())),
    }


def build_manifest(split_summary: dict[str, dict]) -> dict:
    split_summary = {name: split_summary[name] for name in DEFAULT_SPLITS}

    return {
        "name": "anomaly_db_v1",
        "setting": {
            "anomaly_types": list(ANOMALY_TYPES),
            "balanced_classes": True,
            "series_length": 256,
            "background_noise": "random_from_none_light_moderate",
            "split_seeds": {
                split_name: split_config["seed"]
                for split_name, split_config in DEFAULT_SPLITS.items()
            },
            "trend_variant": "default",
            "max_number_of_intervals": 1,
            "labels": ["point_labels", "events", "description"],
            "image_render": {
                "style": "plain",
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "image_dir": IMAGE_DIR_NAME,
            },
        },
        "splits": split_summary,
    }


def write_split_dataset(
    output_dir: Path,
    split_name: str,
    samples: list[dict],
) -> dict:
    render_and_attach_images(output_dir, split_name, samples)
    summary = summarize_split(samples)
    (output_dir / f"{split_name}.json").write_text(
        json.dumps(samples, indent=2),
        encoding="utf-8",
    )
    return summary


def write_manifest(output_dir: Path, split_summary: dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(build_manifest(split_summary), indent=2),
        encoding="utf-8",
    )


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_dataset_files(output_dir: Path) -> list[Path]:
    files = [output_dir / "manifest.json"]
    for split_name in DEFAULT_SPLITS:
        files.append(output_dir / f"{split_name}.json")
    for split_name in DEFAULT_SPLITS:
        split_dir = output_dir / IMAGE_DIR_NAME / split_name
        files.extend(sorted(split_dir.glob("*.png")))
    return files


def write_checksums(output_dir: Path) -> None:
    lines = [
        f"{compute_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in iter_dataset_files(output_dir)
    ]
    (output_dir / CHECKSUM_FILENAME).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    split_summary = {}
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_config in DEFAULT_SPLITS.items():
        samples = generate_split(
            split_name,
            num_samples=split_config["num_samples"],
            seed=split_config["seed"],
        )
        split_summary[split_name] = write_split_dataset(
            DEFAULT_OUTPUT_DIR,
            split_name,
            samples,
        )
    write_manifest(DEFAULT_OUTPUT_DIR, split_summary)
    write_checksums(DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
