import tempfile
import unittest
from pathlib import Path

from dev.scripts.generate_balanced_dataset import (
    CHECKSUM_FILENAME,
    IMAGE_DIR_NAME,
    build_image_path,
    build_manifest,
    build_type_plan,
    write_checksums,
    render_and_attach_images,
)


class BalancedDatasetScriptTests(unittest.TestCase):
    def test_build_type_plan_is_balanced(self) -> None:
        plan = build_type_plan(8)
        self.assertEqual(plan.count("point"), 2)
        self.assertEqual(plan.count("freq"), 2)
        self.assertEqual(plan.count("trend"), 2)
        self.assertEqual(plan.count("range"), 2)

    def test_build_type_plan_requires_divisible_count(self) -> None:
        with self.assertRaises(ValueError):
            build_type_plan(10)

    def test_build_manifest_records_expected_setting(self) -> None:
        manifest = build_manifest(
            {
                "train": {
                    "num_samples": 2,
                    "anomaly_type_counts": {"freq": 1, "point": 1},
                    "background_noise_counts": {"light": 1, "none": 1},
                },
                "val": {
                    "num_samples": 0,
                    "anomaly_type_counts": {},
                    "background_noise_counts": {},
                },
                "test": {
                    "num_samples": 0,
                    "anomaly_type_counts": {},
                    "background_noise_counts": {},
                },
            }
        )
        self.assertEqual(manifest["setting"]["series_length"], 256)
        self.assertEqual(manifest["setting"]["trend_variant"], "default")
        self.assertEqual(manifest["setting"]["max_number_of_intervals"], 1)
        self.assertEqual(manifest["setting"]["split_seeds"]["train"], 101)
        self.assertEqual(manifest["setting"]["image_render"]["style"], "plain")
        self.assertEqual(
            manifest["splits"]["train"]["anomaly_type_counts"],
            {"freq": 1, "point": 1},
        )

    def test_build_image_path_uses_image_directory(self) -> None:
        self.assertEqual(
            build_image_path("train", "train_000001"),
            f"{IMAGE_DIR_NAME}/train/train_000001.png",
        )

    def test_render_and_attach_images_sets_image_path(self) -> None:
        sample = {
            "sample_id": "train_000001",
            "series": [0.0, 0.5, -0.2, 0.1],
            "point_labels": [0, 1, 1, 0],
            "parameters": {"anomaly_type": "point"},
            "context": {"background_noise": "none"},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            render_and_attach_images(output_dir, "train", [sample])
            self.assertEqual(
                sample["image_path"],
                f"{IMAGE_DIR_NAME}/train/train_000001.png",
            )
            self.assertTrue((output_dir / sample["image_path"]).exists())

    def test_write_checksums_creates_checksum_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
            for split_name in ("train", "val", "test"):
                (output_dir / f"{split_name}.json").write_text("[]", encoding="utf-8")
                split_dir = output_dir / IMAGE_DIR_NAME / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                (split_dir / f"{split_name}_000001.png").write_bytes(b"png")
            write_checksums(output_dir)
            checksum_path = output_dir / CHECKSUM_FILENAME
            self.assertTrue(checksum_path.exists())
            content = checksum_path.read_text(encoding="utf-8")
            self.assertIn("manifest.json", content)
            self.assertIn(f"{IMAGE_DIR_NAME}/train/train_000001.png", content)


if __name__ == "__main__":
    unittest.main()
