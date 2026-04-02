import unittest

from pattern_finder import SyntheticGenerator, generate_sample


class GeneratorTests(unittest.TestCase):
    def test_generate_sample_has_series_params_and_description(self) -> None:
        sample = generate_sample(
            sample_id="sample_000001",
            anomaly_type="freq",
            series_length=128,
            seed=7,
        )

        self.assertEqual(sample["sample_id"], "sample_000001")
        self.assertEqual(len(sample["series"]), 128)
        self.assertEqual(len(sample["point_labels"]), 128)
        self.assertGreaterEqual(len(sample["events"]), 1)
        self.assertIn("parameters", sample)

        event = sample["events"][0]
        self.assertEqual(event["type"], "freq")
        self.assertIn("raw_params", event)
        self.assertIn("frequency_multiplier", event["raw_params"])
        self.assertIn("description", event)
        self.assertIn("index", event["description"])
        self.assertNotIn("controls", sample["parameters"])
        self.assertIn("generator_model", sample["parameters"])

    def test_all_supported_types_generate(self) -> None:
        generator = SyntheticGenerator(seed=3)
        anomaly_types = ["point", "freq", "trend", "range"]

        for index, anomaly_type in enumerate(anomaly_types, start=1):
            sample = generator.generate_sample(
                sample_id=f"sample_{index:06d}",
                anomaly_type=anomaly_type,
                series_length=160,
            )
            self.assertEqual(sample["events"][0]["type"], anomaly_type)
            self.assertGreater(sum(sample["point_labels"]), 0)

    def test_trend_flat_variant_is_supported(self) -> None:
        sample = generate_sample(
            sample_id="sample_000005",
            anomaly_type="trend",
            variant="flat-trend",
            series_length=160,
            seed=9,
        )
        self.assertEqual(sample["parameters"]["anomaly_type"], "trend")
        self.assertEqual(sample["parameters"]["variant"], "flat-trend")
        self.assertEqual(
            tuple(sample["parameters"]["generator_model"]["abnormal_slope_range"]),
            (4.5, 6.0),
        )
        self.assertEqual(
            tuple(sample["events"][0]["raw_params"]["abnormal_slope_range"]),
            (4.5, 6.0),
        )

    def test_generate_dataset_accepts_background_noise(self) -> None:
        generator = SyntheticGenerator(seed=11)
        samples = generator.generate_dataset(
            num_samples=2,
            anomaly_type="range",
            series_length=128,
            background_noise="heavy",
        )
        self.assertEqual(len(samples), 2)
        self.assertTrue(
            all(sample["context"]["background_noise"] == "heavy" for sample in samples)
        )

    def test_generate_sample_respects_max_number_of_intervals(self) -> None:
        sample = generate_sample(
            sample_id="sample_000006",
            anomaly_type="point",
            series_length=256,
            max_number_of_intervals=1,
            seed=5,
        )
        self.assertEqual(sample["parameters"]["max_number_of_intervals"], 1)
        self.assertEqual(len(sample["events"]), 1)


if __name__ == "__main__":
    unittest.main()
