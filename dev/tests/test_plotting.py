import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pattern_finder import generate_sample
from pattern_finder.plotting import save_plain_sample_plot


class PlottingTests(unittest.TestCase):
    def test_save_plain_sample_plot_uses_requested_size(self) -> None:
        sample = generate_sample(
            sample_id="sample_000007",
            anomaly_type="freq",
            series_length=128,
            seed=13,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plain.png"
            save_plain_sample_plot(sample, output_path, width=768, height=384)
            self.assertTrue(output_path.exists())
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (768, 384))


if __name__ == "__main__":
    unittest.main()
