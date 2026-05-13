# PatternFinder

PatternFinder generates a synthetic time-series anomaly dataset with point labels, event descriptions, and plain PNG renders for vision-style inputs.

## TS Grounder SFT + RL Framework

The latest downstream training framework for `Thesis/Training-model-yilong` is documented here:

- [TS Grounder SFT + RL Framework](docs/ts_grounder_sft_rl_framework.md)

It covers the full input-data format, VLM SFT JSONL construction, token-level SFT loss masking, optional auxiliary losses, RL reward design, GRPO-style policy update, Slurm workflow, and figure text for the system diagram.

## Default Dataset

- Balanced anomaly types: `point`, `freq`, `trend`, `range`
- Fixed `series_length = 256`
- Random background noise from `none`, `light`, `moderate`
- `trend` uses the `default` variant
- `max_number_of_intervals = 1`
- Plain image render: `768x384`

## Generate Dataset

```bash
.venv/bin/pip install -r requirements-dataset-lock.txt
PYTHONPATH=src .venv/bin/python dev/scripts/generate_balanced_dataset.py
```

## Output

The default output directory is `datasets/anomaly_db_v1/`.

- `train.json`, `val.json`, `test.json`: sample records
- `images_plain_768x384/{split}/*.png`: plain PNG renders
- `manifest.json`: dataset-level settings and split summaries
- `checksums.sha256`: file hashes for reproducibility

Each sample record includes:

- `sample_id`
- `series`
- `point_labels`
- `events`
- `image_path`

The plain PNG render uses a white background with a single time-series curve and no title, axes, or anomaly overlay.
