from __future__ import annotations

import argparse
import json

from .generator import ANOMALY_TYPES, BACKGROUND_NOISE_LEVELS, SyntheticGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic time-series anomaly samples."
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--series-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-number-of-intervals", type=int, default=8)
    parser.add_argument(
        "--background-noise",
        type=str,
        default=None,
        choices=tuple(BACKGROUND_NOISE_LEVELS),
    )
    parser.add_argument(
        "--anomaly-type",
        type=str,
        default=None,
        choices=ANOMALY_TYPES,
    )
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    generator = SyntheticGenerator(seed=args.seed)
    samples = generator.generate_dataset(
        num_samples=args.num_samples,
        output_path=args.output,
        anomaly_type=args.anomaly_type,
        series_length=args.series_length,
        max_number_of_intervals=args.max_number_of_intervals,
        background_noise=args.background_noise,
        variant=args.variant,
    )
    if args.output is None:
        print(json.dumps(samples, indent=2))


if __name__ == "__main__":
    main()
