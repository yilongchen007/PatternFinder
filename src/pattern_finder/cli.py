from __future__ import annotations

import argparse
import json

from .generator import ANOMALY_TYPES, SyntheticGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic time-series anomaly samples."
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--series-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
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
        variant=args.variant,
    )
    if args.output is None:
        print(json.dumps(samples, indent=2))


if __name__ == "__main__":
    main()
