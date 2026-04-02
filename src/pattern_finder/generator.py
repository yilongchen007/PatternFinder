from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ANOMALY_TYPES = (
    "point",
    "freq",
    "trend",
    "range",
)

BACKGROUND_NOISE_LEVELS = {
    "none": 0.0,
    "light": 0.02,
    "moderate": 0.05,
    "heavy": 0.08,
}

POINT_ANOMALY_STD_VALUES = (0.2, 0.5, 0.8)
FREQ_MULTIPLIER_VALUES = (1.5, 2.0, 3.0)
TREND_SLOPE_RANGE_VALUES = ((4.5, 6.0), (6.0, 12.0), (12.0, 20.0))
RANGE_SHIFT_VALUES = ((0.3, 0.45), (0.5, 0.8), (0.8, 1.2))

TEMPLATES = {
    "point": (
        "A {strength} noise anomaly occurs from index {start} to {end}, "
        "and this segment becomes irregular."
    ),
    "freq": (
        "A {strength} frequency anomaly occurs from index {start} to {end}, "
        "and the oscillation becomes {direction}."
    ),
    "trend": (
        "A {strength} trend anomaly occurs from index {start} to {end}, "
        "and the local trend {direction}."
    ),
    "range": (
        "A {strength} level anomaly occurs from index {start} to {end}, "
        "and the segment {direction}."
    ),
}


@dataclass(slots=True)
class Event:
    start: int
    end: int
    type: str
    raw_params: dict[str, Any]
    verbal_tags: dict[str, str]
    description: str


@dataclass(slots=True)
class Sample:
    sample_id: str
    series: list[float]
    point_labels: list[int]
    parameters: dict[str, Any]
    events: list[dict[str, Any]]
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyntheticGenerator:
    """Synthetic generator derived from AnomLLM-style generation logic."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def generate_sample(
        self,
        sample_id: str,
        anomaly_type: str | None = None,
        series_length: int = 1000,
        max_number_of_intervals: int = 8,
        background_noise: str | None = None,
        variant: str | None = None,
    ) -> dict[str, Any]:
        if anomaly_type is None:
            anomaly_type = self.rng.choice(ANOMALY_TYPES)
        if anomaly_type not in ANOMALY_TYPES:
            raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")
        if series_length < 32:
            raise ValueError("series_length must be at least 32")
        if max_number_of_intervals < 1:
            raise ValueError("max_number_of_intervals must be at least 1")

        sample_params = self._sample_raw_params(anomaly_type)
        variant = self._normalize_variant(anomaly_type, variant, sample_params)
        generator_model = self._build_generator_model(
            anomaly_type=anomaly_type,
            raw_params=sample_params,
            series_length=series_length,
            variant=variant,
        )
        series, event_params_list = self._generate_series_from_anomllm_model(
            anomaly_type=anomaly_type,
            generator_model=generator_model,
            series_length=series_length,
            max_number_of_intervals=max_number_of_intervals,
        )

        noise_label = background_noise or self.rng.choice(
            ["none", "light", "moderate"]
        )
        if noise_label not in BACKGROUND_NOISE_LEVELS:
            raise ValueError(f"Unsupported background_noise: {noise_label}")
        noise_scale = BACKGROUND_NOISE_LEVELS[noise_label]
        if noise_scale > 0:
            series = [value + self.rng.gauss(0.0, noise_scale) for value in series]

        events: list[dict[str, Any]] = []
        labels = [0] * series_length
        for event_raw_params in event_params_list:
            start = event_raw_params["start"]
            end = event_raw_params["end"]
            verbal_tags = self._build_verbal_tags(
                anomaly_type=anomaly_type,
                raw_params=event_raw_params,
                generator_model=generator_model,
            )
            description = self._build_description(
                anomaly_type=anomaly_type,
                start=start,
                end=end,
                verbal_tags=verbal_tags,
            )
            for idx in range(start, end):
                labels[idx] = 1
            events.append(
                asdict(
                    Event(
                        start=start,
                        end=end,
                        type=anomaly_type,
                        raw_params=event_raw_params,
                        verbal_tags=verbal_tags,
                        description=description,
                    )
                )
            )

        sample = Sample(
            sample_id=sample_id,
            series=[round(value, 4) for value in series],
            point_labels=labels,
            parameters={
                "anomaly_type": anomaly_type,
                "variant": variant,
                "max_number_of_intervals": max_number_of_intervals,
                "generator_model": generator_model,
            },
            events=events,
            context={
                "background_noise": noise_label,
                "series_length": series_length,
                "sampling_regime": "regular",
                "source_model": "AnomLLM-derived",
            },
        )
        return sample.to_dict()

    def generate_dataset(
        self,
        num_samples: int,
        output_path: str | Path | None = None,
        anomaly_type: str | None = None,
        series_length: int = 1000,
        max_number_of_intervals: int = 8,
        background_noise: str | None = None,
        variant: str | None = None,
    ) -> list[dict[str, Any]]:
        samples = [
            self.generate_sample(
                sample_id=f"sample_{index + 1:06d}",
                anomaly_type=anomaly_type,
                series_length=series_length,
                max_number_of_intervals=max_number_of_intervals,
                background_noise=background_noise,
                variant=variant,
            )
            for index in range(num_samples)
        ]
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
        return samples

    def _sample_raw_params(self, anomaly_type: str) -> dict[str, Any]:
        if anomaly_type == "point":
            return {"anomaly_std": self.rng.choice(POINT_ANOMALY_STD_VALUES)}
        if anomaly_type == "freq":
            return {
                "direction": self.rng.choice(["up", "down"]),
                "frequency_multiplier": self.rng.choice(FREQ_MULTIPLIER_VALUES),
            }
        if anomaly_type == "trend":
            return {
                "trend_mode": self.rng.choice(["steeper_up", "reverse_down"]),
                "abnormal_slope_range": self.rng.choice(TREND_SLOPE_RANGE_VALUES),
            }
        if anomaly_type == "range":
            return {
                "direction": self.rng.choice(["up", "down"]),
                "anomaly_size_range": self.rng.choice(RANGE_SHIFT_VALUES),
            }
        raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")

    def _normalize_variant(
        self,
        anomaly_type: str,
        variant: str | None,
        raw_params: dict[str, Any],
    ) -> str:
        if anomaly_type == "trend":
            normalized = variant or "default"
            if normalized not in {"default", "flat-trend"}:
                raise ValueError(f"Unsupported trend variant: {normalized}")
            if normalized == "flat-trend":
                raw_params["trend_mode"] = "steeper_up"
                raw_params["abnormal_slope_range"] = (4.5, 6.0)
            return normalized
        if variant not in {None, "default"}:
            raise ValueError(f"Unsupported variant for {anomaly_type}: {variant}")
        return "default"

    def _build_generator_model(
        self,
        anomaly_type: str,
        raw_params: dict[str, Any],
        series_length: int,
        variant: str,
    ) -> dict[str, Any]:
        scale = series_length / 1000.0

        def scale_duration(value: float, minimum: int = 1) -> int:
            return max(minimum, int(round(value * scale)))

        def scale_rate(value: float) -> float:
            return max(1.0, value * scale)

        if anomaly_type == "point":
            return {
                "base_frequency": 0.03,
                "normal_duration_rate": scale_rate(800.0),
                "anomaly_duration_rate": scale_rate(30.0),
                "minimum_anomaly_duration": scale_duration(5),
                "minimum_normal_duration": scale_duration(200),
                "anomaly_std": raw_params["anomaly_std"],
            }

        if anomaly_type == "freq":
            return {
                "base_frequency": 0.03,
                "normal_duration_rate": scale_rate(450.0),
                "anomaly_duration_rate": scale_rate(15.0),
                "minimum_anomaly_duration": scale_duration(7),
                "minimum_normal_duration": scale_duration(20),
                "frequency_multiplier": raw_params["frequency_multiplier"],
                "direction": raw_params["direction"],
            }

        if anomaly_type == "trend":
            return {
                "base_frequency": 0.02,
                "normal_duration_rate": scale_rate(1700.0),
                "anomaly_duration_rate": scale_rate(100.0),
                "minimum_anomaly_duration": scale_duration(50),
                "minimum_normal_duration": scale_duration(800),
                "normal_slope": 3.0,
                "abnormal_slope_range": raw_params["abnormal_slope_range"],
                "trend_mode": raw_params["trend_mode"],
                "variant": variant,
            }

        if anomaly_type == "range":
            return {
                "nominal_data_mean": 0.0,
                "nominal_data_std": 0.1,
                "normal_duration_rate": scale_rate(800.0),
                "anomaly_duration_rate": scale_rate(20.0),
                "minimum_anomaly_duration": scale_duration(5),
                "minimum_normal_duration": scale_duration(10),
                "anomaly_size_range": raw_params["anomaly_size_range"],
                "direction": raw_params["direction"],
            }

        raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")

    def _generate_series_from_anomllm_model(
        self,
        anomaly_type: str,
        generator_model: dict[str, Any],
        series_length: int,
        max_number_of_intervals: int,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        if anomaly_type == "point":
            return self._generate_point_series(
                generator_model, series_length, max_number_of_intervals
            )
        if anomaly_type == "freq":
            return self._generate_freq_series(
                generator_model, series_length, max_number_of_intervals
            )
        if anomaly_type == "trend":
            return self._generate_trend_series(
                generator_model, series_length, max_number_of_intervals
            )
        if anomaly_type == "range":
            return self._generate_range_series(
                generator_model, series_length, max_number_of_intervals
            )
        raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")

    def _generate_point_series(
        self,
        config: dict[str, Any],
        series_length: int,
        max_number_of_intervals: int,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        series = [
            math.sin(2 * math.pi * config["base_frequency"] * t)
            for t in range(series_length)
        ]
        intervals = self._sample_intervals(
            series_length=series_length,
            normal_duration_rate=config["normal_duration_rate"],
            anomaly_duration_rate=config["anomaly_duration_rate"],
            minimum_anomaly_duration=config["minimum_anomaly_duration"],
            minimum_normal_duration=config["minimum_normal_duration"],
            max_number_of_intervals=max_number_of_intervals,
        )
        event_params = []
        for start, end in intervals:
            for idx in range(start, end):
                series[idx] = self.rng.gauss(0.0, config["anomaly_std"])
            event_params.append(
                {
                    "start": start,
                    "end": end,
                    "anomaly_std": config["anomaly_std"],
                }
            )
        return series, event_params

    def _generate_freq_series(
        self,
        config: dict[str, Any],
        series_length: int,
        max_number_of_intervals: int,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        freq_function = [config["base_frequency"]] * series_length
        intervals = self._sample_intervals(
            series_length=series_length,
            normal_duration_rate=config["normal_duration_rate"],
            anomaly_duration_rate=config["anomaly_duration_rate"],
            minimum_anomaly_duration=config["minimum_anomaly_duration"],
            minimum_normal_duration=config["minimum_normal_duration"],
            max_number_of_intervals=max_number_of_intervals,
        )
        event_params = []
        for start, end in intervals:
            for idx in range(start, end):
                if config["direction"] == "up":
                    freq_function[idx] *= config["frequency_multiplier"]
                else:
                    freq_function[idx] /= config["frequency_multiplier"]
            event_params.append(
                {
                    "start": start,
                    "end": end,
                    "direction": config["direction"],
                    "frequency_multiplier": config["frequency_multiplier"],
                }
            )

        cumulative = 0.0
        series = []
        for value in freq_function:
            cumulative += value
            series.append(math.sin(2 * math.pi * cumulative))
        return series, event_params

    def _generate_trend_series(
        self,
        config: dict[str, Any],
        series_length: int,
        max_number_of_intervals: int,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        t = list(range(series_length))
        intervals = self._sample_intervals(
            series_length=series_length,
            normal_duration_rate=config["normal_duration_rate"],
            anomaly_duration_rate=config["anomaly_duration_rate"],
            minimum_anomaly_duration=config["minimum_anomaly_duration"],
            minimum_normal_duration=config["minimum_normal_duration"],
            max_number_of_intervals=max_number_of_intervals,
        )
        trend = [0.0] * series_length
        current_value = 0.0
        current_time = 0
        event_params = []

        for start, end in intervals:
            if start > current_time:
                base_t = t[current_time]
                for idx in range(current_time, start):
                    trend[idx] = current_value + config["normal_slope"] * (
                        t[idx] - base_t
                    ) / series_length
                current_value = trend[start - 1]

            abnormal_slope = self._sample_abnormal_slope(config)
            base_t = t[start]
            for idx in range(start, end):
                trend[idx] = current_value + abnormal_slope * (
                    t[idx] - base_t
                ) / series_length
            current_value = trend[end - 1]
            current_time = end
            event_params.append(
                {
                    "start": start,
                    "end": end,
                    "abnormal_slope_range": config["abnormal_slope_range"],
                    "normal_slope": config["normal_slope"],
                    "abnormal_slope": abnormal_slope,
                }
            )

        if current_time < series_length:
            base_t = t[current_time]
            for idx in range(current_time, series_length):
                trend[idx] = current_value + config["normal_slope"] * (
                    t[idx] - base_t
                ) / series_length

        series = [
            math.sin(2 * math.pi * config["base_frequency"] * idx) + trend[idx]
            for idx in range(series_length)
        ]
        return self._normalize_to_unit_range(series), event_params

    def _generate_range_series(
        self,
        config: dict[str, Any],
        series_length: int,
        max_number_of_intervals: int,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        series = [
            self.rng.gauss(config["nominal_data_mean"], config["nominal_data_std"])
            for _ in range(series_length)
        ]
        intervals = self._sample_intervals(
            series_length=series_length,
            normal_duration_rate=config["normal_duration_rate"],
            anomaly_duration_rate=config["anomaly_duration_rate"],
            minimum_anomaly_duration=config["minimum_anomaly_duration"],
            minimum_normal_duration=config["minimum_normal_duration"],
            max_number_of_intervals=max_number_of_intervals,
        )
        direction_sign = 1.0 if config["direction"] == "up" else -1.0
        low, high = config["anomaly_size_range"]
        event_params = []
        for start, end in intervals:
            for idx in range(start, end):
                series[idx] += direction_sign * self.rng.uniform(low, high)
            event_params.append(
                {
                    "start": start,
                    "end": end,
                    "direction": config["direction"],
                    "anomaly_size_range": config["anomaly_size_range"],
                }
            )
        return series, event_params

    def _sample_intervals(
        self,
        series_length: int,
        normal_duration_rate: float,
        anomaly_duration_rate: float,
        minimum_anomaly_duration: int,
        minimum_normal_duration: int,
        max_number_of_intervals: int = 8,
    ) -> list[tuple[int, int]]:
        location = 0
        intervals: list[tuple[int, int]] = []

        for _ in range(max_number_of_intervals):
            normal_duration = max(
                minimum_normal_duration,
                int(self.rng.expovariate(1.0 / normal_duration_rate)),
            )
            anomaly_start = location + normal_duration
            anomaly_duration = max(
                minimum_anomaly_duration,
                int(self.rng.expovariate(1.0 / anomaly_duration_rate)),
            )
            anomaly_end = min(series_length, anomaly_start + anomaly_duration)

            if anomaly_start >= series_length:
                break

            intervals.append((anomaly_start, anomaly_end))
            location = anomaly_end

        if not intervals:
            fallback_duration = min(
                max(minimum_anomaly_duration, 3), max(3, series_length // 6)
            )
            start = max(0, series_length // 2 - fallback_duration // 2)
            intervals.append((start, min(series_length, start + fallback_duration)))

        return intervals

    def _sample_abnormal_slope(self, config: dict[str, Any]) -> float:
        normal_slope = config["normal_slope"]
        min_slope, max_slope = config["abnormal_slope_range"]
        mode = config["trend_mode"]

        if mode == "steeper_up":
            lower = max(normal_slope, min_slope)
            return self.rng.uniform(lower, max_slope)
        if mode == "reverse_down":
            lower_bound = min(-max_slope, min(normal_slope, min_slope))
            return self.rng.uniform(lower_bound, 0.0)
        raise ValueError(f"Unsupported trend_mode: {mode}")

    def _build_verbal_tags(
        self,
        anomaly_type: str,
        raw_params: dict[str, Any],
        generator_model: dict[str, Any],
    ) -> dict[str, str]:
        if anomaly_type == "point":
            return {
                "strength": self._strength_from_anomaly_std(raw_params["anomaly_std"]),
                "direction": "becomes irregular",
            }
        if anomaly_type == "freq":
            return {
                "strength": self._strength_from_frequency_multiplier(
                    raw_params["frequency_multiplier"]
                ),
                "direction": "faster"
                if raw_params["direction"] == "up"
                else "slower",
            }
        if anomaly_type == "trend":
            return {
                "strength": self._strength_from_slope_range(
                    tuple(raw_params["abnormal_slope_range"])
                ),
                "direction": "becomes steeper"
                if raw_params["abnormal_slope"] > generator_model["normal_slope"]
                else "turns downward",
            }
        if anomaly_type == "range":
            return {
                "strength": self._strength_from_anomaly_size_range(
                    tuple(raw_params["anomaly_size_range"])
                ),
                "direction": "shifts upward"
                if raw_params["direction"] == "up"
                else "shifts downward",
            }
        raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")

    def _build_description(
        self,
        anomaly_type: str,
        start: int,
        end: int,
        verbal_tags: dict[str, str],
    ) -> str:
        template = TEMPLATES[anomaly_type]
        return template.format(
            strength=verbal_tags["strength"],
            start=start,
            end=end,
            direction=verbal_tags["direction"],
        )

    def _normalize_to_unit_range(self, values: list[float]) -> list[float]:
        min_value = min(values)
        max_value = max(values)
        if max_value <= min_value:
            return values[:]
        return [
            2 * (value - min_value) / (max_value - min_value) - 1
            for value in values
        ]

    def _strength_from_anomaly_std(self, anomaly_std: float) -> str:
        return {0.2: "mild", 0.5: "obvious", 0.8: "strong"}[round(anomaly_std, 1)]

    def _strength_from_frequency_multiplier(self, frequency_multiplier: float) -> str:
        return {1.5: "mild", 2.0: "obvious", 3.0: "strong"}[
            round(frequency_multiplier, 1)
        ]

    def _strength_from_slope_range(self, abnormal_slope_range: tuple[float, float]) -> str:
        return {
            (4.5, 6.0): "mild",
            (6.0, 12.0): "obvious",
            (12.0, 20.0): "strong",
        }[tuple(round(value, 1) for value in abnormal_slope_range)]

    def _strength_from_anomaly_size_range(
        self,
        anomaly_size_range: tuple[float, float],
    ) -> str:
        rounded = tuple(round(value, 2) for value in anomaly_size_range)
        if rounded == (0.3, 0.45):
            return "mild"
        if rounded == (0.5, 0.8):
            return "obvious"
        if rounded == (0.8, 1.2):
            return "strong"
        raise ValueError(f"Unsupported anomaly_size_range: {anomaly_size_range}")


def generate_sample(
    sample_id: str = "sample_000001",
    anomaly_type: str | None = None,
    series_length: int = 1000,
    max_number_of_intervals: int = 8,
    seed: int | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    generator = SyntheticGenerator(seed=seed)
    return generator.generate_sample(
        sample_id=sample_id,
        anomaly_type=anomaly_type,
        series_length=series_length,
        max_number_of_intervals=max_number_of_intervals,
        variant=variant,
    )
