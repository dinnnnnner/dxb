from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = (
    PROJECT_ROOT
    / "wheel_cog_outputs"
    / "blowout_manual_labeling_package"
    / "labeling_package"
    / "event_time_labels.csv"
)
DEFAULT_BATCH_SUMMARY = (
    PROJECT_ROOT
    / "wheel_cog_outputs"
    / "fast_alarm_batch_outputs"
    / "fast_alarm_batch_summary.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "wheel_cog_outputs" / "augmented_event_dataset"
WHEEL_COUNT = 4


@dataclass(frozen=True)
class EventLabel:
    event_id: str
    file: str
    event_time_s: float


@dataclass(frozen=True)
class SourceSeries:
    times: np.ndarray
    wheels: np.ndarray
    dt: float
    columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grouped wheel-speed augmentation samples around event_time labels."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--series", choices=["raw", "corrected", "ref_comp_on"], default="corrected"
    )
    parser.add_argument("--pre-s", type=float, default=40.0)
    parser.add_argument("--post-s", type=float, default=10.0)
    parser.add_argument("--aug-per-event", type=int, default=50)
    parser.add_argument("--normal-per-event", type=int, default=10)
    parser.add_argument("--normal-guard-s", type=float, default=10.0)
    parser.add_argument("--event-jitter-s", type=float, default=0.5)
    parser.add_argument("--time-scale-min", type=float, default=0.95)
    parser.add_argument("--time-scale-max", type=float, default=1.05)
    parser.add_argument("--speed-scale-min", type=float, default=0.95)
    parser.add_argument("--speed-scale-max", type=float, default=1.05)
    parser.add_argument("--noise-gain-min", type=float, default=0.10)
    parser.add_argument("--noise-gain-max", type=float, default=0.50)
    parser.add_argument("--dropout-probability", type=float, default=0.30)
    parser.add_argument("--max-dropout-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pre_s <= 0.0 or args.post_s <= 0.0:
        raise ValueError("--pre-s and --post-s must be positive")
    if args.aug_per_event < 0 or args.normal_per_event < 0:
        raise ValueError("sample counts cannot be negative")
    if not 0.0 < args.time_scale_min <= args.time_scale_max:
        raise ValueError("invalid time-scale range")
    if not 0.0 < args.speed_scale_min <= args.speed_scale_max:
        raise ValueError("invalid speed-scale range")
    if not 0.0 <= args.noise_gain_min <= args.noise_gain_max:
        raise ValueError("invalid noise-gain range")
    if not 0.0 <= args.dropout_probability <= 1.0:
        raise ValueError("--dropout-probability must be between 0 and 1")
    if args.max_dropout_samples < 0:
        raise ValueError("--max-dropout-samples cannot be negative")


def read_event_labels(path: Path) -> list[EventLabel]:
    labels: list[EventLabel] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            file_name = row["file"].strip()
            if not file_name:
                continue
            if file_name in seen:
                raise ValueError(f"duplicate label for {file_name}")
            seen.add(file_name)
            labels.append(
                EventLabel(
                    event_id=f"E{index:02d}",
                    file=file_name,
                    event_time_s=float(row["event_time_s"]),
                )
            )
    if not labels:
        raise ValueError(f"no labels found in {path}")
    return labels


def read_source_map(path: Path) -> dict[str, Path]:
    source_map: dict[str, Path] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            input_file = row.get("input_file", "").strip()
            wheel_csv = row.get("wheel_speed_csv", "").strip()
            if not input_file or not wheel_csv:
                continue
            name = Path(input_file).name
            if name in source_map and source_map[name] != Path(wheel_csv):
                raise ValueError(f"multiple processed sources found for {name}")
            source_map[name] = Path(wheel_csv)
    return source_map


def load_source(path: Path, series: str) -> SourceSeries:
    frame = pd.read_csv(path)
    columns = [f"wheel{i}_{series}_rad_s" for i in range(WHEEL_COUNT)]
    required = ["time_s", *columns]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")

    values = frame[required].dropna().sort_values("time_s")
    times = values["time_s"].to_numpy(dtype=float)
    wheels = values[columns].to_numpy(dtype=float)
    if len(times) < 2:
        raise ValueError(f"not enough data in {path}")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"time_s must be strictly increasing in {path}")
    dt = float(np.median(np.diff(times)))
    return SourceSeries(times=times, wheels=wheels, dt=dt, columns=columns)


def interpolate_wheels(source: SourceSeries, query_times: np.ndarray) -> np.ndarray:
    if query_times[0] < source.times[0] or query_times[-1] > source.times[-1]:
        raise ValueError(
            f"requested range [{query_times[0]:.2f}, {query_times[-1]:.2f}] "
            f"outside source [{source.times[0]:.2f}, {source.times[-1]:.2f}]"
        )
    return np.column_stack(
        [np.interp(query_times, source.times, source.wheels[:, i]) for i in range(WHEEL_COUNT)]
    )


def normal_noise_std(
    source: SourceSeries, event_time_s: float, guard_s: float
) -> np.ndarray:
    end_s = event_time_s - guard_s
    mask = source.times < end_s
    normal = source.wheels[mask]
    if len(normal) < 100:
        raise ValueError(
            f"not enough pre-event normal data before {event_time_s:.2f}s "
            f"with {guard_s:.2f}s guard"
        )

    # Remove slow vehicle motion, then use the residual to estimate a Gaussian
    # noise scale independently for each wheel.
    smooth = (
        pd.DataFrame(normal)
        .rolling(window=21, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    residual = (normal - smooth)[10:-10]
    return np.std(residual, axis=0, ddof=1)


def draw_gaussian_noise(
    noise_std: np.ndarray, length: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(length, len(noise_std)),
    )


def add_interpolated_dropout(
    wheels: np.ndarray,
    probability: float,
    max_samples: int,
    rng: np.random.Generator,
) -> int:
    if max_samples == 0 or rng.random() >= probability or len(wheels) < max_samples + 2:
        return 0

    gap = int(rng.integers(1, max_samples + 1))
    start = int(rng.integers(1, len(wheels) - gap))
    end = start + gap
    for wheel in range(WHEEL_COUNT):
        wheels[start:end, wheel] = np.linspace(
            wheels[start - 1, wheel],
            wheels[end, wheel],
            gap + 2,
        )[1:-1]
    return gap


def augment_values(
    values: np.ndarray,
    noise_std: np.ndarray,
    speed_scale: float,
    noise_gain: float,
    dropout_probability: float,
    max_dropout_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    augmented = values * speed_scale
    if noise_gain > 0.0:
        noise = draw_gaussian_noise(noise_std, len(augmented), rng)
        augmented = augmented + noise_gain * noise
    dropout_samples = add_interpolated_dropout(
        augmented, dropout_probability, max_dropout_samples, rng
    )
    return np.maximum(augmented, 0.0), dropout_samples


def write_sample(
    path: Path,
    local_times: np.ndarray,
    wheels: np.ndarray,
    columns: list[str],
) -> None:
    data: dict[str, np.ndarray] = {"time_s": local_times}
    data.update({column: wheels[:, index] for index, column in enumerate(columns)})
    pd.DataFrame(data).to_csv(path, index=False, float_format="%.8f")


def event_sample(
    source: SourceSeries,
    label: EventLabel,
    local_times: np.ndarray,
    event_position_s: float,
    time_scale: float,
    speed_scale: float,
    noise_gain: float,
    noise_std: np.ndarray,
    dropout_probability: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float, float]:
    query_times = label.event_time_s + (local_times - event_position_s) / time_scale
    values = interpolate_wheels(source, query_times)
    values, dropout_samples = augment_values(
        values,
        noise_std,
        speed_scale,
        noise_gain,
        dropout_probability,
        args.max_dropout_samples,
        rng,
    )
    return values, dropout_samples, float(query_times[0]), float(query_times[-1])


def normal_sample(
    source: SourceSeries,
    label: EventLabel,
    local_times: np.ndarray,
    time_scale: float,
    speed_scale: float,
    noise_gain: float,
    noise_std: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float, float]:
    source_duration = float(local_times[-1] / time_scale)
    latest_start = label.event_time_s - args.normal_guard_s - source_duration
    if latest_start <= source.times[0]:
        raise ValueError(
            f"not enough normal data in {label.file} for a {local_times[-1]:.2f}s sample"
        )
    source_start = float(rng.uniform(source.times[0], latest_start))
    query_times = source_start + local_times / time_scale
    values = interpolate_wheels(source, query_times)
    values, dropout_samples = augment_values(
        values,
        noise_std,
        speed_scale,
        noise_gain,
        args.dropout_probability,
        args.max_dropout_samples,
        rng,
    )
    return values, dropout_samples, float(query_times[0]), float(query_times[-1])


def uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high)) if high > low else float(low)


def build_dataset(args: argparse.Namespace) -> list[dict[str, object]]:
    validate_args(args)
    labels = read_event_labels(args.labels)
    source_map = read_source_map(args.batch_summary)
    missing = [label.file for label in labels if label.file not in source_map]
    if missing:
        raise FileNotFoundError(f"processed wheel-speed CSV not found for: {missing}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; use --overwrite to replace samples"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    manifest: list[dict[str, object]] = []

    for label in labels:
        source_path = source_map[label.file]
        source = load_source(source_path, args.series)
        duration_s = args.pre_s + args.post_s
        frame_count = int(round(duration_s / source.dt))
        local_times = np.arange(frame_count, dtype=float) * source.dt
        noise_std = normal_noise_std(source, label.event_time_s, args.normal_guard_s)

        event_specs: list[tuple[int, float, float, float, float]] = [
            (0, args.pre_s, 1.0, 1.0, 0.0)
        ]
        for index in range(1, args.aug_per_event + 1):
            event_specs.append(
                (
                    index,
                    args.pre_s
                    + uniform(rng, -args.event_jitter_s, args.event_jitter_s),
                    uniform(rng, args.time_scale_min, args.time_scale_max),
                    uniform(rng, args.speed_scale_min, args.speed_scale_max),
                    uniform(rng, args.noise_gain_min, args.noise_gain_max),
                )
            )

        for index, event_position, time_scale, speed_scale, noise_gain in event_specs:
            sample_id = f"{label.event_id}_event_{index:03d}"
            sample_path = samples_dir / f"{sample_id}.csv"
            values, dropout_samples, source_start, source_end = event_sample(
                source,
                label,
                local_times,
                event_position,
                time_scale,
                speed_scale,
                noise_gain,
                noise_std,
                0.0 if index == 0 else args.dropout_probability,
                args,
                rng,
            )
            write_sample(sample_path, local_times, values, source.columns)
            manifest.append(
                {
                    "sample_id": sample_id,
                    "sample_type": "event",
                    "source_event_id": label.event_id,
                    "source_file": label.file,
                    "sample_file": str(sample_path.relative_to(args.output_dir)),
                    "event_time_in_sample_s": f"{event_position:.8f}",
                    "source_event_time_s": f"{label.event_time_s:.8f}",
                    "source_start_s": f"{source_start:.8f}",
                    "source_end_s": f"{source_end:.8f}",
                    "time_scale": f"{time_scale:.8f}",
                    "speed_scale": f"{speed_scale:.8f}",
                    "noise_gain": f"{noise_gain:.8f}",
                    "dropout_samples": dropout_samples,
                    "is_augmented": int(index != 0),
                }
            )

        for index in range(1, args.normal_per_event + 1):
            time_scale = uniform(rng, args.time_scale_min, args.time_scale_max)
            speed_scale = uniform(rng, args.speed_scale_min, args.speed_scale_max)
            noise_gain = uniform(rng, args.noise_gain_min, args.noise_gain_max)
            values, dropout_samples, source_start, source_end = normal_sample(
                source,
                label,
                local_times,
                time_scale,
                speed_scale,
                noise_gain,
                noise_std,
                args,
                rng,
            )
            sample_id = f"{label.event_id}_normal_{index:03d}"
            sample_path = samples_dir / f"{sample_id}.csv"
            write_sample(sample_path, local_times, values, source.columns)
            manifest.append(
                {
                    "sample_id": sample_id,
                    "sample_type": "normal",
                    "source_event_id": label.event_id,
                    "source_file": label.file,
                    "sample_file": str(sample_path.relative_to(args.output_dir)),
                    "event_time_in_sample_s": "",
                    "source_event_time_s": f"{label.event_time_s:.8f}",
                    "source_start_s": f"{source_start:.8f}",
                    "source_end_s": f"{source_end:.8f}",
                    "time_scale": f"{time_scale:.8f}",
                    "speed_scale": f"{speed_scale:.8f}",
                    "noise_gain": f"{noise_gain:.8f}",
                    "dropout_samples": dropout_samples,
                    "is_augmented": 1,
                }
            )

    return manifest


def write_metadata(args: argparse.Namespace, manifest: list[dict[str, object]]) -> None:
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "sample_count": len(manifest),
            "event_sample_count": sum(row["sample_type"] == "event" for row in manifest),
            "normal_sample_count": sum(row["sample_type"] == "normal" for row in manifest),
            "important": (
                "Split by source_event_id before training. Never place samples with the "
                "same source_event_id in both train and test."
            ),
        }
    )
    (args.output_dir / "dataset_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    write_metadata(args, manifest)
    print(f"wrote {args.output_dir}")
    print(
        f"samples={len(manifest)} "
        f"events={sum(row['sample_type'] == 'event' for row in manifest)} "
        f"normal={sum(row['sample_type'] == 'normal' for row in manifest)}"
    )


if __name__ == "__main__":
    main()
