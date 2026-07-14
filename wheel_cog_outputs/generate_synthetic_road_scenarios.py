from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable


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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "wheel_cog_outputs" / "synthetic_road_scenario_dataset"
)
WHEEL_COUNT = 4
WHEEL_NAMES = ("FL", "FR", "RL", "RR")

SCENARIOS = {
    "hard_brake": ("急刹", "driving_event"),
    "sharp_turn": ("急转弯", "driving_event"),
    "pothole": ("坑洼", "road_event"),
    "speed_bump": ("减速带", "road_event"),
    "low_tire_pressure": ("低胎压", "tire_fault"),
    "slow_leak": ("慢漏气", "tire_fault"),
    "slip": ("打滑", "driving_event"),
    "rough_to_ice": ("粗糙路面进入冰面", "road_transition"),
    "sensor_anomaly": ("传感器异常", "sensor_fault"),
}


@dataclass(frozen=True)
class EventLabel:
    event_id: str
    file: str
    event_time_s: float


@dataclass(frozen=True)
class SourceSeries:
    times: list[float]
    wheels: list[tuple[float, float, float, float]]
    dt: float
    columns: list[str]


@dataclass(frozen=True)
class ScenarioResult:
    scenario_start_s: float
    scenario_peak_s: float
    target_wheels: tuple[int, ...]
    variant: str
    severity: float
    parameters: dict[str, object]


@dataclass(frozen=True)
class ScenarioConfig:
    wheelbase_m: float
    track_width_m: float
    tire_radius_m: float
    turn_lateral_accel_min_m_s2: float
    turn_lateral_accel_max_m_s2: float
    driven_axle: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate non-blowout hard-negative and fault wheel-speed scenarios "
            "from real pre-event driving windows."
        )
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--series", choices=["raw", "corrected", "ref_comp_on"], default="corrected"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument("--samples-per-scenario", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=50.0)
    parser.add_argument("--scenario-time-s", type=float, default=40.0)
    parser.add_argument("--scenario-jitter-s", type=float, default=1.0)
    parser.add_argument("--normal-guard-s", type=float, default=15.0)
    parser.add_argument("--min-mean-speed-rad-s", type=float, default=20.0)
    parser.add_argument("--max-slow-fraction", type=float, default=0.05)
    parser.add_argument("--wheelbase-m", type=float, default=2.75)
    parser.add_argument("--track-width-m", type=float, default=1.60)
    parser.add_argument("--tire-radius-m", type=float, default=0.33)
    parser.add_argument("--turn-lateral-accel-min-m-s2", type=float, default=2.0)
    parser.add_argument("--turn-lateral-accel-max-m-s2", type=float, default=5.0)
    parser.add_argument(
        "--driven-axle", choices=["front", "rear", "all"], default="rear"
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_scenario <= 0:
        raise ValueError("--samples-per-scenario must be positive")
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    if not 0.0 <= args.scenario_time_s < args.duration_s:
        raise ValueError("--scenario-time-s must be inside the sample")
    if args.scenario_time_s + args.scenario_jitter_s >= args.duration_s - 1.0:
        raise ValueError("scenario time and jitter leave too little post-scenario data")
    if args.scenario_time_s - args.scenario_jitter_s < 0.0:
        raise ValueError("scenario time and jitter extend before the sample")
    if args.normal_guard_s < 0.0:
        raise ValueError("--normal-guard-s cannot be negative")
    if args.min_mean_speed_rad_s < 0.0:
        raise ValueError("--min-mean-speed-rad-s cannot be negative")
    if not 0.0 <= args.max_slow_fraction <= 1.0:
        raise ValueError("--max-slow-fraction must be between 0 and 1")
    if args.wheelbase_m <= 0.0 or args.track_width_m <= 0.0:
        raise ValueError("--wheelbase-m and --track-width-m must be positive")
    if args.tire_radius_m <= 0.0:
        raise ValueError("--tire-radius-m must be positive")
    if not (
        0.0
        < args.turn_lateral_accel_min_m_s2
        <= args.turn_lateral_accel_max_m_s2
    ):
        raise ValueError("invalid turn lateral-acceleration range")


def read_event_labels(path: Path) -> list[EventLabel]:
    labels: list[EventLabel] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            file_name = row.get("file", "").strip()
            if not file_name:
                continue
            labels.append(
                EventLabel(
                    event_id=f"E{index:02d}",
                    file=file_name,
                    event_time_s=float(row["event_time_s"]),
                )
            )
    if not labels:
        raise ValueError(f"no labels found in {path}")
    if len({label.file for label in labels}) != len(labels):
        raise ValueError(f"duplicate files found in {path}")
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
            candidate = Path(wheel_csv)
            if name in source_map and source_map[name] != candidate:
                raise ValueError(f"multiple processed sources found for {name}")
            source_map[name] = candidate
    return source_map


def load_source(path: Path, series: str) -> SourceSeries:
    columns = [f"wheel{i}_{series}_rad_s" for i in range(WHEEL_COUNT)]
    required = ["time_s", *columns]
    times: list[float] = []
    wheels: list[tuple[float, float, float, float]] = []

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing columns in {path}: {missing}")
        for row in reader:
            try:
                time_s = float(row["time_s"])
                values = tuple(float(row[name]) for name in columns)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(time_s) or not all(math.isfinite(value) for value in values):
                continue
            times.append(time_s)
            wheels.append(values)  # type: ignore[arg-type]

    if len(times) < 2:
        raise ValueError(f"not enough valid data in {path}")
    differences = [right - left for left, right in zip(times, times[1:])]
    if any(value <= 0.0 for value in differences):
        raise ValueError(f"time_s must be strictly increasing in {path}")
    dt = float(median(differences))
    return SourceSeries(times=times, wheels=wheels, dt=dt, columns=columns)


def choose_baseline_window(
    source: SourceSeries,
    event_time_s: float,
    duration_s: float,
    normal_guard_s: float,
    min_mean_speed_rad_s: float,
    max_slow_fraction: float,
    rng: random.Random,
) -> tuple[list[float], list[list[float]], float, float]:
    frame_count = int(round(duration_s / source.dt))
    latest_end_s = event_time_s - normal_guard_s
    latest_start_s = latest_end_s - (frame_count - 1) * source.dt
    last_start = bisect.bisect_right(source.times, latest_start_s) - 1
    if last_start < 0:
        raise ValueError(
            f"not enough pre-event data for a {duration_s:.2f}s sample before "
            f"{event_time_s:.2f}s"
        )

    best: tuple[float, float, int] | None = None
    attempts = min(500, max(100, last_start + 1))
    for _ in range(attempts):
        start = rng.randint(0, last_start)
        segment = source.wheels[start : start + frame_count]
        if len(segment) != frame_count:
            continue
        frame_means = [sum(row) / WHEEL_COUNT for row in segment]
        mean_speed = sum(frame_means) / len(frame_means)
        slow_fraction = sum(
            value < min_mean_speed_rad_s for value in frame_means
        ) / len(frame_means)
        score = mean_speed - 50.0 * slow_fraction
        if best is None or score > best[0]:
            best = (score, slow_fraction, start)
        if mean_speed >= min_mean_speed_rad_s and slow_fraction <= max_slow_fraction:
            local_times = [index * source.dt for index in range(frame_count)]
            return (
                local_times,
                [list(row) for row in segment],
                source.times[start],
                source.times[start + frame_count - 1],
            )

    if best is None:
        raise ValueError("could not select a baseline window")
    _, best_slow_fraction, _ = best
    raise ValueError(
        "could not find a sufficiently moving baseline window; "
        f"best slow fraction was {best_slow_fraction:.3f}"
    )


def smoothstep(value: float) -> float:
    clipped = min(max(value, 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def random_wheel(rng: random.Random) -> int:
    return rng.randrange(WHEEL_COUNT)


def apply_hard_brake(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    _: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    del config
    duration = rng.uniform(1.5, 3.2)
    deceleration_fraction = rng.uniform(0.18, 0.42)
    abs_slip = rng.uniform(0.015, 0.07)
    abs_frequency_hz = rng.uniform(7.0, 13.0)
    phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(WHEEL_COUNT)]

    for time_s, row in zip(times, wheels):
        if time_s < onset:
            continue
        elapsed = time_s - onset
        progress = min(elapsed / duration, 1.0)
        common_factor = 1.0 - deceleration_fraction * smoothstep(progress)
        abs_envelope = math.sin(math.pi * progress) ** 2 if elapsed <= duration else 0.0
        for wheel in range(WHEEL_COUNT):
            axle_gain = 1.15 if wheel < 2 else 0.85
            modulation = 0.5 + 0.5 * math.sin(
                2.0 * math.pi * abs_frequency_hz * elapsed + phases[wheel]
            )
            row[wheel] *= common_factor * (
                1.0 - abs_slip * axle_gain * abs_envelope * modulation
            )

    return ScenarioResult(
        scenario_start_s=onset,
        scenario_peak_s=onset + duration,
        target_wheels=(0, 1, 2, 3),
        variant="abs_braking",
        severity=deceleration_fraction,
        parameters={
            "duration_s": duration,
            "deceleration_fraction": deceleration_fraction,
            "abs_slip_fraction": abs_slip,
            "abs_frequency_hz": abs_frequency_hz,
        },
    )


def apply_sharp_turn(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    _: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    duration = rng.uniform(2.0, 4.8)
    peak_lateral_accel = rng.uniform(
        config.turn_lateral_accel_min_m_s2,
        config.turn_lateral_accel_max_m_s2,
    )
    turn_name = rng.choice(("left", "right"))
    turn_sign = 1.0 if turn_name == "left" else -1.0
    side_signs = (-1.0, 1.0, -1.0, 1.0)
    peak_geometry: dict[str, float] = {}
    largest_envelope = -1.0

    for time_s, row in zip(times, wheels):
        progress = (time_s - onset) / duration
        if not 0.0 <= progress <= 1.0:
            continue
        envelope = math.sin(math.pi * progress) ** 2
        mean_wheel_speed = sum(row) / WHEEL_COUNT
        vehicle_speed_m_s = mean_wheel_speed * config.tire_radius_m
        lateral_accel = peak_lateral_accel * envelope
        if vehicle_speed_m_s <= 0.0 or lateral_accel <= 1e-12:
            continue

        turn_radius_m = vehicle_speed_m_s * vehicle_speed_m_s / lateral_accel
        turn_radius_m = max(turn_radius_m, config.track_width_m / 2.0 + 0.1)
        for wheel in range(WHEEL_COUNT):
            lateral_radius_m = (
                turn_radius_m
                + turn_sign * side_signs[wheel] * config.track_width_m / 2.0
            )
            path_radius_m = (
                math.hypot(lateral_radius_m, config.wheelbase_m)
                if wheel < 2
                else lateral_radius_m
            )
            row[wheel] *= path_radius_m / turn_radius_m

        if envelope > largest_envelope:
            largest_envelope = envelope
            peak_geometry = {
                "turn_radius_m": turn_radius_m,
                "vehicle_speed_m_s": vehicle_speed_m_s,
                "yaw_rate_deg_s": math.degrees(vehicle_speed_m_s / turn_radius_m),
                "equivalent_steering_angle_deg": math.degrees(
                    math.atan(config.wheelbase_m / turn_radius_m)
                ),
            }

    return ScenarioResult(
        scenario_start_s=onset,
        scenario_peak_s=onset + duration / 2.0,
        target_wheels=(0, 1, 2, 3),
        variant=f"ackermann_{turn_name}_turn",
        severity=peak_lateral_accel,
        parameters={
            "model": "ackermann_no_slip",
            "duration_s": duration,
            "turn_direction": turn_name,
            "peak_lateral_accel_m_s2": peak_lateral_accel,
            "wheelbase_m": config.wheelbase_m,
            "track_width_m": config.track_width_m,
            "tire_radius_m": config.tire_radius_m,
            "peak_geometry": peak_geometry,
        },
    )


def apply_pothole(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    _: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    del config
    target = random_wheel(rng)
    duration = rng.uniform(0.35, 0.85)
    amplitude = rng.uniform(0.08, 0.28)
    cycles = rng.uniform(1.5, 3.5)

    for time_s, row in zip(times, wheels):
        progress = (time_s - onset) / duration
        if not 0.0 <= progress <= 1.0:
            continue
        impact = -amplitude * math.exp(-3.5 * progress) * math.cos(
            2.0 * math.pi * cycles * progress
        )
        row[target] *= 1.0 + impact
        same_axle = target ^ 1
        row[same_axle] *= 1.0 + 0.12 * impact

    return ScenarioResult(
        scenario_start_s=onset,
        scenario_peak_s=onset,
        target_wheels=(target,),
        variant="single_wheel_impact",
        severity=amplitude,
        parameters={
            "duration_s": duration,
            "peak_fraction": amplitude,
            "damped_cycles": cycles,
        },
    )


def apply_speed_bump(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    _: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    del config
    axle_duration = rng.uniform(0.55, 1.1)
    rear_delay = rng.uniform(0.18, 0.45)
    amplitude = rng.uniform(0.045, 0.16)
    axle_asymmetries = (rng.uniform(0.96, 1.04), rng.uniform(0.96, 1.04))

    for time_s, row in zip(times, wheels):
        for axle_index, (axle, axle_onset) in enumerate(
            (((0, 1), onset), ((2, 3), onset + rear_delay))
        ):
            progress = (time_s - axle_onset) / axle_duration
            if not 0.0 <= progress <= 1.0:
                continue
            impact = -amplitude * math.exp(-2.8 * progress) * math.cos(
                4.0 * math.pi * progress
            )
            row[axle[0]] *= 1.0 + impact
            row[axle[1]] *= 1.0 + impact * axle_asymmetries[axle_index]

    return ScenarioResult(
        scenario_start_s=onset,
        scenario_peak_s=onset + rear_delay,
        target_wheels=(0, 1, 2, 3),
        variant="front_then_rear_axle",
        severity=amplitude,
        parameters={
            "axle_duration_s": axle_duration,
            "rear_axle_delay_s": rear_delay,
            "peak_fraction": amplitude,
            "axle_asymmetry_factors": axle_asymmetries,
        },
    )


def apply_low_tire_pressure(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    _: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    del onset, config
    target = random_wheel(rng)
    radius_reduction = rng.uniform(0.008, 0.035)
    drift = rng.uniform(0.0005, 0.002)
    phase = rng.uniform(0.0, 2.0 * math.pi)

    for time_s, row in zip(times, wheels):
        periodic_drift = drift * math.sin(2.0 * math.pi * time_s / 18.0 + phase)
        row[target] *= 1.0 / (1.0 - radius_reduction) + periodic_drift

    return ScenarioResult(
        scenario_start_s=0.0,
        scenario_peak_s=0.0,
        target_wheels=(target,),
        variant="steady_effective_radius_reduction",
        severity=radius_reduction,
        parameters={
            "effective_radius_reduction_fraction": radius_reduction,
            "slow_drift_fraction": drift,
        },
    )


def apply_slow_leak(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    _: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    del config
    target = random_wheel(rng)
    final_radius_reduction = rng.uniform(0.01, 0.035)
    ramp_duration = max(times[-1] - onset, 0.1)

    for time_s, row in zip(times, wheels):
        if time_s < onset:
            continue
        progress = smoothstep((time_s - onset) / ramp_duration)
        radius_reduction = final_radius_reduction * progress
        row[target] *= 1.0 / (1.0 - radius_reduction)

    return ScenarioResult(
        scenario_start_s=onset,
        scenario_peak_s=times[-1],
        target_wheels=(target,),
        variant="time_compressed_radius_drift",
        severity=final_radius_reduction,
        parameters={
            "ramp_duration_s": ramp_duration,
            "final_effective_radius_reduction_fraction": final_radius_reduction,
            "time_compressed": True,
        },
    )


def apply_slip(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    index: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    mode = "drive_spin" if index % 2 == 0 else "braking_slip"
    if mode == "drive_spin":
        candidates = {
            "front": (0, 1),
            "rear": (2, 3),
            "all": (0, 1, 2, 3),
        }[config.driven_axle]
        target = rng.choice(candidates)
        duration = rng.uniform(1.0, 2.4)
        peak_slip_magnitude = rng.uniform(0.10, 0.25)
        rise_fraction = rng.uniform(0.12, 0.25)
        settle_fraction = rng.uniform(0.10, 0.20)
        hold_fraction = rng.uniform(0.20, 0.40)
        control_reduction = rng.uniform(0.25, 0.55)
        signed_peak_slip = peak_slip_magnitude
        variant = "traction_controlled_wheel_spin"
    else:
        target = random_wheel(rng)
        duration = rng.uniform(0.6, 1.6)
        peak_slip_magnitude = rng.uniform(0.35, 0.85)
        rise_fraction = rng.uniform(0.08, 0.18)
        settle_fraction = rng.uniform(0.12, 0.22)
        hold_fraction = rng.uniform(0.20, 0.45)
        control_reduction = rng.uniform(0.20, 0.50)
        signed_peak_slip = -peak_slip_magnitude
        variant = "abs_controlled_braking_slip"

    rise_duration = duration * rise_fraction
    settle_duration = duration * settle_fraction
    hold_duration = duration * hold_fraction
    recovery_duration = duration - rise_duration - settle_duration - hold_duration
    controlled_slip_magnitude = peak_slip_magnitude * (1.0 - control_reduction)

    other_wheels = [wheel for wheel in range(WHEEL_COUNT) if wheel != target]
    calibration_ratios = []
    for time_s, row in zip(times, wheels):
        if onset - 2.0 <= time_s < onset:
            reference = median(row[wheel] for wheel in other_wheels)
            if reference > 1e-9:
                calibration_ratios.append(row[target] / reference)
    target_calibration = (
        min(max(float(median(calibration_ratios)), 0.95), 1.05)
        if calibration_ratios
        else 1.0
    )

    peak_time_s = onset
    actual_peak_magnitude = -1.0
    peak_reference_wheel_speed = 0.0
    for time_s, row in zip(times, wheels):
        elapsed = time_s - onset
        if not 0.0 <= elapsed <= duration:
            continue

        if elapsed <= rise_duration:
            slip_magnitude = peak_slip_magnitude * smoothstep(
                elapsed / rise_duration
            )
        elif elapsed <= rise_duration + settle_duration:
            settle_progress = (elapsed - rise_duration) / settle_duration
            slip_magnitude = peak_slip_magnitude * (
                1.0 - control_reduction * smoothstep(settle_progress)
            )
        elif elapsed <= rise_duration + settle_duration + hold_duration:
            slip_magnitude = controlled_slip_magnitude
        else:
            recovery_progress = (
                elapsed - rise_duration - settle_duration - hold_duration
            ) / recovery_duration
            slip_magnitude = controlled_slip_magnitude * (
                1.0 - smoothstep(recovery_progress)
            )

        reference_wheel_speed = float(median(row[wheel] for wheel in other_wheels))
        free_rolling_target_speed = reference_wheel_speed * target_calibration
        if mode == "drive_spin":
            row[target] = free_rolling_target_speed / (1.0 - slip_magnitude)
        else:
            row[target] = free_rolling_target_speed * (1.0 - slip_magnitude)

        if slip_magnitude > actual_peak_magnitude:
            actual_peak_magnitude = slip_magnitude
            peak_time_s = time_s
            peak_reference_wheel_speed = reference_wheel_speed

    return ScenarioResult(
        scenario_start_s=onset,
        scenario_peak_s=peak_time_s,
        target_wheels=(target,),
        variant=variant,
        severity=actual_peak_magnitude,
        parameters={
            "model": "longitudinal_slip_ratio_with_controller",
            "slip_ratio_definition": (
                "(wheel_circumferential_speed - vehicle_speed) / "
                "max(abs(wheel_circumferential_speed), abs(vehicle_speed))"
            ),
            "mode": mode,
            "duration_s": duration,
            "rise_duration_s": rise_duration,
            "controller_settle_duration_s": settle_duration,
            "controlled_hold_duration_s": hold_duration,
            "recovery_duration_s": recovery_duration,
            "controller_reduction_fraction": control_reduction,
            "requested_peak_slip_ratio": signed_peak_slip,
            "actual_peak_slip_ratio": (
                actual_peak_magnitude
                if mode == "drive_spin"
                else -actual_peak_magnitude
            ),
            "controlled_slip_ratio_magnitude": controlled_slip_magnitude,
            "vehicle_speed_reference": "median_of_three_non_target_wheels",
            "peak_reference_wheel_speed_rad_s": peak_reference_wheel_speed,
            "peak_reference_vehicle_speed_m_s": (
                peak_reference_wheel_speed * config.tire_radius_m
            ),
            "target_calibration_factor": target_calibration,
            "driven_axle": config.driven_axle,
        },
    )


def apply_rough_to_ice(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    index: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    """Model a rough-road segment followed by a persistent low-friction surface.

    Wheel speed alone cannot reveal a friction change during steady free rolling, so
    the variants deliberately include one coast case with no invented slip signal.
    The other cases add bounded wheel-slip signatures caused by throttle, braking,
    or cornering after the road transition.
    """
    variants = (
        "low_mu_drive_tcs",
        "low_mu_brake_abs",
        "low_mu_turn_esc",
        "coast_no_observable_slip",
    )
    variant = variants[index % len(variants)]
    rough_duration = rng.uniform(2.5, 4.5)
    rough_start = max(times[0], onset - rough_duration)
    rough_ramp_duration = min(rng.uniform(0.20, 0.45), rough_duration / 3.0)
    transition_duration = rng.uniform(0.15, 0.45)
    rough_amplitude = rng.uniform(0.008, 0.025)
    rough_common_frequency_hz = rng.uniform(5.0, 11.0)
    rough_wheel_frequencies_hz = [
        rng.uniform(12.0, 26.0) for _ in range(WHEEL_COUNT)
    ]
    rough_phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(WHEEL_COUNT)]
    dry_mu_proxy = rng.uniform(0.65, 0.90)
    ice_mu_proxy = rng.uniform(0.05, 0.18)

    # Preserve each wheel's normal rolling-speed ratio before roughness is injected.
    rolling_ratios: list[float] = []
    for target in range(WHEEL_COUNT):
        ratios: list[float] = []
        for time_s, row in zip(times, wheels):
            if onset - 2.0 <= time_s < onset:
                reference = float(median(row))
                if reference > 1e-9:
                    ratios.append(row[target] / reference)
        rolling_ratios.append(
            min(max(float(median(ratios)), 0.95), 1.05) if ratios else 1.0
        )

    # Rough asphalt is represented by a correlated body component plus a smaller
    # wheel-local component. The disturbance fades as the vehicle reaches the ice.
    for time_s, row in zip(times, wheels):
        if not rough_start <= time_s <= onset + transition_duration:
            continue
        if time_s < rough_start + rough_ramp_duration:
            envelope = smoothstep(
                (time_s - rough_start) / max(rough_ramp_duration, 1e-9)
            )
        elif time_s <= onset:
            envelope = 1.0
        else:
            envelope = 1.0 - smoothstep(
                (time_s - onset) / max(transition_duration, 1e-9)
            )

        common = (
            0.65
            * math.sin(2.0 * math.pi * rough_common_frequency_hz * time_s)
            + 0.35
            * math.sin(
                2.0 * math.pi * rough_common_frequency_hz * 1.73 * time_s + 0.8
            )
        )
        for wheel in range(WHEEL_COUNT):
            local = math.sin(
                2.0 * math.pi * rough_wheel_frequencies_hz[wheel] * time_s
                + rough_phases[wheel]
            )
            perturbation = rough_amplitude * envelope * (0.65 * common + 0.35 * local)
            row[wheel] *= max(0.5, 1.0 + perturbation)

    peak_time_s = onset + transition_duration
    target_wheels: tuple[int, ...]
    maneuver_parameters: dict[str, object]

    if variant == "coast_no_observable_slip":
        target_wheels = (0, 1, 2, 3)
        maneuver_parameters = {
            "wheel_speed_observability": (
                "friction change is not directly observable during steady free rolling"
            ),
            "requested_peak_slip_ratio": 0.0,
            "controlled_slip_ratio_magnitude": 0.0,
        }
    else:
        rise_duration = transition_duration
        settle_duration = rng.uniform(0.25, 0.60)
        hold_duration = rng.uniform(0.80, 1.80)
        recovery_duration = rng.uniform(0.30, 0.80)
        maneuver_duration = (
            rise_duration + settle_duration + hold_duration + recovery_duration
        )

        if variant == "low_mu_drive_tcs":
            target_wheels = {
                "front": (0, 1),
                "rear": (2, 3),
                "all": (0, 1, 2, 3),
            }[config.driven_axle]
            peak_slip = rng.uniform(0.12, 0.30)
            controlled_slip = rng.uniform(0.035, 0.10)
            signed_peak_slip = peak_slip
            controller = "TCS"
        elif variant == "low_mu_brake_abs":
            target_wheels = (0, 1, 2, 3)
            peak_slip = rng.uniform(0.25, 0.55)
            controlled_slip = rng.uniform(0.06, 0.16)
            signed_peak_slip = -peak_slip
            controller = "ABS"
        else:
            target_wheels = (0, 1, 2, 3)
            peak_slip = rng.uniform(0.035, 0.11)
            controlled_slip = rng.uniform(0.01, min(0.045, peak_slip * 0.65))
            signed_peak_slip = peak_slip
            controller = "ESC"

        wheel_gains = [rng.uniform(0.92, 1.08) for _ in range(WHEEL_COUNT)]
        controller_frequency_hz = rng.uniform(6.0, 11.0)
        controller_phases = [
            rng.uniform(0.0, 2.0 * math.pi) for _ in range(WHEEL_COUNT)
        ]
        turn_direction = rng.choice(("left", "right"))
        turn_sign = 1.0 if turn_direction == "left" else -1.0
        side_signs = (-1.0, 1.0, -1.0, 1.0)

        for time_s, row in zip(times, wheels):
            elapsed = time_s - onset
            if not 0.0 <= elapsed <= maneuver_duration:
                continue
            if elapsed <= rise_duration:
                slip = peak_slip * smoothstep(elapsed / max(rise_duration, 1e-9))
            elif elapsed <= rise_duration + settle_duration:
                progress = (elapsed - rise_duration) / settle_duration
                slip = peak_slip + (controlled_slip - peak_slip) * smoothstep(progress)
            elif elapsed <= rise_duration + settle_duration + hold_duration:
                slip = controlled_slip
            else:
                progress = (
                    elapsed - rise_duration - settle_duration - hold_duration
                ) / recovery_duration
                slip = controlled_slip * (1.0 - smoothstep(progress))

            original_row = row.copy()
            reference = float(median(original_row))
            if variant == "low_mu_drive_tcs" and len(target_wheels) < WHEEL_COUNT:
                reference = float(
                    median(
                        original_row[wheel]
                        for wheel in range(WHEEL_COUNT)
                        if wheel not in target_wheels
                    )
                )

            for wheel in target_wheels:
                modulation = 0.90 + 0.10 * math.sin(
                    2.0 * math.pi * controller_frequency_hz * elapsed
                    + controller_phases[wheel]
                )
                wheel_slip = min(slip * wheel_gains[wheel] * modulation, 0.90)
                free_rolling_speed = reference * rolling_ratios[wheel]
                if variant == "low_mu_drive_tcs":
                    row[wheel] = free_rolling_speed / max(1.0 - wheel_slip, 0.10)
                elif variant == "low_mu_brake_abs":
                    row[wheel] = free_rolling_speed * (1.0 - wheel_slip)
                else:
                    axle_gain = 1.15 if wheel < 2 else 0.85
                    lateral_delta = (
                        turn_sign * side_signs[wheel] * wheel_slip * axle_gain
                    )
                    row[wheel] = free_rolling_speed * (1.0 + lateral_delta)

        maneuver_parameters = {
            "controller": controller,
            "maneuver_duration_s": maneuver_duration,
            "rise_duration_s": rise_duration,
            "controller_settle_duration_s": settle_duration,
            "controlled_hold_duration_s": hold_duration,
            "recovery_duration_s": recovery_duration,
            "requested_peak_slip_ratio": signed_peak_slip,
            "controlled_slip_ratio_magnitude": controlled_slip,
            "controller_frequency_hz": controller_frequency_hz,
        }
        if variant == "low_mu_turn_esc":
            maneuver_parameters["turn_direction"] = turn_direction

    severity = 1.0 - ice_mu_proxy / dry_mu_proxy
    return ScenarioResult(
        scenario_start_s=rough_start,
        scenario_peak_s=peak_time_s,
        target_wheels=target_wheels,
        variant=variant,
        severity=severity,
        parameters={
            "model": "rough_surface_to_low_mu_transition",
            "road_sequence": ["rough_asphalt", "ice"],
            "rough_start_s": rough_start,
            "ice_transition_start_s": onset,
            "transition_duration_s": transition_duration,
            "rough_duration_s": rough_duration,
            "rough_ramp_duration_s": rough_ramp_duration,
            "rough_amplitude_fraction": rough_amplitude,
            "rough_common_frequency_hz": rough_common_frequency_hz,
            "rough_wheel_frequencies_hz": rough_wheel_frequencies_hz,
            "dry_mu_proxy": dry_mu_proxy,
            "ice_mu_proxy": ice_mu_proxy,
            "ice_persists_to_sample_end": True,
            "friction_drop_fraction": severity,
            "variant_cycle": list(variants),
            **maneuver_parameters,
        },
    )


def apply_sensor_anomaly(
    times: list[float],
    wheels: list[list[float]],
    onset: float,
    rng: random.Random,
    index: int,
    config: ScenarioConfig,
) -> ScenarioResult:
    del config
    target = random_wheel(rng)
    variants = ("spike", "dropout", "stuck", "bias_step", "noise_burst")
    variant = variants[index % len(variants)]
    start_index = min(range(len(times)), key=lambda position: abs(times[position] - onset))
    parameters: dict[str, object]
    severity: float
    peak_s = onset

    if variant == "spike":
        count = rng.randint(1, 4)
        factor = rng.choice((rng.uniform(0.15, 0.5), rng.uniform(1.5, 2.5)))
        for position in range(start_index, min(start_index + count, len(wheels))):
            wheels[position][target] *= factor
        severity = abs(factor - 1.0)
        parameters = {"sample_count": count, "multiplicative_factor": factor}
    elif variant == "dropout":
        duration = rng.uniform(0.05, 0.35)
        end_s = onset + duration
        for time_s, row in zip(times, wheels):
            if onset <= time_s <= end_s:
                row[target] = 0.0
        severity = 1.0
        peak_s = end_s
        parameters = {"duration_s": duration, "output_rad_s": 0.0}
    elif variant == "stuck":
        duration = rng.uniform(0.6, 2.8)
        held_value = wheels[max(start_index - 1, 0)][target]
        for time_s, row in zip(times, wheels):
            if onset <= time_s <= onset + duration:
                row[target] = held_value
        severity = duration
        peak_s = onset + duration
        parameters = {"duration_s": duration, "held_value_rad_s": held_value}
    elif variant == "bias_step":
        bias = rng.choice((-1.0, 1.0)) * rng.uniform(0.06, 0.22)
        for time_s, row in zip(times, wheels):
            if time_s >= onset:
                row[target] *= 1.0 + bias
        severity = abs(bias)
        peak_s = times[-1]
        parameters = {"bias_fraction": bias, "persistent": True}
    else:
        duration = rng.uniform(0.3, 1.5)
        noise_sigma = rng.uniform(0.06, 0.20)
        for time_s, row in zip(times, wheels):
            if onset <= time_s <= onset + duration:
                row[target] *= max(0.0, 1.0 + rng.gauss(0.0, noise_sigma))
        severity = noise_sigma
        peak_s = onset + duration
        parameters = {"duration_s": duration, "noise_sigma_fraction": noise_sigma}

    return ScenarioResult(
        scenario_start_s=onset,
        scenario_peak_s=peak_s,
        target_wheels=(target,),
        variant=variant,
        severity=severity,
        parameters=parameters,
    )


SCENARIO_FUNCTIONS: dict[
    str,
    Callable[
        [
            list[float],
            list[list[float]],
            float,
            random.Random,
            int,
            ScenarioConfig,
        ],
        ScenarioResult,
    ],
] = {
    "hard_brake": apply_hard_brake,
    "sharp_turn": apply_sharp_turn,
    "pothole": apply_pothole,
    "speed_bump": apply_speed_bump,
    "low_tire_pressure": apply_low_tire_pressure,
    "slow_leak": apply_slow_leak,
    "slip": apply_slip,
    "rough_to_ice": apply_rough_to_ice,
    "sensor_anomaly": apply_sensor_anomaly,
}


def sanitize_wheels(wheels: list[list[float]]) -> None:
    for row in wheels:
        for wheel, value in enumerate(row):
            if not math.isfinite(value):
                raise ValueError("scenario generated a non-finite wheel speed")
            row[wheel] = max(value, 0.0)


def write_sample(
    path: Path, times: list[float], wheels: list[list[float]], columns: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", *columns])
        for time_s, row in zip(times, wheels):
            writer.writerow([f"{time_s:.8f}", *(f"{value:.8f}" for value in row)])


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {path}; use --overwrite to replace it"
            )
        shutil.rmtree(path)
    samples_dir = path / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir


def build_dataset(args: argparse.Namespace) -> list[dict[str, object]]:
    validate_args(args)
    labels = read_event_labels(args.labels)
    source_map = read_source_map(args.batch_summary)
    missing = [label.file for label in labels if label.file not in source_map]
    if missing:
        raise FileNotFoundError(f"processed wheel-speed CSV not found for: {missing}")

    samples_dir = prepare_output_dir(args.output_dir, args.overwrite)
    rng = random.Random(args.seed)
    scenario_config = ScenarioConfig(
        wheelbase_m=args.wheelbase_m,
        track_width_m=args.track_width_m,
        tire_radius_m=args.tire_radius_m,
        turn_lateral_accel_min_m_s2=args.turn_lateral_accel_min_m_s2,
        turn_lateral_accel_max_m_s2=args.turn_lateral_accel_max_m_s2,
        driven_axle=args.driven_axle,
    )
    manifest: list[dict[str, object]] = []

    for source_index, label in enumerate(labels):
        assigned_indices = [
            index
            for index in range(args.samples_per_scenario)
            if index % len(labels) == source_index
        ]
        if not assigned_indices:
            continue
        source = load_source(source_map[label.file], args.series)

        for scenario in args.scenarios:
            scenario_zh, condition_type = SCENARIOS[scenario]
            for sample_index in assigned_indices:
                times, wheels, source_start, source_end = choose_baseline_window(
                    source=source,
                    event_time_s=label.event_time_s,
                    duration_s=args.duration_s,
                    normal_guard_s=args.normal_guard_s,
                    min_mean_speed_rad_s=args.min_mean_speed_rad_s,
                    max_slow_fraction=args.max_slow_fraction,
                    rng=rng,
                )
                requested_onset = args.scenario_time_s + rng.uniform(
                    -args.scenario_jitter_s, args.scenario_jitter_s
                )
                result = SCENARIO_FUNCTIONS[scenario](
                    times,
                    wheels,
                    requested_onset,
                    rng,
                    sample_index,
                    scenario_config,
                )
                sanitize_wheels(wheels)

                sample_id = f"{scenario}_{sample_index + 1:03d}"
                sample_path = samples_dir / f"{sample_id}.csv"
                write_sample(sample_path, times, wheels, source.columns)
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
                        "time_scale": "1.00000000",
                        "speed_scale": "1.00000000",
                        "noise_gain": "0.00000000",
                        "dropout_samples": 0,
                        "is_augmented": 1,
                        "scenario": scenario,
                        "scenario_zh": scenario_zh,
                        "condition_type": condition_type,
                        "scenario_start_s": f"{result.scenario_start_s:.8f}",
                        "scenario_peak_s": f"{result.scenario_peak_s:.8f}",
                        "target_wheels": ";".join(
                            WHEEL_NAMES[index] for index in result.target_wheels
                        ),
                        "variant": result.variant,
                        "severity": f"{result.severity:.8f}",
                        "parameters_json": json.dumps(
                            result.parameters, ensure_ascii=False, sort_keys=True
                        ),
                        "expected_blowout": 0,
                    }
                )

    manifest.sort(key=lambda row: str(row["sample_id"]))
    return manifest


def write_metadata(args: argparse.Namespace, manifest: list[dict[str, object]]) -> None:
    if not manifest:
        raise ValueError("no samples generated")
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    scenario_counts = {
        scenario: sum(row["scenario"] == scenario for row in manifest)
        for scenario in args.scenarios
    }
    config = {
        key: [str(item) for item in value]
        if isinstance(value, list) and value and isinstance(value[0], Path)
        else str(value)
        if isinstance(value, Path)
        else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "sample_count": len(manifest),
            "scenario_counts": scenario_counts,
            "wheel_mapping": {f"wheel{index}": name for index, name in enumerate(WHEEL_NAMES)},
            "expected_blowout": 0,
            "important": (
                "These are synthetic wheel-speed signatures based on real pre-event "
                "windows. Use them for algorithm stress tests, not as evidence of real-road "
                "performance. Split evaluation by source_event_id."
            ),
        }
    )
    (args.output_dir / "dataset_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    write_metadata(args, manifest)
    print(f"wrote {args.output_dir}")
    print(f"samples={len(manifest)} scenarios={len(args.scenarios)}")
    for scenario in args.scenarios:
        count = sum(row["scenario"] == scenario for row in manifest)
        print(f"  {scenario}: {count}")


if __name__ == "__main__":
    main()
