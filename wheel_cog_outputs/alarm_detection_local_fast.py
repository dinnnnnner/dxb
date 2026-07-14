from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


WHEEL_COUNT = 4
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "wheel_cog_outputs" / "wheel_cog_outputs" / "wheel_speed_raw_vs_corrected.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "wheel_cog_outputs" / "wheel_cog_outputs" / "alarm_detection_results_fast.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "wheel_cog_outputs" / "wheel_cog_outputs" / "alarm_detection_summary_fast.json"


def diffs_wheels(wheels: list[float]) -> list[float]:
    fl, fr, rl, rr = wheels
    total = fl + fr + rl + rr
    norm = 4.0 / total if total > 1e-9 else 0.0
    return [
        (fl - fr) * norm,
        (rl - rr) * norm,
        (rl - fl) * norm,
        (rr - fr) * norm,
    ]


def wheel_relative_deviation(wheels: list[float]) -> dict[str, float]:
    names = ("wheel0", "wheel1", "wheel2", "wheel3")
    out: dict[str, float] = {}
    for idx, name in enumerate(names):
        others = [value for j, value in enumerate(wheels) if j != idx]
        ref = sum(others) / max(1, len(others))
        out[name] = wheels[idx] / ref - 1.0 if ref > 1e-9 else 0.0
    return out


@dataclass
class FastAlarmConfig:
    min_avg_speed: float = 20.0
    fast_alpha: float = 0.45
    slow_alpha: float = 0.001046674642901591
    filter_alpha: float = 0.65
    output_scale: float = 200.0
    use_wheel_feature: bool = False
    warmup_frames: int = 3000
 
    noise_alpha: float = 0.00402139496108789
    center_alpha: float = 0.0435206345569064
    noise_floor: float = 0.08
    enter_min: float = 0.7476840122076729
    enter_noise_gain: float = 3.140654072838856
    exit_min: float = 0.1928536858667876
    exit_noise_gain: float = 2.6050079637872177
    evidence_decay: float = 0.9709366109407416
    evidence_on: float = 1.419128576581635
    evidence_off: float = 0.2849447795174631
    evidence_input_cap: float = 1.10
    instant_on: float = 999.0
    freeze_enter: float = 0.38225805635222615

    recovery_enabled: bool = True
    recovery_frames: int = 100
    recovery_holdoff_frames: int = 20
    recovery_slow_alpha: float = 0.08
    recovery_center_alpha: float = 0.20
    recovery_noise_alpha: float = 0.05
    hold_alarm_below_min_speed: bool = True


@dataclass
class WheelSpeedFrame:
    t_sec: float
    wheels: list[float]


@dataclass
class DetectionResult:
    t_sec: float
    wheels: list[float]
    avg_speed: float
    combind: float
    legacy_combind: float
    wheel_feature: float
    feature_baseline: float
    innovation: float
    alarm: bool
    score: float
    off_threshold: float
    on_threshold: float
    evidence: float
    signed_evidence: float
    enter_threshold: float
    exit_threshold: float
    noise: float
    recovery_active: bool
    recovery_frames_left: int
    alarm_wheel: str | None
    alarm_wheel_dev: float | None


@dataclass
class FeatureEstimate:
    selected: float
    legacy_combind: float
    wheel_feature: float
    wheel_name: str
    wheel_dev: float


class EwmaResidualEstimator:
    def __init__(self, cfg: FastAlarmConfig):
        self.cfg = cfg
        self.fast: list[float] | None = None
        self.slow: list[float] | None = None
        self.filtered = 0.0
        self.initialized = False

    def update(
        self,
        wheels: list[float],
        adapt_slow: bool,
        slow_alpha: float | None = None,
    ) -> FeatureEstimate:
        if not self.initialized:
            self.fast = wheels.copy()
            self.slow = wheels.copy()
            self.filtered = 0.0
            self.initialized = True
            return FeatureEstimate(0.0, 0.0, 0.0, "wheel0", 0.0)

        assert self.fast is not None
        assert self.slow is not None

        for idx, value in enumerate(wheels):
            self.fast[idx] += self.cfg.fast_alpha * (value - self.fast[idx])

        raw = self.calculate_residual_value(diffs_wheels(self.fast), diffs_wheels(self.slow))
        self.filtered += self.cfg.filter_alpha * (raw - self.filtered)
        legacy_combind = self.filtered * self.cfg.output_scale

        fast_rel = wheel_relative_deviation(self.fast)
        slow_rel = wheel_relative_deviation(self.slow)
        wheel_residuals = {
            wheel: (fast_rel[wheel] - slow_rel[wheel]) * 100.0
            for wheel in fast_rel
        }
        wheel_name, wheel_feature = max(wheel_residuals.items(), key=lambda item: abs(item[1]))
        if self.cfg.use_wheel_feature:
            selected = wheel_feature if abs(wheel_feature) > abs(legacy_combind) else legacy_combind
        else:
            selected = legacy_combind

        if adapt_slow:
            alpha = self.cfg.slow_alpha if slow_alpha is None else slow_alpha
            for idx, value in enumerate(wheels):
                self.slow[idx] += alpha * (value - self.slow[idx])

        return FeatureEstimate(
            selected=selected,
            legacy_combind=legacy_combind,
            wheel_feature=wheel_feature,
            wheel_name=wheel_name,
            wheel_dev=wheel_feature / 100.0,
        )

    @staticmethod
    def calculate_residual_value(fast_diffs: list[float], slow_diffs: list[float]) -> float:
        residuals = [slow_diffs[i] - fast_diffs[i] for i in range(WHEEL_COUNT)]
        return 0.5 * (residuals[0] + residuals[3] - (residuals[2] + residuals[1]))


class LeakyEvidenceDetector:
    def __init__(self, cfg: FastAlarmConfig):
        self.cfg = cfg
        self.noise = cfg.noise_floor
        self.pos_evidence = 0.0
        self.neg_evidence = 0.0
        self.score = 0.0
        self.alarm = False
        self.center: float | None = None
        self.innovation = 0.0
        self.enter_threshold = cfg.enter_min
        self.exit_threshold = cfg.exit_min

    def update(self, x: float, adapt_noise: bool) -> bool:
        if self.center is None:
            self.center = x
            self.innovation = 0.0
            return False

        self.innovation = x - self.center
        abs_x = abs(self.innovation)
        if adapt_noise and not self.alarm:
            self.noise += self.cfg.noise_alpha * (abs_x - self.noise)
            self.noise = max(self.cfg.noise_floor, self.noise)
            self.center += self.cfg.center_alpha * (x - self.center)

        self.enter_threshold = max(self.cfg.enter_min, self.cfg.enter_noise_gain * self.noise)
        self.exit_threshold = max(self.cfg.exit_min, self.cfg.exit_noise_gain * self.noise)

        evidence_x = max(-self.cfg.evidence_input_cap, min(self.cfg.evidence_input_cap, self.innovation))
        self.pos_evidence = max(0.0, self.cfg.evidence_decay * self.pos_evidence + evidence_x - self.enter_threshold)
        self.neg_evidence = max(0.0, self.cfg.evidence_decay * self.neg_evidence - evidence_x - self.enter_threshold)

        evidence = max(self.pos_evidence, self.neg_evidence)
        self.score = evidence
        if self.alarm:
            self.alarm = evidence > self.cfg.evidence_off or abs_x > self.exit_threshold
        else:
            self.alarm = evidence >= self.cfg.evidence_on or abs_x >= self.cfg.instant_on
        return self.alarm

    def signed_evidence(self) -> float:
        return self.pos_evidence if self.pos_evidence >= self.neg_evidence else -self.neg_evidence

    def feature_baseline(self) -> float:
        return 0.0 if self.center is None else self.center

    def clear_evidence(self) -> None:
        self.pos_evidence = 0.0
        self.neg_evidence = 0.0
        self.score = 0.0

    def recover_towards(self, x: float) -> None:
        if self.center is None:
            self.center = x
            self.innovation = 0.0
        else:
            self.innovation = x - self.center
            abs_x = abs(self.innovation)
            self.noise += self.cfg.recovery_noise_alpha * (abs_x - self.noise)
            self.noise = max(self.cfg.noise_floor, self.noise)
            self.center += self.cfg.recovery_center_alpha * (x - self.center)
        self.enter_threshold = max(self.cfg.enter_min, self.cfg.enter_noise_gain * self.noise)
        self.exit_threshold = max(self.cfg.exit_min, self.cfg.exit_noise_gain * self.noise)
        self.clear_evidence()


class WheelAlarmFastPipeline:
    def __init__(self, cfg: FastAlarmConfig | None = None):
        self.cfg = cfg or FastAlarmConfig()
        self.estimator = EwmaResidualEstimator(self.cfg)
        self.detector = LeakyEvidenceDetector(self.cfg)
        self.frame_count = 0
        self.recovery_frames_left = 0
        self.recovery_holdoff_left = 0

    def push_frame(self, frame: WheelSpeedFrame) -> DetectionResult:
        self.frame_count += 1
        avg_speed = sum(frame.wheels) / WHEEL_COUNT
        speed_valid = avg_speed >= self.cfg.min_avg_speed
        recovery_active = self.recovery_frames_left > 0
        previous_alarm = self.detector.alarm

        # Slow baseline is frozen around possible events, but updated every clean frame.
        adapt_slow_before = (
            speed_valid
            and not self.detector.alarm
            and (recovery_active or self.detector.score < self.cfg.freeze_enter)
        )
        slow_alpha = self.cfg.recovery_slow_alpha if recovery_active else None
        estimate = self.estimator.update(
            frame.wheels,
            adapt_slow=adapt_slow_before,
            slow_alpha=slow_alpha,
        )

        if speed_valid:
            if recovery_active:
                self.detector.recover_towards(estimate.selected)
            alarm = self.detector.update(estimate.selected, adapt_noise=not self.detector.alarm)
            if self.frame_count <= self.cfg.warmup_frames:
                self.detector.clear_evidence()
                self.detector.alarm = False
                alarm = False
            if self.recovery_holdoff_left > 0:
                self.detector.clear_evidence()
                self.detector.alarm = False
                alarm = False
            if self.cfg.recovery_enabled and previous_alarm and not alarm:
                self.recovery_frames_left = max(self.recovery_frames_left, self.cfg.recovery_frames)
                self.recovery_holdoff_left = max(self.recovery_holdoff_left, self.cfg.recovery_holdoff_frames)
        else:
            if previous_alarm and self.cfg.hold_alarm_below_min_speed:
                self.detector.alarm = True
                alarm = True
            else:
                self.detector.clear_evidence()
                self.detector.alarm = False
                alarm = False
            if self.cfg.recovery_enabled and previous_alarm and not alarm:
                self.recovery_frames_left = max(self.recovery_frames_left, self.cfg.recovery_frames)
                self.recovery_holdoff_left = max(self.recovery_holdoff_left, self.cfg.recovery_holdoff_frames)

        if speed_valid and self.recovery_frames_left > 0:
            self.recovery_frames_left -= 1
        if speed_valid and self.recovery_holdoff_left > 0:
            self.recovery_holdoff_left -= 1
        recovery_active = self.recovery_frames_left > 0

        alarm_wheel = None
        alarm_wheel_dev = None
        if alarm:
            alarm_wheel = estimate.wheel_name
            alarm_wheel_dev = estimate.wheel_dev

        return DetectionResult(
            t_sec=frame.t_sec,
            wheels=frame.wheels.copy(),
            avg_speed=avg_speed,
            combind=estimate.selected,
            legacy_combind=estimate.legacy_combind,
            wheel_feature=estimate.wheel_feature,
            feature_baseline=self.detector.feature_baseline(),
            innovation=self.detector.innovation,
            alarm=alarm,
            score=self.detector.score,
            off_threshold=self.cfg.evidence_off,
            on_threshold=self.cfg.evidence_on,
            evidence=self.detector.score,
            signed_evidence=self.detector.signed_evidence(),
            enter_threshold=self.detector.enter_threshold,
            exit_threshold=self.detector.exit_threshold,
            noise=self.detector.noise,
            recovery_active=recovery_active,
            recovery_frames_left=self.recovery_frames_left,
            alarm_wheel=alarm_wheel,
            alarm_wheel_dev=alarm_wheel_dev,
        )


def load_frames(path: Path, series: str) -> list[WheelSpeedFrame]:
    df = pd.read_csv(path)
    columns = [f"wheel{i}_{series}_rad_s" for i in range(WHEEL_COUNT)]
    missing = [col for col in ["time_s", *columns] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return [
        WheelSpeedFrame(float(row[0]), [float(value) for value in row[1:]])
        for row in df[["time_s", *columns]].itertuples(index=False, name=None)
    ]


def run_detection(
    frames: Iterable[WheelSpeedFrame],
    min_avg_speed: float = 20.0,
    cfg: FastAlarmConfig | None = None,
) -> list[DetectionResult]:
    cfg = cfg or FastAlarmConfig(min_avg_speed=min_avg_speed)
    pipeline = WheelAlarmFastPipeline(cfg)
    return [pipeline.push_frame(frame) for frame in frames]


def fallback_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def write_results(path: Path, results: list[DetectionResult]) -> Path:
    rows = []
    for result in results:
        rows.append({
            "time_s": result.t_sec,
            "wheel0_rad_s": result.wheels[0],
            "wheel1_rad_s": result.wheels[1],
            "wheel2_rad_s": result.wheels[2],
            "wheel3_rad_s": result.wheels[3],
            "avg_speed_rad_s": result.avg_speed,
            "combind": result.combind,
            "legacy_combind": result.legacy_combind,
            "wheel_feature": result.wheel_feature,
            "feature_baseline": result.feature_baseline,
            "innovation": result.innovation,
            "alarm": int(result.alarm),
            "score": result.score,
            "off_threshold": result.off_threshold,
            "on_threshold": result.on_threshold,
            "evidence": result.evidence,
            "signed_evidence": result.signed_evidence,
            "enter_threshold": result.enter_threshold,
            "exit_threshold": result.exit_threshold,
            "noise": result.noise,
            "recovery_active": int(result.recovery_active),
            "recovery_frames_left": result.recovery_frames_left,
            "alarm_wheel": result.alarm_wheel,
            "alarm_wheel_dev": result.alarm_wheel_dev,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.8f")
        return path
    except PermissionError:
        unlocked_path = fallback_path(path)
        pd.DataFrame(rows).to_csv(unlocked_path, index=False, float_format="%.8f")
        return unlocked_path


def summarize(results: list[DetectionResult]) -> dict[str, object]:
    alarm_indices = [i for i, result in enumerate(results) if result.alarm]
    segments = []
    if alarm_indices:
        start = prev = alarm_indices[0]
        for idx in alarm_indices[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            segments.append({
                "start_time_s": results[start].t_sec,
                "end_time_s": results[prev].t_sec,
                "frame_count": prev - start + 1,
            })
            start = prev = idx
        segments.append({
            "start_time_s": results[start].t_sec,
            "end_time_s": results[prev].t_sec,
            "frame_count": prev - start + 1,
        })
    return {
        "frames": len(results),
        "alarm_frames": len(alarm_indices),
        "alarm_segments": len(segments),
        "first_alarm_time_s": results[alarm_indices[0]].t_sec if alarm_indices else None,
        "last_alarm_time_s": results[alarm_indices[-1]].t_sec if alarm_indices else None,
        "segments": segments,
    }


def write_summary(path: Path, summary: dict[str, object]) -> Path:
    content = json.dumps(summary, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding="utf-8")
        return path
    except PermissionError:
        unlocked_path = fallback_path(path)
        unlocked_path.write_text(content, encoding="utf-8")
        return unlocked_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-latency wheel speed alarm detection.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--series", choices=["raw", "corrected", "ref_comp_on"], default="corrected")
    parser.add_argument("--min-avg-speed", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FastAlarmConfig(min_avg_speed=args.min_avg_speed)
    frames = load_frames(args.input, args.series)
    results = run_detection(frames, cfg=cfg)
    output_path = write_results(args.output, results)
    summary = summarize(results)
    summary.update({
        "input_file": str(args.input),
        "output_file": str(output_path),
        "series": args.series,
        "min_avg_speed": args.min_avg_speed,
        "algorithm": "fast_ewma_leaky_evidence",
        "config": cfg.__dict__,
    })
    summary_path = write_summary(args.summary, summary)
    summary["summary_file"] = str(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
