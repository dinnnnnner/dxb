from __future__ import annotations
import math
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# Constants from the uploaded Rust flow
WHEEL_COUNT = 4
SAMPLE_TIME_SEC = 0.01
LOW_SPEED_COG_COUNT_THRESHOLD = 10
FRONT_AXLE_COG_COUNT = 48
REAR_AXLE_COG_COUNT = 48
LOSE_COG_DETECT_THRESHOLD_PERCENT = 50
QUICK_LEARN_LAP_THRESHOLD = 20
QUICK_UPDATE_SPEED = 0.95
TIMER_RESOLUTION_NS = 1000.0
DEFAULT_COG_COUNT = float(FRONT_AXLE_COG_COUNT)
TIMER_WRAP_US = 65_536
TIMESTAMP_HISTORY_LEN = 17
DELTA_HISTORY_LEN = 70
LOW_SPEED_MEMORY_LEN = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / 'ly' / '0116' / '20260116_yuan_baotai_rr100_45kmh.txt'
OUT_DIR = PROJECT_ROOT / 'wheel_cog_outputs' / 'wheel_cog_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_i64_fields(line: str) -> list[int]:
    out = []
    for part in line.split():
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def clamp_u16(x: int) -> int:
    return max(0, min(65535, int(x)))


def parse_frames(path: Path) -> list[list[list[int]]]:
    in_data = False
    raw_rows: list[list[int]] = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            trimmed = line.strip()
            if not in_data:
                in_data = (trimmed.lower() == 'marks end')
                continue
            if not trimmed:
                continue
            fields = parse_i64_fields(trimmed)
            if fields:
                raw_rows.append(fields)
    frames = []
    usable = (len(raw_rows) // 5) * 5
    for i in range(0, usable, 5):
        chunk = raw_rows[i:i+5]
        frame = []
        for wheel in range(WHEEL_COUNT):
            row = chunk[wheel]
            frame.append([clamp_u16(v) for v in row[1:TIMESTAMP_HISTORY_LEN]])
        frames.append(frame)
    return frames


def timer_delta(prev: int, curr: int) -> int:
    delta = int(curr) - int(prev)
    if delta <= 0:
        delta += TIMER_WRAP_US
    return max(0, min(65535, delta))


def next_idx(idx: int, length: int) -> int:
    return 0 if idx + 1 >= length else idx + 1

def previous_idx(idx: int, length: int) -> int:
    return length - 1 if idx == 0 else idx - 1

def wrap_add(idx: int, add: int, length: int) -> int:
    return (idx + add) % length

def ring_len(start: int, end: int, length: int) -> int:
    return end - start + 1 if end >= start else end + length - start + 1

def ring_distance(start: int, end: int, length: int) -> int:
    return end - start if end >= start else end + length - start

def blend(current: float, previous: float, update_speed: float) -> float:
    return previous * (1.0 - update_speed) + current * update_speed


class RustLikeWheelState:
    def __init__(self, enable_comp: bool = False, fixed_lose_cog_or: bool = False, comp_sign: str = 'rust_minus'):
        self.enable_comp = enable_comp
        self.fixed_lose_cog_or = fixed_lose_cog_or
        self.comp_sign = comp_sign  # 'rust_minus' or 'plus'
        self.raw_timestamps = [0] * TIMESTAMP_HISTORY_LEN
        self.timestamp_end = TIMESTAMP_HISTORY_LEN - 1
        self.has_previous_timestamp = False
        self.raw_delta = [0] * DELTA_HISTORY_LEN
        self.compensated_delta = [0] * DELTA_HISTORY_LEN
        self.delta_end = DELTA_HISTORY_LEN - 1
        self.cog_error_rad = [0.0] * max(FRONT_AXLE_COG_COUNT, REAR_AXLE_COG_COUNT)
        self.cog_error_mark_start = 0
        self.learned_laps = 0
        self.cog_count_window = [0] * LOW_SPEED_MEMORY_LEN
        self.cog_count_window_idx = 0
        self.cog_count_window_sum = 0
        # stats
        self.lose_cog_frames = 0
        self.low_speed_frames = 0
        self.delta_frames = 0

    def ensure_cog_error_capacity(self, cog_count: int):
        if len(self.cog_error_rad) < cog_count:
            self.cog_error_rad.extend([0.0] * (cog_count - len(self.cog_error_rad)))

    def update_low_speed_window(self, timestamp_count: int) -> bool:
        forgetting = self.cog_count_window[self.cog_count_window_idx]
        self.cog_count_window_sum = max(0, self.cog_count_window_sum - forgetting) + timestamp_count
        self.cog_count_window[self.cog_count_window_idx] = timestamp_count
        self.cog_count_window_idx = next_idx(self.cog_count_window_idx, LOW_SPEED_MEMORY_LEN)
        return self.cog_count_window_sum < LOW_SPEED_COG_COUNT_THRESHOLD

    def reset_delta_history(self):
        self.delta_end = DELTA_HISTORY_LEN - 1
        self.cog_error_mark_start = 0

    def detect_lose_cog(self, delta_start: int, delta_end: int) -> bool:
        if delta_start == delta_end:
            return False
        lose_cog = False
        for offset in range(ring_len(delta_start, delta_end, DELTA_HISTORY_LEN)):
            delta_idx = wrap_add(delta_start, offset, DELTA_HISTORY_LEN)
            prev_idx = previous_idx(delta_idx, DELTA_HISTORY_LEN)
            prev = self.raw_delta[prev_idx]
            curr = self.raw_delta[delta_idx]
            margin = round(prev * LOSE_COG_DETECT_THRESHOLD_PERCENT / 100.0)
            upper_limit = prev + margin
            lower_limit = max(0, prev - margin)
            flag = (curr > upper_limit or curr < lower_limit)
            if self.fixed_lose_cog_or:
                lose_cog = lose_cog or flag
            else:
                lose_cog = flag  # exact Rust code behavior: overwritten each loop
        return bool(lose_cog)

    def sum_delta(self, start: int, count: int) -> int:
        return sum(self.raw_delta[wrap_add(start, i, DELTA_HISTORY_LEN)] for i in range(count))

    def update_cog_error_table(self, lap_time_us: int, cog_count: int):
        standard_angle = 2.0 * math.pi / cog_count
        standard_time = lap_time_us / cog_count
        for idx in range(cog_count):
            delta_idx = wrap_add(self.cog_error_mark_start, idx, DELTA_HISTORY_LEN)
            current_angle = self.raw_delta[delta_idx] * standard_angle / standard_time
            learned_error = current_angle - standard_angle
            self.cog_error_rad[idx] = blend(learned_error, self.cog_error_rad[idx], QUICK_UPDATE_SPEED)

    def learn_cog_error(self, delta_start: int, delta_end: int, cog_count: int):
        if self.learned_laps >= QUICK_LEARN_LAP_THRESHOLD:
            return
        while ring_len(self.cog_error_mark_start, delta_end, DELTA_HISTORY_LEN) >= cog_count:
            lap_time_us = self.sum_delta(self.cog_error_mark_start, cog_count)
            if lap_time_us > 0:
                self.update_cog_error_table(lap_time_us, cog_count)
            self.cog_error_mark_start = wrap_add(self.cog_error_mark_start, cog_count, DELTA_HISTORY_LEN)
            self.learned_laps += 1
        if self.learned_laps == 0 and ring_len(delta_start, delta_end, DELTA_HISTORY_LEN) >= cog_count:
            self.cog_error_mark_start = delta_start

    def compensate_delta(self, delta_start: int, delta_end: int, cog_count: int):
        standard_angle = 2.0 * math.pi / cog_count
        for offset in range(ring_len(delta_start, delta_end, DELTA_HISTORY_LEN)):
            delta_idx = wrap_add(delta_start, offset, DELTA_HISTORY_LEN)
            err_idx = ring_distance(self.cog_error_mark_start, delta_idx, DELTA_HISTORY_LEN) % cog_count
            raw = self.raw_delta[delta_idx]
            prev_idx = previous_idx(delta_idx, DELTA_HISTORY_LEN)
            if raw == self.raw_delta[prev_idx]:
                self.compensated_delta[delta_idx] = self.compensated_delta[prev_idx]
            else:
                if self.comp_sign == 'plus':
                    denom = standard_angle + self.cog_error_rad[err_idx]
                else:
                    denom = standard_angle - self.cog_error_rad[err_idx]
                if abs(denom) < 1e-15:
                    val = raw
                else:
                    val = round(raw * standard_angle / denom)
                self.compensated_delta[delta_idx] = max(0, min(65535, int(val)))

    def angular_velocity_from_delta(self, delta_start: int, delta_end: int, cog_count: int) -> float:
        count = ring_len(delta_start, delta_end, DELTA_HISTORY_LEN)
        if count == 0:
            return 0.0
        delta_sum_us = 0
        for offset in range(count):
            idx = wrap_add(delta_start, offset, DELTA_HISTORY_LEN)
            delta = self.compensated_delta[idx] if self.enable_comp else self.raw_delta[idx]
            delta_sum_us += delta
        if delta_sum_us == 0:
            return 0.0
        delta_sec = delta_sum_us * TIMER_RESOLUTION_NS * 1e-9
        return count * 2.0 * math.pi / cog_count / delta_sec

    def push_frame(self, timestamps: list[int], cog_count: int) -> float:
        self.ensure_cog_error_capacity(cog_count)
        low_speed = self.update_low_speed_window(len(timestamps))
        delta_start = next_idx(self.delta_end, DELTA_HISTORY_LEN)
        delta_count = 0
        for timestamp in timestamps:
            previous_timestamp = self.raw_timestamps[self.timestamp_end]
            self.timestamp_end = next_idx(self.timestamp_end, TIMESTAMP_HISTORY_LEN)
            self.raw_timestamps[self.timestamp_end] = timestamp
            if self.has_previous_timestamp:
                if delta_count == 0:
                    delta_start = next_idx(self.delta_end, DELTA_HISTORY_LEN)
                delta = timer_delta(previous_timestamp, timestamp)
                self.delta_end = next_idx(self.delta_end, DELTA_HISTORY_LEN)
                self.raw_delta[self.delta_end] = delta
                self.compensated_delta[self.delta_end] = delta
                delta_count += 1
            else:
                self.has_previous_timestamp = True
        if low_speed or len(timestamps) == 0:
            self.low_speed_frames += 1
            self.reset_delta_history()
            return 0.0
        if delta_count == 0:
            return 0.0
        self.delta_frames += 1
        lose_cog = self.detect_lose_cog(delta_start, self.delta_end)
        if lose_cog:
            self.lose_cog_frames += 1
        if (not low_speed) and (not lose_cog):
            self.learn_cog_error(delta_start, self.delta_end, cog_count)
            self.compensate_delta(delta_start, self.delta_end, cog_count)
        return self.angular_velocity_from_delta(delta_start, self.delta_end, cog_count)


def cog_count_for_wheel(wheel: int, cli_cog_count: float = DEFAULT_COG_COUNT) -> int:
    if abs(cli_cog_count - DEFAULT_COG_COUNT) > 1e-15:
        return max(1, round(cli_cog_count))
    return FRONT_AXLE_COG_COUNT if wheel < 2 else REAR_AXLE_COG_COUNT


def compute_rust_like(frames: list[list[list[int]]], enable_comp: bool = False, fixed_lose_cog_or: bool = False, comp_sign: str = 'rust_minus'):
    states = [RustLikeWheelState(enable_comp, fixed_lose_cog_or, comp_sign) for _ in range(WHEEL_COUNT)]
    speeds = np.zeros((len(frames), WHEEL_COUNT), dtype=float)
    for i, frame in enumerate(frames):
        for wheel in range(WHEEL_COUNT):
            speeds[i, wheel] = states[wheel].push_frame(frame[wheel], cog_count_for_wheel(wheel))
    return speeds, states


@dataclass
class DeltaEvent:
    frame_idx: int
    delta_us: int
    phase: int
    timestamp: int


def extract_delta_events(frames: list[list[list[int]]], cog_count: int = 48):
    events_by_wheel: list[list[DeltaEvent]] = [[] for _ in range(WHEEL_COUNT)]
    phase_by_wheel = [0] * WHEEL_COUNT
    prev_by_wheel: list[Optional[int]] = [None] * WHEEL_COUNT
    for frame_idx, frame in enumerate(frames):
        for w in range(WHEEL_COUNT):
            for ts in frame[w]:
                prev = prev_by_wheel[w]
                if prev is not None:
                    delta = timer_delta(prev, ts)
                    # phase denotes the interval ending at current tooth; choose sequential modulo cog_count.
                    ph = phase_by_wheel[w] % cog_count
                    events_by_wheel[w].append(DeltaEvent(frame_idx, delta, ph, ts))
                    phase_by_wheel[w] += 1
                prev_by_wheel[w] = ts
    return events_by_wheel


def learn_phase_factors(events: list[DeltaEvent], cog_count: int = 48):
    """Offline robust tooth-period correction factors.

    factor[phase] ~= measured interval / lap mean. Corrected interval = raw / factor.
    Full laps with likely lost teeth are excluded using a +/-50% adjacent/median filter.
    """
    if not events:
        return np.ones(cog_count), {"lap_count": 0, "accepted_laps": 0, "rejected_laps": 0}
    # Use contiguous chunks of cog_count events. Starting phase may be arbitrary but consistent.
    ratios_by_phase = [[] for _ in range(cog_count)]
    lap_count = 0
    accepted = 0
    rejected = 0
    # Align laps to phase 0 boundaries where possible for stable mapping.
    # Find first event with phase==0.
    start_idx = next((i for i, e in enumerate(events) if e.phase == 0), 0)
    for s in range(start_idx, len(events) - cog_count + 1, cog_count):
        lap = events[s:s+cog_count]
        if len(lap) < cog_count:
            break
        # Require phases cycle 0..cog_count-1 relative to the first phase.
        ok_phase = all(lap[i].phase == ((lap[0].phase + i) % cog_count) for i in range(cog_count))
        if not ok_phase:
            rejected += 1
            continue
        deltas = np.array([e.delta_us for e in lap], dtype=float)
        lap_count += 1
        med = float(np.median(deltas))
        if med <= 0:
            rejected += 1
            continue
        # reject laps with probable missed/double teeth or severe acceleration artifact
        if np.any(deltas < 0.5 * med) or np.any(deltas > 1.5 * med):
            rejected += 1
            continue
        lap_mean = float(np.mean(deltas))
        if lap_mean <= 0:
            rejected += 1
            continue
        # Reject extremely sparse/slow laps that are outside meaningful speed range; not necessary here.
        for e, d in zip(lap, deltas):
            ratios_by_phase[e.phase].append(d / lap_mean)
        accepted += 1
    factors = np.ones(cog_count, dtype=float)
    counts = []
    for ph in range(cog_count):
        vals = np.array(ratios_by_phase[ph], dtype=float)
        counts.append(int(vals.size))
        if vals.size:
            # Trim mild outliers per phase.
            q1, q3 = np.percentile(vals, [10, 90])
            trimmed = vals[(vals >= q1) & (vals <= q3)] if vals.size >= 10 else vals
            factors[ph] = float(np.median(trimmed if trimmed.size else vals))
    # Normalize to keep one-lap average exactly 1.
    mean_factor = float(np.mean(factors))
    if mean_factor > 0:
        factors /= mean_factor
    meta = {
        "lap_count": int(lap_count),
        "accepted_laps": int(accepted),
        "rejected_laps": int(rejected),
        "per_phase_min_samples": int(min(counts) if counts else 0),
        "per_phase_median_samples": float(np.median(counts) if counts else 0),
        "factor_min": float(np.min(factors)),
        "factor_max": float(np.max(factors)),
        "factor_peak_to_peak_pct": float((np.max(factors) - np.min(factors)) * 100.0),
    }
    return factors, meta


def compute_phase_corrected_speeds(frames: list[list[list[int]]], factors_by_wheel: list[np.ndarray], cog_count: int = 48):
    prev_by_wheel: list[Optional[int]] = [None] * WHEEL_COUNT
    phase_by_wheel = [0] * WHEEL_COUNT
    speeds = np.zeros((len(frames), WHEEL_COUNT), dtype=float)
    raw_frame_speed = np.zeros((len(frames), WHEEL_COUNT), dtype=float)
    counts_by_frame = np.zeros((len(frames), WHEEL_COUNT), dtype=int)
    # Reuse low-speed window behavior from Rust, but corrected speed uses corrected deltas per frame.
    low_windows = [[0]*LOW_SPEED_MEMORY_LEN for _ in range(WHEEL_COUNT)]
    low_idx = [0]*WHEEL_COUNT
    low_sum = [0]*WHEEL_COUNT
    for frame_idx, frame in enumerate(frames):
        for w in range(WHEEL_COUNT):
            timestamps = frame[w]
            forgetting = low_windows[w][low_idx[w]]
            low_sum[w] = max(0, low_sum[w] - forgetting) + len(timestamps)
            low_windows[w][low_idx[w]] = len(timestamps)
            low_idx[w] = next_idx(low_idx[w], LOW_SPEED_MEMORY_LEN)
            low_speed = low_sum[w] < LOW_SPEED_COG_COUNT_THRESHOLD
            raw_deltas = []
            corr_deltas = []
            for ts in timestamps:
                prev = prev_by_wheel[w]
                if prev is not None:
                    d = timer_delta(prev, ts)
                    ph = phase_by_wheel[w] % cog_count
                    factor = float(factors_by_wheel[w][ph]) if factors_by_wheel[w][ph] > 1e-12 else 1.0
                    raw_deltas.append(d)
                    corr_deltas.append(d / factor)
                    phase_by_wheel[w] += 1
                prev_by_wheel[w] = ts
            counts_by_frame[frame_idx, w] = len(raw_deltas)
            if low_speed or len(timestamps) == 0 or len(raw_deltas) == 0:
                speeds[frame_idx, w] = 0.0
                raw_frame_speed[frame_idx, w] = 0.0
                continue
            count = len(raw_deltas)
            raw_sum = float(np.sum(raw_deltas))
            corr_sum = float(np.sum(corr_deltas))
            raw_frame_speed[frame_idx, w] = count * 2.0 * math.pi / cog_count / (raw_sum * 1e-6) if raw_sum > 0 else 0.0
            speeds[frame_idx, w] = count * 2.0 * math.pi / cog_count / (corr_sum * 1e-6) if corr_sum > 0 else 0.0
    return speeds, raw_frame_speed, counts_by_frame


def format_time(value: float) -> str:
    s = f"{value:.2f}"
    while '.' in s and s.endswith('0'):
        s = s[:-1]
    if s.endswith('.'):
        s = s[:-1]
    return s


def format_speed(value: float) -> str:
    return '0' if abs(value) < 0.000005 else f"{value:.5f}"


def write_no_header(path: Path, speeds: np.ndarray):
    with path.open('w', encoding='utf-8', newline='') as f:
        for i, row in enumerate(speeds):
            f.write(','.join([format_time(i*SAMPLE_TIME_SEC)] + [format_speed(x) for x in row]) + '\n')


def local_residual_metrics(speeds: np.ndarray, window: int = 101):
    import pandas as pd
    out = []
    for w in range(WHEEL_COUNT):
        s = pd.Series(speeds[:, w])
        valid = s > 1e-9
        med = s.where(valid).rolling(window, center=True, min_periods=max(5, window//4)).median()
        res = ((s - med) / med).replace([np.inf, -np.inf], np.nan)
        res = res[valid & med.notna() & (med > 1e-9)].dropna()
        out.append({
            'wheel': w,
            'valid_points': int(valid.sum()),
            'mean_rad_s': float(s[valid].mean()) if valid.any() else 0.0,
            'median_rad_s': float(s[valid].median()) if valid.any() else 0.0,
            'residual_std_pct': float(res.std(ddof=0) * 100.0) if len(res) else None,
            'residual_p95_abs_pct': float(res.abs().quantile(0.95) * 100.0) if len(res) else None,
            'residual_p99_abs_pct': float(res.abs().quantile(0.99) * 100.0) if len(res) else None,
        })
    return pd.DataFrame(out)


def process_file(input_path: Path = INPUT, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = parse_frames(input_path)
    # Raw Rust-like speeds. This is the reference output with ENABLE_COG_ERROR_COMPENSATION=false.
    raw_speeds, raw_states = compute_rust_like(frames, enable_comp=False)
    # Reference-code compensation switched on, retaining Rust sign and ring-index behavior.
    ref_comp_speeds, ref_states = compute_rust_like(frames, enable_comp=True, fixed_lose_cog_or=False, comp_sign='rust_minus')
    # Reference-code with plus sign and OR fixed, just for diagnostic comparison.
    ref_plus_speeds, ref_plus_states = compute_rust_like(frames, enable_comp=True, fixed_lose_cog_or=True, comp_sign='plus')

    # Offline robust phase factor correction.
    events_by_wheel = extract_delta_events(frames, cog_count=48)
    factors_by_wheel = []
    factor_meta = []
    for w, events in enumerate(events_by_wheel):
        factors, meta = learn_phase_factors(events, cog_count=48)
        factors_by_wheel.append(factors)
        meta['wheel'] = w
        meta['event_count'] = len(events)
        factor_meta.append(meta)
    corrected_speeds, frame_raw_from_phase, counts_by_frame = compute_phase_corrected_speeds(frames, factors_by_wheel, cog_count=48)

    # Write files.
    write_no_header(out_dir / 'wheel_speed_raw_reference_rad_s.txt', raw_speeds)
    write_no_header(out_dir / 'wheel_speed_corrected_phase_rad_s.txt', corrected_speeds)
    write_no_header(out_dir / 'wheel_speed_reference_compensation_on_rad_s.txt', ref_comp_speeds)

    # Analysis CSV with headers.
    df = pd.DataFrame({'time_s': np.arange(len(frames)) * SAMPLE_TIME_SEC})
    for w in range(WHEEL_COUNT):
        df[f'wheel{w}_raw_rad_s'] = raw_speeds[:, w]
        df[f'wheel{w}_corrected_rad_s'] = corrected_speeds[:, w]
        df[f'wheel{w}_ref_comp_on_rad_s'] = ref_comp_speeds[:, w]
        df[f'wheel{w}_delta_count'] = counts_by_frame[:, w]
    df.to_csv(out_dir / 'wheel_speed_raw_vs_corrected.csv', index=False, float_format='%.8f')

    factors_df = pd.DataFrame({'phase': np.arange(48)})
    for w, factors in enumerate(factors_by_wheel):
        factors_df[f'wheel{w}_factor'] = factors
        # angle error equivalent: factor - 1 times nominal angle, as percent of one tooth.
        factors_df[f'wheel{w}_tooth_error_pct_of_pitch'] = (factors - 1.0) * 100.0
    factors_df.to_csv(out_dir / 'learned_tooth_correction_factors.csv', index=False, float_format='%.8f')

    metrics_raw = local_residual_metrics(raw_speeds); metrics_raw['series'] = 'raw_reference'
    metrics_corr = local_residual_metrics(corrected_speeds); metrics_corr['series'] = 'phase_corrected'
    metrics_ref = local_residual_metrics(ref_comp_speeds); metrics_ref['series'] = 'reference_compensation_on'
    metrics_plus = local_residual_metrics(ref_plus_speeds); metrics_plus['series'] = 'reference_plus_or_diagnostic'
    metrics = pd.concat([metrics_raw, metrics_corr, metrics_ref, metrics_plus], ignore_index=True)
    metrics.to_csv(out_dir / 'correction_quality_metrics.csv', index=False, float_format='%.8f')

    # Extra summary.
    summary = {
        'input_file': str(input_path),
        'output_dir': str(out_dir),
        'frames': len(frames),
        'duration_s': (len(frames) - 1) * SAMPLE_TIME_SEC if frames else 0,
        'raw_rows_after_marks_end_usable': len(frames) * 5,
        'sample_time_s': SAMPLE_TIME_SEC,
        'cog_count': 48,
        'wheel_count': 4,
        'factor_meta': factor_meta,
        'rust_raw_state': [
            {'wheel': w, 'low_speed_frames': s.low_speed_frames, 'lose_cog_frames': s.lose_cog_frames,
             'learned_laps': s.learned_laps, 'delta_frames': s.delta_frames}
            for w, s in enumerate(raw_states)
        ],
        'rust_ref_comp_state': [
            {'wheel': w, 'low_speed_frames': s.low_speed_frames, 'lose_cog_frames': s.lose_cog_frames,
             'learned_laps': s.learned_laps, 'delta_frames': s.delta_frames}
            for w, s in enumerate(ref_states)
        ]
    }
    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        'summary': summary,
        'metrics': metrics,
        'wheel_speed_csv': out_dir / 'wheel_speed_raw_vs_corrected.csv',
    }


def main():
    result = process_file(INPUT, OUT_DIR)
    summary = result['summary']
    metrics = result['metrics']

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    print('\nMetrics:')
    print(metrics.to_string(index=False))

if __name__ == '__main__':
    main()
