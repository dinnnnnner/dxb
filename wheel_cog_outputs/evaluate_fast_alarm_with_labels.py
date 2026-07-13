from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "wheel_cog_outputs" / "fast_alarm_event_eval"


@dataclass(frozen=True)
class EventLabel:
    file: str
    event_time_s: float


@dataclass(frozen=True)
class AlarmInterval:
    start_s: float
    end_s: float


def read_labels(path: Path) -> dict[str, EventLabel]:
    labels: dict[str, EventLabel] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "event_time_s"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing label columns in {path}: {sorted(missing)}")

        for row in reader:
            file_name = row["file"].strip()
            event_time_value = row["event_time_s"].strip()
            if not file_name or not event_time_value:
                continue
            if file_name in labels:
                raise ValueError(f"duplicate event label for {file_name}")
            labels[file_name] = EventLabel(
                file=file_name,
                event_time_s=float(event_time_value),
            )
    if not labels:
        raise ValueError(f"no event labels found in {path}")
    return labels


def read_batch_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_alarm_results(path: Path) -> tuple[list[float], list[int]]:
    times: list[float] = []
    alarms: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"time_s", "alarm"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing alarm columns in {path}: {sorted(missing)}")
        for row in reader:
            times.append(float(row["time_s"]))
            alarms.append(int(row["alarm"]))
    if not times:
        raise ValueError(f"no alarm results in {path}")
    return times, alarms


def sample_period_s(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    positive_diffs = [
        current - previous
        for previous, current in zip(times, times[1:])
        if current > previous
    ]
    return median(positive_diffs) if positive_diffs else 0.0


def alarm_intervals(
    times: list[float], alarms: list[int], sample_period: float
) -> list[AlarmInterval]:
    intervals: list[AlarmInterval] = []
    start: float | None = None
    for t_sec, alarm in zip(times, alarms):
        if alarm and start is None:
            start = t_sec
        elif not alarm and start is not None:
            intervals.append(AlarmInterval(start, t_sec))
            start = None
    if start is not None:
        intervals.append(AlarmInterval(start, times[-1] + sample_period))
    return intervals


def overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def first_detection_start(
    intervals: list[AlarmInterval], earliest_valid_s: float
) -> float | None:
    starts = [item.start_s for item in intervals if item.start_s >= earliest_valid_s]
    return min(starts) if starts else None


def evaluate_case(
    batch_row: dict[str, str],
    label: EventLabel,
    detection_window_s: float,
    early_tolerance_s: float,
    ignore_initial_s: float,
) -> dict[str, object]:
    alarm_output = batch_row.get("alarm_output_file", "").strip()
    if not alarm_output:
        raise ValueError(f"missing alarm_output_file for {label.file}")

    times, alarms = read_alarm_results(Path(alarm_output))
    dt = sample_period_s(times)
    intervals = alarm_intervals(times, alarms, dt)
    record_start_s = times[0]
    record_end_s = times[-1] + dt
    event_time_s = label.event_time_s
    if not record_start_s <= event_time_s < record_end_s:
        raise ValueError(
            f"event_time_s={event_time_s:.3f} is outside "
            f"[{record_start_s:.3f}, {record_end_s:.3f}) for {label.file}"
        )

    earliest_valid_detection_s = event_time_s - early_tolerance_s
    detection_time_s = first_detection_start(intervals, earliest_valid_detection_s)
    delay_s = (
        round(detection_time_s - event_time_s, 8)
        if detection_time_s is not None
        else None
    )

    evaluation_start_s = max(record_start_s, ignore_initial_s)
    normal_end_s = max(evaluation_start_s, earliest_valid_detection_s)
    false_alarm_intervals = [
        item
        for item in intervals
        if overlap_s(evaluation_start_s, normal_end_s, item.start_s, item.end_s) > 0.0
    ]
    false_alarm_duration_s = sum(
        overlap_s(evaluation_start_s, normal_end_s, item.start_s, item.end_s)
        for item in false_alarm_intervals
    )
    normal_evaluated_s = max(0.0, normal_end_s - evaluation_start_s)

    alarm_active_at_event = any(
        item.start_s <= event_time_s < item.end_s for item in intervals
    )

    def detected_by(deadline_s: float) -> bool:
        return delay_s is not None and -early_tolerance_s <= delay_s <= deadline_s

    return {
        "case_name": batch_row.get("case_name", ""),
        "file": label.file,
        "event_time_s": event_time_s,
        "record_end_s": round(record_end_s, 8),
        "first_alarm_s": intervals[0].start_s if intervals else None,
        "detection_time_s": detection_time_s,
        "delay_s": delay_s,
        "detected_within_window": detected_by(detection_window_s),
        "detected_1s": detected_by(1.0),
        "detected_2s": detected_by(2.0),
        "detected_5s": detected_by(5.0),
        "alarm_active_at_event": alarm_active_at_event,
        "false_alarm_events_before_event": len(false_alarm_intervals),
        "false_alarm_duration_s_before_event": round(false_alarm_duration_s, 8),
        "normal_evaluated_s": round(normal_evaluated_s, 8),
        "total_alarm_events": len(intervals),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, object]], detection_window_s: float
) -> dict[str, object]:
    case_count = len(rows)
    normal_s = sum(float(row["normal_evaluated_s"]) for row in rows)
    false_alarm_events = sum(
        int(row["false_alarm_events_before_event"]) for row in rows
    )
    timely_delays = [
        float(row["delay_s"])
        for row in rows
        if row["detected_within_window"] and row["delay_s"] is not None
    ]

    def deadline_metrics(column: str) -> dict[str, object]:
        detected = sum(bool(row[column]) for row in rows)
        return {
            "detected_events": detected,
            "missed_events": case_count - detected,
            "recall": detected / case_count if case_count else None,
        }

    configured = deadline_metrics("detected_within_window")
    return {
        "cases": case_count,
        "detection_window_s": detection_window_s,
        "detected_events_within_window": configured["detected_events"],
        "missed_events_within_window": configured["missed_events"],
        "event_recall_within_window": configured["recall"],
        "deadlines": {
            "1s": deadline_metrics("detected_1s"),
            "2s": deadline_metrics("detected_2s"),
            "5s": deadline_metrics("detected_5s"),
        },
        "mean_delay_s_timely_only": mean(timely_delays) if timely_delays else None,
        "median_delay_s_timely_only": median(timely_delays) if timely_delays else None,
        "max_delay_s_timely_only": max(timely_delays) if timely_delays else None,
        "alarm_active_at_event_cases": sum(
            bool(row["alarm_active_at_event"]) for row in rows
        ),
        "normal_evaluated_s": normal_s,
        "normal_evaluated_hours": normal_s / 3600.0,
        "false_alarm_events_before_event": false_alarm_events,
        "false_alarm_duration_s_before_event": sum(
            float(row["false_alarm_duration_s_before_event"]) for row in rows
        ),
        "false_alarm_events_per_hour": (
            false_alarm_events / (normal_s / 3600.0) if normal_s > 0.0 else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fast alarm rising edges against event_time labels."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--detection-window-s",
        type=float,
        default=2.0,
        help="Maximum accepted alarm delay for the primary event recall metric.",
    )
    parser.add_argument(
        "--early-tolerance-s",
        type=float,
        default=0.0,
        help="Allow an alarm rising edge this many seconds before event_time.",
    )
    parser.add_argument(
        "--ignore-initial-s",
        type=float,
        default=30.0,
        help="Exclude detector warmup from normal exposure and false-alarm metrics.",
    )
    args = parser.parse_args()
    if args.detection_window_s < 0.0:
        parser.error("--detection-window-s cannot be negative")
    if args.early_tolerance_s < 0.0:
        parser.error("--early-tolerance-s cannot be negative")
    if args.ignore_initial_s < 0.0:
        parser.error("--ignore-initial-s cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    labels_by_file = read_labels(args.labels)
    batch_rows = read_batch_summary(args.batch_summary)
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for batch_row in batch_rows:
        input_file = batch_row.get("input_file", "").strip()
        if not input_file:
            continue
        file_name = Path(input_file).name
        label = labels_by_file.get(file_name)
        if label is None:
            skipped.append(file_name)
            continue
        rows.append(
            evaluate_case(
                batch_row,
                label,
                detection_window_s=args.detection_window_s,
                early_tolerance_s=args.early_tolerance_s,
                ignore_initial_s=args.ignore_initial_s,
            )
        )

    evaluated_files = {str(row["file"]) for row in rows}
    missing_results = sorted(set(labels_by_file).difference(evaluated_files))
    if missing_results:
        raise FileNotFoundError(
            f"no batch alarm results found for labeled files: {missing_results}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_case_path = args.output_dir / "fast_alarm_event_eval_per_case.csv"
    summary_path = args.output_dir / "fast_alarm_event_eval_summary.json"
    write_csv(per_case_path, rows)
    summary = {
        "labels": str(args.labels),
        "batch_summary": str(args.batch_summary),
        "per_case_csv": str(per_case_path),
        "early_tolerance_s": args.early_tolerance_s,
        "ignore_initial_s": args.ignore_initial_s,
        "skipped_unlabeled_files": skipped,
        "metrics": aggregate(rows, args.detection_window_s),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = summary["metrics"]
    recall = metrics["event_recall_within_window"]
    false_alarm_rate = metrics["false_alarm_events_per_hour"]
    recall_text = "n/a" if recall is None else f"{recall:.4f}"
    false_alarm_rate_text = (
        "n/a" if false_alarm_rate is None else f"{false_alarm_rate:.4f}"
    )
    print(
        f"cases={metrics['cases']} "
        f"detected_{args.detection_window_s:g}s="
        f"{metrics['detected_events_within_window']} "
        f"recall={recall_text}"
    )
    print(
        f"false_alarm_events={metrics['false_alarm_events_before_event']} "
        f"normal_hours={metrics['normal_evaluated_hours']:.4f} "
        f"false_alarms_per_hour={false_alarm_rate_text}"
    )
    print(f"wrote {per_case_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
