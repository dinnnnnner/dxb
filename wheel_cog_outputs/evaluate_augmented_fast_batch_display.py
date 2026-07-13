from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import alarm_detection_local_fast as alarm_model
from process_wheel_cog import PROJECT_ROOT


DEFAULT_DATASET_DIR = PROJECT_ROOT / "wheel_cog_outputs" / "augmented_event_dataset"
HTML_MODES = ("none", "hard", "baseline", "hard+baseline", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fast alarm on augmented samples and create fast-batch-style "
            "interactive displays."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <dataset-dir>/fast_batch_display",
    )
    parser.add_argument(
        "--series", choices=["raw", "corrected", "ref_comp_on"], default="corrected"
    )
    parser.add_argument("--detection-window-s", type=float, default=2.0)
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--html-mode", choices=HTML_MODES, default="hard+baseline")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Process only the first N manifest rows; 0 processes all rows.",
    )
    args = parser.parse_args()
    if args.detection_window_s < 0.0:
        parser.error("--detection-window-s cannot be negative")
    if args.warmup_s < 0.0:
        parser.error("--warmup-s cannot be negative")
    if args.max_samples < 0:
        parser.error("--max-samples cannot be negative")
    if args.output_dir is None:
        args.output_dir = args.dataset_dir / "fast_batch_display"
    return args


def alarm_intervals(
    results: list[alarm_model.DetectionResult],
) -> list[tuple[float, float]]:
    if not results:
        return []
    dt = results[1].t_sec - results[0].t_sec if len(results) > 1 else 0.0
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for result in results:
        if result.alarm and start is None:
            start = result.t_sec
        elif not result.alarm and start is not None:
            intervals.append((start, result.t_sec))
            start = None
    if start is not None:
        intervals.append((start, results[-1].t_sec + dt))
    return intervals


def evaluate_results(
    manifest_row: dict[str, object],
    results: list[alarm_model.DetectionResult],
    detection_window_s: float,
) -> dict[str, object]:
    intervals = alarm_intervals(results)
    starts = [start for start, _ in intervals]
    sample_type = str(manifest_row["sample_type"])
    total_alarm_duration_s = sum(end - start for start, end in intervals)

    common: dict[str, object] = {
        "first_alarm_s": starts[0] if starts else None,
        "alarm_events": len(intervals),
        "alarm_duration_s": round(total_alarm_duration_s, 8),
        "pre_event_alarm_events": 0,
        "alarm_active_at_event": False,
        "detection_time_s": None,
        "delay_s": None,
        "detected_1s": False,
        "detected_2s": False,
        "detected_5s": False,
    }

    if sample_type == "normal":
        common["status"] = "NORMAL_OK" if not intervals else "FALSE_ALARM"
        return common

    event_time_s = float(manifest_row["event_time_in_sample_s"])
    pre_event = [start for start in starts if start < event_time_s]
    detections = [start for start in starts if start >= event_time_s]
    detection_time_s = min(detections) if detections else None
    delay_s = (
        round(detection_time_s - event_time_s, 8)
        if detection_time_s is not None
        else None
    )
    alarm_active_at_event = any(
        start <= event_time_s < end for start, end in intervals
    )
    common.update(
        {
            "pre_event_alarm_events": len(pre_event),
            "alarm_active_at_event": alarm_active_at_event,
            "detection_time_s": detection_time_s,
            "delay_s": delay_s,
            "detected_1s": delay_s is not None and 0.0 <= delay_s <= 1.0,
            "detected_2s": delay_s is not None and 0.0 <= delay_s <= 2.0,
            "detected_5s": delay_s is not None and 0.0 <= delay_s <= 5.0,
        }
    )

    if pre_event:
        status = "EARLY_ALARM"
    elif delay_s is None:
        status = "MISS"
    elif delay_s <= detection_window_s:
        status = "PASS"
    else:
        status = "LATE"
    common["status"] = status
    return common


def should_write_html(row: dict[str, object], mode: str) -> bool:
    is_hard = row["status"] not in {"PASS", "NORMAL_OK"}
    is_baseline = row["sample_type"] == "event" and int(row["is_augmented"]) == 0
    if mode == "none":
        return False
    if mode == "hard":
        return is_hard
    if mode == "baseline":
        return is_baseline
    if mode == "hard+baseline":
        return is_hard or is_baseline
    return True


def add_context_shapes(
    fig: go.Figure,
    intervals: list[tuple[float, float]],
    event_time_s: float | None,
    detection_window_s: float,
    warmup_s: float,
) -> None:
    for subplot_row in range(1, 4):
        if event_time_s is not None:
            fig.add_vrect(
                x0=event_time_s,
                x1=event_time_s + detection_window_s,
                fillcolor="#2ca02c",
                opacity=0.08,
                line_width=0,
                row=subplot_row,
                col=1,
            )
            fig.add_vline(
                x=event_time_s,
                line_color="#2ca02c",
                line_dash="dash",
                line_width=2,
                row=subplot_row,
                col=1,
            )
        fig.add_vline(
            x=warmup_s,
            line_color="#7f7f7f",
            line_dash="dot",
            line_width=1,
            row=subplot_row,
            col=1,
        )
        for start_s, end_s in intervals:
            fig.add_vrect(
                x0=start_s,
                x1=end_s,
                fillcolor="#d62728",
                opacity=0.22,
                line_color="#d62728",
                line_width=1,
                row=subplot_row,
                col=1,
            )


def build_display_figure(
    row: dict[str, object],
    results: list[alarm_model.DetectionResult],
    detection_window_s: float,
    warmup_s: float,
) -> go.Figure:
    times = [item.t_sec for item in results]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.50, 0.25, 0.25],
        subplot_titles=("wheel speed", "innovation and thresholds", "evidence"),
    )

    for wheel in range(4):
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=[item.wheels[wheel] for item in results],
                mode="lines",
                name=f"wheel{wheel}",
                line={"width": 1},
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scattergl(
            x=times,
            y=[item.innovation for item in results],
            mode="lines",
            name="innovation",
            line={"color": "#1f77b4", "width": 1},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=[item.enter_threshold for item in results],
            mode="lines",
            name="+enter threshold",
            line={"color": "#ff7f0e", "dash": "dash", "width": 1},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=[-item.enter_threshold for item in results],
            mode="lines",
            name="-enter threshold",
            line={"color": "#ff7f0e", "dash": "dash", "width": 1},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=times,
            y=[item.signed_evidence for item in results],
            mode="lines",
            name="signed evidence",
            line={"color": "#9467bd", "width": 1},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=[item.on_threshold for item in results],
            mode="lines",
            name="+alarm threshold",
            line={"color": "#8c564b", "dash": "dash", "width": 1},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=[-item.on_threshold for item in results],
            mode="lines",
            name="-alarm threshold",
            line={"color": "#8c564b", "dash": "dash", "width": 1},
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    event_time_value = row.get("event_time_in_sample_s")
    event_time_s = (
        None
        if event_time_value is None or pd.isna(event_time_value)
        else float(event_time_value)
    )
    add_context_shapes(
        fig,
        alarm_intervals(results),
        event_time_s,
        detection_window_s,
        warmup_s,
    )

    delay = row.get("delay_s")
    delay_text = "n/a" if delay is None or pd.isna(delay) else f"{float(delay):.3f}s"
    title = (
        f"{row['sample_id']} | {row['status']} | "
        f"source={row['source_event_id']} | delay={delay_text}"
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=900,
        hovermode="x unified",
        dragmode="pan",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    fig.update_yaxes(title_text="rad/s", row=1, col=1)
    fig.update_yaxes(title_text="innovation", row=2, col=1)
    fig.update_yaxes(title_text="evidence", row=3, col=1)
    fig.update_xaxes(title_text="time_s", rangeslider={"visible": True}, row=3, col=1)
    return fig


def write_display_html(
    path: Path,
    row: dict[str, object],
    results: list[alarm_model.DetectionResult],
    detection_window_s: float,
    warmup_s: float,
) -> None:
    fig = build_display_figure(
        row,
        results,
        detection_window_s=detection_window_s,
        warmup_s=warmup_s,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        },
    )


def calculate_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    events = [row for row in rows if row["sample_type"] == "event"]
    augmented_events = [row for row in events if int(row["is_augmented"]) == 1]
    baseline_events = [row for row in events if int(row["is_augmented"]) == 0]
    normal = [row for row in rows if row["sample_type"] == "normal"]
    valid_delays = [
        float(row["delay_s"])
        for row in events
        if row.get("delay_s") is not None and not pd.isna(row["delay_s"])
    ]

    def event_metrics(items: list[dict[str, object]]) -> dict[str, object]:
        return {
            "samples": len(items),
            "detected_2s": sum(bool(row["detected_2s"]) for row in items),
            "recall_2s": (
                sum(bool(row["detected_2s"]) for row in items) / len(items)
                if items
                else None
            ),
            "early_alarm_samples": sum(
                int(row["pre_event_alarm_events"]) > 0 for row in items
            ),
            "missed_samples": sum(row["status"] == "MISS" for row in items),
            "late_samples": sum(row["status"] == "LATE" for row in items),
        }

    return {
        "all_event_samples": event_metrics(events),
        "baseline_event_samples": event_metrics(baseline_events),
        "augmented_event_samples": event_metrics(augmented_events),
        "normal_samples": {
            "samples": len(normal),
            "samples_with_false_alarm": sum(
                row["status"] == "FALSE_ALARM" for row in normal
            ),
            "false_alarm_events": sum(int(row["alarm_events"]) for row in normal),
        },
        "detection_delay_s": {
            "mean": mean(valid_delays) if valid_delays else None,
            "median": median(valid_delays) if valid_delays else None,
            "p95": float(np.percentile(valid_delays, 95)) if valid_delays else None,
            "max": max(valid_delays) if valid_delays else None,
        },
        "status_counts": {
            str(status): int(count)
            for status, count in pd.Series(
                [row["status"] for row in rows]
            ).value_counts().items()
        },
    }


def write_index(
    path: Path,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    baseline = summary["baseline_event_samples"]
    augmented = summary["augmented_event_samples"]
    normal = summary["normal_samples"]
    cards = [
        ("Baseline 2s", f"{baseline['detected_2s']}/{baseline['samples']}"),
        ("Augmented 2s", f"{augmented['detected_2s']}/{augmented['samples']}"),
        ("Early alarm", str(augmented["early_alarm_samples"])),
        ("Miss", str(augmented["missed_samples"])),
        ("Normal false alarm", f"{normal['samples_with_false_alarm']}/{normal['samples']}"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    table_rows = []
    for row in rows:
        link = row.get("display_html", "")
        sample = html.escape(str(row["sample_id"]))
        if link:
            sample = f'<a href="{html.escape(str(link))}">{sample}</a>'
        delay = row.get("delay_s")
        delay_text = "" if delay is None or pd.isna(delay) else f"{float(delay):.3f}"
        table_rows.append(
            "<tr>"
            f"<td>{sample}</td>"
            f"<td>{html.escape(str(row['sample_type']))}</td>"
            f"<td>{html.escape(str(row['source_event_id']))}</td>"
            f"<td class=\"status {html.escape(str(row['status']))}\">"
            f"{html.escape(str(row['status']))}</td>"
            f"<td>{delay_text}</td>"
            f"<td>{row['alarm_events']}</td>"
            f"<td>{row['pre_event_alarm_events']}</td>"
            f"<td>{float(row['time_scale']):.3f}</td>"
            f"<td>{float(row['speed_scale']):.3f}</td>"
            f"<td>{float(row['noise_gain']):.3f}</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Augmented fast alarm display</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1 {{ margin-bottom: 8px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; min-width: 140px; }}
.label {{ color: #666; font-size: 13px; }}
.value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 7px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ position: sticky; top: 0; background: #f5f5f5; }}
.status {{ font-weight: 700; }}
.PASS, .NORMAL_OK {{ color: #208020; }}
.EARLY_ALARM, .FALSE_ALARM, .MISS, .LATE, .ERROR {{ color: #c02020; }}
</style>
</head>
<body>
<h1>Augmented fast alarm display</h1>
<p>点击带链接的 sample_id 查看交互曲线。红色区域是算法报警，绿色虚线是 event_time，绿色浅色区域是检测时限。</p>
<div class="cards">{card_html}</div>
<table>
<thead><tr><th>sample</th><th>type</th><th>source</th><th>status</th><th>delay_s</th><th>alarms</th><th>early</th><th>time scale</th><th>speed scale</th><th>noise</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if args.max_samples:
        manifest = manifest.head(args.max_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    display_dir = args.output_dir / "html"
    rows: list[dict[str, object]] = []
    total = len(manifest)

    for position, manifest_row in enumerate(manifest.to_dict(orient="records"), start=1):
        sample_id = str(manifest_row["sample_id"])
        sample_path = args.dataset_dir / str(manifest_row["sample_file"])
        try:
            frames = alarm_model.load_frames(sample_path, series=args.series)
            results = alarm_model.run_detection(frames, cfg=alarm_model.FastAlarmConfig())
            evaluation = evaluate_results(
                manifest_row,
                results,
                detection_window_s=args.detection_window_s,
            )
            row = {**manifest_row, **evaluation, "error": "", "display_html": ""}
            if should_write_html(row, args.html_mode):
                html_path = display_dir / f"{sample_id}.html"
                write_display_html(
                    html_path,
                    row,
                    results,
                    detection_window_s=args.detection_window_s,
                    warmup_s=args.warmup_s,
                )
                row["display_html"] = str(html_path.relative_to(args.output_dir))
        except Exception as exc:
            row = {
                **manifest_row,
                "status": "ERROR",
                "error": repr(exc),
                "display_html": "",
                "first_alarm_s": None,
                "alarm_events": 0,
                "alarm_duration_s": 0.0,
                "pre_event_alarm_events": 0,
                "alarm_active_at_event": False,
                "detection_time_s": None,
                "delay_s": None,
                "detected_1s": False,
                "detected_2s": False,
                "detected_5s": False,
            }
        rows.append(row)
        if position == 1 or position % 25 == 0 or position == total:
            print(f"[{position}/{total}] {sample_id} {row['status']}")

    evaluation_path = args.output_dir / "augmented_fast_alarm_evaluation.csv"
    with evaluation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = calculate_summary(rows)
    summary.update(
        {
            "dataset_dir": str(args.dataset_dir),
            "manifest": str(manifest_path),
            "series": args.series,
            "detection_window_s": args.detection_window_s,
            "warmup_s": args.warmup_s,
            "html_mode": args.html_mode,
            "displayed_samples": sum(bool(row["display_html"]) for row in rows),
        }
    )
    summary_path = args.output_dir / "augmented_fast_alarm_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    index_path = args.output_dir / "index.html"
    write_index(index_path, rows, summary)

    baseline = summary["baseline_event_samples"]
    augmented = summary["augmented_event_samples"]
    normal = summary["normal_samples"]
    print(
        f"baseline_2s={baseline['detected_2s']}/{baseline['samples']} "
        f"augmented_2s={augmented['detected_2s']}/{augmented['samples']} "
        f"normal_false_alarm={normal['samples_with_false_alarm']}/{normal['samples']}"
    )
    print(f"wrote {evaluation_path}")
    print(f"wrote {summary_path}")
    print(f"open {index_path}")


if __name__ == "__main__":
    main()
