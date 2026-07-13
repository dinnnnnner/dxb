from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import alarm_detection_local_fast as alarm_model
from process_wheel_cog import PROJECT_ROOT


DEFAULT_INPUT = PROJECT_ROOT / "wheel_cog_outputs" / "wheel_speeds.csv"
BATCH_OUT_DIR = PROJECT_ROOT / "wheel_cog_outputs" / "fast_alarm_batch_outputs"
DEFAULT_WHEEL_RADIUS_M = 0.31
MPH_TO_M_S = 0.44704
WHEEL_MPH_COLUMNS = ["wheel_1_mph", "wheel_2_mph", "wheel_3_mph", "wheel_4_mph"]


def safe_name(path: Path) -> str:
    name = path.with_suffix("").name
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def load_mph_frames(input_path: Path, wheel_radius_m: float) -> list[alarm_model.WheelSpeedFrame]:
    df = pd.read_csv(input_path)
    required = ["t", *WHEEL_MPH_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {input_path}: {missing}")

    rows = df.loc[:, required].apply(pd.to_numeric, errors="coerce").dropna()
    if rows.empty:
        raise ValueError(f"No numeric wheel-speed rows in {input_path}")

    first_time = float(rows["t"].iloc[0])
    frames = []
    for row in rows.itertuples(index=False, name=None):
        t_sec = float(row[0]) - first_time
        wheels = [float(value) * MPH_TO_M_S / wheel_radius_m for value in row[1:]]
        frames.append(alarm_model.WheelSpeedFrame(t_sec, wheels))
    return frames


def write_fast_batch_wheel_csv(path: Path, frames: list[alarm_model.WheelSpeedFrame]) -> Path:
    rows = []
    for frame in frames:
        out_row: dict[str, float | int] = {"time_s": frame.t_sec}
        for idx, wheel in enumerate(frame.wheels):
            out_row[f"wheel{idx}_raw_rad_s"] = wheel
            out_row[f"wheel{idx}_corrected_rad_s"] = wheel
            out_row[f"wheel{idx}_ref_comp_on_rad_s"] = wheel
            out_row[f"wheel{idx}_delta_count"] = 0
        rows.append(out_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.8f")
    return path


def write_batch_summary_csv(summary_csv: Path, row: dict[str, object]) -> None:
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        if "case_name" in summary.columns:
            summary = summary[summary["case_name"] != row["case_name"]]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)


def write_batch_summary_json_if_text(summary_json: Path, row: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    if summary_json.exists():
        head = summary_json.read_bytes()[:128]
        if b"E-SafeNet" in head and b"LOCK" in head:
            print(f"Skipped locked batch summary JSON: {summary_json}")
            return
        try:
            loaded = json.loads(summary_json.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = [item for item in loaded if isinstance(item, dict)]
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(f"Skipped non-JSON batch summary file: {summary_json}")
            return

    rows = [item for item in rows if item.get("case_name") != row["case_name"]]
    rows.append(row)
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def process_one(
    input_path: Path,
    batch_out_dir: Path = BATCH_OUT_DIR,
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
) -> dict[str, object]:
    case_name = safe_name(input_path)
    case_dir = batch_out_dir / case_name
    wheel_csv = write_fast_batch_wheel_csv(
        case_dir / "wheel_speed_raw_vs_corrected.csv",
        load_mph_frames(input_path, wheel_radius_m=wheel_radius_m),
    )

    cfg = alarm_model.FastAlarmConfig()
    frames = alarm_model.load_frames(wheel_csv, series="corrected")
    alarm_results = alarm_model.run_detection(frames, cfg=cfg)
    alarm_output = alarm_model.write_results(case_dir / "alarm_detection_results_fast.csv", alarm_results)

    alarm_summary = alarm_model.summarize(alarm_results)
    alarm_summary.update({
        "input_file": str(input_path.resolve()),
        "case_name": case_name,
        "case_dir": str(case_dir.resolve()),
        "wheel_speed_csv": str(wheel_csv.resolve()),
        "alarm_output_file": str(alarm_output.resolve()),
        "series": "corrected",
        "input_units": "mph",
        "wheel_radius_m": wheel_radius_m,
        "algorithm": "fast_ewma_leaky_evidence",
        "config": cfg.__dict__,
    })
    alarm_summary_path = alarm_model.write_summary(case_dir / "alarm_detection_summary_fast.json", alarm_summary)

    return {
        "case_name": case_name,
        "input_file": str(input_path.resolve()),
        "case_dir": str(case_dir.resolve()),
        "wheel_speed_csv": str(wheel_csv.resolve()),
        "alarm_output_file": str(alarm_output.resolve()),
        "alarm_summary_file": str(alarm_summary_path.resolve()),
        "frames": alarm_summary["frames"],
        "alarm_frames": alarm_summary["alarm_frames"],
        "alarm_segments": alarm_summary["alarm_segments"],
        "first_alarm_time_s": alarm_summary["first_alarm_time_s"],
        "last_alarm_time_s": alarm_summary["last_alarm_time_s"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a wheel_1_mph..wheel_4_mph CSV into fast_batch_display-compatible output."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=BATCH_OUT_DIR)
    parser.add_argument("--wheel-radius-m", type=float, default=DEFAULT_WHEEL_RADIUS_M)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = process_one(
        input_path=args.input,
        batch_out_dir=args.out_dir,
        wheel_radius_m=args.wheel_radius_m,
    )

    summary_csv = args.out_dir / "fast_alarm_batch_summary.csv"
    summary_json = args.out_dir / "fast_alarm_batch_summary.json"
    write_batch_summary_csv(summary_csv, row)
    write_batch_summary_json_if_text(summary_json, row)

    print(f"Wrote case: {row['case_dir']}")
    print(f"Wrote: {summary_csv}")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
