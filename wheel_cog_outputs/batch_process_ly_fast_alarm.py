from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

import alarm_detection_local_fast as alarm_model
from process_wheel_cog import PROJECT_ROOT, process_file
from process_wheel_speeds_fast_batch_display import (
    DEFAULT_INPUT as WHEEL_SPEEDS_CSV,
    process_one as process_wheel_speeds_case,
)


LY_ROOT = PROJECT_ROOT / "ly"
BATCH_OUT_DIR = PROJECT_ROOT / "wheel_cog_outputs" / "fast_alarm_batch_outputs"


def safe_name(path: Path) -> str:
    relative = path.relative_to(LY_ROOT).with_suffix("")
    name = "__".join(relative.parts)
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def discover_input_files(root: Path = LY_ROOT) -> list[Path]:
    return sorted(path for path in root.rglob("*.txt") if path.is_file() and not path.name.endswith("~"))


def process_one(input_path: Path, batch_out_dir: Path = BATCH_OUT_DIR, series: str = "corrected") -> dict[str, object]:
    case_name = safe_name(input_path)
    case_dir = batch_out_dir / case_name
    wheel_result = process_file(input_path, case_dir)

    wheel_csv = wheel_result["wheel_speed_csv"]
    alarm_output = case_dir / "alarm_detection_results_fast.csv"
    alarm_summary_path = case_dir / "alarm_detection_summary_fast.json"

    cfg = alarm_model.FastAlarmConfig()
    frames = alarm_model.load_frames(wheel_csv, series=series)
    alarm_results = alarm_model.run_detection(frames, cfg=cfg)
    alarm_output = alarm_model.write_results(alarm_output, alarm_results)
    alarm_summary = alarm_model.summarize(alarm_results)
    alarm_summary.update({
        "input_file": str(input_path),
        "case_name": case_name,
        "case_dir": str(case_dir),
        "wheel_speed_csv": str(wheel_csv),
        "alarm_output_file": str(alarm_output),
        "series": series,
        "algorithm": "fast_ewma_leaky_evidence",
        "config": cfg.__dict__,
    })
    alarm_summary_path = alarm_model.write_summary(alarm_summary_path, alarm_summary)

    return {
        "case_name": case_name,
        "input_file": str(input_path),
        "case_dir": str(case_dir),
        "wheel_speed_csv": str(wheel_csv),
        "alarm_output_file": str(alarm_output),
        "alarm_summary_file": str(alarm_summary_path),
        "frames": alarm_summary["frames"],
        "alarm_frames": alarm_summary["alarm_frames"],
        "alarm_segments": alarm_summary["alarm_segments"],
        "first_alarm_time_s": alarm_summary["first_alarm_time_s"],
        "last_alarm_time_s": alarm_summary["last_alarm_time_s"],
    }


def main() -> None:
    input_files = discover_input_files()
    BATCH_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, input_path in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] {input_path}")
        try:
            rows.append(process_one(input_path))
        except Exception as exc:
            rows.append({
                "case_name": safe_name(input_path),
                "input_file": str(input_path),
                "error": repr(exc),
            })

    if WHEEL_SPEEDS_CSV.exists():
        print(f"[extra] {WHEEL_SPEEDS_CSV}")
        try:
            rows.append(process_wheel_speeds_case(WHEEL_SPEEDS_CSV, BATCH_OUT_DIR))
        except Exception as exc:
            rows.append({
                "case_name": WHEEL_SPEEDS_CSV.stem,
                "input_file": str(WHEEL_SPEEDS_CSV),
                "error": repr(exc),
            })

    summary_csv = BATCH_OUT_DIR / "fast_alarm_batch_summary.csv"
    summary_json = BATCH_OUT_DIR / "fast_alarm_batch_summary.json"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote: {summary_csv}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
