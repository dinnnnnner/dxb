from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_SUMMARY = (
    PROJECT_ROOT
    / "wheel_cog_outputs"
    / "fast_alarm_batch_outputs"
    / "fast_alarm_batch_summary.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "wheel_cog_outputs"
    / "blowout_manual_labeling_package"
    / "labeling_package"
    / "event_time_labels_reference.csv"
)
LY_ROOT = PROJECT_ROOT / "ly"


def first_alarm_time(path: Path) -> float | None:
    """Return the current algorithm's first confirmed alarm timestamp."""
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["alarm"]) == 1:
                return float(row["time_s"])
    return None


def is_ly_input(path: Path) -> bool:
    try:
        path.resolve().relative_to(LY_ROOT.resolve())
        return True
    except ValueError:
        return False


def generate_reference_rows(batch_summary: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with batch_summary.open(newline="", encoding="utf-8") as handle:
        for batch_row in csv.DictReader(handle):
            input_value = batch_row.get("input_file", "").strip()
            alarm_value = batch_row.get("alarm_output_file", "").strip()
            if not input_value or not alarm_value:
                continue

            input_path = Path(input_value)
            if not is_ly_input(input_path):
                continue

            alarm_path = Path(alarm_value)
            if not alarm_path.exists():
                raise FileNotFoundError(f"Alarm output not found: {alarm_path}")

            event_time = first_alarm_time(alarm_path)
            rows.append(
                {
                    "file": input_path.name,
                    "event_time_s": "" if event_time is None else f"{event_time:.2f}",
                }
            )

    return rows


def write_reference(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "event_time_s"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an event-time-only reference label file from the current "
            "fast alarm algorithm's first confirmed alarm."
        )
    )
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = generate_reference_rows(args.batch_summary)
    write_reference(args.output, rows)
    missing = sum(1 for row in rows if row["event_time_s"] == "")
    print(f"wrote {args.output}")
    print(f"cases={len(rows)} event_times={len(rows) - missing} missing={missing}")


if __name__ == "__main__":
    main()
