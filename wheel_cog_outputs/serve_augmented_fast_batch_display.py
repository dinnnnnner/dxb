from __future__ import annotations

import argparse
import html
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import pandas as pd

import alarm_detection_local_fast as alarm_model
from evaluate_augmented_fast_batch_display import (
    build_display_figure,
    evaluate_results,
)
from process_wheel_cog import PROJECT_ROOT


DEFAULT_DATASET_DIR = PROJECT_ROOT / "wheel_cog_outputs" / "augmented_event_dataset"
PLOT_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve augmented fast-alarm Plotly displays only when a sample is opened."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument(
        "--series", choices=["raw", "corrected", "ref_comp_on"], default="corrected"
    )
    parser.add_argument("--detection-window-s", type=float, default=2.0)
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-size", type=int, default=12)
    args = parser.parse_args()
    if args.evaluation is None:
        args.evaluation = (
            args.dataset_dir
            / "fast_batch_display"
            / "augmented_fast_alarm_evaluation.csv"
        )
    if args.detection_window_s < 0.0:
        parser.error("--detection-window-s cannot be negative")
    if args.warmup_s < 0.0:
        parser.error("--warmup-s cannot be negative")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.cache_size < 0:
        parser.error("--cache-size cannot be negative")
    return args


def is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def value_text(value: object, digits: int | None = None) -> str:
    if value is None or pd.isna(value):
        return ""
    if digits is not None:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            pass
    return str(value)


class ViewerState:
    def __init__(
        self,
        dataset_dir: Path,
        evaluation_path: Path,
        series: str,
        detection_window_s: float,
        warmup_s: float,
        cache_size: int,
    ) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.evaluation_path = evaluation_path.resolve()
        self.series = series
        self.detection_window_s = detection_window_s
        self.warmup_s = warmup_s

        manifest = pd.read_csv(self.dataset_dir / "manifest.csv")
        self.evaluation_available = self.evaluation_path.exists()
        if self.evaluation_available:
            evaluation = pd.read_csv(self.evaluation_path)
            evaluation_columns = [
                column
                for column in [
                    "sample_id",
                    "status",
                    "delay_s",
                    "alarm_events",
                    "pre_event_alarm_events",
                    "detected_2s",
                ]
                if column in evaluation.columns
            ]
            table = manifest.merge(
                evaluation[evaluation_columns],
                on="sample_id",
                how="left",
            )
        else:
            table = manifest.copy()
            table["status"] = "ON_DEMAND"
            table["delay_s"] = pd.NA
            table["alarm_events"] = pd.NA
            table["pre_event_alarm_events"] = pd.NA
            table["detected_2s"] = False

        table["status"] = table["status"].fillna("ON_DEMAND")
        self.rows = table.to_dict(orient="records")
        self.rows_by_id = {str(row["sample_id"]): row for row in self.rows}
        self.sample_ids = [str(row["sample_id"]) for row in self.rows]
        self.positions = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        self._cached_render = lru_cache(maxsize=cache_size)(self._render_sample)

    def render_sample(self, sample_id: str) -> str:
        return self._cached_render(sample_id)

    def _render_sample(self, sample_id: str) -> str:
        row = self.rows_by_id.get(sample_id)
        if row is None:
            raise KeyError(sample_id)

        sample_path = self.dataset_dir / str(row["sample_file"])
        frames = alarm_model.load_frames(sample_path, series=self.series)
        results = alarm_model.run_detection(frames, cfg=alarm_model.FastAlarmConfig())
        evaluation = evaluate_results(
            row,
            results,
            detection_window_s=self.detection_window_s,
        )
        display_row = {**row, **evaluation}
        figure = build_display_figure(
            display_row,
            results,
            detection_window_s=self.detection_window_s,
            warmup_s=self.warmup_s,
        )
        plot_html = figure.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            config=PLOT_CONFIG,
        )

        position = self.positions[sample_id]
        previous_link = (
            f'/sample/{quote(self.sample_ids[position - 1], safe="")}'
            if position > 0
            else ""
        )
        next_link = (
            f'/sample/{quote(self.sample_ids[position + 1], safe="")}'
            if position + 1 < len(self.sample_ids)
            else ""
        )
        metadata_fields = [
            "sample_id",
            "sample_type",
            "source_event_id",
            "source_file",
            "event_time_in_sample_s",
            "status",
            "detection_time_s",
            "delay_s",
            "alarm_events",
            "pre_event_alarm_events",
            "time_scale",
            "speed_scale",
            "noise_gain",
            "dropout_samples",
        ]
        metadata = "".join(
            "<tr>"
            f"<th>{html.escape(field)}</th>"
            f"<td>{html.escape(value_text(display_row.get(field), 6))}</td>"
            "</tr>"
            for field in metadata_fields
        )
        navigation = ["<a href=\"/\">← 返回总览</a>"]
        if previous_link:
            navigation.append(f'<a href="{previous_link}">← 上一个</a>')
        if next_link:
            navigation.append(f'<a href="{next_link}">下一个 →</a>')

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(sample_id)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 16px; color: #222; }}
.nav {{ display: flex; gap: 18px; margin-bottom: 10px; }}
.meta {{ border-collapse: collapse; margin: 8px 0 18px; font-size: 13px; }}
.meta th, .meta td {{ border-bottom: 1px solid #ddd; padding: 5px 10px; text-align: left; }}
.meta th {{ color: #666; }}
</style>
</head>
<body>
<div class="nav">{''.join(navigation)}</div>
<table class="meta">{metadata}</table>
{plot_html}
</body>
</html>
"""

    def index_html(self) -> str:
        baseline = [
            row
            for row in self.rows
            if row["sample_type"] == "event" and int(row["is_augmented"]) == 0
        ]
        augmented = [
            row
            for row in self.rows
            if row["sample_type"] == "event" and int(row["is_augmented"]) == 1
        ]
        normal = [row for row in self.rows if row["sample_type"] == "normal"]
        baseline_detected = sum(is_true(row["detected_2s"]) for row in baseline)
        augmented_detected = sum(is_true(row["detected_2s"]) for row in augmented)
        false_alarm_normal = sum(row["status"] == "FALSE_ALARM" for row in normal)
        early = sum(row["status"] == "EARLY_ALARM" for row in augmented)
        missed = sum(row["status"] == "MISS" for row in augmented)
        cards = [
            ("总样本", str(len(self.rows))),
            ("Baseline 2s", f"{baseline_detected}/{len(baseline)}"),
            ("Augmented 2s", f"{augmented_detected}/{len(augmented)}"),
            ("Early alarm", str(early)),
            ("Miss", str(missed)),
            ("Normal false alarm", f"{false_alarm_normal}/{len(normal)}"),
        ]
        card_html = "".join(
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div></div>'
            for label, value in cards
        )

        table_rows = []
        for row in self.rows:
            sample_id = str(row["sample_id"])
            href = f'/sample/{quote(sample_id, safe="")}'
            searchable = " ".join(
                str(row.get(field, ""))
                for field in [
                    "sample_id",
                    "sample_type",
                    "source_event_id",
                    "source_file",
                    "status",
                ]
            ).lower()
            table_rows.append(
                f'<tr data-search="{html.escape(searchable)}" '
                f'data-status="{html.escape(str(row["status"]))}" '
                f'data-type="{html.escape(str(row["sample_type"]))}">'
                f'<td><a href="{href}">{html.escape(sample_id)}</a></td>'
                f'<td>{html.escape(str(row["sample_type"]))}</td>'
                f'<td>{html.escape(str(row["source_event_id"]))}</td>'
                f'<td class="status {html.escape(str(row["status"]))}">'
                f'{html.escape(str(row["status"]))}</td>'
                f'<td>{html.escape(value_text(row.get("delay_s"), 3))}</td>'
                f'<td>{html.escape(value_text(row.get("alarm_events")))}</td>'
                f'<td>{html.escape(value_text(row.get("time_scale"), 3))}</td>'
                f'<td>{html.escape(value_text(row.get("noise_gain"), 3))}</td>'
                "</tr>"
            )

        evaluation_note = (
            f"已加载评价表：{html.escape(str(self.evaluation_path))}"
            if self.evaluation_available
            else "尚无批量评价表；点击样本仍会即时运行算法。"
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dynamic augmented fast alarm viewer</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; min-width: 125px; }}
.label {{ color: #666; font-size: 13px; }}
.value {{ font-size: 23px; font-weight: 700; margin-top: 4px; }}
.filters {{ display: flex; gap: 10px; margin: 18px 0; }}
input, select {{ padding: 8px; border: 1px solid #bbb; border-radius: 5px; }}
input {{ min-width: 310px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 7px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ position: sticky; top: 0; background: #f5f5f5; }}
.status {{ font-weight: 700; }}
.PASS, .NORMAL_OK {{ color: #208020; }}
.EARLY_ALARM, .FALSE_ALARM, .MISS, .LATE, .ERROR {{ color: #c02020; }}
.ON_DEMAND {{ color: #777; }}
</style>
</head>
<body>
<h1>动态增强样本检测查看器</h1>
<p>点击 sample_id 时才加载轮速、运行算法并生成交互图。{evaluation_note}</p>
<div class="cards">{card_html}</div>
<div class="filters">
<input id="search" placeholder="搜索 sample、source、文件名……">
<select id="status"><option value="">全部状态</option><option>PASS</option><option>EARLY_ALARM</option><option>MISS</option><option>FALSE_ALARM</option><option>NORMAL_OK</option><option>ON_DEMAND</option></select>
<select id="type"><option value="">全部类型</option><option>event</option><option>normal</option></select>
</div>
<table><thead><tr><th>sample</th><th>type</th><th>source</th><th>status</th><th>delay_s</th><th>alarms</th><th>time scale</th><th>noise</th></tr></thead>
<tbody id="rows">{''.join(table_rows)}</tbody></table>
<script>
const search = document.getElementById('search');
const status = document.getElementById('status');
const type = document.getElementById('type');
function filterRows() {{
  const q = search.value.toLowerCase();
  for (const row of document.querySelectorAll('#rows tr')) {{
    const visible = row.dataset.search.includes(q)
      && (!status.value || row.dataset.status === status.value)
      && (!type.value || row.dataset.type === type.value);
    row.style.display = visible ? '' : 'none';
  }}
}}
search.addEventListener('input', filterRows);
status.addEventListener('change', filterRows);
type.addEventListener('change', filterRows);
</script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    state: ViewerState

    def send_html(self, content: str, status: int = 200) -> None:
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        route = urlparse(self.path).path
        if route == "/":
            self.send_html(self.state.index_html())
            return
        if route == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if route.startswith("/sample/"):
            sample_id = unquote(route.removeprefix("/sample/"))
            try:
                self.send_html(self.state.render_sample(sample_id))
            except KeyError:
                self.send_html("<h1>404 sample not found</h1>", status=404)
            except Exception as exc:
                self.send_html(
                    "<h1>500 sample render failed</h1>"
                    f"<pre>{html.escape(repr(exc))}</pre>",
                    status=500,
                )
            return
        self.send_html("<h1>404 not found</h1>", status=404)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    args = parse_args()
    state = ViewerState(
        dataset_dir=args.dataset_dir,
        evaluation_path=args.evaluation,
        series=args.series,
        detection_window_s=args.detection_window_s,
        warmup_s=args.warmup_s,
        cache_size=args.cache_size,
    )
    ViewerHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"samples={len(state.rows)} evaluation={state.evaluation_available}")
    print(f"open http://localhost:{args.port}")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping viewer")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
