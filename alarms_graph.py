# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "cairosvg",
# ]
# ///
"""
Graphs rocket/missile alert frequency for a configurable area in Israel.

Primary data: yuval-harpaz/alarms CSV on GitHub (~122K rows, 2019–present).
Cached locally and refreshed periodically.

Usage:
    uv run alarms_graph.py
    uv run alarms_graph.py --area "אשקלון" --label "Ashkelon" --start 2025-01-01
    uv run alarms_graph.py --area "" --label "All Areas" --bin-hours 3
"""

import argparse
import datetime
import pathlib
from zoneinfo import ZoneInfo

import requests

from alarms_core import (
    ALARMS_CSV_URL,
    CITY_TRANSLATIONS,
    DEFAULT_AREA_FILTER,
    DEFAULT_BIN_HOURS,
    DEFAULT_START,
    TZEVAADOM_API_URL,
    load_alerts,
    load_api_alerts,
    load_alerts_rich,
    load_api_alerts_rich,
    render_chart,
)

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

CACHE_FILE = pathlib.Path("alarms_cache.csv")
CACHE_TIME_FILE = pathlib.Path("alarms_cache_time.txt")
CACHE_MAX_AGE_MINUTES = 30

API_CACHE_FILE = pathlib.Path("alerts_cache.json")
API_CACHE_MAX_AGE_MINUTES = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--area",
        default=DEFAULT_AREA_FILTER,
        help="Hebrew substring to filter cities (empty = all)",
    )
    p.add_argument(
        "--label",
        default=None,
        help="English label for chart title (default: translated from the area)",
    )
    p.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Start date YYYY-MM-DD (default: %(default)s)",
    )
    p.add_argument(
        "--bin-hours",
        type=int,
        default=DEFAULT_BIN_HOURS,
        help="Bin size in hours (default: %(default)s)",
    )
    p.add_argument(
        "--threat",
        type=int,
        default=0,
        help="Threat type: 0=missiles, 5=UAV/intrusion, -1=all (default: 0)",
    )
    p.add_argument(
        "--output",
        default="alarms_frequency.svg",
        help="Output file path (default: %(default)s)",
    )
    p.add_argument(
        "--style",
        choices=["dots", "lines"],
        default="lines",
        help="dots: binned scatter (default); lines: narrow tick at exact alert time",
    )
    p.add_argument(
        "--forecast",
        choices=["off", "simple", "advanced", "ridge"],
        default="off",
        help="Forecast method: off, simple (direct), advanced (rate-sub), ridge (34-feature Ridge)",
    )
    return p.parse_args()


def _parse_last_modified(header: str) -> datetime.datetime:
    """Parse HTTP Last-Modified (UTC) and return as naive Israel-local datetime."""
    utc = datetime.datetime.strptime(header, "%a, %d %b %Y %H:%M:%S GMT").replace(
        tzinfo=datetime.timezone.utc
    )
    return utc.astimezone(ISRAEL_TZ).replace(tzinfo=None)


def fetch_csv() -> tuple[str, datetime.datetime]:
    """Download the alarms CSV, caching locally. Returns (csv_text, github_file_time)."""
    if CACHE_FILE.exists():
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            CACHE_FILE.stat().st_mtime
        )
        if age < datetime.timedelta(minutes=CACHE_MAX_AGE_MINUTES):
            print(f"Using cached CSV ({int(age.total_seconds() / 60)}m old).")
            if CACHE_TIME_FILE.exists():
                ts = datetime.datetime.fromisoformat(
                    CACHE_TIME_FILE.read_text().strip()
                )
            else:
                ts = datetime.datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
            return CACHE_FILE.read_text(encoding="utf-8"), ts

    print("Downloading alarms CSV from GitHub...")
    resp = requests.get(ALARMS_CSV_URL, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    CACHE_FILE.write_text(resp.text, encoding="utf-8")

    lm = resp.headers.get("Last-Modified")
    ts = _parse_last_modified(lm) if lm else datetime.datetime.now()
    CACHE_TIME_FILE.write_text(ts.isoformat())

    print(
        f"Saved {CACHE_FILE} ({len(resp.text) // 1024}KB). GitHub file time: {ts:%Y-%m-%d %H:%M} IL"
    )
    return resp.text, ts


def fetch_api_data() -> list[dict]:
    """Fetch recent alerts from tzevaadom API, with short-lived cache."""
    import json

    if API_CACHE_FILE.exists():
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            API_CACHE_FILE.stat().st_mtime
        )
        if age < datetime.timedelta(minutes=API_CACHE_MAX_AGE_MINUTES):
            return json.loads(API_CACHE_FILE.read_text(encoding="utf-8"))

    print("Fetching recent alerts from tzevaadom API...")
    resp = requests.get(TZEVAADOM_API_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    API_CACHE_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Got {len(data)} alert groups from API.")
    return data


if __name__ == "__main__":
    args = parse_args()
    if args.label is None:
        args.label = CITY_TRANSLATIONS.get(args.area, args.area or "All Areas")

    csv_text, data_cutoff = fetch_csv()
    try:
        api_data = fetch_api_data()
    except Exception:
        print("Fetching latest alarms failed")
        api_data = []

    times, seen_ids = load_alerts(csv_text, args.area, args.threat, args.start)
    api_times = load_api_alerts(api_data, args.area, args.threat, args.start, seen_ids)
    if api_times:
        print(f"  +{len(api_times)} alerts from tzevaadom API.")
    times = sorted(times + api_times)
    print(f"Matched {len(times)} alerts for '{args.label}' (since {args.start}).")

    # Rich loading for ridge forecast (all cities, needed for global features)
    all_records = None
    if args.forecast == "ridge":
        all_records, rich_seen = load_alerts_rich(csv_text, args.threat, args.start)
        api_rich = load_api_alerts_rich(api_data, args.threat, args.start, rich_seen)
        all_records = all_records + api_rich
        print(f"Rich load: {len(all_records)} city-records for ridge forecast.")

    svg_bytes = render_chart(
        times, args.label, args.bin_hours, args.start, data_cutoff, args.style,
        forecast=args.forecast,
        all_records=all_records,
        city_filter=args.area if args.forecast == "ridge" else None,
    )

    output = pathlib.Path(args.output)
    if output.suffix.lower() == ".png":
        import base64
        import re
        import cairosvg

        # Embed local ETBembo fonts so cairosvg renders them correctly
        # (it cannot fetch the Google Fonts @import from the network).
        font_dir = pathlib.Path.home() / ".local/share/fonts/et-book"
        faces = [
            ("ETBembo", "normal", "400", "et-book-roman-line-figures.ttf"),
            ("ETBembo", "normal", "700", "et-book-bold-line-figures.ttf"),
            ("ETBembo", "italic", "400", "et-book-display-italic-old-style-figures.ttf"),
        ]
        font_css = "".join(
            f'@font-face{{font-family:"ETBembo";font-style:{s};font-weight:{w};'
            f'src:url("data:font/truetype;base64,{base64.b64encode((font_dir / f).read_bytes()).decode()}") format("truetype");}}'
            for _, s, w, f in faces
            if (font_dir / f).exists()
        )
        if font_css:
            svg_bytes = re.sub(rb'@import url\([^)]+\);?', font_css.encode(), svg_bytes)

        output.write_bytes(cairosvg.svg2png(bytestring=svg_bytes, scale=6))
    else:
        output.write_bytes(svg_bytes)
    print(f"Saved: {args.output}")

    # Display the chart
    import webbrowser

    webbrowser.open(output.resolve().as_uri())
