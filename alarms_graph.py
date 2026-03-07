# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "matplotlib",
#   "seaborn",
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
import csv
import datetime
import io
import pathlib

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import requests
import seaborn as sns

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_AREA_FILTER = "תל אביב - מרכז העיר"
DEFAULT_AREA_LABEL  = "Tel Aviv - City Center"
DEFAULT_BIN_HOURS   = 1
DEFAULT_START       = "2026-02-28"

ALARMS_CSV_URL = "https://raw.githubusercontent.com/yuval-harpaz/alarms/master/data/alarms.csv"
CACHE_FILE = pathlib.Path("alarms_cache.csv")
CACHE_MAX_AGE_MINUTES = 30

BG_COLOR  = "#f0ede3"
DOT_COLOR = "#333333"
DOT_S     = 28  # scatter area for a single-count dot (points²)
# ─────────────────────────────────────────────────────────────────────────────

# Register ET-Book fonts once at import time
for _f in pathlib.Path.home().glob(".local/share/fonts/et-book/*.ttf"):
    fm.fontManager.addfont(str(_f))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--area",  default=DEFAULT_AREA_FILTER,
                   help="Hebrew substring to filter cities (empty = all)")
    p.add_argument("--label", default=DEFAULT_AREA_LABEL,
                   help="English label for chart title")
    p.add_argument("--start", default=DEFAULT_START,
                   help="Start date YYYY-MM-DD (default: %(default)s)")
    p.add_argument("--bin-hours", type=int, default=DEFAULT_BIN_HOURS,
                   help="Bin size in hours (default: %(default)s)")
    p.add_argument("--threat", type=int, default=0,
                   help="Threat type: 0=missiles, 5=UAV/intrusion, -1=all (default: 0)")
    p.add_argument("--output", default="alarms_frequency.png",
                   help="Output file path (default: %(default)s)")
    return p.parse_args()


def fetch_csv() -> str:
    """Download the alarms CSV, caching locally."""
    if CACHE_FILE.exists():
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        if age < datetime.timedelta(minutes=CACHE_MAX_AGE_MINUTES):
            print(f"Using cached CSV ({int(age.total_seconds() / 60)}m old).")
            return CACHE_FILE.read_text(encoding="utf-8")

    print("Downloading alarms CSV from GitHub...")
    resp = requests.get(ALARMS_CSV_URL, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    CACHE_FILE.write_text(resp.text, encoding="utf-8")
    print(f"Saved {CACHE_FILE} ({len(resp.text) // 1024}KB).")
    return resp.text


def load_alerts(csv_text: str, area_filter: str, threat: int, start: str
                ) -> list[datetime.datetime]:
    """Parse CSV and return deduplicated alert times matching filters."""
    cutoff = datetime.datetime.strptime(start, "%Y-%m-%d")
    seen_ids: set[str] = set()
    times = []

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        # Time filter (quick skip for old rows)
        dt = datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
        if dt < cutoff:
            continue

        # Threat filter
        if threat >= 0:
            try:
                row_threat = int(row["threat"])
            except (ValueError, KeyError):
                continue
            if row_threat != threat:
                continue

        # Area filter
        if area_filter and area_filter not in row.get("cities", ""):
            continue

        # Deduplicate by alert id — one event counts once per area match
        alert_id = row.get("id", "")
        if alert_id in seen_ids:
            continue
        seen_ids.add(alert_id)

        times.append(dt)

    return sorted(times)


def plot(times: list[datetime.datetime], area_label: str, bin_hours: int,
         output: str, start_date: str = DEFAULT_START):
    """One row per day, x = hour of day (0–24). Compact vertical layout."""
    if not times:
        print("No alerts found.")
        return

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = times[-1].date()
    days  = []
    d = start
    while d <= end:
        days.append(d)
        d += datetime.timedelta(days=1)

    # Count per (date, hour-bin)
    bins: dict[tuple, int] = {}
    for t in times:
        key = (t.date(), (t.hour // bin_hours) * bin_hours)
        bins[key] = bins.get(key, 0) + 1

    sns.set_theme(style="ticks", font_scale=1.0)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["ETBembo", "Palatino", "Georgia", "DejaVu Serif"]

    n_days = len(days)
    fig, ax = plt.subplots(figsize=(8, max(3, n_days * 0.5 + 1.5)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    NIGHT_COLOR = "#e2dfd5"
    ax.axvspan(0, 7, color=NIGHT_COLOR, zorder=0, linewidth=0)
    ax.axvspan(21, 24, color=NIGHT_COLOR, zorder=0, linewidth=0)

    for i, day in enumerate(days):
        y = -i
        ax.axhline(y, color="#cccccc", linewidth=0.4, zorder=1)
        for h in range(0, 24, bin_hours):
            count = bins.get((day, h), 0)
            if count > 0:
                ax.scatter([h + bin_hours / 2], [y], s=DOT_S * count,
                           color=DOT_COLOR, zorder=3, clip_on=False)

    ax.set_xlim(0, 24)
    ax.set_ylim(-n_days + 0.5, 0.5)
    ax.set_yticks(range(0, -n_days, -1))
    ax.set_yticklabels([d.strftime("%a %-d %b") for d in days], fontsize=8, color="#555555")
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=8, color="#555555")
    ax.tick_params(axis="x", colors="#555555", labelsize=9)
    sns.despine(ax=ax, left=True, right=True, top=True, bottom=False, offset=6)

    grey = "#888888"
    date_range = f"{times[0].strftime('%b %d')} – {times[-1].strftime('%b %d, %Y')}"
    ax.set_title(f"Rocket alert frequency — {area_label}", loc="left",
                 fontsize=13, fontweight="bold", pad=26)
    ax.text(0, 1.04, f"{date_range}   ({len(times)} alerts)",
            transform=ax.transAxes, fontsize=9, color=grey, va="bottom")

    # ── Dot-size legend (built right-to-left) ────────────────────────────
    leg_label = f"alerts per {bin_hours}h:"
    leg_x = 0.98
    leg_y = 1.065
    for c, lbl in [(5, "5"), (1, "1")]:
        ax.text(leg_x, leg_y, lbl, transform=ax.transAxes,
                fontsize=9, color=grey, va="center", ha="right")
        leg_x -= 0.022
        ax.scatter([leg_x], [leg_y], s=DOT_S * c, color=DOT_COLOR,
                   transform=ax.transAxes, clip_on=False, zorder=4)
        leg_x -= 0.03
    ax.text(leg_x, leg_y, leg_label, transform=ax.transAxes,
            fontsize=9, color=grey, va="center", ha="right")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    csv_text = fetch_csv()
    times = load_alerts(csv_text, args.area, args.threat, args.start)
    print(f"Matched {len(times)} alerts for '{args.label}' (since {args.start}).")
    plot(times, args.label, args.bin_hours, args.output, args.start)
