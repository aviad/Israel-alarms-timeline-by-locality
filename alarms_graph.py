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
import math
import pathlib
from zoneinfo import ZoneInfo

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

import matplotlib.font_manager as fm
from matplotlib.path import Path
import matplotlib.pyplot as plt
import requests
import seaborn as sns

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_AREA_FILTER = "תל אביב - מרכז העיר"
DEFAULT_AREA_LABEL = "Tel Aviv - City Center"
DEFAULT_BIN_HOURS = 1
DEFAULT_START = "2026-02-28"

ALARMS_CSV_URL = (
    "https://raw.githubusercontent.com/yuval-harpaz/alarms/master/data/alarms.csv"
)
CACHE_FILE = pathlib.Path("alarms_cache.csv")
CACHE_TIME_FILE = pathlib.Path("alarms_cache_time.txt")
CACHE_MAX_AGE_MINUTES = 30

TZEVAADOM_API_URL = "https://api.tzevaadom.co.il/alerts-history/"
API_CACHE_FILE = pathlib.Path("alerts_cache.json")
API_CACHE_MAX_AGE_MINUTES = 2

BG_COLOR = "#f0ede3"
NIGHT_DOT_COLOR = "#333333"  # dark for night hours (0–7, 21–24)
DAY_DOT_COLOR = "#888888"    # lighter for daytime hours
DOT_COLOR = NIGHT_DOT_COLOR  # alias used in bar/text modes
DOT_S = 28  # scatter area for a single-count dot (points²)
# ─────────────────────────────────────────────────────────────────────────────

# Register ET-Book fonts once at import time
for _f in pathlib.Path.home().glob(".local/share/fonts/et-book/*.ttf"):
    fm.fontManager.addfont(str(_f))


def _make_wedge_marker(start_frac: float, end_frac: float, n: int = 32) -> Path:
    """Pie-wedge Path for scatter marker: clockwise from top, fractions 0–1."""
    t0 = math.pi / 2 - start_frac * 2 * math.pi
    t1 = math.pi / 2 - end_frac * 2 * math.pi
    thetas = [t0 + (t1 - t0) * k / (n - 1) for k in range(n)]
    verts = [(0.0, 0.0)] + [(math.cos(t), math.sin(t)) for t in thetas] + [(0.0, 0.0)]
    codes = [Path.MOVETO] + [Path.LINETO] * n + [Path.CLOSEPOLY]
    return Path(verts, codes)


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
        "--label", default=DEFAULT_AREA_LABEL, help="English label for chart title"
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
        default="alarms_frequency.png",
        help="Output file path (default: %(default)s)",
    )
    p.add_argument(
        "--daily-total",
        default="dot",
        choices=["none", "text", "bar", "dot"],
        help="Show daily totals: none | text (count label) | bar (mini bar chart) | dot (sized dot)",
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


def load_alerts(
    csv_text: str, area_filter: str, threat: int, start: str
) -> tuple[list[datetime.datetime], set[str]]:
    """Parse CSV and return (deduplicated alert times matching filters, seen ids)."""
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

    return sorted(times), seen_ids


def load_api_alerts(
    api_data: list[dict], area_filter: str, threat: int, start: str, seen_ids: set[str]
) -> list[datetime.datetime]:
    """Return alert times from API data not already in seen_ids."""
    cutoff = datetime.datetime.strptime(start, "%Y-%m-%d")
    times = []
    for group in api_data:
        gid = str(group["id"])
        if gid in seen_ids:
            continue
        for alert in group.get("alerts", []):
            dt = datetime.datetime.fromtimestamp(alert["time"])
            if dt < cutoff:
                continue
            if threat >= 0 and alert.get("threat") != threat:
                continue
            cities = " ".join(alert.get("cities", []))
            if area_filter and area_filter not in cities:
                continue
            # First matching alert in this group — record it
            seen_ids.add(gid)
            times.append(dt)
            break
    return times


def plot(
    times: list[datetime.datetime],
    area_label: str,
    bin_hours: int,
    output: str,
    start_date: str = DEFAULT_START,
    daily_total: str = "none",
    data_cutoff: datetime.datetime | None = None,
):
    """One row per day, x = hour of day (0–24). Compact vertical layout."""
    if not times:
        print("No alerts found.")
        return

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = times[-1].date()
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += datetime.timedelta(days=1)

    # Count per (date, hour-bin)
    bins: dict[tuple, int] = {}
    for t in times:
        key = (t.date(), (t.hour // bin_hours) * bin_hours)
        bins[key] = bins.get(key, 0) + 1

    daily_totals = {
        day: sum(bins.get((day, h), 0) for h in range(0, 24, bin_hours)) for day in days
    }
    max_daily = max(daily_totals.values(), default=1)
    daily_night = {
        day: sum(bins.get((day, h), 0) for h in range(0, 24, bin_hours) if h < 7 or h >= 21)
        for day in days
    }

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

    now = datetime.datetime.now()
    cutoff_date = now.date()
    cutoff_hour = now.hour + now.minute / 60

    for i, day in enumerate(days):
        y = -i
        line_end = cutoff_hour if day == cutoff_date else 24
        ax.plot([0, line_end], [y, y], color="#cccccc", linewidth=0.4, zorder=1)
        for h in range(0, 24, bin_hours):
            count = bins.get((day, h), 0)
            if count > 0:
                dot_color = NIGHT_DOT_COLOR if (h < 7 or h >= 21) else DAY_DOT_COLOR
                ax.scatter(
                    [h + bin_hours / 2],
                    [y],
                    s=DOT_S * count,
                    color=dot_color,
                    zorder=3,
                    clip_on=False,
                )

    # ── Daily total annotations ───────────────────────────────────────────
    BAR_SEP = 24.8  # x where bar area starts
    BAR_WIDTH = 3.0  # max bar length in data units
    if daily_total == "text":
        x_end = 27.5
    elif daily_total == "bar":
        x_end = BAR_SEP + BAR_WIDTH + 1.5
    elif daily_total == "dot":
        x_end = 26.5
    else:
        x_end = 24

    ax.set_xlim(0, x_end)
    ax.set_ylim(-n_days + 0.5, 0.5)
    ax.set_yticks(range(0, -n_days, -1))
    ax.set_yticklabels(
        [d.strftime("%a %-d %b") for d in days], fontsize=8, color="#555555"
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=8, color="#555555"
    )
    ax.tick_params(axis="x", colors="#555555", labelsize=9)
    sns.despine(ax=ax, left=True, right=True, top=True, bottom=False, offset=6)

    grey = "#888888"
    if daily_total == "text":
        ax.text(25.0, 0.55, "total", fontsize=7, color=grey, va="bottom", ha="left")
        for i, day in enumerate(days):
            tot = daily_totals[day]
            ax.text(
                25.0,
                -i,
                str(tot) if tot else "–",
                fontsize=7,
                color=DOT_COLOR if tot else grey,
                va="center",
                ha="left",
            )

    elif daily_total == "bar":
        ax.text(
            BAR_SEP + 0.1, 0.55, "total", fontsize=7, color=grey, va="bottom", ha="left"
        )
        for i, day in enumerate(days):
            tot = daily_totals[day]
            if tot:
                width = (tot / max_daily) * BAR_WIDTH
                ax.barh(
                    -i,
                    width,
                    left=BAR_SEP + 0.1,
                    height=0.45,
                    color=DOT_COLOR,
                    alpha=0.35,
                    zorder=2,
                )
                ax.text(
                    BAR_SEP + 0.1 + width + 0.1,
                    -i,
                    str(tot),
                    fontsize=7,
                    color=DOT_COLOR,
                    va="center",
                    ha="left",
                )

    elif daily_total == "dot":
        DOT_X = 25.5
        ax.text(DOT_X, 0.55, "total", fontsize=7, color=grey, va="bottom", ha="center")
        for i, day in enumerate(days):
            tot = daily_totals[day]
            if tot:
                night = daily_night[day]
                night_frac = night / tot
                s = DOT_S * tot
                if night_frac <= 0:
                    ax.scatter([DOT_X], [-i], s=s, color=DAY_DOT_COLOR, zorder=3, clip_on=False)
                elif night_frac >= 1:
                    ax.scatter([DOT_X], [-i], s=s, color=NIGHT_DOT_COLOR, zorder=3, clip_on=False)
                else:
                    ax.scatter([DOT_X], [-i], s=s, marker=_make_wedge_marker(0, night_frac),
                               color=NIGHT_DOT_COLOR, zorder=3, clip_on=False)
                    ax.scatter([DOT_X], [-i], s=s, marker=_make_wedge_marker(night_frac, 1.0),
                               color=DAY_DOT_COLOR, zorder=3, clip_on=False)
                ax.text(DOT_X, -i, str(tot), fontsize=6, color="white",
                        va="center", ha="center", zorder=4)

    date_range = f"{times[0].strftime('%b %d')} – {times[-1].strftime('%b %d, %Y')}"
    ax.set_title(
        f"Rocket alert frequency — {area_label}",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=26,
    )
    ax.text(
        0,
        1.04,
        f"{date_range}   ({len(times)} alerts)",
        transform=ax.transAxes,
        fontsize=9,
        color=grey,
        va="bottom",
    )

    # ── Dot-size legend (built right-to-left) ────────────────────────────
    leg_label = f"alerts per {bin_hours}h:"
    leg_x = 24 / x_end
    leg_y = 1.065
    for c, lbl in [(5, "5"), (1, "1")]:
        ax.text(
            leg_x,
            leg_y,
            lbl,
            transform=ax.transAxes,
            fontsize=9,
            color=grey,
            va="center",
            ha="right",
        )
        leg_x -= 0.022
        ax.scatter(
            [leg_x],
            [leg_y],
            s=DOT_S * c,
            color=NIGHT_DOT_COLOR,
            transform=ax.transAxes,
            clip_on=False,
            zorder=4,
        )
        leg_x -= 0.03
    ax.text(
        leg_x,
        leg_y,
        leg_label,
        transform=ax.transAxes,
        fontsize=9,
        color=grey,
        va="center",
        ha="right",
    )

    # Night / day colour key — same row, to the left of "alerts per Xh:" label
    # leg_x at this point is where "alerts per Xh:" right-aligns; step past its text width
    lx = leg_x - 0.18
    ax.scatter([lx], [leg_y], s=DOT_S * 2, marker=_make_wedge_marker(0.5, 1.0),
               color=NIGHT_DOT_COLOR, transform=ax.transAxes, clip_on=False, zorder=4)
    ax.scatter([lx], [leg_y], s=DOT_S * 2, marker=_make_wedge_marker(0, 0.5),
               color=DAY_DOT_COLOR, transform=ax.transAxes, clip_on=False, zorder=4)
    ax.text(lx - 0.012, leg_y, "night/day:", fontsize=9, color=grey,
            va="center", ha="right", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    csv_text, data_cutoff = fetch_csv()
    times, seen_ids = load_alerts(csv_text, args.area, args.threat, args.start)
    api_data = fetch_api_data()
    api_times = load_api_alerts(api_data, args.area, args.threat, args.start, seen_ids)
    if api_times:
        print(f"  +{len(api_times)} alerts from tzevaadom API.")
    times = sorted(times + api_times)
    print(f"Matched {len(times)} alerts for '{args.label}' (since {args.start}).")
    plot(
        times,
        args.label,
        args.bin_hours,
        args.output,
        args.start,
        args.daily_total,
        data_cutoff,
    )
