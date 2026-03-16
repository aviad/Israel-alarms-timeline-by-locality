"""
alarms_core.py — shared pure logic for alarms_graph.py (CLI) and worker/entry.py (Cloudflare).

No I/O, no CLI, no side effects at module level.
"""

import datetime
import math

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_AREA_FILTER = "תל אביב - מרכז העיר"
DEFAULT_BIN_HOURS = 1
DEFAULT_START = "2026-02-28"

ALARMS_CSV_URL = (
    "https://raw.githubusercontent.com/yuval-harpaz/alarms/master/data/alarms.csv"
)
TZEVAADOM_API_URL = "https://api.tzevaadom.co.il/alerts-history/"

BG_COLOR = "#f0ede3"
NIGHT_DOT_COLOR = "#333333"
DAY_DOT_COLOR = "#888888"
# ─────────────────────────────────────────────────────────────────────────────

# ── Re-exports for backward compatibility ─────────────────────────────────────
from city_translations import CITY_TRANSLATIONS          # noqa: F401, E402
from israel_time import _israel_utc_offset, _epoch_to_israel  # noqa: F401, E402
from data_loading import (                               # noqa: F401, E402
    ROCKET_DESC, load_alerts, load_api_alerts,
    load_alerts_rich, load_api_alerts_rich,
)

# Prediction logic lives in forecast.py; re-export for backward compatibility.
from forecast import predict_remaining, predict_remaining_ridge  # noqa: F401, E402
from forecast import predict_night_ridge, predict_night_rolling  # noqa: F401, E402
from forecast import _compute_global_features, _day_start_7am  # noqa: F401, E402

def _svg_wedge(cx: float, cy: float, r: float, start_frac: float, end_frac: float, color: str) -> str:
    """SVG arc path for a pie wedge, fractions 0–1 clockwise from top."""
    t0 = -math.pi / 2 + start_frac * 2 * math.pi
    t1 = -math.pi / 2 + end_frac * 2 * math.pi
    x0 = cx + r * math.cos(t0)
    y0 = cy + r * math.sin(t0)
    x1 = cx + r * math.cos(t1)
    y1 = cy + r * math.sin(t1)
    large = 1 if (end_frac - start_frac) > 0.5 else 0
    return (
        f'<path d="M {cx:.1f},{cy:.1f} L {x0:.2f},{y0:.2f} '
        f'A {r:.1f},{r:.1f} 0 {large},1 {x1:.2f},{y1:.2f} Z" '
        f'fill="{color}"/>'
    )


def compute_prediction(
    times: list[datetime.datetime],
    all_records: list[dict] | None,
    city_filter: str | None,
    forecast: str,
    global_features_cache: dict | None = None,
    now: datetime.datetime | None = None,
) -> tuple[float, float, str] | None:
    """Compute remaining-alert prediction. Returns (pred_remaining, pred_sigma, label) or None."""
    if forecast == "off":
        return None
    if now is None:
        _utc = datetime.datetime.utcnow()
        now = _utc + datetime.timedelta(hours=_israel_utc_offset(_utc))

    _night_mode = forecast == "ridge" and (now.hour >= 20 or now.hour < 6)
    label = "tonight (until 7am)" if _night_mode else "rest of today (est.)"

    if forecast == "ridge":
        if city_filter and all_records is not None:
            if _night_mode and now.hour < 6:
                rem, sig = predict_night_ridge(
                    all_records, city_filter, now=now,
                    global_features_cache=global_features_cache,
                )
            elif _night_mode:
                rem, sig = predict_night_rolling(times, now=now)
            else:
                rem, sig = predict_remaining_ridge(
                    all_records, city_filter, now=now,
                    global_features_cache=global_features_cache,
                )
        else:
            rem, sig = predict_remaining(times, now=now, method="advanced")
    else:
        rem, sig = predict_remaining(times, now=now, method=forecast)

    return rem, sig, label


def render_chart(
    times: list[datetime.datetime],
    area_label: str,
    bin_hours: int,
    start_date: str = DEFAULT_START,
    data_cutoff: datetime.datetime | None = None,
    style: str = "lines",
    fmt: str = "svg",
    threat_label: str = "Rocket",
    forecast: str = "off",
    all_records: list[dict] | None = None,
    city_filter: str | None = None,
    global_features_cache: dict | None = None,
    now: datetime.datetime | None = None,
) -> bytes:
    """Generate chart and return SVG bytes. Pure Python, no dependencies."""
    if not times:
        raise ValueError("No alerts found — cannot render chart.")

    # ── Data ─────────────────────────────────────────────────────────────────
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = max(times[-1].date(), datetime.date.today())
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += datetime.timedelta(days=1)
    n_days = len(days)

    bins: dict[tuple, int] = {}
    for t in times:
        key = (t.date(), (t.hour // bin_hours) * bin_hours)
        bins[key] = bins.get(key, 0) + 1

    daily_totals = {
        day: sum(bins.get((day, h), 0) for h in range(0, 24, bin_hours)) for day in days
    }
    daily_night = {
        day: sum(
            bins.get((day, h), 0) for h in range(0, 24, bin_hours) if h < 7 or h >= 21
        )
        for day in days
    }

    times_by_day: dict = {}
    for t in times:
        times_by_day.setdefault(t.date(), []).append(t)

    if now is None:
        _utc = datetime.datetime.utcnow()
        now = _utc + datetime.timedelta(hours=_israel_utc_offset(_utc))
    cutoff_date = now.date()
    cutoff_hour = now.hour + now.minute / 60

    # ── Prediction for today / tonight ───────────────────────────────────────
    _night_mode = forecast == "ridge" and (now.hour >= 20 or now.hour < 6)
    _pred_result = compute_prediction(
        times, all_records, city_filter, forecast,
        global_features_cache=global_features_cache, now=now,
    )
    if _pred_result is not None:
        pred_remaining, pred_sigma, _pred_label = _pred_result
        today_so_far = daily_totals.get(cutoff_date, 0)
        pred_total = today_so_far + pred_remaining
        has_prediction = int(round(pred_total)) > today_so_far
    else:
        has_prediction = False

    # ── Layout ───────────────────────────────────────────────────────────────
    ROW_H = 20
    LEFT_MARGIN = 68
    TOP_MARGIN = 58
    HOUR_W = 22
    CHART_W = 24 * HOUR_W
    DOT_COL_X = LEFT_MARGIN + CHART_W + 30
    SVG_W = LEFT_MARGIN + CHART_W + 56
    BOTTOM_MARGIN = 30 + (20 if has_prediction else 0)
    CHART_H = n_days * ROW_H
    SVG_H = TOP_MARGIN + CHART_H + BOTTOM_MARGIN
    NIGHT_BG = "#e2dfd5"
    grey = "#888888"
    tick_h = ROW_H * 0.12  # half-height of alert tick marks

    def xpx(h: float) -> float:
        return LEFT_MARGIN + h * HOUR_W

    def ypx(i: int) -> float:
        return TOP_MARGIN + i * ROW_H + ROW_H / 2

    max_daily = max(daily_totals.values()) if daily_totals else 1
    def dot_r(n: int) -> float:
        # area ∝ n: r = k*sqrt(n), k chosen so max daily total → radius ROW_H*0.35
        return math.sqrt(n / max_daily) * ROW_H * 0.504

    # ── Build SVG ─────────────────────────────────────────────────────────────
    o: list[str] = []
    o.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" '
        f'style="background:{BG_COLOR}">'
    )
    o.append(
        '<style>'
        '@import url("https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,700;1,400&amp;display=swap");'
        'text{font-family:ETBembo,"EB Garamond",Georgia,Palatino,serif}'
        '</style>'
    )
    # Embed prediction for JS to extract (pipe-separated: remaining|sigma|label)
    if _pred_result is not None:
        o.append(
            f'<desc id="pred-data">'
            f'{round(pred_remaining, 1)}|{round(pred_sigma, 1)}|{_pred_label}'
            f'</desc>'
        )

    # Night shading bands
    o.append(
        f'<rect x="{xpx(0):.0f}" y="{TOP_MARGIN}" '
        f'width="{7 * HOUR_W}" height="{CHART_H}" fill="{NIGHT_BG}"/>'
    )
    o.append(
        f'<rect x="{xpx(21):.0f}" y="{TOP_MARGIN}" '
        f'width="{3 * HOUR_W}" height="{CHART_H}" fill="{NIGHT_BG}"/>'
    )

    # Per-row baselines, date labels, alert marks
    for i, day in enumerate(days):
        yc = ypx(i)
        line_end = cutoff_hour if day == cutoff_date else 24.0
        o.append(
            f'<line x1="{xpx(0):.0f}" y1="{yc:.1f}" '
            f'x2="{xpx(line_end):.1f}" y2="{yc:.1f}" '
            f'stroke="#cccccc" stroke-width="0.4"/>'
        )
        label = day.strftime("%a %-d %b")
        o.append(
            f'<text x="{LEFT_MARGIN - 6}" y="{yc:.1f}" text-anchor="end" '
            f'dominant-baseline="middle" font-size="9" fill="#555555">{label}</text>'
        )
        if style == "lines":
            for t in times_by_day.get(day, []):
                xh = t.hour + t.minute / 60 + t.second / 3600
                col = NIGHT_DOT_COLOR if (t.hour < 7 or t.hour >= 21) else DAY_DOT_COLOR
                xp = xpx(xh)
                o.append(
                    f'<line x1="{xp:.2f}" y1="{yc - tick_h:.1f}" '
                    f'x2="{xp:.2f}" y2="{yc + tick_h:.1f}" '
                    f'stroke="{col}" stroke-width="0.8" stroke-linecap="butt"/>'
                )
        else:
            for h in range(0, 24, bin_hours):
                count = bins.get((day, h), 0)
                if count > 0:
                    col = NIGHT_DOT_COLOR if (h < 7 or h >= 21) else DAY_DOT_COLOR
                    xc = xpx(h + bin_hours / 2)
                    o.append(
                        f'<circle cx="{xc:.1f}" cy="{yc:.1f}" r="{dot_r(count):.1f}" fill="{col}"/>'
                    )

    # Bottom axis line + x-axis labels (Tufte-style: offset, bold, thick)
    AXIS_OFFSET = 6  # px gap separating axis from chart area
    axis_y = TOP_MARGIN + CHART_H + AXIS_OFFSET
    TICK_W = 1.0
    o.append(
        f'<line x1="{xpx(0) - TICK_W / 2:.1f}" y1="{axis_y}" '
        f'x2="{xpx(24) + TICK_W / 2:.1f}" y2="{axis_y}" stroke="#444444" stroke-width="1.5"/>'
    )
    for h in range(0, 25, 3):
        xp = xpx(h)
        o.append(
            f'<line x1="{xp:.0f}" y1="{axis_y}" x2="{xp:.0f}" y2="{axis_y + 6}" '
            f'stroke="#444444" stroke-width="{TICK_W}"/>'
        )
        o.append(
            f'<text x="{xp:.0f}" y="{axis_y + 16}" text-anchor="middle" '
            f'font-size="8" font-weight="bold" fill="#444444">{h:02d}:00</text>'
        )

    # Total-count column header
    o.append(
        f'<text x="{DOT_COL_X:.0f}" y="{TOP_MARGIN - 6}" text-anchor="middle" '
        f'font-size="7" fill="{grey}">total</text>'
    )
    is_today_row = has_prediction and cutoff_date in days
    for i, day in enumerate(days):
        tot = daily_totals[day]
        if not tot:
            continue
        yc = ypx(i)
        cx = DOT_COL_X
        night_frac = daily_night[day] / tot
        r = dot_r(tot)
        if night_frac <= 0:
            o.append(
                f'<circle cx="{cx:.0f}" cy="{yc:.1f}" r="{r:.1f}" fill="{DAY_DOT_COLOR}"/>'
            )
        elif night_frac >= 1:
            o.append(
                f'<circle cx="{cx:.0f}" cy="{yc:.1f}" r="{r:.1f}" fill="{NIGHT_DOT_COLOR}"/>'
            )
        else:
            o.append(_svg_wedge(cx, yc, r, 0, night_frac, NIGHT_DOT_COLOR))
            o.append(_svg_wedge(cx, yc, r, night_frac, 1.0, DAY_DOT_COLOR))
        o.append(
            f'<text x="{cx:.0f}" y="{yc:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-size="6" fill="white">{tot}</text>'
        )

    # Prediction: +N label + economist curve
    if is_today_row:
        i_today = days.index(cutoff_date)
        yc_today = ypx(i_today)
        pred_remaining_int = max(0, int(round(pred_remaining)))

        if _night_mode and now.hour < 6:
            # Midnight–6am: place annotation just before 7am (right side of night zone)
            nl_x = xpx(6.7)
            o.append(
                f'<text x="{nl_x:.1f}" y="{yc_today:.1f}" text-anchor="end" '
                f'dominant-baseline="middle" font-size="7" fill="#555555">'
                f'+{pred_remaining_int}</text>'
            )
            # Economist curve: start just below the number, arc down-left to label
            # Label sits at the LEFT of the night zone so the curve never crosses it
            ax0, ay0 = nl_x, yc_today + 4
            ax1, ay1 = 172, SVG_H - 10
            path = (
                f'M {ax0:.1f},{ay0:.1f} '
                f'C {ax0:.1f},{ay0 + 35:.1f} {ax1 + 40:.1f},{ay1:.1f} {ax1:.1f},{ay1:.1f}'
            )
            o.append(f'<path d="{path}" fill="none" stroke="{grey}" stroke-width="0.6"/>')
            o.append(
                f'<text x="{ax1 - 2:.1f}" y="{ay1:.1f}" text-anchor="end" '
                f'dominant-baseline="middle" font-size="7" font-style="italic" fill="{grey}">'
                f'{_pred_label}</text>'
            )
        else:
            # Daytime / 8pm–midnight: +N right of totals dot, curve to bottom-right
            lbl_x = DOT_COL_X + dot_r(today_so_far) + 4
            o.append(
                f'<text x="{lbl_x:.1f}" y="{yc_today:.1f}" text-anchor="start" '
                f'dominant-baseline="middle" font-size="7" fill="#555555">'
                f'+{pred_remaining_int}</text>'
            )
            ax0, ay0 = lbl_x + 4, yc_today + 5
            ax1, ay1 = DOT_COL_X - 50, SVG_H - 10
            path = (
                f'M {ax0:.1f},{ay0:.1f} '
                f'C {ax0:.1f},{ay0 + 35:.1f} {ax1 + 40:.1f},{ay1:.1f} {ax1:.1f},{ay1:.1f}'
            )
            o.append(f'<path d="{path}" fill="none" stroke="{grey}" stroke-width="0.6"/>')
            o.append(
                f'<text x="{ax1 - 2:.1f}" y="{ay1:.1f}" text-anchor="end" '
                f'dominant-baseline="middle" font-size="7" font-style="italic" fill="{grey}">'
                f'{_pred_label}</text>'
            )

    # Title + subtitle
    date_range = f"{times[0].strftime('%b %d')} \u2013 {times[-1].strftime('%b %d, %Y')}"
    o.append(
        f'<text x="{xpx(0):.0f}" y="15" font-size="14" font-weight="bold" fill="#222222">'
        f'{threat_label} frequency \u2014 {area_label}</text>'
    )
    o.append(
        f'<text x="{xpx(0):.0f}" y="31" font-size="9" fill="{grey}">'
        f'{date_range}   ({len(times)} alerts)</text>'
    )

    # Legend (upper-right area, right-aligned to chart edge)
    leg_y = 45.0
    leg_r = 4.5
    icon_x = xpx(24) - 4.0
    nd_x = icon_x
    # night/day: to the left of +forecast (or rightmost if no prediction)
    o.append(_svg_wedge(nd_x, leg_y, leg_r, 0.0, 0.5, NIGHT_DOT_COLOR))
    o.append(_svg_wedge(nd_x, leg_y, leg_r, 0.5, 1.0, DAY_DOT_COLOR))
    o.append(
        f'<text x="{nd_x - leg_r - 3:.1f}" y="{leg_y}" text-anchor="end" '
        f'dominant-baseline="middle" font-size="9" fill="{grey}">day/night:</text>'
    )
    if style == "dots":
        dot_leg_x = icon_x - 100.0
        for cnt, lbl in [(5, "5"), (1, "1")]:
            r = dot_r(cnt)
            o.append(
                f'<circle cx="{dot_leg_x:.1f}" cy="{leg_y:.0f}" r="{r:.1f}" fill="{NIGHT_DOT_COLOR}"/>'
            )
            o.append(
                f'<text x="{dot_leg_x + r + 2:.1f}" y="{leg_y}" '
                f'dominant-baseline="middle" font-size="9" fill="{grey}">{lbl}</text>'
            )
            dot_leg_x -= r * 2 + 24
        o.append(
            f'<text x="{dot_leg_x:.1f}" y="{leg_y}" text-anchor="end" '
            f'dominant-baseline="middle" font-size="9" fill="{grey}">alerts per {bin_hours}h:</text>'
        )

    o.append('</svg>')
    return ''.join(o).encode('utf-8')
