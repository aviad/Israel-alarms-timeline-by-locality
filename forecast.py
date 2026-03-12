"""
forecast.py — prediction logic for remaining alarms today.

Pure Python, no external dependencies. Shared by CLI and Cloudflare Worker.

All datetimes are naive but represent Israel local time (Asia/Jerusalem).
We avoid zoneinfo/tzdata for Cloudflare Workers compatibility and instead
use _israel_utc_offset() for UTC → Israel conversion.
"""

import datetime
import math


def _israel_utc_offset(utc_dt: datetime.datetime) -> int:
    """Return Israel UTC offset (2 or 3) for a given UTC datetime.
    DST: last Friday of March 02:00 → last Sunday of October 02:00 (Israel rule).
    """
    y = utc_dt.year
    mar31_wd = datetime.date(y, 3, 31).weekday()
    dst_start = datetime.datetime(y, 3, 31 - (mar31_wd + 3) % 7, 2)
    oct31_wd = datetime.date(y, 10, 31).weekday()
    dst_end = datetime.datetime(y, 10, 31 - (oct31_wd + 1) % 7, 2)
    return 3 if dst_start <= utc_dt < dst_end else 2


def _now_israel() -> datetime.datetime:
    """Current naive datetime in Israel local time (no tzdata needed)."""
    utc = datetime.datetime.utcnow()
    return utc + datetime.timedelta(hours=_israel_utc_offset(utc))


def _solve_normal_equation(
    X: list[list[float]], y: list[float], alpha: float = 0.0
) -> list[float]:
    """Solve (XᵀX + αI)⁻¹Xᵀy for Ridge regression. X must include bias column."""
    n, p = len(X), len(X[0])
    XtX = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    if alpha > 0:
        for i in range(p):
            XtX[i][i] += alpha
    Xty = [sum(X[k][i] * y[k] for k in range(n)) for i in range(p)]
    # Gauss-Jordan inversion
    aug = [XtX[i][:] + [1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    for col in range(p):
        max_row = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            return [0.0] * p
        for j in range(2 * p):
            aug[col][j] /= pivot
        for row in range(p):
            if row != col:
                f = aug[row][col]
                for j in range(2 * p):
                    aug[row][j] -= f * aug[col][j]
    inv = [aug[i][p:] for i in range(p)]
    return [sum(inv[i][j] * Xty[j] for j in range(p)) for i in range(p)]


def _build_training_data(
    by_day: dict[datetime.date, list[datetime.datetime]],
    train_days: list[datetime.date],
    recent_days: int,
) -> tuple[list[list[float]], list[float], list[float]]:
    """Build feature matrix and targets for both simple and advanced methods.

    All datetimes in by_day are Israel local time.
    Returns (X, y_remaining, y_daily_total).
    """
    X: list[list[float]] = []
    y_remaining: list[float] = []
    y_daily_total: list[float] = []

    for day in train_days:
        day_times = by_day[day]
        day_total = len(day_times)
        window_start = day - datetime.timedelta(days=recent_days)
        recent_counts = [
            len(by_day.get(d, [])) for d in train_days if window_start <= d < day
        ]
        recent_rate = sum(recent_counts) / max(len(recent_counts), 1)
        for hour in range(24):
            so_far = sum(1 for t in day_times if t.hour < hour)
            X.append([1.0, 24 - hour, so_far, recent_rate])
            y_remaining.append(day_total - so_far)
            y_daily_total.append(day_total)

    return X, y_remaining, y_daily_total


def predict_remaining(
    times: list[datetime.datetime],
    now: datetime.datetime | None = None,
    recent_days: int = 7,
    method: str = "simple",
) -> tuple[float, float]:
    """Predict how many alerts remain today after *now*.

    All datetimes (times and now) must be naive Israel local time.

    method="simple": direct prediction (target = remaining alerts).
    method="advanced": rate-subtraction (target = daily total, subtract observed).

    Returns (expected_remaining, std_dev).
    Returns (0.0, 0.0) if insufficient data.
    """
    if now is None:
        now = _now_israel()

    today = now.date()
    current_hour = now.hour

    by_day: dict[datetime.date, list[datetime.datetime]] = {}
    for t in times:
        by_day.setdefault(t.date(), []).append(t)

    all_days = sorted(by_day.keys())
    train_days = [d for d in all_days if d < today]
    if len(train_days) < 2:
        return 0.0, 0.0

    X, y_remaining, y_daily_total = _build_training_data(by_day, train_days, recent_days)

    y = y_daily_total if method == "advanced" else y_remaining
    beta = _solve_normal_equation(X, y)

    # Features for now (Israel local time)
    today_times = by_day.get(today, [])
    alerts_so_far = sum(1 for t in today_times if t.hour < current_hour)
    window_start = today - datetime.timedelta(days=recent_days)
    recent_counts = [
        len(by_day.get(d, [])) for d in train_days if window_start <= d < today
    ]
    recent_rate = sum(recent_counts) / max(len(recent_counts), 1)

    hours_left = 24 - current_hour
    x = [1.0, hours_left, alerts_so_far, recent_rate]
    raw_pred = sum(a * b for a, b in zip(x, beta))

    if method == "advanced":
        # Rate-sub: predicted daily total minus observed so far
        pred = max(0.0, raw_pred - alerts_so_far)
    else:
        pred = max(0.0, raw_pred)
        hourly_rate = recent_rate / 24
        pred = min(pred, hours_left * hourly_rate * 2)

    # Residual std, scaled down as day closes
    n, p = len(y), len(beta)
    ss = sum((y[i] - sum(X[i][j] * beta[j] for j in range(p))) ** 2 for i in range(n))
    sigma = math.sqrt(ss / max(n - p, 1))
    sigma = sigma * hours_left / 24

    return round(pred, 1), round(sigma, 1)


# ── 34-feature Ridge predictor ────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    # City-level
    "city_alarms_so_far", "city_minutes_since_last", "city_prev_day_total", "city_historical_avg",
    # Today's context
    "hour_of_day", "today_alarms_vs_prev_day", "today_events_so_far", "today_cities_so_far",
    "hours_since_first_alarm", "minutes_since_last_alarm", "rocket_frac",
    # Campaign
    "campaign_day", "prev_day_total_vs_avg",
    # Rate baseline
    "city_rate_pred", "adjusted_rate_pred", "city_last_24h", "intensity_ratio",
    # City profile
    "city_hit_rate", "city_rank_pct", "city_today_vs_hist", "city_ema_avg", "city_event_frac",
    # Wave structure
    "n_waves", "wave_active", "alarm_rate_last_1h", "alarm_rate_last_3h", "rate_accel", "avg_wave_gap",
    # Event/salvo context
    "avg_cities_per_event", "avg_event_gap", "events_last_1h",
    # Interaction
    "alarms_x_hours_rem", "intensity_x_hours_rem", "hist_avg_x_hours_rem",
]


def _wave_stats(sorted_mins: list[int], cutoff_min: int, wave_gap: int = 30) -> dict:
    """Wave/rate features from sorted alarm minutes (before cutoff)."""
    if not sorted_mins:
        return {
            "n_waves": 0, "wave_active": 0,
            "alarm_rate_last_1h": 0.0, "alarm_rate_last_3h": 0.0,
            "rate_accel": 0.0, "avg_wave_gap": 0.0,
        }
    n_waves = 1
    wave_starts = [sorted_mins[0]]
    for i in range(1, len(sorted_mins)):
        if sorted_mins[i] - sorted_mins[i - 1] > wave_gap:
            n_waves += 1
            wave_starts.append(sorted_mins[i])
    avg_wave_gap = 0.0
    if len(wave_starts) > 1:
        gaps = [wave_starts[j + 1] - wave_starts[j] for j in range(len(wave_starts) - 1)]
        avg_wave_gap = sum(gaps) / len(gaps)
    wave_active = 1 if (cutoff_min - sorted_mins[-1]) < wave_gap else 0
    rate_1h = float(sum(1 for m in sorted_mins if m >= cutoff_min - 60))
    rate_3h = sum(1 for m in sorted_mins if m >= cutoff_min - 180) / 3.0
    return {
        "n_waves": n_waves,
        "wave_active": wave_active,
        "alarm_rate_last_1h": rate_1h,
        "alarm_rate_last_3h": rate_3h,
        "rate_accel": rate_1h - rate_3h,
        "avg_wave_gap": avg_wave_gap,
    }


def _compute_global_features(records: list[dict], now: datetime.datetime) -> dict:
    """Compute 18 global features from event-level records as of *now*.

    records: list of {time, cities: list[str], event_id, is_rocket} — one per event.
    All datetimes are naive Israel time.

    Also returns private fields _today_count and _global_daily_avg (total alarm
    counts, i.e. sum of len(cities) across events) used by _compute_interaction_features.
    All values are JSON-serializable.
    """
    today = now.date()
    cutoff_min = now.hour * 60 + now.minute

    by_day: dict = {}
    for r in records:
        by_day.setdefault(r["time"].date(), []).append(r)

    all_days = sorted(by_day.keys())
    prior_days = [d for d in all_days if d < today]

    today_recs = [r for r in by_day.get(today, []) if r["time"] < now]
    today_mins = sorted(r["time"].hour * 60 + r["time"].minute for r in today_recs)

    # Total alarm count = sum of cities across events (matches reference global_before_count)
    def _day_alarm_count(recs):
        return sum(len(r["cities"]) for r in recs)

    today_count = _day_alarm_count(today_recs)
    global_daily_avg = (
        sum(_day_alarm_count(by_day[d]) for d in prior_days) / len(prior_days)
        if prior_days else float(today_count)
    )

    today_events_so_far = len(today_recs)  # one rec per event
    today_cities_so_far = len({c for r in today_recs for c in r["cities"]})

    if today_mins:
        hours_since_first = max(0.0, (cutoff_min - today_mins[0]) / 60.0)
        minutes_since_last = float(cutoff_min - today_mins[-1])
    else:
        hours_since_first = 0.0
        minutes_since_last = -1.0

    rocket_frac = (
        sum(1 for r in today_recs if r.get("is_rocket")) / len(today_recs)
        if today_recs else 0.5
    )

    campaign_day = (all_days.index(today) + 1) if today in by_day else len(all_days) + 1

    if prior_days:
        prev_count = _day_alarm_count(by_day[prior_days[-1]])
        today_alarms_vs_prev_day = today_count / max(prev_count, 1)
    else:
        prev_count = today_count
        today_alarms_vs_prev_day = 1.0

    if len(prior_days) >= 2:
        avg_earlier = (
            sum(_day_alarm_count(by_day[d]) for d in prior_days[:-1]) / (len(prior_days) - 1)
        )
        prev_day_total_vs_avg = prev_count / max(avg_earlier, 1)
    else:
        prev_day_total_vs_avg = 1.0

    wave = _wave_stats(today_mins, cutoff_min)

    # Event context: each record IS one event; cities list gives avg_cities_per_event
    if today_recs:
        avg_cities_per_event = sum(len(r["cities"]) for r in today_recs) / len(today_recs)
        evt_first = sorted(r["time"].hour * 60 + r["time"].minute for r in today_recs)
        avg_event_gap = (
            sum(evt_first[j + 1] - evt_first[j] for j in range(len(evt_first) - 1)) / (len(evt_first) - 1)
            if len(evt_first) > 1 else 0.0
        )
        events_last_1h = sum(1 for m in evt_first if m >= cutoff_min - 60)
    else:
        avg_cities_per_event = 0.0
        avg_event_gap = 0.0
        events_last_1h = 0

    return {
        "hour_of_day": now.hour,
        "today_alarms_vs_prev_day": today_alarms_vs_prev_day,
        "today_events_so_far": today_events_so_far,
        "today_cities_so_far": today_cities_so_far,
        "hours_since_first_alarm": hours_since_first,
        "minutes_since_last_alarm": minutes_since_last,
        "rocket_frac": rocket_frac,
        "campaign_day": campaign_day,
        "prev_day_total_vs_avg": prev_day_total_vs_avg,
        **wave,
        "avg_cities_per_event": avg_cities_per_event,
        "avg_event_gap": avg_event_gap,
        "events_last_1h": events_last_1h,
        # Private fields for interaction computation (not in FEATURE_COLS)
        "_today_count": today_count,
        "_global_daily_avg": global_daily_avg,
    }


def _compute_city_features(records: list[dict], city: str, now: datetime.datetime) -> dict:
    """Compute 9 city-specific features as of *now* (naive Israel time).

    records: list of {time, cities: list[str], event_id, is_rocket} — one per event.
    city: city name to compute features for.
    """
    today = now.date()
    cutoff_min = now.hour * 60 + now.minute
    ema_alpha = 1 - 0.5 ** (1 / 3)

    by_day_all: dict = {}
    city_by_day: dict = {}  # date -> events that hit `city`
    for r in records:
        d = r["time"].date()
        by_day_all.setdefault(d, []).append(r)
        if city in r["cities"]:
            city_by_day.setdefault(d, []).append(r)

    all_days = sorted(by_day_all.keys())
    prior_days = [d for d in all_days if d < today]

    today_city = [r for r in city_by_day.get(today, []) if r["time"] < now]
    city_alarms_so_far = len(today_city)  # events hitting city today before now
    city_mins_today = sorted(r["time"].hour * 60 + r["time"].minute for r in today_city)
    city_minutes_since_last = float(cutoff_min - city_mins_today[-1]) if city_mins_today else -1.0

    hist_counts = [len(city_by_day.get(d, [])) for d in prior_days]
    city_prev_day_total = hist_counts[-1] if hist_counts else 0
    city_historical_avg = sum(hist_counts) / max(len(hist_counts), 1) if hist_counts else 0.0
    city_hit_rate = sum(1 for c in hist_counts if c > 0) / max(len(prior_days), 1)

    # city_rank_pct: city's rank among all cities by total prior alarm events
    city_totals: dict = {}
    for d in prior_days:
        for r in by_day_all[d]:
            for c in r["cities"]:
                city_totals[c] = city_totals.get(c, 0) + 1
    if city_totals:
        sorted_c = sorted(city_totals, key=lambda c2: city_totals[c2])
        n_r = max(len(sorted_c) - 1, 1)
        city_rank_pct = {c2: i / n_r for i, c2 in enumerate(sorted_c)}.get(city, 0.0)
    else:
        city_rank_pct = 0.5

    city_today_vs_hist = city_alarms_so_far / max(city_historical_avg, 0.1)

    ema = 0.0
    for d in prior_days:
        ct = len(city_by_day.get(d, []))
        ema = ema_alpha * ct + (1 - ema_alpha) * ema

    total_events_today = len([r for r in by_day_all.get(today, []) if r["time"] < now])
    city_event_frac = city_alarms_so_far / max(total_events_today, 1)

    return {
        "city_alarms_so_far": city_alarms_so_far,
        "city_minutes_since_last": city_minutes_since_last,
        "city_prev_day_total": city_prev_day_total,
        "city_historical_avg": city_historical_avg,
        "city_hit_rate": city_hit_rate,
        "city_rank_pct": city_rank_pct,
        "city_today_vs_hist": city_today_vs_hist,
        "city_ema_avg": ema,
        "city_event_frac": city_event_frac,
    }


def _compute_interaction_features(
    global_feats: dict, city_feats: dict, hours_remaining: float
) -> dict:
    """Compute 7 interaction/rate features combining global + city features."""
    city_alarms_so_far = city_feats["city_alarms_so_far"]
    city_historical_avg = city_feats["city_historical_avg"]
    # city_last_24h: approximate as city_alarms_so_far (no 24h lookback available)
    city_last_24h = float(city_alarms_so_far)
    # intensity_ratio: current global intensity vs historical daily avg
    today_count = global_feats.get("_today_count", global_feats.get("today_events_so_far", 1))
    global_daily_avg = global_feats.get("_global_daily_avg", 1.0)
    intensity_ratio = today_count / max(global_daily_avg, 1)
    city_rate_pred = city_last_24h / 24 * hours_remaining
    adjusted_rate_pred = intensity_ratio * city_historical_avg * hours_remaining / 24
    return {
        "city_rate_pred": city_rate_pred,
        "adjusted_rate_pred": adjusted_rate_pred,
        "city_last_24h": city_last_24h,
        "intensity_ratio": intensity_ratio,
        "alarms_x_hours_rem": city_alarms_so_far * hours_remaining,
        "intensity_x_hours_rem": intensity_ratio * hours_remaining,
        "hist_avg_x_hours_rem": city_historical_avg * hours_remaining,
    }


def predict_remaining_ridge(
    all_records: list[dict],
    city: str,
    now: datetime.datetime | None = None,
    alpha: float = 10.0,
    global_features_cache: dict | None = None,
) -> tuple[float, float]:
    """Predict remaining alarms today for *city* using 34-feature Ridge regression.

    all_records: list of {time, city, event_id, is_rocket} dicts (naive Israel time).
    city: city name to predict for.
    now: current naive Israel datetime (defaults to actual current time).
    alpha: Ridge regularization strength (L2).
    global_features_cache: pre-computed global features dict (skips re-computation).

    Training: for each (historical_day, cutoff_hour in {0,3,6,9,12,15,18,21}),
    compute 34 features with simulated now = that (day, hour); target = city daily total.
    Prediction: rate-sub framing: max(0, predicted_daily_total - city_alarms_so_far).

    Returns (expected_remaining, std_dev).
    """
    if now is None:
        now = _now_israel()

    today = now.date()
    CUTOFF_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)

    all_days_set: set = {r["time"].date() for r in all_records}
    train_days = sorted(d for d in all_days_set if d < today)

    if len(train_days) < 2:
        return 0.0, 0.0

    X: list[list[float]] = []
    y_daily: list[float] = []

    for date in train_days:
        # Filter to records up to this training day (no future data leakage)
        train_records = [r for r in all_records if r["time"].date() <= date]
        # City's daily total = events that hit this city on this training day
        city_day_recs = [r for r in train_records if r["time"].date() == date and city in r["cities"]]
        city_daily_total = float(len(city_day_recs))

        for cutoff_hour in CUTOFF_HOURS:
            fake_now = datetime.datetime(date.year, date.month, date.day, cutoff_hour, 0)
            hours_remaining = float(24 - cutoff_hour)

            gf = _compute_global_features(train_records, fake_now)
            cf = _compute_city_features(train_records, city, fake_now)
            intf = _compute_interaction_features(gf, cf, hours_remaining)

            row_feats = {**gf, **cf, **intf}
            X.append([1.0] + [float(row_feats.get(f, 0.0)) for f in FEATURE_COLS])
            y_daily.append(city_daily_total)

    if len(X) < 5:
        return 0.0, 0.0

    beta = _solve_normal_equation(X, y_daily, alpha=alpha)

    # Prediction for actual now
    hours_remaining = max(0.0, 24.0 - now.hour - now.minute / 60.0)
    gf_pred = global_features_cache if global_features_cache is not None else _compute_global_features(all_records, now)
    cf_pred = _compute_city_features(all_records, city, now)
    intf_pred = _compute_interaction_features(gf_pred, cf_pred, hours_remaining)

    pred_feats = {**gf_pred, **cf_pred, **intf_pred}
    x_pred = [1.0] + [float(pred_feats.get(f, 0.0)) for f in FEATURE_COLS]
    raw_pred = sum(a * b for a, b in zip(x_pred, beta))

    # Blend rate-sub and rate predictions weighted by hours_remaining/24.
    # rate-sub dominates early (preserves observed momentum),
    # rate dominates late (decays to near-zero at end of day).
    city_alarms_so_far = cf_pred["city_alarms_so_far"]
    rate_sub = max(0.0, raw_pred - city_alarms_so_far)
    rate     = max(0.0, raw_pred * hours_remaining / 24)
    w = hours_remaining / 24
    pred_remaining = w * rate_sub + (1 - w) * rate

    # Residual sigma scaled by hours remaining
    n, p = len(X), len(beta)
    ss = sum(
        (y_daily[i] - sum(X[i][j] * beta[j] for j in range(p))) ** 2
        for i in range(n)
    )
    sigma = math.sqrt(ss / max(n - p, 1)) * hours_remaining / 24

    return round(pred_remaining, 1), round(sigma, 1)
