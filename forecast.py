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
