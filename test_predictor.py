"""
Backtest the linear regression predictor against historical data.

Train on Mar 2-7, test on Mar 8-9. Multiple areas.
"""
import datetime
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alarms_core import load_alerts

# ── Load CSV ─────────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "alarms_cache.csv"
if not CSV_PATH.exists():
    import urllib.request
    from alarms_core import ALARMS_CSV_URL
    data = urllib.request.urlopen(ALARMS_CSV_URL).read().decode("utf-8")
    CSV_PATH.write_text(data)
else:
    data = CSV_PATH.read_text()

START = "2026-03-02"
TEST_DAYS = {datetime.date(2026, 3, 8), datetime.date(2026, 3, 9)}

AREAS = [
    "תל אביב - מרכז העיר",
    "חיפה - כרמל ועיר תחתית",
    "ירושלים - מרכז",
    "באר שבע - מערב",
    "ראשון לציון - מערב",
    "פתח תקווה",
    "נתניה - מערב",
    "אשדוד - א,ב,ג,ד,ו",
    "רמת גן - מערב",
    "הרצליה",
]

# ── Predictor helpers ────────────────────────────────────────────────────────

def build_training_data(
    times_by_day: dict[datetime.date, list[datetime.datetime]],
    exclude_days: set[datetime.date],
    recent_days: int = 7,
) -> tuple[list[list[float]], list[float]]:
    all_days = sorted(times_by_day.keys())
    X, y = [], []
    for day in all_days:
        if day in exclude_days:
            continue
        day_times = times_by_day[day]
        day_total = len(day_times)
        window_start = day - datetime.timedelta(days=recent_days)
        recent_counts = [
            len(times_by_day.get(d, []))
            for d in all_days
            if window_start <= d < day and d not in exclude_days
        ]
        recent_rate = sum(recent_counts) / max(len(recent_counts), 1)
        for hour in range(24):
            alerts_so_far = sum(1 for t in day_times if t.hour < hour)
            alerts_remaining = day_total - alerts_so_far
            X.append([hours_remaining := 24 - hour, alerts_so_far, recent_rate])
            y.append(alerts_remaining)
    return X, y


def solve_normal_equation(X, y):
    n, p = len(X), len(X[0])
    XtX = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    Xty = [sum(X[k][i] * y[k] for k in range(n)) for i in range(p)]
    aug = [XtX[i][:] + [1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    for col in range(p):
        max_row = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")
        for j in range(2 * p):
            aug[col][j] /= pivot
        for row in range(p):
            if row != col:
                f = aug[row][col]
                for j in range(2 * p):
                    aug[row][j] -= f * aug[col][j]
    inv = [aug[i][p:] for i in range(p)]
    return [sum(inv[i][j] * Xty[j] for j in range(p)) for i in range(p)]


def predict(x, beta):
    return sum(a * b for a, b in zip(x, beta))


def residual_std(X, y, beta):
    n, p = len(y), len(beta)
    ss = sum((y[i] - predict(X[i], beta)) ** 2 for i in range(n))
    return math.sqrt(ss / max(n - p, 1))


# ── Run backtest per area ────────────────────────────────────────────────────

test_hours = [0, 6, 9, 12, 15, 18, 21]
all_area_errors = []

for area in AREAS:
    times, _ = load_alerts(data, area, threat=-1, start=START)
    if not times:
        print(f"\n{'='*60}")
        print(f"AREA: {area} — NO ALERTS, skipping")
        continue

    times_by_day: dict[datetime.date, list[datetime.datetime]] = {}
    for t in times:
        times_by_day.setdefault(t.date(), []).append(t)

    all_days = sorted(times_by_day.keys())
    actual_test_days = sorted(d for d in all_days if d in TEST_DAYS)
    if not actual_test_days:
        print(f"\n{'='*60}")
        print(f"AREA: {area} — no alerts on test days, skipping")
        continue

    # Train on non-test days
    X_train, y_train = build_training_data(times_by_day, exclude_days=TEST_DAYS)
    if not X_train:
        print(f"\n{'='*60}")
        print(f"AREA: {area} — no training data, skipping")
        continue

    X_train_b = [[1.0] + row for row in X_train]
    beta = solve_normal_equation(X_train_b, y_train)
    sigma = residual_std(X_train_b, y_train, beta)

    print(f"\n{'='*60}")
    print(f"AREA: {area}")
    print(f"  Training days: {[d.isoformat() for d in all_days if d not in TEST_DAYS]}")
    print(f"  Alerts/day: {', '.join(f'{d}={len(times_by_day[d])}' for d in all_days)}")
    print(f"  β=[bias={beta[0]:.2f}, hrs_rem={beta[1]:.2f}, today={beta[2]:.2f}, rate={beta[3]:.2f}]  σ={sigma:.1f}")
    print(f"  {'Day':<12} {'Hour':>4} {'Actual':>7} {'Pred':>6} {'Err':>6}")
    print(f"  {'-'*42}")

    area_errors = []
    for test_day in actual_test_days:
        day_times = times_by_day.get(test_day, [])
        day_total = len(day_times)

        window_start = test_day - datetime.timedelta(days=7)
        recent_counts = [
            len(times_by_day.get(d, []))
            for d in all_days
            if window_start <= d < test_day and d not in TEST_DAYS
        ]
        recent_rate = sum(recent_counts) / max(len(recent_counts), 1)

        for hour in test_hours:
            alerts_so_far = sum(1 for t in day_times if t.hour < hour)
            actual_remaining = day_total - alerts_so_far
            x = [1.0, 24 - hour, alerts_so_far, recent_rate]
            pred = max(0, predict(x, beta))
            error = pred - actual_remaining
            area_errors.append(error)
            all_area_errors.append(error)
            print(f"  {test_day}  {hour:>4}h  {actual_remaining:>5}  {pred:>6.1f}  {error:>+5.1f}")

    a_mae = sum(abs(e) for e in area_errors) / len(area_errors)
    a_bias = sum(area_errors) / len(area_errors)
    print(f"  ── area MAE={a_mae:.2f}, bias={a_bias:+.2f}")

# ── Grand summary ────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("GRAND SUMMARY (all areas, Mar 8-9 only)")
if all_area_errors:
    mae = sum(abs(e) for e in all_area_errors) / len(all_area_errors)
    rmse = math.sqrt(sum(e**2 for e in all_area_errors) / len(all_area_errors))
    bias = sum(all_area_errors) / len(all_area_errors)
    print(f"  Samples: {len(all_area_errors)}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Bias: {bias:+.2f}")
else:
    print("  No test data available.")
