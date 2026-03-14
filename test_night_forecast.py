"""
test_night_forecast.py — Steps 1 & 2: data exploration + correlation analysis.

Night windows (all times are Israel local):
  whole_night  = 21:00 on day D → 07:00 on day D+1  (10 h)
  late_night   = 00:00 on day D+1 → 07:00 on day D+1  (7 h)

Prediction trigger: 20:00 on day D (8 pm).
Pre-trigger window for correlation: 17:00–20:00 on day D (5–8 pm, 3 h).
Daytime: 07:00–20:00 on day D (13 h).

Runs across all AREAS and pools results for aggregate stats.
"""
import datetime
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from alarms_core import load_alerts_rich, ALARMS_CSV_URL, DEFAULT_START
from forecast import predict_night_ridge, _day_start_7am

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

MIN_HISTORY = 7  # nights needed before making a baseline prediction

# ── Load CSV ──────────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "alarms_cache.csv"
if not CSV_PATH.exists():
    import urllib.request
    print("Downloading CSV …")
    raw = urllib.request.urlopen(ALARMS_CSV_URL).read().decode("utf-8")
    CSV_PATH.write_text(raw)
else:
    raw = CSV_PATH.read_text()

# Load all events, threat=0 (rockets), from DEFAULT_START (no area filter yet)
all_records, _ = load_alerts_rich(raw, threat=0, start=DEFAULT_START)

# ── Helper functions ──────────────────────────────────────────────────────────

def h(d: datetime.date, hour: int) -> datetime.datetime:
    return datetime.datetime(d.year, d.month, d.day, hour)


def count_window(times: list[datetime.datetime],
                 start_dt: datetime.datetime,
                 end_dt: datetime.datetime) -> int:
    return sum(1 for t in times if start_dt <= t < end_dt)


def build_nights(times: list[datetime.datetime]) -> list[dict]:
    """Build per-night feature dicts for a sorted time series."""
    by_day: dict[datetime.date, list[datetime.datetime]] = {}
    for t in times:
        by_day.setdefault(t.date(), []).append(t)
    all_days = sorted(by_day.keys())

    nights = []
    # Collect all calendar dates spanned by the data, not just alert days
    if not all_days:
        return nights
    span_start = all_days[0]
    span_end = all_days[-1]
    d = span_start
    while d < span_end:
        d1 = d + datetime.timedelta(days=1)
        nights.append({
            "date": d,
            "whole":       count_window(times, h(d, 21), h(d1, 7)),
            "late":        count_window(times, h(d1, 0), h(d1, 7)),
            "early":       count_window(times, h(d, 21), h(d1, 0)),
            "daytime":     count_window(times, h(d, 7),  h(d, 20)),
            "pre_trigger": count_window(times, h(d, 17), h(d, 20)),
        })
        d = d1
    return nights


def stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mu = sum(vals) / n
    sv = sorted(vals)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(n - 1, 1))
    return {"n": n, "mean": mu, "median": med, "std": sd,
            "min": sv[0], "max": sv[-1],
            "zero_pct": 100 * sum(1 for v in vals if v == 0) / n}


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / denom if denom > 1e-12 else float("nan")


def rolling_baseline_errors(nights: list[dict], key: str) -> list[float]:
    """Absolute errors of rolling 7-night mean vs actual."""
    errors = []
    for i, rec in enumerate(nights):
        past = nights[max(0, i - 7): i]
        if len(past) < MIN_HISTORY:
            continue
        pred = sum(p[key] for p in past) / len(past)
        errors.append(abs(pred - rec[key]))
    return errors


def rate_baseline_errors(nights: list[dict], key: str, window_hours: float) -> list[float]:
    errors = []
    for i, rec in enumerate(nights):
        past = nights[max(0, i - 7): i]
        if len(past) < MIN_HISTORY:
            continue
        total_7d = sum(p["daytime"] + p["whole"] for p in past)
        pred = (total_7d / (7 * 24)) * window_hours
        errors.append(abs(pred - rec[key]))
    return errors


# ── Per-area analysis ─────────────────────────────────────────────────────────

pooled_nights: list[dict] = []   # all (area, night) rows
area_results: list[dict] = []

print(f"{'='*65}")
print("PER-AREA RESULTS")
print(f"{'='*65}")
print(f"{'Area':<35} {'N':>3}  {'WN mean':>7}  {'LN mean':>7}  "
      f"{'r(day→WN)':>9}  {'r(pre→WN)':>9}  {'r(early→late)':>13}")
print(f"{'-'*95}")

for area in AREAS:
    area_recs = [r for r in all_records if area in r["cities"]]
    if not area_recs:
        continue
    area_times = sorted(r["time"] for r in area_recs)
    nights = build_nights(area_times)
    if len(nights) < 3:
        continue

    whole  = [float(n["whole"])       for n in nights]
    late   = [float(n["late"])        for n in nights]
    early  = [float(n["early"])       for n in nights]
    day    = [float(n["daytime"])     for n in nights]
    pre    = [float(n["pre_trigger"]) for n in nights]

    r1 = pearson(day,   whole)
    r2 = pearson(early, late)
    r3 = pearson(pre,   whole)

    sw = stats(whole)
    sl = stats(late)

    print(f"  {area:<33} {sw['n']:>3}  {sw['mean']:>7.2f}  {sl['mean']:>7.2f}  "
          f"{r1:>+9.3f}  {r3:>+9.3f}  {r2:>+13.3f}")

    area_results.append({"area": area, "nights": nights, "n": sw["n"],
                         "r_day_wn": r1, "r_pre_wn": r3, "r_early_late": r2,
                         "wn_mean": sw["mean"], "ln_mean": sl["mean"]})
    pooled_nights.extend(nights)

# ── Pooled aggregate stats ────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"POOLED AGGREGATE  (n={len(pooled_nights)} area-night pairs, {len(area_results)} areas)")
print(f"{'='*65}")

p_whole  = [float(n["whole"])       for n in pooled_nights]
p_late   = [float(n["late"])        for n in pooled_nights]
p_early  = [float(n["early"])       for n in pooled_nights]
p_day    = [float(n["daytime"])     for n in pooled_nights]
p_pre    = [float(n["pre_trigger"]) for n in pooled_nights]

sw = stats(p_whole)
sl = stats(p_late)

print(f"\n  Whole night (9pm–7am)")
print(f"    Mean={sw['mean']:.2f}  Median={sw['median']:.1f}  Std={sw['std']:.2f}  "
      f"Min={sw['min']:.0f}  Max={sw['max']:.0f}  Zero={sw['zero_pct']:.0f}%")
print(f"\n  Late night (12am–7am)")
print(f"    Mean={sl['mean']:.2f}  Median={sl['median']:.1f}  Std={sl['std']:.2f}  "
      f"Min={sl['min']:.0f}  Max={sl['max']:.0f}  Zero={sl['zero_pct']:.0f}%")

r1 = pearson(p_day,   p_whole)
r2 = pearson(p_early, p_late)
r3 = pearson(p_pre,   p_whole)
print(f"\n  Corr(daytime 7am–8pm  → whole night)  : {r1:+.3f}")
print(f"  Corr(pre-trigger 5pm–8pm → whole night): {r3:+.3f}")
print(f"  Corr(early night 9pm–12am → late night): {r2:+.3f}")

# ── Baselines (pooled) ────────────────────────────────────────────────────────

print(f"\n  Baselines (per-area rolling 7-night, pooled errors):")

roll_w, roll_l, rate_w, rate_l = [], [], [], []
for ar in area_results:
    roll_w.extend(rolling_baseline_errors(ar["nights"], "whole"))
    roll_l.extend(rolling_baseline_errors(ar["nights"], "late"))
    rate_w.extend(rate_baseline_errors(ar["nights"], "whole", 10))
    rate_l.extend(rate_baseline_errors(ar["nights"], "late",  7))

def mae_rmse(errs: list[float]) -> str:
    if not errs:
        return "n/a"
    mae  = sum(errs) / len(errs)
    rmse = math.sqrt(sum(e**2 for e in errs) / len(errs))
    return f"MAE={mae:.2f}  RMSE={rmse:.2f}  (n={len(errs)})"

print(f"    Rolling 7-night mean — whole: {mae_rmse(roll_w)}")
print(f"    Rolling 7-night mean — late : {mae_rmse(roll_l)}")
print(f"    Rate baseline        — whole: {mae_rmse(rate_w)}")
print(f"    Rate baseline        — late : {mae_rmse(rate_l)}")

# ── Day-of-week breakdown (pooled whole-night) ────────────────────────────────

print(f"\n{'='*65}")
print("DAY-OF-WEEK BREAKDOWN (pooled whole-night counts)")
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_dow: dict[int, list[float]] = {}
for n in pooled_nights:
    by_dow.setdefault(n["date"].weekday(), []).append(float(n["whole"]))

print(f"  {'Day':<6} {'N':>3}  {'Mean':>6}  {'Median':>7}  {'Zero%':>6}")
print(f"  {'-'*36}")
for dow in range(7):
    vals = by_dow.get(dow, [])
    if not vals:
        continue
    mu  = sum(vals) / len(vals)
    med = sorted(vals)[len(vals) // 2]
    z   = 100 * sum(1 for v in vals if v == 0) / len(vals)
    print(f"  {DOW[dow]:<6} {len(vals):>3}  {mu:>6.2f}  {med:>7.1f}  {z:>5.0f}%")

# ── Summary verdict ───────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("VERDICT")
avg_r1 = sum(a["r_day_wn"] for a in area_results if not math.isnan(a["r_day_wn"])) / max(1, sum(1 for a in area_results if not math.isnan(a["r_day_wn"])))
avg_r3 = sum(a["r_pre_wn"] for a in area_results if not math.isnan(a["r_pre_wn"])) / max(1, sum(1 for a in area_results if not math.isnan(a["r_pre_wn"])))
print(f"  Mean r(daytime→WN) across areas  : {avg_r1:+.3f}")
print(f"  Mean r(pre-trigger→WN) across areas: {avg_r3:+.3f}")
if abs(avg_r1) > 0.4 or abs(avg_r3) > 0.4:
    print("  >> Consistent signal found — Steps 3–5 worthwhile.")
else:
    print("  >> Weak signal — naive rolling average likely sufficient.")


# ── Step E: Backtest predict_night_ridge ──────────────────────────────────────

print(f"\n{'='*65}")
print("BACKTEST: predict_night_ridge")
print(f"{'='*65}")
print("Simulating 8pm and midnight predictions per area-night pair")
print("(Needs ≥7 prior 7am-days of data; skips earlier nights)\n")

# Collect all unique 7am-day-start dates in the dataset
all_7am_days = sorted({_day_start_7am(r["time"]) for r in all_records})

bt_8pm_errors: list[float] = []   # whole night (8pm→7am prediction vs actual)
bt_mid_errors: list[float] = []   # late night (midnight→7am prediction vs actual)
bt_8pm_naive: list[float] = []    # rolling 7-night baseline for comparison
bt_mid_naive: list[float] = []

MIN_TRAIN_DAYS = 7  # minimum 7am-days needed before backtesting

for area in AREAS:
    area_recs = [r for r in all_records if area in r["cities"]]
    if not area_recs:
        continue

    area_days = sorted({_day_start_7am(r["time"]) for r in area_recs})

    # Map 7am-day-start → actual whole-night count for this area
    def _night_count(day_start: datetime.date, start_h: int, end_day_offset: int, end_h: int) -> int:
        d1 = day_start + datetime.timedelta(days=end_day_offset)
        return sum(
            1 for r in area_recs
            if datetime.datetime(day_start.year, day_start.month, day_start.day, start_h)
            <= r["time"]
            < datetime.datetime(d1.year, d1.month, d1.day, end_h)
        )

    area_night_history: list[int] = []  # per 7am-day whole-night count (for rolling baseline)

    for idx, day_start in enumerate(area_days):
        # whole night = 9pm day_start → 7am day_start+1
        whole_actual = _night_count(day_start, 21, 1, 7)
        # late night  = midnight day_start+1 → 7am day_start+1
        late_actual  = _night_count(
            day_start + datetime.timedelta(days=1), 0, 0, 7
        )

        # Need ≥ MIN_TRAIN_DAYS prior days
        prior_nights = area_night_history[-7:] if len(area_night_history) >= MIN_HISTORY else []
        area_night_history.append(whole_actual)

        if len(prior_nights) < MIN_HISTORY:
            continue

        # Simulate 8pm prediction: all records known up to 8pm on day_start
        sim_8pm = datetime.datetime(day_start.year, day_start.month, day_start.day, 20, 0)
        known_8pm = [r for r in all_records if r["time"] < sim_8pm]
        if not known_8pm:
            continue

        pred_8pm, _ = predict_night_ridge(known_8pm, area, now=sim_8pm)

        # remaining at 8pm = everything from 8pm to 7am next day
        remaining_8pm = _night_count(day_start, 20, 1, 7)
        bt_8pm_errors.append(abs(pred_8pm - remaining_8pm))
        bt_8pm_naive.append(abs(sum(prior_nights) / len(prior_nights) - whole_actual))

        # Simulate midnight prediction: all records known up to midnight
        sim_mid = datetime.datetime(
            (day_start + datetime.timedelta(days=1)).year,
            (day_start + datetime.timedelta(days=1)).month,
            (day_start + datetime.timedelta(days=1)).day,
            0, 0,
        )
        known_mid = [r for r in all_records if r["time"] < sim_mid]
        if not known_mid:
            continue

        pred_mid, _ = predict_night_ridge(known_mid, area, now=sim_mid)
        bt_mid_errors.append(abs(pred_mid - late_actual))
        bt_mid_naive.append(abs(sum(p for p in prior_nights) / len(prior_nights) - late_actual))

def _mae_rmse(errs: list[float], label: str) -> None:
    if not errs:
        print(f"  {label}: no data")
        return
    mae  = sum(errs) / len(errs)
    rmse = math.sqrt(sum(e ** 2 for e in errs) / len(errs))
    print(f"  {label}: MAE={mae:.2f}  RMSE={rmse:.2f}  (n={len(errs)})")

_mae_rmse(bt_8pm_errors, "Ridge 8pm→7am (whole night)")
_mae_rmse(bt_8pm_naive,  "Naive 7-night  (whole night)")
_mae_rmse(bt_mid_errors, "Ridge 12am→7am (late night) ")
_mae_rmse(bt_mid_naive,  "Naive 7-night  (late night) ")

if bt_8pm_errors and bt_8pm_naive:
    improvement = (sum(bt_8pm_naive) - sum(bt_8pm_errors)) / sum(bt_8pm_naive) * 100
    print(f"\n  8pm Ridge vs naive improvement: {improvement:+.1f}%  (target >+15%)")
    verdict = "PASS" if improvement > 15 else "FAIL — use rolling average fallback"
    print(f"  Verdict: {verdict}")
if bt_mid_errors and bt_mid_naive:
    improvement_mid = (sum(bt_mid_naive) - sum(bt_mid_errors)) / sum(bt_mid_naive) * 100
    print(f"  Midnight Ridge vs naive improvement: {improvement_mid:+.1f}%  (target >+15%)")

