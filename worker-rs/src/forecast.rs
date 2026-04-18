/// forecast.rs — prediction models, ported from forecast.py.
///
/// All inputs are naive Israel local time (IsraelTime). No external dependencies.

use std::collections::HashMap;
use crate::data_loading::{IsraelTime, AlertRecord};
use crate::israel_time::{ymd_to_epoch, days_to_ymd};

// ── Date helpers ─────────────────────────────────────────────────────────────

/// A calendar date derived from IsraelTime (year, month, day).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

impl Date {
    pub fn from_israel(t: &IsraelTime) -> Self {
        Date { year: t.year, month: t.month, day: t.day }
    }

    /// Add `delta` days (may be negative).
    pub fn add_days(self, delta: i64) -> Self {
        // ymd_to_epoch gives seconds; work in units of days.
        let day_epoch = ymd_to_epoch(self.year, self.month, self.day) / 86400 + delta;
        let (y, m, d) = days_to_ymd(day_epoch);
        Date { year: y, month: m, day: d }
    }
}

/// 7am-day-start for t: records in [D 07:00, D+1 07:00) belong to 7am-day D.
fn day_start_7am(t: &IsraelTime) -> Date {
    let d = Date::from_israel(t);
    if t.hour >= 7 { d } else { d.add_days(-1) }
}

// ── Linear algebra ────────────────────────────────────────────────────────────

/// Solve (XᵀX + αI)⁻¹Xᵀy via Gauss-Jordan elimination.
/// X must contain a bias column (first column = 1.0).
/// Returns a zero vector if the system is singular (pivot < 1e-12).
/// Used by predict_remaining (α=0) and Ridge (α>0, step 5).
pub fn solve_normal_equation(x: &[Vec<f64>], y: &[f64], alpha: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let p = x[0].len();

    // XᵀX (p×p)
    let mut xtx = vec![vec![0.0f64; p]; p];
    for k in 0..n {
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] += x[k][i] * x[k][j];
            }
        }
    }
    if alpha > 0.0 {
        for i in 0..p {
            xtx[i][i] += alpha;
        }
    }

    // Xᵀy
    let mut xty = vec![0.0f64; p];
    for k in 0..n {
        for i in 0..p {
            xty[i] += x[k][i] * y[k];
        }
    }

    // Augmented matrix [XᵀX | I] for Gauss-Jordan inversion
    let mut aug: Vec<Vec<f64>> = (0..p)
        .map(|i| {
            let mut row = xtx[i].clone();
            for j in 0..p {
                row.push(if i == j { 1.0 } else { 0.0 });
            }
            row
        })
        .collect();

    for col in 0..p {
        // Partial pivoting
        let max_row = (col..p)
            .max_by(|&a, &b| {
                aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap()
            })
            .unwrap();
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            return vec![0.0; p];
        }
        for j in 0..2 * p {
            aug[col][j] /= pivot;
        }
        for row in 0..p {
            if row != col {
                let f = aug[row][col];
                for j in 0..2 * p {
                    aug[row][j] -= f * aug[col][j];
                }
            }
        }
    }

    // β = inv · Xᵀy
    (0..p)
        .map(|i| (0..p).map(|j| aug[i][p + j] * xty[j]).sum())
        .collect()
}

// ── Training data builder ─────────────────────────────────────────────────────

/// Group IsraelTimes by calendar date.
fn group_by_date<'a>(times: &'a [IsraelTime]) -> HashMap<Date, Vec<&'a IsraelTime>> {
    let mut map: HashMap<Date, Vec<&'a IsraelTime>> = HashMap::new();
    for t in times {
        map.entry(Date::from_israel(t)).or_default().push(t);
    }
    map
}

/// Build feature matrix and targets for predict_remaining.
/// Returns (X rows, y_remaining, y_daily_total).
/// Features per row: [1.0, hours_left, alarms_so_far, recent_rate].
fn build_training_data(
    by_day: &HashMap<Date, Vec<&IsraelTime>>,
    train_days: &[Date],
    recent_days: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let mut x_mat = Vec::new();
    let mut y_rem = Vec::new();
    let mut y_tot = Vec::new();

    for &day in train_days {
        let day_times = by_day.get(&day).map(|v| v.as_slice()).unwrap_or(&[]);
        let day_total = day_times.len() as f64;

        // Average count over the last `recent_days` *training* days before this day.
        // Only days present in train_days (i.e. with alarms) are averaged — matches Python.
        let window_start = day.add_days(-(recent_days as i64));
        let recent_counts: Vec<usize> = train_days
            .iter()
            .filter(|&&d| d >= window_start && d < day)
            .map(|d| by_day.get(d).map(|v| v.len()).unwrap_or(0))
            .collect();
        let recent_rate = if recent_counts.is_empty() {
            0.0
        } else {
            recent_counts.iter().sum::<usize>() as f64 / recent_counts.len() as f64
        };

        for hour in 0u32..24 {
            let so_far = day_times.iter().filter(|t| t.hour < hour).count() as f64;
            x_mat.push(vec![1.0, (24 - hour) as f64, so_far, recent_rate]);
            y_rem.push(day_total - so_far);
            y_tot.push(day_total);
        }
    }
    (x_mat, y_rem, y_tot)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Predict how many alerts remain today after `now` (naive Israel local time).
///
/// `method = "simple"`: direct regression (target = remaining alerts).
/// `method = "advanced"`: rate-subtraction (target = daily total, subtract observed).
/// Returns (expected_remaining, std_dev). Returns (0.0, 0.0) if < 2 training days.
pub fn predict_remaining(
    times: &[IsraelTime],
    now: &IsraelTime,
    recent_days: usize,
    method: &str,
) -> (f64, f64) {
    let today = Date::from_israel(now);
    let current_hour = now.hour;

    let by_day = group_by_date(times);

    let mut all_days: Vec<Date> = by_day.keys().copied().collect();
    all_days.sort();
    let train_days: Vec<Date> = all_days.iter().copied().filter(|&d| d < today).collect();

    if train_days.len() < 2 {
        return (0.0, 0.0);
    }

    let (x_mat, y_rem, y_tot) = build_training_data(&by_day, &train_days, recent_days);

    let y = if method == "advanced" { &y_tot } else { &y_rem };
    let beta = solve_normal_equation(&x_mat, y, 0.0);

    // Features for `now`
    let today_times = by_day.get(&today).map(|v| v.as_slice()).unwrap_or(&[]);
    let alerts_so_far = today_times.iter().filter(|t| t.hour < current_hour).count() as f64;

    let window_start = today.add_days(-(recent_days as i64));
    let recent_counts: Vec<usize> = train_days
        .iter()
        .filter(|&&d| d >= window_start && d < today)
        .map(|d| by_day.get(&d).map(|v| v.len()).unwrap_or(0))
        .collect();
    let recent_rate = if recent_counts.is_empty() {
        0.0
    } else {
        recent_counts.iter().sum::<usize>() as f64 / recent_counts.len() as f64
    };

    let hours_left = (24 - current_hour) as f64;
    let x_pred = [1.0, hours_left, alerts_so_far, recent_rate];
    let raw_pred: f64 = x_pred.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();

    let pred = if method == "advanced" {
        (raw_pred - alerts_so_far).max(0.0)
    } else {
        let hourly_rate = recent_rate / 24.0;
        raw_pred.max(0.0).min(hours_left * hourly_rate * 2.0)
    };

    // Residual std scaled by hours remaining
    let n = x_mat.len();
    let p = beta.len();
    let ss: f64 = x_mat
        .iter()
        .zip(y.iter())
        .map(|(xi, &yi)| {
            let p_i: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
            (yi - p_i).powi(2)
        })
        .sum();
    let sigma = (ss / n.saturating_sub(p).max(1) as f64).sqrt() * hours_left / 24.0;

    (round1(pred), round1(sigma))
}

/// Predict remaining night alerts (until 7am) using rolling average.
///
/// Computes rolling mean/std of whole-night (9pm→7am) counts over the last
/// `recent_days` 7am-days, pro-rated by `hours_remaining / 10.0`
/// (full 9pm–7am window = 10h).
///
/// Returns (expected_remaining, std_dev). Returns (0.0, 0.0) if no past data.
pub fn predict_night_rolling(
    times: &[IsraelTime],
    now: &IsraelTime,
    recent_days: usize,
) -> (f64, f64) {
    let current_day_start = day_start_7am(now);

    let mut by_7am_day: HashMap<Date, Vec<&IsraelTime>> = HashMap::new();
    for t in times {
        by_7am_day.entry(day_start_7am(t)).or_default().push(t);
    }

    let mut past_days: Vec<Date> = by_7am_day
        .keys()
        .copied()
        .filter(|&d| d < current_day_start)
        .collect();
    past_days.sort();

    if past_days.is_empty() {
        return (0.0, 0.0);
    }

    // Whole-night count for 7am-day D: alarms in [21:00 D, 07:00 D+1).
    // Within by_7am_day[D], these are entries with hour >= 21 (evening) or hour < 7 (early morning).
    let whole_night = |d: Date| -> f64 {
        by_7am_day
            .get(&d)
            .map(|v| v.iter().filter(|t| t.hour >= 21 || t.hour < 7).count() as f64)
            .unwrap_or(0.0)
    };

    let recent_start = past_days.len().saturating_sub(recent_days);
    let recent = &past_days[recent_start..];
    let night_counts: Vec<f64> = recent.iter().map(|&d| whole_night(d)).collect();

    let avg = night_counts.iter().sum::<f64>() / night_counts.len() as f64;
    let variance = night_counts
        .iter()
        .map(|&c| (c - avg).powi(2))
        .sum::<f64>()
        / night_counts.len().saturating_sub(1).max(1) as f64;
    let std = variance.sqrt();

    // hours_remaining within the 10h night window (9pm → 7am)
    // Same formula as Python: max(0, 24 - (now.hour - 7) % 24 - now.minute / 60)
    let elapsed = ((now.hour as f64 - 7.0).rem_euclid(24.0)) + now.min as f64 / 60.0;
    let hours_remaining = (24.0 - elapsed).max(0.0);
    let scale = (hours_remaining / 10.0).min(1.0);

    (round1(avg * scale), round1(std * scale))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Round to 1 decimal place (matches Python `round(x, 1)`).
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

// ── 34-feature Ridge predictors ───────────────────────────────────────────────

/// Feature column order — must match Python FEATURE_COLS exactly.
const FEATURE_COLS: &[&str] = &[
    "city_alarms_so_far", "city_minutes_since_last", "city_prev_day_total", "city_historical_avg",
    "hour_of_day", "today_alarms_vs_prev_day", "today_events_so_far", "today_cities_so_far",
    "hours_since_first_alarm", "minutes_since_last_alarm", "rocket_frac",
    "campaign_day", "prev_day_total_vs_avg",
    "city_rate_pred", "adjusted_rate_pred", "city_last_24h", "intensity_ratio",
    "city_hit_rate", "city_rank_pct", "city_today_vs_hist", "city_ema_avg", "city_event_frac",
    "n_waves", "wave_active", "alarm_rate_last_1h", "alarm_rate_last_3h", "rate_accel", "avg_wave_gap",
    "avg_cities_per_event", "avg_event_gap", "events_last_1h",
    "alarms_x_hours_rem", "intensity_x_hours_rem", "hist_avg_x_hours_rem",
];

/// Wave/rate features from sorted alarm minutes (before cutoff_min).
fn wave_stats(sorted_mins: &[i32], cutoff_min: i32, wave_gap: i32) -> HashMap<&'static str, f64> {
    let mut out = HashMap::new();
    if sorted_mins.is_empty() {
        out.insert("n_waves", 0.0);
        out.insert("wave_active", 0.0);
        out.insert("alarm_rate_last_1h", 0.0);
        out.insert("alarm_rate_last_3h", 0.0);
        out.insert("rate_accel", 0.0);
        out.insert("avg_wave_gap", 0.0);
        return out;
    }
    let mut n_waves = 1i32;
    let mut wave_starts = vec![sorted_mins[0]];
    for i in 1..sorted_mins.len() {
        if sorted_mins[i] - sorted_mins[i - 1] > wave_gap {
            n_waves += 1;
            wave_starts.push(sorted_mins[i]);
        }
    }
    let avg_wave_gap = if wave_starts.len() > 1 {
        let gaps: Vec<i32> = wave_starts.windows(2).map(|w| w[1] - w[0]).collect();
        gaps.iter().sum::<i32>() as f64 / gaps.len() as f64
    } else {
        0.0
    };
    let wave_active = if cutoff_min - sorted_mins.last().unwrap() < wave_gap { 1.0 } else { 0.0 };
    let rate_1h = sorted_mins.iter().filter(|&&m| m >= cutoff_min - 60).count() as f64;
    let rate_3h = sorted_mins.iter().filter(|&&m| m >= cutoff_min - 180).count() as f64 / 3.0;
    out.insert("n_waves", n_waves as f64);
    out.insert("wave_active", wave_active);
    out.insert("alarm_rate_last_1h", rate_1h);
    out.insert("alarm_rate_last_3h", rate_3h);
    out.insert("rate_accel", rate_1h - rate_3h);
    out.insert("avg_wave_gap", avg_wave_gap);
    out
}

/// Global features (18 keys + 2 private: _today_count, _global_daily_avg).
fn compute_global_features(records: &[AlertRecord], now: &IsraelTime) -> HashMap<&'static str, f64> {
    let today = Date::from_israel(now);
    let cutoff_min = (now.hour * 60 + now.min) as i32;

    // Group records by calendar date
    let mut by_day: HashMap<Date, Vec<&AlertRecord>> = HashMap::new();
    for r in records {
        by_day.entry(Date::from_israel(&r.time)).or_default().push(r);
    }
    let mut all_days: Vec<Date> = by_day.keys().copied().collect();
    all_days.sort();
    let prior_days: Vec<Date> = all_days.iter().copied().filter(|&d| d < today).collect();

    // today_recs: events strictly before now
    let today_recs: Vec<&AlertRecord> = by_day
        .get(&today)
        .map(|v| v.iter().copied().filter(|r| r.time < *now).collect())
        .unwrap_or_default();

    let mut today_mins: Vec<i32> = today_recs
        .iter()
        .map(|r| (r.time.hour * 60 + r.time.min) as i32)
        .collect();
    today_mins.sort();

    fn day_alarm_count(recs: &[&AlertRecord]) -> usize {
        recs.iter().map(|r| r.cities.len()).sum()
    }

    let today_count = day_alarm_count(&today_recs) as f64;
    let global_daily_avg = if !prior_days.is_empty() {
        prior_days.iter().map(|d| {
            day_alarm_count(by_day.get(d).map(|v| v.as_slice()).unwrap_or(&[])) as f64
        }).sum::<f64>() / prior_days.len() as f64
    } else {
        today_count
    };

    let today_events_so_far = today_recs.len() as f64;
    let today_cities_so_far: usize = today_recs
        .iter()
        .flat_map(|r| r.cities.iter().map(|s| s.as_str()))
        .collect::<std::collections::HashSet<_>>()
        .len();

    let (hours_since_first, minutes_since_last) = if !today_mins.is_empty() {
        (
            (cutoff_min - today_mins[0]).max(0) as f64 / 60.0,
            (cutoff_min - today_mins.last().unwrap()) as f64,
        )
    } else {
        (0.0, -1.0)
    };

    let rocket_frac = if !today_recs.is_empty() {
        today_recs.iter().filter(|r| r.is_rocket).count() as f64 / today_recs.len() as f64
    } else {
        0.5
    };

    // campaign_day: 1-based index of today in all_days that have records
    let campaign_day = if let Some(pos) = all_days.iter().position(|&d| d == today) {
        (pos + 1) as f64
    } else {
        all_days.len() as f64 + 1.0
    };

    let (today_alarms_vs_prev_day, prev_count) = if !prior_days.is_empty() {
        let pc = day_alarm_count(
            by_day.get(prior_days.last().unwrap()).map(|v| v.as_slice()).unwrap_or(&[])
        ) as f64;
        (today_count / pc.max(1.0), pc)
    } else {
        (1.0, today_count)
    };

    let prev_day_total_vs_avg = if prior_days.len() >= 2 {
        let earlier_avg = prior_days[..prior_days.len()-1].iter().map(|d| {
            day_alarm_count(by_day.get(d).map(|v| v.as_slice()).unwrap_or(&[])) as f64
        }).sum::<f64>() / (prior_days.len() - 1) as f64;
        prev_count / earlier_avg.max(1.0)
    } else {
        1.0
    };

    let wave = wave_stats(&today_mins, cutoff_min, 30);

    let (avg_cities_per_event, avg_event_gap, events_last_1h) = if !today_recs.is_empty() {
        let avg_c = today_recs.iter().map(|r| r.cities.len()).sum::<usize>() as f64
            / today_recs.len() as f64;
        let mut evt_mins: Vec<i32> = today_recs
            .iter()
            .map(|r| (r.time.hour * 60 + r.time.min) as i32)
            .collect();
        evt_mins.sort();
        let avg_gap = if evt_mins.len() > 1 {
            evt_mins.windows(2).map(|w| (w[1] - w[0]) as f64).sum::<f64>()
                / (evt_mins.len() - 1) as f64
        } else {
            0.0
        };
        let last_1h = evt_mins.iter().filter(|&&m| m >= cutoff_min - 60).count() as f64;
        (avg_c, avg_gap, last_1h)
    } else {
        (0.0, 0.0, 0.0)
    };

    let mut out = HashMap::new();
    out.insert("hour_of_day", now.hour as f64);
    out.insert("today_alarms_vs_prev_day", today_alarms_vs_prev_day);
    out.insert("today_events_so_far", today_events_so_far);
    out.insert("today_cities_so_far", today_cities_so_far as f64);
    out.insert("hours_since_first_alarm", hours_since_first);
    out.insert("minutes_since_last_alarm", minutes_since_last);
    out.insert("rocket_frac", rocket_frac);
    out.insert("campaign_day", campaign_day);
    out.insert("prev_day_total_vs_avg", prev_day_total_vs_avg);
    out.insert("avg_cities_per_event", avg_cities_per_event);
    out.insert("avg_event_gap", avg_event_gap);
    out.insert("events_last_1h", events_last_1h);
    // Wave features
    for (k, v) in &wave {
        out.insert(k, *v);
    }
    // Private fields
    out.insert("_today_count", today_count);
    out.insert("_global_daily_avg", global_daily_avg);
    out
}

/// City-specific features (9 keys).
fn compute_city_features(records: &[AlertRecord], city: &str, now: &IsraelTime) -> HashMap<&'static str, f64> {
    let today = Date::from_israel(now);
    let cutoff_min = (now.hour * 60 + now.min) as i32;
    let ema_alpha = 1.0 - 0.5f64.powf(1.0 / 3.0);

    let mut by_day_all: HashMap<Date, Vec<&AlertRecord>> = HashMap::new();
    let mut city_by_day: HashMap<Date, Vec<&AlertRecord>> = HashMap::new();
    for r in records {
        let d = Date::from_israel(&r.time);
        by_day_all.entry(d).or_default().push(r);
        if r.cities.iter().any(|c| c == city) {
            city_by_day.entry(d).or_default().push(r);
        }
    }
    let mut all_days: Vec<Date> = by_day_all.keys().copied().collect();
    all_days.sort();
    let prior_days: Vec<Date> = all_days.iter().copied().filter(|&d| d < today).collect();

    let today_city: Vec<&AlertRecord> = city_by_day
        .get(&today)
        .map(|v| v.iter().copied().filter(|r| r.time < *now).collect())
        .unwrap_or_default();

    let city_alarms_so_far = today_city.len() as f64;
    let city_mins_today: Vec<i32> = today_city
        .iter()
        .map(|r| (r.time.hour * 60 + r.time.min) as i32)
        .collect();
    let city_minutes_since_last = city_mins_today.iter().max()
        .map(|&m| (cutoff_min - m) as f64)
        .unwrap_or(-1.0);

    let hist_counts: Vec<usize> = prior_days.iter()
        .map(|d| city_by_day.get(d).map(|v| v.len()).unwrap_or(0))
        .collect();
    let city_prev_day_total = hist_counts.last().copied().unwrap_or(0) as f64;
    let city_historical_avg = if !hist_counts.is_empty() {
        hist_counts.iter().sum::<usize>() as f64 / hist_counts.len() as f64
    } else {
        0.0
    };
    let city_hit_rate = if !prior_days.is_empty() {
        hist_counts.iter().filter(|&&c| c > 0).count() as f64 / prior_days.len() as f64
    } else {
        0.0
    };

    // city_rank_pct: preserve insertion order (chronological) to match Python dict iteration.
    let mut city_totals_keys: Vec<String> = Vec::new();
    let mut city_totals: HashMap<String, usize> = HashMap::new();
    for d in &prior_days {
        // by_day_all values are in records-order (chronological); walk sorted days.
        if let Some(recs) = by_day_all.get(d) {
            for r in recs {
                for c in &r.cities {
                    if !city_totals.contains_key(c.as_str()) {
                        city_totals_keys.push(c.clone());
                        city_totals.insert(c.clone(), 0);
                    }
                    *city_totals.get_mut(c).unwrap() += 1;
                }
            }
        }
    }
    let city_rank_pct = if !city_totals.is_empty() {
        // Stable-sort by count asc; equal-count ties preserve insertion order — matches Python.
        let mut sorted_c: Vec<&String> = city_totals_keys.iter().collect();
        sorted_c.sort_by_key(|c| city_totals[*c]);
        let n_r = (sorted_c.len() - 1).max(1);
        sorted_c.iter().enumerate()
            .find(|(_, c)| c.as_str() == city)
            .map(|(i, _)| i as f64 / n_r as f64)
            .unwrap_or(0.0)
    } else {
        0.5
    };

    let city_today_vs_hist = city_alarms_so_far / city_historical_avg.max(0.1);

    let mut ema = 0.0f64;
    for d in &prior_days {
        let ct = city_by_day.get(d).map(|v| v.len()).unwrap_or(0) as f64;
        ema = ema_alpha * ct + (1.0 - ema_alpha) * ema;
    }

    let total_events_today = by_day_all.get(&today)
        .map(|v| v.iter().filter(|r| r.time < *now).count())
        .unwrap_or(0) as f64;
    let city_event_frac = city_alarms_so_far / total_events_today.max(1.0);

    let mut out = HashMap::new();
    out.insert("city_alarms_so_far", city_alarms_so_far);
    out.insert("city_minutes_since_last", city_minutes_since_last);
    out.insert("city_prev_day_total", city_prev_day_total);
    out.insert("city_historical_avg", city_historical_avg);
    out.insert("city_hit_rate", city_hit_rate);
    out.insert("city_rank_pct", city_rank_pct);
    out.insert("city_today_vs_hist", city_today_vs_hist);
    out.insert("city_ema_avg", ema);
    out.insert("city_event_frac", city_event_frac);
    out
}

/// Interaction/rate features (7 keys).
fn compute_interaction_features(
    global: &HashMap<&'static str, f64>,
    city: &HashMap<&'static str, f64>,
    hours_remaining: f64,
) -> HashMap<&'static str, f64> {
    let city_alarms_so_far = global.get("_today_count").copied().unwrap_or(0.0); // note: this is city field
    // Python: city_last_24h = city_alarms_so_far (city feature, not global)
    let city_last_24h = city.get("city_alarms_so_far").copied().unwrap_or(0.0);
    let city_historical_avg = city.get("city_historical_avg").copied().unwrap_or(0.0);
    let today_count = global.get("_today_count").copied().unwrap_or(0.0);
    let global_daily_avg = global.get("_global_daily_avg").copied().unwrap_or(1.0);
    let intensity_ratio = today_count / global_daily_avg.max(1.0);
    let city_rate_pred = city_last_24h / 24.0 * hours_remaining;
    let adjusted_rate_pred = intensity_ratio * city_historical_avg * hours_remaining / 24.0;
    let city_so_far = city.get("city_alarms_so_far").copied().unwrap_or(0.0);

    let mut out = HashMap::new();
    out.insert("city_rate_pred", city_rate_pred);
    out.insert("adjusted_rate_pred", adjusted_rate_pred);
    out.insert("city_last_24h", city_last_24h);
    out.insert("intensity_ratio", intensity_ratio);
    out.insert("alarms_x_hours_rem", city_so_far * hours_remaining);
    out.insert("intensity_x_hours_rem", intensity_ratio * hours_remaining);
    out.insert("hist_avg_x_hours_rem", city_historical_avg * hours_remaining);
    let _ = city_alarms_so_far; // suppress unused warning
    out
}

/// Build a 35-element feature row (bias + 34 features) from the merged feature maps.
fn build_feature_row(
    global: &HashMap<&'static str, f64>,
    city: &HashMap<&'static str, f64>,
    interaction: &HashMap<&'static str, f64>,
) -> Vec<f64> {
    let mut row = vec![1.0f64];
    for &col in FEATURE_COLS {
        let v = city.get(col)
            .or_else(|| global.get(col))
            .or_else(|| interaction.get(col))
            .copied()
            .unwrap_or(0.0);
        row.push(v);
    }
    row
}

/// Finish a ridge prediction: blend rate-sub and rate, compute residual sigma.
fn ridge_finish(
    raw_pred: f64,
    city_so_far: f64,
    hours_remaining: f64,
    x_mat: &[Vec<f64>],
    y: &[f64],
    beta: &[f64],
) -> (f64, f64) {
    let rate_sub = (raw_pred - city_so_far).max(0.0);
    let rate = (raw_pred * hours_remaining / 24.0).max(0.0);
    let w = hours_remaining / 24.0;
    let pred = w * rate_sub + (1.0 - w) * rate;

    let n = x_mat.len();
    let p = beta.len();
    let ss: f64 = x_mat.iter().zip(y.iter()).map(|(xi, &yi)| {
        let p_i: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        (yi - p_i).powi(2)
    }).sum();
    let sigma = (ss / n.saturating_sub(p).max(1) as f64).sqrt() * hours_remaining / 24.0;

    (round1(pred), round1(sigma))
}

/// Predict remaining alarms today for `city` using 34-feature Ridge regression.
///
/// Training: 8 cutoff hours (0,3,6,9,12,15,18,21) × prior days.
/// Rate-sub framing blended with rate by hours_remaining/24.
/// Returns (expected_remaining, std_dev). Returns (0.0, 0.0) if < 2 training days.
pub fn predict_remaining_ridge(
    all_records: &[AlertRecord],
    city: &str,
    now: &IsraelTime,
    alpha: f64,
) -> (f64, f64) {
    const CUTOFF_HOURS: &[u32] = &[0, 3, 6, 9, 12, 15, 18, 21];
    let today = Date::from_israel(now);

    let all_dates: std::collections::HashSet<Date> = all_records.iter()
        .map(|r| Date::from_israel(&r.time))
        .collect();
    let mut train_days: Vec<Date> = all_dates.iter().copied().filter(|&d| d < today).collect();
    train_days.sort();

    if train_days.len() < 2 {
        return (0.0, 0.0);
    }

    let mut x_mat: Vec<Vec<f64>> = Vec::new();
    let mut y_daily: Vec<f64> = Vec::new();

    for &date in &train_days {
        // No future-data leakage: training records are those up to and including this day.
        let train_records: Vec<&AlertRecord> = all_records.iter()
            .filter(|r| Date::from_israel(&r.time) <= date)
            .collect();
        let city_daily_total = train_records.iter()
            .filter(|r| Date::from_israel(&r.time) == date && r.cities.iter().any(|c| c == city))
            .count() as f64;

        for &ch in CUTOFF_HOURS {
            let fake_now = IsraelTime { year: date.year, month: date.month, day: date.day,
                                        hour: ch, min: 0, sec: 0 };
            let hrs = (24 - ch) as f64;
            let train_refs: Vec<AlertRecord> = train_records.iter().map(|r| (*r).clone()).collect();
            let gf = compute_global_features(&train_refs, &fake_now);
            let cf = compute_city_features(&train_refs, city, &fake_now);
            let intf = compute_interaction_features(&gf, &cf, hrs);
            x_mat.push(build_feature_row(&gf, &cf, &intf));
            y_daily.push(city_daily_total);
        }
    }

    if x_mat.len() < 5 {
        return (0.0, 0.0);
    }

    let beta = solve_normal_equation(&x_mat, &y_daily, alpha);
    let hours_remaining = (24.0 - now.hour as f64 - now.min as f64 / 60.0).max(0.0);
    let gf_pred = compute_global_features(all_records, now);
    let cf_pred = compute_city_features(all_records, city, now);
    let intf_pred = compute_interaction_features(&gf_pred, &cf_pred, hours_remaining);
    let x_pred = build_feature_row(&gf_pred, &cf_pred, &intf_pred);
    let raw_pred: f64 = x_pred.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
    let city_so_far = cf_pred.get("city_alarms_so_far").copied().unwrap_or(0.0);
    ridge_finish(raw_pred, city_so_far, hours_remaining, &x_mat, &y_daily, &beta)
}

/// Predict remaining alarms tonight (until 7am) for `city` using 34-feature Ridge.
///
/// Uses 7am→7am day boundary. Training cutoffs: (7,9,12,15,18,20).
/// Returns (expected_remaining, std_dev). Returns (0.0, 0.0) if < 2 training days.
pub fn predict_night_ridge(
    all_records: &[AlertRecord],
    city: &str,
    now: &IsraelTime,
    alpha: f64,
) -> (f64, f64) {
    const CUTOFF_HOURS: &[u32] = &[7, 9, 12, 15, 18, 20];
    let current_day_start = day_start_7am(now);

    let all_7am_starts: std::collections::HashSet<Date> = all_records.iter()
        .map(|r| day_start_7am(&r.time))
        .collect();
    let mut train_days: Vec<Date> = all_7am_starts.iter().copied()
        .filter(|&d| d < current_day_start)
        .collect();
    train_days.sort();

    if train_days.len() < 2 {
        return (0.0, 0.0);
    }

    let mut x_mat: Vec<Vec<f64>> = Vec::new();
    let mut y_daily: Vec<f64> = Vec::new();

    for &date in &train_days {
        // Feature context: records up to and including calendar day `date`
        let train_records: Vec<&AlertRecord> = all_records.iter()
            .filter(|r| Date::from_israel(&r.time) <= date)
            .collect();
        // Target: city's full 7am-day total (may include overnight rows on date+1)
        let city_daily_total = all_records.iter()
            .filter(|r| day_start_7am(&r.time) == date && r.cities.iter().any(|c| c == city))
            .count() as f64;

        for &ch in CUTOFF_HOURS {
            let fake_now = IsraelTime { year: date.year, month: date.month, day: date.day,
                                        hour: ch, min: 0, sec: 0 };
            let hrs = (24 - (ch + 24 - 7) % 24) as f64;
            let train_refs: Vec<AlertRecord> = train_records.iter().map(|r| (*r).clone()).collect();
            let gf = compute_global_features(&train_refs, &fake_now);
            let cf = compute_city_features(&train_refs, city, &fake_now);
            let intf = compute_interaction_features(&gf, &cf, hrs);
            x_mat.push(build_feature_row(&gf, &cf, &intf));
            y_daily.push(city_daily_total);
        }
    }

    if x_mat.len() < 5 {
        return (0.0, 0.0);
    }

    let beta = solve_normal_equation(&x_mat, &y_daily, alpha);
    let hours_remaining = {
        let elapsed = ((now.hour as f64 - 7.0).rem_euclid(24.0)) + now.min as f64 / 60.0;
        (24.0 - elapsed).max(0.0)
    };
    let gf_pred = compute_global_features(all_records, now);
    let cf_pred = compute_city_features(all_records, city, now);
    let intf_pred = compute_interaction_features(&gf_pred, &cf_pred, hours_remaining);
    let x_pred = build_feature_row(&gf_pred, &cf_pred, &intf_pred);
    let raw_pred: f64 = x_pred.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();

    // city_so_far: records in current 7am-day for this city, before now
    let city_so_far = all_records.iter()
        .filter(|r| day_start_7am(&r.time) == current_day_start
            && r.cities.iter().any(|c| c == city)
            && r.time < *now)
        .count() as f64;

    ridge_finish(raw_pred, city_so_far, hours_remaining, &x_mat, &y_daily, &beta)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn it(year: i32, month: u32, day: u32, hour: u32, min: u32) -> IsraelTime {
        IsraelTime { year, month, day, hour, min, sec: 0 }
    }

    // Build `n` evenly-spaced alarms across all 24 hours of a day.
    fn day_alarms(year: i32, month: u32, day: u32, n: usize) -> Vec<IsraelTime> {
        (0..n)
            .map(|i| it(year, month, day, (i * 24 / n.max(1)) as u32, 0))
            .collect()
    }

    // ── solve_normal_equation ─────────────────────────────────────────────────

    #[test]
    fn test_snq_identity() {
        // X = I₂, y = [3, 4], α = 0 → β = [3, 4]
        let x = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let y = vec![3.0, 4.0];
        let beta = solve_normal_equation(&x, &y, 0.0);
        assert!((beta[0] - 3.0).abs() < 1e-10, "beta[0]={}", beta[0]);
        assert!((beta[1] - 4.0).abs() < 1e-10, "beta[1]={}", beta[1]);
    }

    #[test]
    fn test_snq_ridge_shrinks() {
        // Large α forces coefficients toward zero.
        let x = vec![vec![1.0, 2.0], vec![1.0, 3.0], vec![1.0, 4.0]];
        let y = vec![5.0, 6.0, 7.0];
        let beta_ols = solve_normal_equation(&x, &y, 0.0);
        let beta_ridge = solve_normal_equation(&x, &y, 1000.0);
        assert!(
            beta_ridge[1].abs() < beta_ols[1].abs(),
            "ridge should shrink: {} vs {}",
            beta_ridge[1],
            beta_ols[1]
        );
    }

    #[test]
    fn test_snq_singular_returns_zero() {
        // Linearly dependent columns → singular → zero vector.
        let x = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let y = vec![1.0, 2.0];
        let beta = solve_normal_equation(&x, &y, 0.0);
        assert_eq!(beta, vec![0.0, 0.0]);
    }

    #[test]
    fn test_snq_empty_x() {
        let beta = solve_normal_equation(&[], &[], 0.0);
        assert!(beta.is_empty());
    }

    // ── Date arithmetic ───────────────────────────────────────────────────────

    #[test]
    fn test_date_add_days_forward() {
        let d = Date { year: 2026, month: 3, day: 29 };
        assert_eq!(d.add_days(3), Date { year: 2026, month: 4, day: 1 });
    }

    #[test]
    fn test_date_add_days_backward() {
        let d = Date { year: 2026, month: 3, day: 1 };
        assert_eq!(d.add_days(-1), Date { year: 2026, month: 2, day: 28 });
    }

    #[test]
    fn test_day_start_7am_morning() {
        // 3am on Mar 5 → 7am-day is Mar 4
        let t = it(2026, 3, 5, 3, 0);
        assert_eq!(day_start_7am(&t), Date { year: 2026, month: 3, day: 4 });
    }

    #[test]
    fn test_day_start_7am_afternoon() {
        // 14:00 on Mar 5 → 7am-day is Mar 5
        let t = it(2026, 3, 5, 14, 0);
        assert_eq!(day_start_7am(&t), Date { year: 2026, month: 3, day: 5 });
    }

    // ── predict_remaining: degenerate inputs ──────────────────────────────────

    #[test]
    fn test_pr_empty_times() {
        let now = it(2026, 3, 3, 12, 0);
        assert_eq!(predict_remaining(&[], &now, 7, "simple"), (0.0, 0.0));
    }

    #[test]
    fn test_pr_only_today() {
        let now = it(2026, 3, 3, 12, 0);
        let times = vec![it(2026, 3, 3, 10, 0), it(2026, 3, 3, 11, 0)];
        assert_eq!(predict_remaining(&times, &now, 7, "simple"), (0.0, 0.0));
    }

    #[test]
    fn test_pr_one_train_day() {
        let now = it(2026, 3, 3, 12, 0);
        let mut times = day_alarms(2026, 3, 2, 10);
        times.push(it(2026, 3, 3, 10, 0));
        assert_eq!(predict_remaining(&times, &now, 7, "simple"), (0.0, 0.0));
    }

    #[test]
    fn test_pr_now_before_any_alarm() {
        // now is earlier than all training data
        let now = it(2026, 3, 1, 0, 0);
        let times: Vec<IsraelTime> = [
            day_alarms(2025, 12, 1, 5),
            day_alarms(2025, 12, 2, 5),
        ]
        .concat();
        // Both train days are before today (2026-03-01); train_days.len() >= 2 → may return result
        let (pred, sigma) = predict_remaining(&times, &now, 7, "simple");
        assert!(pred >= 0.0, "pred={}", pred);
        assert!(sigma >= 0.0, "sigma={}", sigma);
    }

    // ── predict_remaining: basic behavior ─────────────────────────────────────

    #[test]
    fn test_pr_simple_nonneg() {
        let mut times: Vec<IsraelTime> = day_alarms(2026, 3, 1, 12);
        times.extend(day_alarms(2026, 3, 2, 12));
        let now = it(2026, 3, 3, 12, 0);
        let (pred, sigma) = predict_remaining(&times, &now, 7, "simple");
        assert!(pred >= 0.0, "pred={}", pred);
        assert!(sigma >= 0.0, "sigma={}", sigma);
    }

    #[test]
    fn test_pr_advanced_nonneg() {
        let mut times: Vec<IsraelTime> = day_alarms(2026, 3, 1, 12);
        times.extend(day_alarms(2026, 3, 2, 12));
        let now = it(2026, 3, 3, 12, 0);
        let (pred, sigma) = predict_remaining(&times, &now, 7, "advanced");
        assert!(pred >= 0.0, "pred={}", pred);
        assert!(sigma >= 0.0, "sigma={}", sigma);
    }

    #[test]
    fn test_pr_end_of_day_capped() {
        // At hour 23, simple method caps at hours_left * hourly_rate * 2 = 1 * rate * 2.
        // With 12 alarms/day training: hourly_rate = 12/24 = 0.5, cap = 1 * 0.5 * 2 = 1.0.
        let mut times: Vec<IsraelTime> = day_alarms(2026, 3, 1, 12);
        times.extend(day_alarms(2026, 3, 2, 12));
        let now = it(2026, 3, 3, 23, 0);
        let (pred, _) = predict_remaining(&times, &now, 7, "simple");
        // cap = 1 * (12/24) * 2 = 1.0; result must be ≤ 1.0 + rounding tolerance
        assert!(pred <= 1.1, "expected ≤ 1.0, got {}", pred);
    }

    #[test]
    fn test_pr_simple_vs_advanced_differ() {
        // The two methods use different regression targets so can produce different values.
        let mut times: Vec<IsraelTime> = day_alarms(2026, 3, 1, 8);
        times.extend(day_alarms(2026, 3, 2, 16));
        let now = it(2026, 3, 3, 6, 0);
        let (ps, _) = predict_remaining(&times, &now, 7, "simple");
        let (pa, _) = predict_remaining(&times, &now, 7, "advanced");
        // Not asserting equal — they use different model targets
        assert!(ps >= 0.0 && pa >= 0.0);
    }

    // ── predict_night_rolling: degenerate inputs ──────────────────────────────

    #[test]
    fn test_pnr_empty() {
        let now = it(2026, 3, 3, 22, 0);
        assert_eq!(predict_night_rolling(&[], &now, 7), (0.0, 0.0));
    }

    #[test]
    fn test_pnr_only_current_day() {
        // All times in the current 7am-day → no past days
        let now = it(2026, 3, 3, 22, 0); // current 7am-day = 2026-03-03
        let times = vec![it(2026, 3, 3, 22, 0), it(2026, 3, 3, 23, 0)];
        assert_eq!(predict_night_rolling(&times, &now, 7), (0.0, 0.0));
    }

    // ── predict_night_rolling: uniform nights ─────────────────────────────────

    #[test]
    fn test_pnr_uniform_at_9pm() {
        // 7 past 7am-days, each with exactly 10 night alarms at hour 22.
        // now = 7am-day 8, at 21:00 → hours_remaining = 24 - 14 = 10, scale = 1.0.
        // Expected: avg=10, std=0 → (10.0, 0.0)
        let mut times = Vec::new();
        for day in 1u32..=7 {
            for _ in 0..10 {
                times.push(it(2026, 3, day, 22, 0));
            }
        }
        let now = it(2026, 3, 8, 21, 0);
        let (pred, sigma) = predict_night_rolling(&times, &now, 7);
        assert!((pred - 10.0).abs() < 0.05, "pred={}", pred);
        assert!((sigma - 0.0).abs() < 0.05, "sigma={}", sigma);
    }

    #[test]
    fn test_pnr_scale_at_10pm() {
        // Same setup, now at 22:00 → hours_remaining = 24 - 15 = 9, scale = 0.9.
        // Expected pred ≈ 9.0
        let mut times = Vec::new();
        for day in 1u32..=7 {
            for _ in 0..10 {
                times.push(it(2026, 3, day, 22, 0));
            }
        }
        let now = it(2026, 3, 8, 22, 0);
        let (pred, _) = predict_night_rolling(&times, &now, 7);
        assert!((pred - 9.0).abs() < 0.05, "pred={}", pred);
    }

    #[test]
    fn test_pnr_at_midnight() {
        // now at 00:00 → hours_remaining = 24 - (0-7)%24 = 24 - 17 = 7, scale = 7/10 = 0.7.
        let mut times = Vec::new();
        for day in 1u32..=7 {
            for _ in 0..10 {
                times.push(it(2026, 3, day, 22, 0));
            }
        }
        // 7am-day for Mar 8 00:00 is Mar 7 (hour < 7 → day-1)
        // So current_day_start = Mar 7, past days = [Mar 1 .. Mar 6]
        let now = it(2026, 3, 8, 0, 0);
        let (pred, _) = predict_night_rolling(&times, &now, 7);
        // 6 past days (Mar 1..6), each 10 alarms; scale = 7/10 = 0.7; pred = 7.0
        assert!((pred - 7.0).abs() < 0.05, "pred={}", pred);
    }

    #[test]
    fn test_pnr_fewer_than_recent_days() {
        // Only 3 past 7am-days with 6 night alarms each; recent_days=7 → uses all 3.
        let mut times = Vec::new();
        for day in 1u32..=3 {
            for _ in 0..6 {
                times.push(it(2026, 3, day, 22, 0));
            }
        }
        let now = it(2026, 3, 4, 21, 0);
        let (pred, sigma) = predict_night_rolling(&times, &now, 7);
        assert!((pred - 6.0).abs() < 0.05, "pred={}", pred);
        assert!((sigma - 0.0).abs() < 0.05, "sigma={}", sigma);
    }

    #[test]
    fn test_pnr_more_than_recent_days() {
        // 10 past days, but recent_days=3 → only last 3 count.
        // Days 1-7: 4 alarms/night. Days 8-10: 10 alarms/night. recent = days 8,9,10.
        let mut times = Vec::new();
        for day in 1u32..=7 {
            for _ in 0..4 {
                times.push(it(2026, 3, day, 22, 0));
            }
        }
        for day in 8u32..=10 {
            for _ in 0..10 {
                times.push(it(2026, 3, day, 22, 0));
            }
        }
        let now = it(2026, 3, 11, 21, 0);
        let (pred, sigma) = predict_night_rolling(&times, &now, 3);
        // avg of [10,10,10] = 10, std = 0, scale = 1.0
        assert!((pred - 10.0).abs() < 0.05, "pred={}", pred);
        assert!((sigma - 0.0).abs() < 0.05, "sigma={}", sigma);
    }

    #[test]
    fn test_pnr_std_nonzero() {
        // Variable night counts → nonzero std.
        let times = vec![
            it(2026, 3, 1, 22, 0), // 1 alarm
            it(2026, 3, 2, 22, 0), it(2026, 3, 2, 23, 0), // 2 alarms
            it(2026, 3, 3, 22, 0), it(2026, 3, 3, 22, 30), it(2026, 3, 3, 23, 0), // 3 alarms
        ];
        let now = it(2026, 3, 4, 21, 0);
        let (pred, sigma) = predict_night_rolling(&times, &now, 7);
        // avg = 2.0, var = ((1-2)²+(2-2)²+(3-2)²)/2 = 1.0, std = 1.0, scale = 1.0
        assert!((pred - 2.0).abs() < 0.05, "pred={}", pred);
        assert!((sigma - 1.0).abs() < 0.05, "sigma={}", sigma);
    }

    // ── Ridge: degenerate inputs ─────────────────────────────────────────────

    fn make_alert(time_str: &str, cities: Vec<&str>, event_id: &str, is_rocket: bool) -> AlertRecord {
        AlertRecord {
            time: IsraelTime::parse(time_str).unwrap(),
            cities: cities.into_iter().map(str::to_string).collect(),
            event_id: event_id.to_string(),
            is_rocket,
        }
    }

    fn multi_day_alerts(days: &[(&str, usize)]) -> Vec<AlertRecord> {
        let mut out = Vec::new();
        for (date, n) in days {
            for i in 0..*n {
                let h = (i * 24 / n.max(&1)) as u32;
                let ts = format!("{} {:02}:00:00", date, h);
                out.push(make_alert(&ts, vec!["testcity"], &format!("{}-{}", date, i), true));
            }
        }
        out
    }

    #[test]
    fn test_ridge_empty_records() {
        let now = it(2026, 3, 15, 14, 0);
        assert_eq!(predict_remaining_ridge(&[], "testcity", &now, 10.0), (0.0, 0.0));
    }

    #[test]
    fn test_ridge_only_today() {
        let now = it(2026, 3, 15, 14, 0);
        let recs = vec![make_alert("2026-03-15 10:00:00", vec!["testcity"], "e1", true)];
        assert_eq!(predict_remaining_ridge(&recs, "testcity", &now, 10.0), (0.0, 0.0));
    }

    #[test]
    fn test_ridge_one_train_day() {
        let now = it(2026, 3, 15, 14, 0);
        let recs = multi_day_alerts(&[("2026-03-14", 5)]);
        assert_eq!(predict_remaining_ridge(&recs, "testcity", &now, 10.0), (0.0, 0.0));
    }

    #[test]
    fn test_ridge_nonneg_with_data() {
        let now = it(2026, 3, 15, 14, 0);
        let mut recs = multi_day_alerts(&[
            ("2026-03-13", 6), ("2026-03-14", 8),
        ]);
        recs.push(make_alert("2026-03-15 10:00:00", vec!["testcity"], "today-e1", true));
        let (pred, sigma) = predict_remaining_ridge(&recs, "testcity", &now, 10.0);
        assert!(pred >= 0.0, "pred={}", pred);
        assert!(sigma >= 0.0, "sigma={}", sigma);
    }

    #[test]
    fn test_night_ridge_nonneg_with_data() {
        let now = it(2026, 3, 15, 21, 0); // 9pm → night mode
        let recs = multi_day_alerts(&[
            ("2026-03-13", 6), ("2026-03-14", 8),
        ]);
        let (pred, sigma) = predict_night_ridge(&recs, "testcity", &now, 10.0);
        assert!(pred >= 0.0, "pred={}", pred);
        assert!(sigma >= 0.0, "sigma={}", sigma);
    }

    #[test]
    fn test_night_ridge_empty() {
        let now = it(2026, 3, 15, 21, 0);
        assert_eq!(predict_night_ridge(&[], "testcity", &now, 10.0), (0.0, 0.0));
    }

    // ── Ridge: fixture parity ─────────────────────────────────────────────────

    fn load_fixture() -> serde_json::Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ridge_intermediates.json");
        let s = std::fs::read_to_string(path).expect("fixture file missing");
        serde_json::from_str(&s).expect("fixture parse error")
    }

    fn records_from_fixture(fix: &serde_json::Value) -> Vec<AlertRecord> {
        fix["records"].as_array().unwrap().iter().map(|r| {
            let cities: Vec<String> = r["cities"].as_array().unwrap()
                .iter().map(|c| c.as_str().unwrap().to_string()).collect();
            AlertRecord {
                time: IsraelTime::parse(r["time"].as_str().unwrap()).unwrap(),
                cities,
                event_id: r["event_id"].as_str().unwrap().to_string(),
                is_rocket: r["is_rocket"].as_bool().unwrap_or(false),
            }
        }).collect()
    }

    #[test]
    fn test_ridge_day_parity() {
        let fix = load_fixture();
        let records = records_from_fixture(&fix);
        let city = fix["city"].as_str().unwrap();
        let now = IsraelTime::parse(fix["now"].as_str().unwrap()).unwrap();
        let alpha = fix["alpha"].as_f64().unwrap();

        let expected = fix["day"]["expected"].as_f64().unwrap();
        let expected_sigma = fix["day"]["sigma"].as_f64().unwrap();

        let (pred, sigma) = predict_remaining_ridge(&records, city, &now, alpha);
        assert!(
            (pred - expected).abs() < 1e-4,
            "day pred mismatch: rust={} python={}", pred, expected
        );
        assert!(
            (sigma - expected_sigma).abs() < 1e-4,
            "day sigma mismatch: rust={} python={}", sigma, expected_sigma
        );
    }

    #[test]
    fn test_ridge_night_parity() {
        let fix = load_fixture();
        let records = records_from_fixture(&fix);
        let city = fix["city"].as_str().unwrap();
        let now = IsraelTime::parse(fix["now"].as_str().unwrap()).unwrap();
        let alpha = fix["alpha"].as_f64().unwrap();

        let expected = fix["night"]["expected"].as_f64().unwrap();
        let expected_sigma = fix["night"]["sigma"].as_f64().unwrap();

        let (pred, sigma) = predict_night_ridge(&records, city, &now, alpha);
        assert!(
            (pred - expected).abs() < 1e-4,
            "night pred mismatch: rust={} python={}", pred, expected
        );
        assert!(
            (sigma - expected_sigma).abs() < 1e-4,
            "night sigma mismatch: rust={} python={}", sigma, expected_sigma
        );
    }

    #[test]
    fn test_ridge_beta_parity() {
        // Pin the coefficient vector against the Python-generated fixture.
        let fix = load_fixture();
        let records = records_from_fixture(&fix);
        let city = fix["city"].as_str().unwrap();
        let now = IsraelTime::parse(fix["now"].as_str().unwrap()).unwrap();
        let alpha = fix["alpha"].as_f64().unwrap();

        let expected_beta: Vec<f64> = fix["day"]["beta"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        let expected_x_pred: Vec<f64> = fix["day"]["x_pred"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();

        // Recompute beta and x_pred manually to pin intermediates
        let today = Date::from_israel(&now);
        let all_dates: std::collections::HashSet<Date> = records.iter()
            .map(|r| Date::from_israel(&r.time)).collect();
        let mut train_days: Vec<Date> = all_dates.iter().copied().filter(|&d| d < today).collect();
        train_days.sort();

        let mut x_mat: Vec<Vec<f64>> = Vec::new();
        let mut y_daily: Vec<f64> = Vec::new();
        for &date in &train_days {
            let train_records: Vec<AlertRecord> = records.iter()
                .filter(|r| Date::from_israel(&r.time) <= date)
                .cloned().collect();
            let city_daily_total = train_records.iter()
                .filter(|r| Date::from_israel(&r.time) == date && r.cities.iter().any(|c| c == city))
                .count() as f64;
            for &ch in &[0u32, 3, 6, 9, 12, 15, 18, 21] {
                let fake_now = IsraelTime { year: date.year, month: date.month, day: date.day,
                                            hour: ch, min: 0, sec: 0 };
                let hrs = (24 - ch) as f64;
                let gf = compute_global_features(&train_records, &fake_now);
                let cf = compute_city_features(&train_records, city, &fake_now);
                let intf = compute_interaction_features(&gf, &cf, hrs);
                x_mat.push(build_feature_row(&gf, &cf, &intf));
                y_daily.push(city_daily_total);
            }
        }
        let beta = solve_normal_equation(&x_mat, &y_daily, alpha);

        let hrs_rem = (24.0 - now.hour as f64 - now.min as f64 / 60.0).max(0.0);
        let gf = compute_global_features(&records, &now);
        let cf = compute_city_features(&records, city, &now);
        let intf = compute_interaction_features(&gf, &cf, hrs_rem);
        let x_pred = build_feature_row(&gf, &cf, &intf);

        assert_eq!(beta.len(), expected_beta.len(), "beta length mismatch");
        for (i, (&r, &p)) in beta.iter().zip(expected_beta.iter()).enumerate() {
            assert!((r - p).abs() < 1e-4, "beta[{}] rust={} python={}", i, r, p);
        }
        assert_eq!(x_pred.len(), expected_x_pred.len(), "x_pred length mismatch");
        for (i, (&r, &p)) in x_pred.iter().zip(expected_x_pred.iter()).enumerate() {
            assert!((r - p).abs() < 1e-9, "x_pred[{}] rust={} python={}", i, r, p);
        }
    }
}
