/// svg.rs — SVG chart rendering, ported from alarms_core.py render_chart().
///
/// Pure string building via std::fmt::Write. No external dependencies.
/// Numeric format strings mirror Python's exactly so SVG output is byte-identical.

use std::collections::HashMap;
use std::fmt::Write;
use crate::data_loading::{IsraelTime, AlertRecord};
use crate::forecast::{predict_remaining, predict_night_rolling, predict_remaining_ridge,
                      predict_night_ridge, Date};
use crate::israel_time::ymd_to_epoch;

// ── Constants ────────────────────────────────────────────────────────────────

const BG_COLOR: &str = "#f0ede3";
const NIGHT_DOT_COLOR: &str = "#333333";
const DAY_DOT_COLOR: &str = "#888888";

pub const DEFAULT_AREA_FILTER: &str = "תל אביב - מרכז העיר";
pub const DEFAULT_BIN_HOURS: u32 = 1;
pub const DEFAULT_START: &str = "2026-02-28";

// ── Date formatting helpers ──────────────────────────────────────────────────

/// Weekday abbreviation (Mon=0..Sun=6) for a given calendar date.
fn weekday_abbr(year: i32, month: u32, day: u32) -> &'static str {
    let epoch = ymd_to_epoch(year, month, day);
    let days = epoch / 86400;
    let wd = (days + 3).rem_euclid(7) as usize;
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wd]
}

fn month_abbr(m: u32) -> &'static str {
    const MONTHS: [&str; 12] = ["Jan","Feb","Mar","Apr","May","Jun",
                                  "Jul","Aug","Sep","Oct","Nov","Dec"];
    MONTHS[(m as usize).saturating_sub(1).min(11)]
}

/// Format like Python strftime("%a %-d %b"): "Mon 3 Apr" (no leading zero on day).
fn fmt_date_label(d: &Date) -> String {
    format!("{} {} {}", weekday_abbr(d.year, d.month, d.day), d.day, month_abbr(d.month))
}

/// Format like Python strftime("%b %d"): "Apr 03" (zero-padded day).
fn fmt_month_day(month: u32, day: u32) -> String {
    format!("{} {:02}", month_abbr(month), day)
}

// ── SVG wedge ────────────────────────────────────────────────────────────────

/// SVG arc path for a pie wedge, fractions 0–1 clockwise from top.
/// Matches Python's _svg_wedge formatting exactly.
pub fn svg_wedge(cx: f64, cy: f64, r: f64, start_frac: f64, end_frac: f64, color: &str) -> String {
    use std::f64::consts::PI;
    let t0 = -PI / 2.0 + start_frac * 2.0 * PI;
    let t1 = -PI / 2.0 + end_frac * 2.0 * PI;
    let x0 = cx + r * t0.cos();
    let y0 = cy + r * t0.sin();
    let x1 = cx + r * t1.cos();
    let y1 = cy + r * t1.sin();
    let large = if (end_frac - start_frac) > 0.5 { 1 } else { 0 };
    format!(
        "<path d=\"M {cx:.1},{cy:.1} L {x0:.2},{y0:.2} \
         A {r:.1},{r:.1} 0 {large},1 {x1:.2},{y1:.2} Z\" fill=\"{color}\"/>"
    )
}

// ── Prediction ───────────────────────────────────────────────────────────────

pub struct PredResult {
    pub remaining: f64,
    pub sigma: f64,
    pub label: String,
}

/// Compute remaining-alert prediction. Returns None if forecast == "off".
/// Mirrors compute_prediction() in alarms_core.py.
pub fn compute_prediction(
    times: &[IsraelTime],
    all_records: Option<&[AlertRecord]>,
    city_filter: Option<&str>,
    forecast: &str,
    now: &IsraelTime,
) -> Option<PredResult> {
    if forecast == "off" {
        return None;
    }

    let night_mode = forecast == "ridge" && (now.hour >= 20 || now.hour < 6);
    let label = if night_mode {
        "tonight (until 7am)".to_string()
    } else {
        "rest of today (est.)".to_string()
    };

    let (rem, sig) = if forecast == "ridge" {
        match (city_filter, all_records) {
            (Some(city), Some(records)) => {
                if night_mode && now.hour < 6 {
                    predict_night_ridge(records, city, now, 10.0)
                } else if night_mode {
                    predict_night_rolling(times, now, 7)
                } else {
                    predict_remaining_ridge(records, city, now, 10.0)
                }
            }
            _ => predict_remaining(times, now, 7, "advanced"),
        }
    } else {
        predict_remaining(times, now, 7, forecast)
    };

    Some(PredResult { remaining: rem, sigma: sig, label })
}

// ── Render chart ─────────────────────────────────────────────────────────────

pub struct RenderParams<'a> {
    pub times: &'a [IsraelTime],
    pub area_label: &'a str,
    pub bin_hours: u32,
    pub start_date: &'a str,       // "YYYY-MM-DD"
    pub style: &'a str,             // "lines" | "dots"
    pub threat_label: &'a str,      // "Rocket", "UAV", etc.
    pub forecast: &'a str,          // "off" | "simple" | "advanced" | "ridge"
    pub all_records: Option<&'a [AlertRecord]>,
    pub city_filter: Option<&'a str>,
    pub now: &'a IsraelTime,
}

/// Generate SVG chart string. Pure string building, no I/O.
/// Returns Err if times is empty or start_date is malformed.
pub fn render_chart(p: &RenderParams) -> Result<String, String> {
    if p.times.is_empty() {
        return Err("No alerts found — cannot render chart.".to_string());
    }

    // ── Date range ───────────────────────────────────────────────────────────
    let start = parse_date(p.start_date).ok_or("invalid start_date")?;
    let last_t = p.times.last().unwrap();
    let last_date = Date { year: last_t.year, month: last_t.month, day: last_t.day };
    let today = Date { year: p.now.year, month: p.now.month, day: p.now.day };
    let end = last_date.max(today);

    let mut days: Vec<Date> = Vec::new();
    let mut d = start;
    while d <= end {
        days.push(d);
        d = d.add_days(1);
    }
    let n_days = days.len();

    // ── Bin alerts ───────────────────────────────────────────────────────────
    let mut bins: HashMap<(Date, u32), u32> = HashMap::new();
    for t in p.times {
        let date = Date { year: t.year, month: t.month, day: t.day };
        let h_bin = (t.hour / p.bin_hours) * p.bin_hours;
        *bins.entry((date, h_bin)).or_insert(0) += 1;
    }

    let mut daily_totals: HashMap<Date, u32> = HashMap::new();
    let mut daily_night: HashMap<Date, u32> = HashMap::new();
    for &day in &days {
        let tot: u32 = (0..24u32).step_by(p.bin_hours as usize)
            .map(|h| bins.get(&(day, h)).copied().unwrap_or(0))
            .sum();
        let night: u32 = (0..24u32).step_by(p.bin_hours as usize)
            .filter(|&h| h < 7 || h >= 21)
            .map(|h| bins.get(&(day, h)).copied().unwrap_or(0))
            .sum();
        daily_totals.insert(day, tot);
        daily_night.insert(day, night);
    }

    let mut times_by_day: HashMap<Date, Vec<&IsraelTime>> = HashMap::new();
    for t in p.times {
        let date = Date { year: t.year, month: t.month, day: t.day };
        times_by_day.entry(date).or_default().push(t);
    }

    let cutoff_date = today;
    let cutoff_hour = p.now.hour as f64 + p.now.min as f64 / 60.0;

    // ── Prediction ───────────────────────────────────────────────────────────
    let night_mode = p.forecast == "ridge" && (p.now.hour >= 20 || p.now.hour < 6);
    let pred_result = compute_prediction(p.times, p.all_records, p.city_filter, p.forecast, p.now);

    let today_so_far = daily_totals.get(&cutoff_date).copied().unwrap_or(0);
    let has_prediction = pred_result.as_ref().map_or(false, |pr| {
        let pred_total = today_so_far as f64 + pr.remaining;
        pred_total.round() as i64 > today_so_far as i64
    });

    // ── Layout ───────────────────────────────────────────────────────────────
    let row_h: f64 = 20.0;
    let left_margin: f64 = 68.0;
    let top_margin: f64 = 58.0;
    let hour_w: f64 = 22.0;
    let chart_w = 24.0 * hour_w;
    let dot_col_x = left_margin + chart_w + 30.0;
    let svg_w = left_margin + chart_w + 56.0;
    let bottom_margin = 30.0 + if has_prediction { 20.0 } else { 0.0 };
    let chart_h = n_days as f64 * row_h;
    let svg_h = top_margin + chart_h + bottom_margin;
    let night_bg = "#e2dfd5";
    let grey = "#888888";
    let tick_h = row_h * 0.12;

    let xpx = |h: f64| left_margin + h * hour_w;
    let ypx = |i: usize| top_margin + i as f64 * row_h + row_h / 2.0;

    let max_daily = daily_totals.values().copied().max().unwrap_or(1).max(1) as f64;
    let dot_r = |n: u32| -> f64 { (n as f64 / max_daily).sqrt() * row_h * 0.504 };

    // ── Build SVG ─────────────────────────────────────────────────────────────
    let mut o = String::new();

    write!(o,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{svg_w}\" height=\"{svg_h}\" \
         style=\"background:{BG_COLOR}\">"
    ).unwrap();
    write!(o,
        "<style>\
         @import url(\"https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,700;1,400&amp;display=swap\");\
         text{{font-family:ETBembo,\"EB Garamond\",Georgia,Palatino,serif}}\
         </style>"
    ).unwrap();

    // Embed prediction for JS (pipe-separated: remaining|sigma|label)
    if let Some(ref pr) = pred_result {
        let rem_r = (pr.remaining * 10.0).round() / 10.0;
        let sig_r = (pr.sigma * 10.0).round() / 10.0;
        write!(o, "<desc id=\"pred-data\">{rem_r}|{sig_r}|{}</desc>", pr.label).unwrap();
    }

    // Night shading bands (00:00–07:00 and 21:00–24:00)
    write!(o,
        "<rect x=\"{:.0}\" y=\"{top_margin}\" width=\"{}\" height=\"{chart_h}\" fill=\"{night_bg}\"/>",
        xpx(0.0), 7.0 * hour_w
    ).unwrap();
    write!(o,
        "<rect x=\"{:.0}\" y=\"{top_margin}\" width=\"{}\" height=\"{chart_h}\" fill=\"{night_bg}\"/>",
        xpx(21.0), 3.0 * hour_w
    ).unwrap();

    // Per-row: baseline, date label, alert marks
    for (i, &day) in days.iter().enumerate() {
        let yc = ypx(i);
        let line_end = if day == cutoff_date { cutoff_hour } else { 24.0 };
        write!(o,
            "<line x1=\"{:.0}\" y1=\"{yc:.1}\" x2=\"{:.1}\" y2=\"{yc:.1}\" \
             stroke=\"#cccccc\" stroke-width=\"0.4\"/>",
            xpx(0.0), xpx(line_end)
        ).unwrap();
        write!(o,
            "<text x=\"{}\" y=\"{yc:.1}\" text-anchor=\"end\" \
             dominant-baseline=\"middle\" font-size=\"9\" fill=\"#555555\">{}</text>",
            left_margin - 6.0, fmt_date_label(&day)
        ).unwrap();

        if p.style == "lines" {
            for t in times_by_day.get(&day).map(|v| v.as_slice()).unwrap_or(&[]) {
                let xh = t.hour as f64 + t.min as f64 / 60.0 + t.sec as f64 / 3600.0;
                let col = if t.hour < 7 || t.hour >= 21 { NIGHT_DOT_COLOR } else { DAY_DOT_COLOR };
                let xp = xpx(xh);
                write!(o,
                    "<line x1=\"{xp:.2}\" y1=\"{:.1}\" x2=\"{xp:.2}\" y2=\"{:.1}\" \
                     stroke=\"{col}\" stroke-width=\"0.8\" stroke-linecap=\"butt\"/>",
                    yc - tick_h, yc + tick_h
                ).unwrap();
            }
        } else {
            for h in (0..24u32).step_by(p.bin_hours as usize) {
                let count = bins.get(&(day, h)).copied().unwrap_or(0);
                if count > 0 {
                    let col = if h < 7 || h >= 21 { NIGHT_DOT_COLOR } else { DAY_DOT_COLOR };
                    let xc = xpx(h as f64 + p.bin_hours as f64 / 2.0);
                    write!(o,
                        "<circle cx=\"{xc:.1}\" cy=\"{yc:.1}\" r=\"{:.1}\" fill=\"{col}\"/>",
                        dot_r(count)
                    ).unwrap();
                }
            }
        }
    }

    // Bottom axis line + Tufte x-axis tick marks and labels
    let axis_offset: f64 = 6.0;
    let tick_w: f64 = 1.0;
    let axis_y = top_margin + chart_h + axis_offset;
    write!(o,
        "<line x1=\"{:.1}\" y1=\"{axis_y}\" x2=\"{:.1}\" y2=\"{axis_y}\" \
         stroke=\"#444444\" stroke-width=\"1.5\"/>",
        xpx(0.0) - tick_w / 2.0, xpx(24.0) + tick_w / 2.0
    ).unwrap();
    for h in (0..=24u32).step_by(3) {
        let xp = xpx(h as f64);
        write!(o,
            "<line x1=\"{xp:.0}\" y1=\"{axis_y}\" x2=\"{xp:.0}\" y2=\"{:.0}\" \
             stroke=\"#444444\" stroke-width=\"{tick_w}\"/>",
            axis_y + 6.0
        ).unwrap();
        write!(o,
            "<text x=\"{xp:.0}\" y=\"{:.0}\" text-anchor=\"middle\" \
             font-size=\"8\" font-weight=\"bold\" fill=\"#444444\">{h:02}:00</text>",
            axis_y + 16.0
        ).unwrap();
    }

    // Total-count column header
    write!(o,
        "<text x=\"{dot_col_x:.0}\" y=\"{:.0}\" text-anchor=\"middle\" \
         font-size=\"7\" fill=\"{grey}\">total</text>",
        top_margin - 6.0
    ).unwrap();

    // Per-row summary dot (day/night pie)
    let is_today_row = has_prediction && days.contains(&cutoff_date);
    for (i, &day) in days.iter().enumerate() {
        let tot = daily_totals[&day];
        if tot == 0 { continue; }
        let yc = ypx(i);
        let cx = dot_col_x;
        let night_frac = daily_night[&day] as f64 / tot as f64;
        let r = dot_r(tot);
        if night_frac <= 0.0 {
            write!(o, "<circle cx=\"{cx:.0}\" cy=\"{yc:.1}\" r=\"{r:.1}\" fill=\"{DAY_DOT_COLOR}\"/>").unwrap();
        } else if night_frac >= 1.0 {
            write!(o, "<circle cx=\"{cx:.0}\" cy=\"{yc:.1}\" r=\"{r:.1}\" fill=\"{NIGHT_DOT_COLOR}\"/>").unwrap();
        } else {
            o.push_str(&svg_wedge(cx, yc, r, 0.0, night_frac, NIGHT_DOT_COLOR));
            o.push_str(&svg_wedge(cx, yc, r, night_frac, 1.0, DAY_DOT_COLOR));
        }
        write!(o,
            "<text x=\"{cx:.0}\" y=\"{yc:.1}\" text-anchor=\"middle\" \
             dominant-baseline=\"middle\" font-size=\"6\" fill=\"white\">{tot}</text>"
        ).unwrap();
    }

    // Prediction: +N label + economist curved annotation
    if is_today_row {
        let i_today = days.iter().position(|&d| d == cutoff_date).unwrap();
        let yc_today = ypx(i_today);
        let pr = pred_result.as_ref().unwrap();
        let pred_remaining_int = pr.remaining.round().max(0.0) as i64;
        let pred_label = &pr.label;

        if night_mode && p.now.hour < 6 {
            // Midnight–6am: annotation before 7am (right side of night zone)
            let nl_x = xpx(6.7);
            write!(o,
                "<text x=\"{nl_x:.1}\" y=\"{yc_today:.1}\" text-anchor=\"end\" \
                 dominant-baseline=\"middle\" font-size=\"7\" fill=\"#555555\">\
                 +{pred_remaining_int}</text>"
            ).unwrap();
            let ax0 = nl_x;
            let ay0 = yc_today + 4.0;
            let ax1 = 172.0_f64;
            let ay1 = svg_h - 10.0;
            write!(o,
                "<path d=\"M {ax0:.1},{ay0:.1} C {ax0:.1},{:.1} {:.1},{ay1:.1} {ax1:.1},{ay1:.1}\" \
                 fill=\"none\" stroke=\"{grey}\" stroke-width=\"0.6\"/>",
                ay0 + 35.0, ax1 + 40.0
            ).unwrap();
            write!(o,
                "<text x=\"{:.1}\" y=\"{ay1:.1}\" text-anchor=\"end\" \
                 dominant-baseline=\"middle\" font-size=\"7\" font-style=\"italic\" fill=\"{grey}\">\
                 {pred_label}</text>",
                ax1 - 2.0
            ).unwrap();
        } else {
            // Daytime / 8pm–midnight: +N right of total dot, curve to bottom-right
            let lbl_x = dot_col_x + dot_r(today_so_far) + 4.0;
            write!(o,
                "<text x=\"{lbl_x:.1}\" y=\"{yc_today:.1}\" text-anchor=\"start\" \
                 dominant-baseline=\"middle\" font-size=\"7\" fill=\"#555555\">\
                 +{pred_remaining_int}</text>"
            ).unwrap();
            let ax0 = lbl_x + 4.0;
            let ay0 = yc_today + 5.0;
            let ax1 = dot_col_x - 50.0;
            let ay1 = svg_h - 10.0;
            write!(o,
                "<path d=\"M {ax0:.1},{ay0:.1} C {ax0:.1},{:.1} {:.1},{ay1:.1} {ax1:.1},{ay1:.1}\" \
                 fill=\"none\" stroke=\"{grey}\" stroke-width=\"0.6\"/>",
                ay0 + 35.0, ax1 + 40.0
            ).unwrap();
            write!(o,
                "<text x=\"{:.1}\" y=\"{ay1:.1}\" text-anchor=\"end\" \
                 dominant-baseline=\"middle\" font-size=\"7\" font-style=\"italic\" fill=\"{grey}\">\
                 {pred_label}</text>",
                ax1 - 2.0
            ).unwrap();
        }
    }

    // Title + subtitle
    let first_t = p.times.first().unwrap();
    let date_range = format!(
        "{} \u{2013} {}, {}",
        fmt_month_day(first_t.month, first_t.day),
        fmt_month_day(last_t.month, last_t.day),
        last_t.year,
    );
    write!(o,
        "<text x=\"{:.0}\" y=\"15\" font-size=\"14\" font-weight=\"bold\" fill=\"#222222\">\
         {} frequency \u{2014} {}</text>",
        xpx(0.0), p.threat_label, p.area_label
    ).unwrap();
    write!(o,
        "<text x=\"{:.0}\" y=\"31\" font-size=\"9\" fill=\"{grey}\">\
         {date_range}   ({} alerts)</text>",
        xpx(0.0), p.times.len()
    ).unwrap();

    // Legend (upper right)
    let leg_y: f64 = 45.0;
    let leg_r: f64 = 4.5;
    let icon_x = xpx(24.0) - 4.0;
    let nd_x = icon_x;
    o.push_str(&svg_wedge(nd_x, leg_y, leg_r, 0.0, 0.5, NIGHT_DOT_COLOR));
    o.push_str(&svg_wedge(nd_x, leg_y, leg_r, 0.5, 1.0, DAY_DOT_COLOR));
    write!(o,
        "<text x=\"{:.1}\" y=\"{leg_y}\" text-anchor=\"end\" \
         dominant-baseline=\"middle\" font-size=\"9\" fill=\"{grey}\">day/night:</text>",
        nd_x - leg_r - 3.0
    ).unwrap();

    if p.style == "dots" {
        let mut dot_leg_x = icon_x - 100.0;
        for (cnt, lbl) in [(5u32, "5"), (1u32, "1")] {
            let r = dot_r(cnt);
            write!(o,
                "<circle cx=\"{dot_leg_x:.1}\" cy=\"{leg_y:.0}\" r=\"{r:.1}\" fill=\"{NIGHT_DOT_COLOR}\"/>"
            ).unwrap();
            write!(o,
                "<text x=\"{:.1}\" y=\"{leg_y}\" dominant-baseline=\"middle\" \
                 font-size=\"9\" fill=\"{grey}\">{lbl}</text>",
                dot_leg_x + r + 2.0
            ).unwrap();
            dot_leg_x -= r * 2.0 + 24.0;
        }
        write!(o,
            "<text x=\"{dot_leg_x:.1}\" y=\"{leg_y}\" text-anchor=\"end\" \
             dominant-baseline=\"middle\" font-size=\"9\" fill=\"{grey}\">alerts per {}h:</text>",
            p.bin_hours
        ).unwrap();
    }

    o.push_str("</svg>");
    Ok(o)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_date(s: &str) -> Option<Date> {
    let b = s.as_bytes();
    if b.len() < 10 { return None; }
    let year = std::str::from_utf8(&b[0..4]).ok()?.parse().ok()?;
    let month = std::str::from_utf8(&b[5..7]).ok()?.parse().ok()?;
    let day = std::str::from_utf8(&b[8..10]).ok()?.parse().ok()?;
    Some(Date { year, month, day })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── svg_wedge ────────────────────────────────────────────────────────────

    #[test]
    fn wedge_full_circle_large_arc() {
        // A wedge spanning > 0.5 must set large-arc flag = 1
        let s = svg_wedge(100.0, 100.0, 10.0, 0.0, 0.75, "#333333");
        assert!(s.contains(",1 "), "expected large-arc flag 1: {s}");
    }

    #[test]
    fn wedge_small_arc_flag() {
        let s = svg_wedge(100.0, 100.0, 10.0, 0.0, 0.25, "#888888");
        assert!(s.contains(",0,1 ") || s.contains(" 0,1 "), "expected large-arc flag 0: {s}");
    }

    #[test]
    fn wedge_starts_with_move_to() {
        let s = svg_wedge(50.0, 60.0, 5.0, 0.0, 0.5, "#ff0000");
        assert!(s.starts_with("<path d=\"M 50.0,60.0"), "unexpected start: {s}");
    }

    #[test]
    fn wedge_half_from_top_ends_at_bottom() {
        // Half circle from top (frac 0.0) to bottom (frac 0.5): end point should be cx, cy+r
        // start_frac=0 → t0=-π/2 → x0=cx, y0=cy-r  (top)
        // end_frac=0.5 → t1=π/2 → x1=cx, y1=cy+r   (bottom)
        let s = svg_wedge(100.0, 100.0, 10.0, 0.0, 0.5, "#333333");
        // x1=100.00, y1=110.00
        assert!(s.contains("100.00,110.00"), "end point wrong: {s}");
    }

    // ── fmt helpers ──────────────────────────────────────────────────────────

    #[test]
    fn weekday_abbr_known() {
        // 2026-04-13 is Monday
        assert_eq!(weekday_abbr(2026, 4, 13), "Mon");
        // 2026-04-14 is Tuesday
        assert_eq!(weekday_abbr(2026, 4, 14), "Tue");
        // 2026-04-19 is Sunday
        assert_eq!(weekday_abbr(2026, 4, 19), "Sun");
    }

    #[test]
    fn fmt_date_label_format() {
        let d = Date { year: 2026, month: 4, day: 3 };
        // 2026-04-03 is Friday
        let s = fmt_date_label(&d);
        assert_eq!(s, "Fri 3 Apr");
    }

    #[test]
    fn fmt_month_day_zero_padded() {
        assert_eq!(fmt_month_day(4, 3), "Apr 03");
        assert_eq!(fmt_month_day(12, 31), "Dec 31");
        assert_eq!(fmt_month_day(1, 1), "Jan 01");
    }

    // ── parse_date ───────────────────────────────────────────────────────────

    #[test]
    fn parse_date_valid() {
        let d = parse_date("2026-02-28").unwrap();
        assert_eq!((d.year, d.month, d.day), (2026, 2, 28));
    }

    #[test]
    fn parse_date_invalid_short() {
        assert!(parse_date("2026-02").is_none());
    }

    // ── render_chart ─────────────────────────────────────────────────────────

    fn make_time(year: i32, month: u32, day: u32, hour: u32, min: u32) -> IsraelTime {
        IsraelTime { year, month, day, hour, min, sec: 0 }
    }

    #[test]
    fn render_empty_times_returns_err() {
        let now = make_time(2026, 4, 10, 12, 0);
        let p = RenderParams {
            times: &[],
            area_label: "Tel Aviv",
            bin_hours: 1,
            start_date: "2026-04-08",
            style: "lines",
            threat_label: "Rocket",
            forecast: "off",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        assert!(render_chart(&p).is_err());
    }

    #[test]
    fn render_returns_svg_root() {
        let times = vec![
            make_time(2026, 4, 8, 10, 0),
            make_time(2026, 4, 8, 14, 30),
            make_time(2026, 4, 9, 22, 0),
        ];
        let now = make_time(2026, 4, 10, 12, 0);
        let p = RenderParams {
            times: &times,
            area_label: "Tel Aviv",
            bin_hours: 1,
            start_date: "2026-04-08",
            style: "lines",
            threat_label: "Rocket",
            forecast: "off",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        let svg = render_chart(&p).unwrap();
        assert!(svg.starts_with("<svg "), "should start with <svg>: {}", &svg[..50]);
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("background:#f0ede3"));
    }

    #[test]
    fn render_contains_title() {
        let times = vec![make_time(2026, 4, 8, 10, 0)];
        let now = make_time(2026, 4, 10, 12, 0);
        let p = RenderParams {
            times: &times,
            area_label: "Test City",
            bin_hours: 1,
            start_date: "2026-04-08",
            style: "lines",
            threat_label: "Rocket",
            forecast: "off",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        let svg = render_chart(&p).unwrap();
        assert!(svg.contains("Rocket frequency"), "title missing: {}", &svg[..200]);
        assert!(svg.contains("Test City"));
    }

    #[test]
    fn render_night_shading_present() {
        let times = vec![make_time(2026, 4, 8, 10, 0)];
        let now = make_time(2026, 4, 10, 12, 0);
        let p = RenderParams {
            times: &times,
            area_label: "X",
            bin_hours: 1,
            start_date: "2026-04-08",
            style: "lines",
            threat_label: "Rocket",
            forecast: "off",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        let svg = render_chart(&p).unwrap();
        // Both night shading rects should be present
        assert_eq!(svg.matches("#e2dfd5").count(), 2, "expected 2 night-shading rects");
    }

    #[test]
    fn render_axis_ticks_count() {
        let times = vec![make_time(2026, 4, 8, 10, 0)];
        let now = make_time(2026, 4, 10, 12, 0);
        let p = RenderParams {
            times: &times,
            area_label: "X",
            bin_hours: 1,
            start_date: "2026-04-08",
            style: "lines",
            threat_label: "Rocket",
            forecast: "off",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        let svg = render_chart(&p).unwrap();
        // 9 tick labels: 00:00, 03:00, ..., 24:00
        assert_eq!(svg.matches(":00</text>").count(), 9, "expected 9 hour labels");
    }

    #[test]
    fn render_dots_style_uses_circles() {
        let times = vec![
            make_time(2026, 4, 8, 10, 0),
            make_time(2026, 4, 8, 10, 5),
        ];
        let now = make_time(2026, 4, 10, 12, 0);
        let p = RenderParams {
            times: &times,
            area_label: "X",
            bin_hours: 1,
            start_date: "2026-04-08",
            style: "dots",
            threat_label: "Rocket",
            forecast: "off",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        let svg = render_chart(&p).unwrap();
        // In dots mode the alert is rendered as a circle (not a <line>)
        // There should be circle elements for the dot and for the total column
        assert!(svg.contains("<circle"), "dots mode should produce circles");
    }

    #[test]
    fn render_pred_desc_when_forecast_on() {
        // enough past alerts for simple forecast to produce non-zero prediction
        let mut times: Vec<IsraelTime> = Vec::new();
        // 5 past days, 10 alerts each at 09:00
        for day in 1..=5u32 {
            for _ in 0..10 {
                times.push(make_time(2026, 4, day, 9, 0));
            }
        }
        // today so far: 3 alerts
        for _ in 0..3 {
            times.push(make_time(2026, 4, 6, 9, 0));
        }
        let now = make_time(2026, 4, 6, 12, 0); // noon on day 6
        let p = RenderParams {
            times: &times,
            area_label: "X",
            bin_hours: 1,
            start_date: "2026-04-01",
            style: "lines",
            threat_label: "Rocket",
            forecast: "simple",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        let svg = render_chart(&p).unwrap();
        // pred-data desc should be present when forecast != "off"
        assert!(svg.contains("pred-data"), "expected pred-data desc element");
    }

    #[test]
    fn render_no_pred_desc_when_forecast_off() {
        let times = vec![make_time(2026, 4, 8, 10, 0)];
        let now = make_time(2026, 4, 10, 12, 0);
        let p = RenderParams {
            times: &times,
            area_label: "X",
            bin_hours: 1,
            start_date: "2026-04-08",
            style: "lines",
            threat_label: "Rocket",
            forecast: "off",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        let svg = render_chart(&p).unwrap();
        assert!(!svg.contains("pred-data"), "no pred-data when forecast=off");
    }

    #[test]
    fn render_svg_width_correct() {
        // SVG_W = LEFT_MARGIN + CHART_W + 56 = 68 + 528 + 56 = 652
        let times = vec![make_time(2026, 4, 8, 10, 0)];
        let now = make_time(2026, 4, 10, 12, 0);
        let p = RenderParams {
            times: &times,
            area_label: "X",
            bin_hours: 1,
            start_date: "2026-04-08",
            style: "lines",
            threat_label: "Rocket",
            forecast: "off",
            all_records: None,
            city_filter: None,
            now: &now,
        };
        let svg = render_chart(&p).unwrap();
        assert!(svg.contains("width=\"652\""), "expected width=652: {}", &svg[..100]);
    }
}
