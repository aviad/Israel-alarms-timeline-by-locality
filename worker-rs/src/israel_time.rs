/// Israel UTC offset and epoch conversion.
/// DST: last Friday of March 02:00 → last Sunday of October 02:00 (Israel rule).

/// Return Israel UTC offset in hours (2 or 3) for a given UTC epoch (seconds).
pub fn israel_utc_offset(epoch: i64) -> i32 {
    // Compute UTC date parts from epoch using integer arithmetic.
    let secs = epoch;
    // Days since Unix epoch (floor division — works for negative too)
    let days = if secs >= 0 { secs / 86400 } else { (secs - 86399) / 86400 };
    let (year, month, day) = days_to_ymd(days);

    // Find last Friday of March: start from Mar 31
    let dst_start = last_weekday_of_month(year, 3, 31, 4); // weekday 4 = Friday (Mon=0)
    // Find last Sunday of October: start from Oct 31
    let dst_end = last_weekday_of_month(year, 10, 31, 6); // weekday 6 = Sunday

    // Epoch of dst_start and dst_end at 02:00 UTC
    let dst_start_epoch = ymd_to_epoch(dst_start.0, dst_start.1, dst_start.2) + 2 * 3600;
    let dst_end_epoch = ymd_to_epoch(dst_end.0, dst_end.1, dst_end.2) + 2 * 3600;

    let _ = (year, month, day); // suppress unused warning
    if epoch >= dst_start_epoch && epoch < dst_end_epoch { 3 } else { 2 }
}

/// Convert Unix timestamp to Israel local time components: (year, month, day, hour, minute, second).
pub fn epoch_to_israel(epoch: i64) -> (i32, u32, u32, u32, u32, u32) {
    let offset = israel_utc_offset(epoch) as i64;
    let local = epoch + offset * 3600;
    let secs_of_day = local.rem_euclid(86400) as u32;
    let days = local.div_euclid(86400);
    let (y, m, d) = days_to_ymd(days);
    let h = secs_of_day / 3600;
    let mn = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    (y, m, d, h, mn, s)
}

// --- helpers ---

/// Find the last weekday (Mon=0..Sun=6) in a month, searching back from `start_day`.
fn last_weekday_of_month(year: i32, month: u32, start_day: u32, target_wd: i64) -> (i32, u32, u32) {
    let epoch = ymd_to_epoch(year, month, start_day);
    let wd = epoch_weekday(epoch); // Mon=0..Sun=6
    // How many days to go back
    let back = ((wd - target_wd).rem_euclid(7)) as u32;
    let day = start_day - back;
    (year, month, day)
}

/// Weekday (Mon=0..Sun=6) for a UTC epoch (seconds).
fn epoch_weekday(epoch: i64) -> i64 {
    // Unix epoch Jan 1 1970 was a Thursday = 3
    let days = if epoch >= 0 { epoch / 86400 } else { (epoch - 86399) / 86400 };
    (days + 3).rem_euclid(7)
}

/// Seconds since Unix epoch for the start of a UTC day (year, month, day).
pub fn ymd_to_epoch(year: i32, month: u32, day: u32) -> i64 {
    // Days from epoch to start of year
    let y = year as i64 - 1970;
    // Leap years between 1970 and year (exclusive)
    let leaps = leap_years_since_1970(year);
    let year_days = y * 365 + leaps;
    let month_days: i64 = [0i64, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        .iter()
        .take(month as usize)
        .sum();
    let feb_extra = if month > 2 && is_leap(year) { 1 } else { 0 };
    (year_days + month_days + feb_extra + day as i64 - 1) * 86400
}

/// Convert days since Unix epoch to (year, month, day).
pub(crate) fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Proleptic Gregorian — works for dates around 1970–2100
    let mut d = days + 719468; // shift to civil epoch (Mar 1, year 0)
    let era = if d >= 0 { d / 146097 } else { (d - 146096) / 146097 };
    d -= era * 146097;
    let yoe = (d - d / 1460 + d / 36524 - d / 146096) / 365;
    let y = yoe + era * 400;
    let doy = d - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year as i32, month as u32, day as u32)
}

fn is_leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn leap_years_since_1970(year: i32) -> i64 {
    // Count leap years in [1970, year)
    let y = year as i64 - 1;
    let count = |n: i64| n / 4 - n / 100 + n / 400;
    count(y) - count(1969)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build epoch from (y, m, d, h, min, s) UTC
    fn utc(y: i32, mo: u32, d: u32, h: u32, mn: u32, s: u32) -> i64 {
        ymd_to_epoch(y, mo, d) + (h as i64) * 3600 + (mn as i64) * 60 + s as i64
    }

    #[test]
    fn test_midsummer_is_dst() {
        // July 1 — always DST
        assert_eq!(israel_utc_offset(utc(2025, 7, 1, 12, 0, 0)), 3);
        assert_eq!(israel_utc_offset(utc(2026, 7, 1, 12, 0, 0)), 3);
    }

    #[test]
    fn test_midwinter_is_standard() {
        // January 1 — always standard time
        assert_eq!(israel_utc_offset(utc(2025, 1, 1, 12, 0, 0)), 2);
        assert_eq!(israel_utc_offset(utc(2026, 1, 1, 12, 0, 0)), 2);
    }

    #[test]
    fn test_dst_start_boundaries() {
        // 2026: last Friday of March is Mar 27
        // Before 02:00 UTC on Mar 27 → offset 2
        assert_eq!(israel_utc_offset(utc(2026, 3, 27, 1, 59, 59)), 2);
        // At 02:00 UTC on Mar 27 → offset 3
        assert_eq!(israel_utc_offset(utc(2026, 3, 27, 2, 0, 0)), 3);
    }

    #[test]
    fn test_dst_end_boundaries() {
        // 2026: last Sunday of October is Oct 25
        // Before 02:00 UTC on Oct 25 → offset 3
        assert_eq!(israel_utc_offset(utc(2026, 10, 25, 1, 59, 59)), 3);
        // At 02:00 UTC on Oct 25 → offset 2
        assert_eq!(israel_utc_offset(utc(2026, 10, 25, 2, 0, 0)), 2);
    }

    #[test]
    fn test_dst_start_2024() {
        // 2024: last Friday of March is Mar 29
        assert_eq!(israel_utc_offset(utc(2024, 3, 29, 1, 59, 59)), 2);
        assert_eq!(israel_utc_offset(utc(2024, 3, 29, 2, 0, 0)), 3);
    }

    #[test]
    fn test_dst_start_2025() {
        // 2025: last Friday of March is Mar 28
        assert_eq!(israel_utc_offset(utc(2025, 3, 28, 1, 59, 59)), 2);
        assert_eq!(israel_utc_offset(utc(2025, 3, 28, 2, 0, 0)), 3);
    }

    #[test]
    fn test_epoch_to_israel_basic() {
        // 2026-04-15 12:00:00 UTC → DST → +3 → 15:00:00 Israel
        let ep = utc(2026, 4, 15, 12, 0, 0);
        let (y, mo, d, h, mn, s) = epoch_to_israel(ep);
        assert_eq!((y, mo, d, h, mn, s), (2026, 4, 15, 15, 0, 0));
    }

    #[test]
    fn test_epoch_to_israel_winter() {
        // 2026-01-01 22:00:00 UTC → +2 → 2026-01-02 00:00:00 Israel
        let ep = utc(2026, 1, 1, 22, 0, 0);
        let (y, mo, d, h, mn, s) = epoch_to_israel(ep);
        assert_eq!((y, mo, d, h, mn, s), (2026, 1, 2, 0, 0, 0));
    }

    #[test]
    fn test_ymd_roundtrip() {
        for (y, m, d) in [(2024, 2, 29), (2025, 3, 1), (2026, 10, 25), (2000, 1, 1)] {
            let ep = ymd_to_epoch(y, m, d);
            let (ry, rm, rd) = days_to_ymd(ep / 86400);
            assert_eq!((ry, rm, rd), (y, m, d), "roundtrip failed for {y}-{m:02}-{d:02}");
        }
    }
}
