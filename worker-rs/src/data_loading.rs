/// data_loading.rs — CSV and API alert loading, ported from data_loading.py.

use std::collections::{HashMap, HashSet};

use crate::israel_time::epoch_to_israel;

pub const ROCKET_DESC: &str = "ירי רקטות וטילים";

/// Israel local datetime (parsed from CSV "YYYY-MM-DD HH:MM:SS" or built from epoch).
/// Derived Ord is lexicographic on fields in declaration order — correct for dates.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IsraelTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub min: u32,
    pub sec: u32,
}

impl IsraelTime {
    pub fn from_epoch(epoch: i64) -> Self {
        let (y, mo, d, h, mn, s) = epoch_to_israel(epoch);
        IsraelTime { year: y, month: mo, day: d, hour: h, min: mn, sec: s }
    }

    /// Parse "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD" (time defaults to 00:00:00).
    pub fn parse(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() < 10 {
            return None;
        }
        let year = parse_digits(&b[0..4])? as i32;
        let month = parse_digits(&b[5..7])?;
        let day = parse_digits(&b[8..10])?;
        let (hour, min, sec) = if b.len() >= 19 {
            (parse_digits(&b[11..13])?, parse_digits(&b[14..16])?, parse_digits(&b[17..19])?)
        } else {
            (0, 0, 0)
        };
        Some(IsraelTime { year, month, day, hour, min, sec })
    }
}

fn parse_digits(b: &[u8]) -> Option<u32> {
    std::str::from_utf8(b).ok()?.parse().ok()
}

/// Event-level record for Ridge regression (load_alerts_rich / load_api_alerts_rich).
#[derive(Clone, Debug)]
pub struct AlertRecord {
    pub time: IsraelTime,
    pub cities: Vec<String>,
    pub event_id: String,
    pub is_rocket: bool,
}

// --- CSV helpers ---

struct CsvIndices {
    time: Option<usize>,
    threat: Option<usize>,
    cities: Option<usize>,
    id: Option<usize>,
    description: Option<usize>,
}

impl CsvIndices {
    fn from_headers(headers: &csv::StringRecord) -> Self {
        let find = |name: &str| headers.iter().position(|h| h == name);
        CsvIndices {
            time: find("time"),
            threat: find("threat"),
            cities: find("cities"),
            id: find("id"),
            description: find("description"),
        }
    }

    fn get<'a>(&self, rec: &'a csv::StringRecord, idx: Option<usize>) -> &'a str {
        idx.and_then(|i| rec.get(i)).unwrap_or("")
    }
}

// --- Public API ---

/// Parse CSV and return `(sorted deduplicated alert times, seen_ids)`.
///
/// `threat = -1` disables threat filter. `area_filter = ""` disables area filter.
/// Matches Python `load_alerts(csv_text, area_filter, threat, start)`.
pub fn load_alerts(
    csv_text: &str,
    area_filter: &str,
    threat: i32,
    start: &str,
) -> (Vec<IsraelTime>, HashSet<String>) {
    let cutoff = match IsraelTime::parse(start) {
        Some(t) => t,
        None => return (vec![], HashSet::new()),
    };
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut times: Vec<IsraelTime> = Vec::new();

    let mut reader = csv::Reader::from_reader(csv_text.as_bytes());
    let idx = {
        let headers = match reader.headers() {
            Ok(h) => h.clone(),
            Err(_) => return (vec![], HashSet::new()),
        };
        CsvIndices::from_headers(&headers)
    };

    for result in reader.records() {
        let rec = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let time_str = idx.get(&rec, idx.time);
        let dt = match IsraelTime::parse(time_str) {
            Some(t) => t,
            None => continue,
        };
        if dt < cutoff {
            continue;
        }

        if threat >= 0 {
            let row_threat: i32 = match idx.get(&rec, idx.threat).parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if row_threat != threat {
                continue;
            }
        }

        let cities = idx.get(&rec, idx.cities);
        if !area_filter.is_empty() && !cities.contains(area_filter) {
            continue;
        }

        let alert_id = idx.get(&rec, idx.id).to_string();
        if seen_ids.contains(&alert_id) {
            continue;
        }
        seen_ids.insert(alert_id);
        times.push(dt);
    }

    times.sort();
    (times, seen_ids)
}

/// Return alert times from API JSON not already in `seen_ids` (mutated in-place).
///
/// Matches Python `load_api_alerts(api_data, area_filter, threat, start, seen_ids)`.
pub fn load_api_alerts(
    api_data: &serde_json::Value,
    area_filter: &str,
    threat: i32,
    start: &str,
    seen_ids: &mut HashSet<String>,
) -> Vec<IsraelTime> {
    let cutoff = match IsraelTime::parse(start) {
        Some(t) => t,
        None => return vec![],
    };
    let groups = match api_data.as_array() {
        Some(a) => a,
        None => return vec![],
    };
    let mut times = Vec::new();

    for group in groups {
        let gid = group["id"].as_i64().map(|n| n.to_string())
            .or_else(|| group["id"].as_str().map(str::to_string))
            .unwrap_or_default();
        if seen_ids.contains(&gid) {
            continue;
        }
        let alerts = match group["alerts"].as_array() {
            Some(a) => a,
            None => continue,
        };
        for alert in alerts {
            let ts = match alert["time"].as_i64() {
                Some(t) => t,
                None => continue,
            };
            let dt = IsraelTime::from_epoch(ts);
            if dt < cutoff {
                continue;
            }
            if threat >= 0 {
                let row_threat = match alert["threat"].as_i64() {
                    Some(t) => t as i32,
                    None => continue,
                };
                if row_threat != threat {
                    continue;
                }
            }
            let cities: String = alert["cities"].as_array()
                .map(|arr| arr.iter().filter_map(|c| c.as_str()).collect::<Vec<_>>().join(" "))
                .unwrap_or_default();
            if !area_filter.is_empty() && !cities.contains(area_filter) {
                continue;
            }
            seen_ids.insert(gid.clone());
            times.push(dt);
            break; // one alert per group, same as Python
        }
    }
    times
}

/// Parse CSV and return `(event-level records, seen_ids)`. No area_filter.
///
/// Cities are aggregated per unique event_id. Matches Python `load_alerts_rich`.
pub fn load_alerts_rich(
    csv_text: &str,
    threat: i32,
    start: &str,
) -> (Vec<AlertRecord>, HashSet<String>) {
    let cutoff = match IsraelTime::parse(start) {
        Some(t) => t,
        None => return (vec![], HashSet::new()),
    };
    let mut seen_ids: HashSet<String> = HashSet::new();
    // Preserve insertion order with a Vec for keys + HashMap for data.
    let mut order: Vec<String> = Vec::new();
    let mut by_event: HashMap<String, (IsraelTime, Vec<String>, bool)> = HashMap::new();

    let mut reader = csv::Reader::from_reader(csv_text.as_bytes());
    let idx = {
        let headers = match reader.headers() {
            Ok(h) => h.clone(),
            Err(_) => return (vec![], HashSet::new()),
        };
        CsvIndices::from_headers(&headers)
    };

    for result in reader.records() {
        let rec = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let time_str = idx.get(&rec, idx.time);
        let dt = match IsraelTime::parse(time_str) {
            Some(t) => t,
            None => continue,
        };
        if dt < cutoff {
            continue;
        }

        if threat >= 0 {
            let row_threat: i32 = match idx.get(&rec, idx.threat).parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if row_threat != threat {
                continue;
            }
        }

        let event_id = idx.get(&rec, idx.id).to_string();
        let city = idx.get(&rec, idx.cities).trim().to_string();
        if city.is_empty() {
            continue;
        }
        let is_rocket = idx.get(&rec, idx.description) == ROCKET_DESC;

        if let Some(entry) = by_event.get_mut(&event_id) {
            entry.1.push(city);
        } else {
            by_event.insert(event_id.clone(), (dt, vec![city], is_rocket));
            order.push(event_id.clone());
            seen_ids.insert(event_id);
        }
    }

    let records = order.into_iter().map(|eid| {
        let (time, cities, is_rocket) = by_event.remove(&eid).unwrap();
        AlertRecord { time, cities, event_id: eid, is_rocket }
    }).collect();

    (records, seen_ids)
}

/// Return event-level records from API JSON not already in `seen_ids`. No area_filter.
///
/// Matches Python `load_api_alerts_rich`.
pub fn load_api_alerts_rich(
    api_data: &serde_json::Value,
    threat: i32,
    start: &str,
    seen_ids: &mut HashSet<String>,
) -> Vec<AlertRecord> {
    let cutoff = match IsraelTime::parse(start) {
        Some(t) => t,
        None => return vec![],
    };
    let groups = match api_data.as_array() {
        Some(a) => a,
        None => return vec![],
    };
    let mut records = Vec::new();

    for group in groups {
        let gid = group["id"].as_i64().map(|n| n.to_string())
            .or_else(|| group["id"].as_str().map(str::to_string))
            .unwrap_or_default();
        if seen_ids.contains(&gid) {
            continue;
        }
        let alerts = match group["alerts"].as_array() {
            Some(a) => a,
            None => continue,
        };
        for alert in alerts {
            let ts = match alert["time"].as_i64() {
                Some(t) => t,
                None => continue,
            };
            let dt = IsraelTime::from_epoch(ts);
            if dt < cutoff {
                continue;
            }
            if threat >= 0 {
                let row_threat = match alert["threat"].as_i64() {
                    Some(t) => t as i32,
                    None => continue,
                };
                if row_threat != threat {
                    continue;
                }
            }
            let cities: Vec<String> = alert["cities"].as_array()
                .map(|arr| arr.iter().filter_map(|c| c.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            if cities.is_empty() {
                continue;
            }
            let is_rocket = alert["threat"].as_i64().map_or(false, |t| t == 0);
            seen_ids.insert(gid.clone());
            records.push(AlertRecord { time: dt, cities, event_id: gid.clone(), is_rocket });
            break;
        }
    }
    records
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal CSV with the columns the code reads.
    fn make_csv(rows: &[(&str, &str, &str, &str, &str)]) -> String {
        let mut s = "time,threat,cities,id,description\n".to_string();
        for (time, threat, cities, id, desc) in rows {
            s.push_str(&format!("{},{},{},{},{}\n", time, threat, cities, id, desc));
        }
        s
    }

    #[test]
    fn test_parse_datetime() {
        let t = IsraelTime::parse("2026-03-15 14:30:00").unwrap();
        assert_eq!(t, IsraelTime { year: 2026, month: 3, day: 15, hour: 14, min: 30, sec: 0 });
    }

    #[test]
    fn test_parse_date_only() {
        let t = IsraelTime::parse("2026-03-15").unwrap();
        assert_eq!(t, IsraelTime { year: 2026, month: 3, day: 15, hour: 0, min: 0, sec: 0 });
    }

    #[test]
    fn test_israel_time_ord() {
        let a = IsraelTime::parse("2026-03-01 00:00:00").unwrap();
        let b = IsraelTime::parse("2026-03-01 12:00:00").unwrap();
        let c = IsraelTime::parse("2026-04-01 00:00:00").unwrap();
        assert!(a < b && b < c);
    }

    #[test]
    fn test_load_alerts_basic() {
        let csv = make_csv(&[
            ("2026-03-01 10:00:00", "0", "תל אביב", "id1", "ירי רקטות וטילים"),
            ("2026-03-02 11:00:00", "0", "חיפה", "id2", "ירי רקטות וטילים"),
        ]);
        let (times, seen) = load_alerts(&csv, "", -1, "2026-03-01");
        assert_eq!(times.len(), 2);
        assert!(seen.contains("id1") && seen.contains("id2"));
        assert!(times[0] < times[1]); // sorted
    }

    #[test]
    fn test_load_alerts_dedup() {
        // Same id twice — second row dropped.
        let csv = make_csv(&[
            ("2026-03-01 10:00:00", "0", "תל אביב", "id1", ""),
            ("2026-03-01 11:00:00", "0", "חיפה", "id1", ""),
        ]);
        let (times, seen) = load_alerts(&csv, "", -1, "2026-03-01");
        assert_eq!(times.len(), 1);
        assert_eq!(seen.len(), 1);
    }

    #[test]
    fn test_load_alerts_start_filter() {
        let csv = make_csv(&[
            ("2026-02-27 10:00:00", "0", "תל אביב", "id1", ""),
            ("2026-02-28 10:00:00", "0", "תל אביב", "id2", ""),
        ]);
        let (times, _) = load_alerts(&csv, "", -1, "2026-02-28");
        assert_eq!(times.len(), 1);
        assert_eq!(times[0], IsraelTime::parse("2026-02-28 10:00:00").unwrap());
    }

    #[test]
    fn test_load_alerts_threat_filter() {
        let csv = make_csv(&[
            ("2026-03-01 10:00:00", "0", "תל אביב", "id1", ""),
            ("2026-03-01 11:00:00", "1", "חיפה", "id2", ""),
        ]);
        let (times, _) = load_alerts(&csv, "", 0, "2026-03-01");
        assert_eq!(times.len(), 1);
        assert_eq!(times[0].hour, 10);
    }

    #[test]
    fn test_load_alerts_area_filter() {
        let csv = make_csv(&[
            ("2026-03-01 10:00:00", "0", "תל אביב", "id1", ""),
            ("2026-03-01 11:00:00", "0", "חיפה", "id2", ""),
        ]);
        let (times, _) = load_alerts(&csv, "חיפה", -1, "2026-03-01");
        assert_eq!(times.len(), 1);
        assert_eq!(times[0].hour, 11);
    }

    #[test]
    fn test_load_api_alerts_basic() {
        // epoch 1773568800 → 2026-03-15 12:00:00 Israel time
        let data = serde_json::json!([
            {
                "id": 101,
                "alerts": [{"time": 1773568800i64, "threat": 0, "cities": ["תל אביב"]}]
            }
        ]);
        let mut seen = HashSet::new();
        let times = load_api_alerts(&data, "", -1, "2026-01-01", &mut seen);
        assert_eq!(times.len(), 1);
        assert!(seen.contains("101"));
    }

    #[test]
    fn test_load_api_alerts_dedup_by_seen() {
        let data = serde_json::json!([
            {
                "id": 101,
                "alerts": [{"time": 1773568800i64, "threat": 0, "cities": ["תל אביב"]}]
            }
        ]);
        let mut seen = HashSet::from(["101".to_string()]);
        let times = load_api_alerts(&data, "", -1, "2026-01-01", &mut seen);
        assert_eq!(times.len(), 0);
    }

    #[test]
    fn test_load_alerts_rich_aggregates_cities() {
        // Two rows with same event_id → cities aggregated.
        let csv = make_csv(&[
            ("2026-03-01 10:00:00", "0", "תל אביב", "ev1", "ירי רקטות וטילים"),
            ("2026-03-01 10:00:00", "0", "חיפה", "ev1", "ירי רקטות וטילים"),
            ("2026-03-01 11:00:00", "1", "ירושלים", "ev2", "כטב\"מ"),
        ]);
        let (records, seen) = load_alerts_rich(&csv, -1, "2026-03-01");
        assert_eq!(records.len(), 2);
        assert_eq!(seen.len(), 2);

        let ev1 = records.iter().find(|r| r.event_id == "ev1").unwrap();
        assert_eq!(ev1.cities, vec!["תל אביב", "חיפה"]);
        assert!(ev1.is_rocket);

        let ev2 = records.iter().find(|r| r.event_id == "ev2").unwrap();
        assert_eq!(ev2.cities, vec!["ירושלים"]);
        assert!(!ev2.is_rocket);
    }

    #[test]
    fn test_load_alerts_rich_threat_filter() {
        let csv = make_csv(&[
            ("2026-03-01 10:00:00", "0", "תל אביב", "ev1", ""),
            ("2026-03-01 11:00:00", "1", "חיפה", "ev2", ""),
        ]);
        let (records, _) = load_alerts_rich(&csv, 0, "2026-03-01");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].event_id, "ev1");
    }

    #[test]
    fn test_load_api_alerts_rich_basic() {
        // epoch 1773568800 → 2026-03-15 12:00:00 Israel time
        let data = serde_json::json!([
            {
                "id": 200,
                "alerts": [{"time": 1773568800i64, "threat": 0, "cities": ["תל אביב", "יפו"]}]
            }
        ]);
        let mut seen = HashSet::new();
        let records = load_api_alerts_rich(&data, -1, "2026-01-01", &mut seen);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].cities, vec!["תל אביב", "יפו"]);
        assert_eq!(records[0].event_id, "200");
        assert!(records[0].is_rocket); // threat == 0
        assert!(seen.contains("200"));
    }
}
