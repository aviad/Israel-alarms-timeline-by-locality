// entry.rs — Cloudflare Worker entrypoint.
//
// Routes:
//   GET /           -> landing page HTML
//   GET /chart.svg  -> SVG chart
//   GET /chart.png  -> SVG chart (same content; PNG rasterized client-side)
//   GET /healthz    -> 200 OK
//   *               -> 404
//
// KV key prefix "rs:" avoids collisions with the Python worker during parallel run.

use worker::*;

use crate::city_translations;
use crate::data_loading::{load_alerts, load_alerts_rich, load_api_alerts, load_api_alerts_rich,
                          IsraelTime};
use crate::kv_cache;
use crate::svg::{render_chart, RenderParams, DEFAULT_AREA_FILTER, DEFAULT_BIN_HOURS, DEFAULT_START};

const ALARMS_CSV_URL: &str =
    "https://raw.githubusercontent.com/yuval-harpaz/alarms/master/data/alarms.csv";
const TZEVAADOM_API_URL: &str = "https://api.tzevaadom.co.il/alerts-history/";

/// CSV rows before this date are dropped (keeps KV payload small).
const CSV_CUTOFF: &str = "2026-02-27";

/// TTL values (seconds).
const CSV_TTL: u64 = 30 * 60;
const API_TTL: u64 = 2 * 60;

// ── Entrypoint ───────────────────────────────────────────────────────────────

#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let url = req.url()?;
    let path = url.path();

    match &path[..] {
        "/" | "" => serve_index(),
        "/healthz" => Response::ok("ok"),
        "/chart.svg" | "/chart.png" => serve_chart(&url, &env).await,
        _ => Response::error("Not found", 404),
    }
}

// ── Landing page ─────────────────────────────────────────────────────────────

fn serve_index() -> Result<Response> {
    let cities_json = build_cities_json();
    let html = include_str!("../assets/landing.html")
        .replace("__DEFAULT_AREA__", DEFAULT_AREA_FILTER)
        .replace("__DEFAULT_START__", DEFAULT_START)
        .replace("__CITIES_JSON__", &cities_json);

    let headers = Headers::new();
    headers.set("Content-Type", "text/html; charset=utf-8")?;
    Ok(Response::from_body(ResponseBody::Body(html.into_bytes()))?.with_headers(headers))
}

/// Build the JS cities array: [["", "כל האזורים / All Areas"], ["אבו גוש", "Abu Ghosh"], ...]
fn build_cities_json() -> String {
    // The Python worker sorts by Hebrew key; city_translations::CITIES is already sorted.
    // We expose it via the private slice — replicate by iterating the public translate fn
    // won't work for all entries. Instead, embed the same slice that city_translations uses.
    // Since CITIES is pub(crate) effectively (static in the module), we use the module directly.

    // Collect all entries from the static slice via translate on every known key — not practical.
    // Instead, re-expose the raw slice from city_translations as pub.
    // Workaround: use city_translations::all_entries() helper we add, OR just iterate.
    //
    // For now, call city_translations::all_cities() which we define below as a thin re-export.
    let entries = city_translations::all_cities();

    let mut json = String::from("[[\"\",\"כל האזורים / All Areas\"]");
    for (he, en) in entries {
        json.push_str(",[");
        push_json_str(&mut json, he);
        json.push(',');
        push_json_str(&mut json, en);
        json.push(']');
    }
    json.push(']');
    json
}

/// Minimal JSON string escaping (no external dep needed for simple strings).
fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
}

// ── Chart endpoint ───────────────────────────────────────────────────────────

async fn serve_chart(url: &Url, env: &Env) -> Result<Response> {
    let params = parse_qs(url.query().unwrap_or(""));

    let area = params.get("area").map(|s| s.as_str()).unwrap_or(DEFAULT_AREA_FILTER);
    let start = params.get("start").map(|s| s.as_str()).unwrap_or(DEFAULT_START);
    let style = params.get("style").map(|s| s.as_str()).unwrap_or("lines");
    let threat: i32 = params.get("threat")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let bin_hours: u32 = params.get("bin_hours")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BIN_HOURS);
    let forecast = params.get("forecast").map(|s| s.as_str()).unwrap_or("off");
    let forecast = if matches!(forecast, "off" | "simple" | "advanced" | "ridge") {
        forecast
    } else {
        "off"
    };

    let label = if area.is_empty() {
        "Israel".to_string()
    } else {
        city_translations::translate(area)
            .unwrap_or(area)
            .to_string()
    };
    let threat_label = match threat {
        5  => "UAV alert",
        -1 => "All threats alert",
        _  => "Rocket alert",
    };

    let kv = env.kv("CACHE")?;
    let now = now_israel();

    // ── Fetch CSV ──────────────────────────────────────────────────────────
    let csv_text = match fetch_csv(&kv).await {
        Ok(t) => t,
        Err(e) => return Response::error(format!("CSV fetch failed: {e}"), 500),
    };

    // ── Fetch API data ─────────────────────────────────────────────────────
    let api_json: serde_json::Value = match fetch_api(&kv).await {
        Ok(v) => v,
        Err(_) => serde_json::Value::Array(vec![]),
    };

    // ── Load alert times ───────────────────────────────────────────────────
    let (mut times, mut seen_ids) = load_alerts(&csv_text, area, threat, start);
    let api_times = load_api_alerts(&api_json, area, threat, start, &mut seen_ids);
    times.extend(api_times);
    times.sort();

    // ── Load rich records for Ridge ────────────────────────────────────────
    let all_records = if forecast == "ridge" {
        let (mut records, mut rich_seen) = load_alerts_rich(&csv_text, threat, start);
        let api_rich = load_api_alerts_rich(&api_json, threat, start, &mut rich_seen);
        records.extend(api_rich);
        Some(records)
    } else {
        None
    };

    // ── Render ────────────────────────────────────────────────────────────
    let params = RenderParams {
        times: &times,
        area_label: &label,
        bin_hours,
        start_date: start,
        style,
        threat_label,
        forecast,
        all_records: all_records.as_deref(),
        city_filter: if forecast == "ridge" && !area.is_empty() { Some(area) } else { None },
        now: &now,
    };

    let svg = match render_chart(&params) {
        Ok(s) => s,
        Err(e) => return Response::error(e, 400),
    };

    let headers = Headers::new();
    headers.set("Content-Type", "image/svg+xml; charset=utf-8")?;
    headers.set("Cache-Control", "public, max-age=120")?;
    Ok(Response::from_body(ResponseBody::Body(svg.into_bytes()))?.with_headers(headers))
}

// ── Data fetching ────────────────────────────────────────────────────────────

/// Fetch CSV from KV cache (zlib-compressed) or upstream GitHub, filter rows ≥ CSV_CUTOFF.
async fn fetch_csv(kv: &worker::kv::KvStore) -> Result<String> {
    const KEY: &str = "rs:csv:alarms:v1";

    if let Some(raw) = kv_cache::get_zlib(kv, KEY).await? {
        return String::from_utf8(raw)
            .map_err(|e| Error::RustError(format!("CSV utf8: {e}")));
    }

    let mut resp = Fetch::Url(ALARMS_CSV_URL.parse()?).send().await?;
    let body = resp.bytes().await?;

    let filtered = filter_csv_bytes(&body);

    // Store compressed; don't fail the request if caching fails
    let _ = kv_cache::put_zlib(kv, KEY, filtered.as_bytes(), CSV_TTL).await;

    Ok(filtered)
}

/// Keep the CSV header + rows where the `time` column is ≥ CSV_CUTOFF.
fn filter_csv_bytes(data: &[u8]) -> String {
    let text = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };

    let mut lines = text.lines();
    let header = match lines.next() {
        None => return String::new(),
        Some(h) => h,
    };

    let time_idx = header.split(',')
        .position(|c| c.trim_matches('"') == "time")
        .unwrap_or(1);

    let mut result = String::with_capacity(data.len() / 2);
    result.push_str(header);
    result.push('\n');

    for line in lines {
        // Split only enough to reach the time field
        let fields: Vec<&str> = line.splitn(time_idx + 2, ',').collect();
        if let Some(ts) = fields.get(time_idx) {
            let ts = ts.trim_matches('"');
            if ts.len() >= 10 && &ts[..10] >= CSV_CUTOFF {
                result.push_str(line);
                result.push('\n');
            }
        }
    }
    result
}

/// Fetch API JSON from KV (1-minute TTL bucket) or live endpoint.
async fn fetch_api(kv: &worker::kv::KvStore) -> Result<serde_json::Value> {
    // Bucket key changes every minute (same strategy as Python worker).
    let key = format!("rs:api:v1:{}", epoch_minute());

    if let Some(cached) = kv_cache::get_text(kv, &key).await? {
        if let Ok(v) = serde_json::from_str(&cached) {
            return Ok(v);
        }
    }

    let mut resp = Fetch::Url(TZEVAADOM_API_URL.parse()?).send().await?;
    let text = resp.text().await?;

    let _ = kv_cache::put_text(kv, &key, &text, API_TTL).await;

    serde_json::from_str(&text).map_err(|e| Error::RustError(format!("API JSON parse: {e}")))
}

// ── Utilities ────────────────────────────────────────────────────────────────

/// Current Israel local time using the Cloudflare worker clock.
fn now_israel() -> IsraelTime {
    let ms = Date::now().as_millis();
    let epoch = (ms / 1000) as i64;
    IsraelTime::from_epoch(epoch)
}

/// UTC epoch in whole minutes — used as a 1-minute KV cache bucket key.
fn epoch_minute() -> u64 {
    Date::now().as_millis() / 1000 / 60
}

/// Parse a URL query string into a simple key→last-value map.
fn parse_qs(query: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for part in query.split('&') {
        if part.is_empty() { continue; }
        let (k, v) = match part.find('=') {
            Some(i) => (&part[..i], &part[i + 1..]),
            None => (part, ""),
        };
        map.insert(url_decode(k), url_decode(v));
    }
    map
}

/// URL percent-decode (handles `+` as space and `%XX` sequences).
///
/// Collects decoded bytes then converts to UTF-8, which is required for
/// multi-byte characters like Hebrew (e.g. `%D7%AA` → `ת`).
fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut decoded: Vec<u8> = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'+' {
            decoded.push(b' ');
            i += 1;
        } else if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    decoded.push(byte);
                    i += 3;
                    continue;
                }
            }
            decoded.push(bytes[i]);
            i += 1;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(decoded).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_decode_ascii() {
        assert_eq!(url_decode("hello+world"), "hello world");
        assert_eq!(url_decode("a%3Db"), "a=b");
        assert_eq!(url_decode("no+encoding"), "no encoding");
    }

    #[test]
    fn url_decode_hebrew() {
        // "תל אביב" URL-encoded (UTF-8 percent-encoding)
        // ת=%D7%AA, ל=%D7%9C, (space)=%20, א=%D7%90, ב=%D7%91, י=%D7%99, ב=%D7%91
        let encoded = "%D7%AA%D7%9C+%D7%90%D7%91%D7%99%D7%91";
        assert_eq!(url_decode(encoded), "תל אביב");
    }

    #[test]
    fn parse_qs_basic() {
        let m = parse_qs("area=%D7%AA%D7%9C&start=2026-02-28&forecast=off");
        assert_eq!(m.get("area").map(String::as_str), Some("תל"));
        assert_eq!(m.get("start").map(String::as_str), Some("2026-02-28"));
    }

    #[test]
    fn filter_csv_keeps_header_and_cutoff() {
        let csv = "id,time,cities\n1,2026-02-26 10:00:00,foo\n2,2026-02-27 00:00:00,bar\n3,2026-03-01 12:00:00,baz\n";
        let filtered = filter_csv_bytes(csv.as_bytes());
        assert!(filtered.contains("2026-02-27"), "cutoff row should be kept");
        assert!(!filtered.contains("2026-02-26"), "pre-cutoff row should be dropped");
        assert!(filtered.contains("2026-03-01"), "post-cutoff row should be kept");
    }
}
