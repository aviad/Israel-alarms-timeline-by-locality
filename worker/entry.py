"""
Cloudflare Python Worker entry point for alarms-graph.

Serves:
  GET /          → landing page (HTML form)
  GET /chart.svg → SVG chart (also accepts /chart.png)
"""

import io
import json
from urllib.parse import urlparse, parse_qs

from js import fetch as js_fetch
from pyodide.ffi import to_js
from workers import WorkerEntrypoint, Response

from alarms_core import (
    ALARMS_CSV_URL,
    CITY_TRANSLATIONS,
    DEFAULT_AREA_FILTER,
    DEFAULT_BIN_HOURS,
    DEFAULT_START,
    TZEVAADOM_API_URL,
    load_alerts,
    load_api_alerts,
    load_alerts_rich,
    load_api_alerts_rich,
    render_chart,
)
from forecast import _compute_global_features, _now_israel


def _build_landing_html() -> str:
    cities_data = [["", "כל האזורים / All Areas"]] + sorted(CITY_TRANSLATIONS.items())
    cities_js = json.dumps(cities_data)

    return f"""<!DOCTYPE html>
<html lang="he">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f0ede3">
  <title>Alarms Graph — Israel Rocket Alert Frequency</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,700;1,400&family=Alef&display=swap" rel="stylesheet">
  <style>
    body {{
      font-family: "EB Garamond", Georgia, Palatino, serif;
      background: #f0ede3; color: #333;
      max-width: 860px; margin: 40px auto; padding: 0 20px;
      direction: ltr;
    }}
    h1 {{ font-size: 1.6rem; font-weight: bold; margin-bottom: 0.3em; }}
    p.sub {{ color: #888; margin-top: 0; }}
    form {{ margin: 1.5em 0; display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }}
    label.field {{ display: flex; flex-direction: column; font-size: 0.9rem; color: #555; gap: 4px; }}
    input[type=date], .combo-inp {{
      font-family: inherit; font-size: 1rem;
      padding: 4px 8px; border: 1px solid #ccc;
      background: #faf9f5; border-radius: 3px; min-width: 200px;
      box-sizing: border-box; height: 2rem; line-height: 1;
    }}
    .combo {{ position: relative; }}
    .combo-inp {{ width: 100%; box-sizing: border-box; direction: rtl; }}
    .combo-drop {{
      display: none; position: absolute; top: 100%; left: 0; right: 0;
      background: #faf9f5; border: 1px solid #ccc; border-top: none;
      max-height: 220px; overflow-y: auto; z-index: 100;
    }}
    .combo-opt {{
      padding: 5px 8px; cursor: pointer; direction: rtl; font-size: 0.95rem;
    }}
    .combo-opt:hover, .combo-opt.hi {{ background: #e8e5db; }}
    .combo-opt .en {{ color: #aaa; font-size: 0.78rem; margin-right: 8px; direction: ltr; display: inline-block; }}
    .radios {{ display: flex; gap: 12px; align-items: center; }}
    .options-group {{ display: flex; flex-direction: column; gap: 0; }}
    .option-row {{
      display: flex; gap: 10px; align-items: center;
      font-size: 0.9rem; color: #333;
      padding: 5px 0;
    }}
    .option-row + .option-row {{ border-top: 1px solid #e0ddd5; }}
    .opt-lbl {{ color: #555; min-width: 52px; font-size: 0.9rem; }}
    .options-group .radios label {{ min-width: 72px; }}
    button.go {{
      font-family: inherit; font-size: 1rem;
      padding: 6px 20px; background: #555; color: #f0ede3;
      border: none; border-radius: 3px; cursor: pointer;
    }}
    button.go:hover {{ background: #333; }}
    #chart-wrap {{ margin-top: 1.5em; overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    #chart-wrap object {{ max-width: none; display: block; border: 1px solid #ddd; }}
    #rotate-hint {{
      display: none; margin: 0.5em 0; font-size: 0.85rem; color: #888;
    }}
    @media (orientation: portrait) and (max-width: 700px) {{
      #rotate-hint {{ display: block; }}
    }}
    @media (max-width: 600px) {{
      body {{ margin: 16px auto; padding: 0 12px; }}
      form {{ gap: 10px; }}
      label.field, .options-group {{ width: 100%; min-width: 0; }}
      input[type=date], .combo-inp {{ min-width: 0; min-height: 44px; padding: 8px; }}
      .combo-opt {{ padding: 10px 8px; }}
      button.go {{ width: 100%; padding: 12px; font-size: 1.1rem; }}
      .dl-btn {{ padding: 6px 14px; font-size: 1rem; }}
    }}
    .dl-bar {{ margin-top: 8px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .dl-btn {{
      font-family: inherit; font-size: 0.9rem; color: #555; text-decoration: none;
      border: 1px solid #ccc; padding: 3px 12px;
      border-radius: 3px; background: #faf9f5; cursor: pointer;
      display: inline-flex; align-items: center; line-height: 1;
    }}
    .dl-btn:hover {{ background: #e8e5db; }}
    .dl-btn:disabled {{ opacity: 0.6; cursor: default; }}
    .copy-overlay {{
      position: absolute; top: 8px; right: 8px; z-index: 10;
      opacity: 0.55; padding: 4px 8px;
    }}
    .copy-overlay:hover {{ opacity: 1; }}
  </style>
</head>
<body>
  <h1>Rocket Alert Frequency</h1>

  <form id="form">
    <label class="field">
      אזור / Area
      <div class="combo">
        <input class="combo-inp" id="area-inp" type="text" autocomplete="off"
               placeholder="חפש עיר… / search city" value="{DEFAULT_AREA_FILTER}">
        <div class="combo-drop" id="area-drop"></div>
      </div>
      <input type="hidden" name="area" id="area-val" value="{DEFAULT_AREA_FILTER}">
    </label>

    <label class="field">
      Start date
      <input type="date" name="start" id="start" value="{DEFAULT_START}">
    </label>

    <div class="options-group">
      <div class="option-row">
        <span class="opt-lbl">Style</span>
        <div class="radios">
          <label><input type="radio" name="style" value="lines" checked> Lines</label>
          <label><input type="radio" name="style" value="dots"> Dots</label>
        </div>
      </div>
      <div class="option-row">
        <span class="opt-lbl">Threat</span>
        <div class="radios">
          <label><input type="radio" name="threat" value="0" checked> Rockets</label>
          <label><input type="radio" name="threat" value="5"> UAVs</label>
          <label><input type="radio" name="threat" value="-1"> Both</label>
        </div>
      </div>
      <div class="option-row">
        <span class="opt-lbl">Forecast</span>
        <div class="radios">
          <label><input type="radio" name="forecast" value="off" checked> Off</label>
          <label><input type="radio" name="forecast" value="simple"> Simple</label>
          <label><input type="radio" name="forecast" value="ridge"> Full model</label>
        </div>
      </div>
    </div>

    <button class="go" type="submit">Generate chart</button>
  </form>

  <p id="rotate-hint">↻ Rotate to landscape for best view</p>
  <div id="chart-wrap"></div>

  <script>
    const CITIES = {cities_js};
    const inp  = document.getElementById('area-inp');
    const drop = document.getElementById('area-drop');
    const hidden = document.getElementById('area-val');
    let hi = -1;

    function renderDrop(items) {{
      hi = -1;
      drop.innerHTML = items.slice(0, 80).map(([he, en]) =>
        `<div class="combo-opt" data-v="${{he}}">${{he || '<em>כל האזורים</em>'}} <span class="en">${{en}}</span></div>`
      ).join('');
      drop.style.display = items.length ? 'block' : 'none';
    }}

    function filterCities() {{
      const q = inp.value.trim().toLowerCase();
      renderDrop(q ? CITIES.filter(([he, en]) => he.includes(q) || en.toLowerCase().includes(q)) : CITIES);
    }}

    inp.addEventListener('focus', () => {{ inp.value = ''; filterCities(); }});
    inp.addEventListener('input', filterCities);

    inp.addEventListener('keydown', e => {{
      const opts = drop.querySelectorAll('.combo-opt');
      if      (e.key === 'ArrowDown')  hi = Math.min(hi + 1, opts.length - 1);
      else if (e.key === 'ArrowUp')    hi = Math.max(hi - 1, 0);
      else if (e.key === 'Enter' && drop.style.display !== 'none') {{ e.preventDefault(); opts[Math.max(hi, 0)].click(); return; }}
      else if (e.key === 'Escape')     {{ drop.style.display = 'none'; return; }}
      else return;
      opts.forEach((o, i) => o.classList.toggle('hi', i === hi));
      if (hi >= 0) opts[hi].scrollIntoView({{block: 'nearest'}});
    }});

    drop.addEventListener('mousedown', e => {{
      const opt = e.target.closest('.combo-opt');
      if (!opt) return;
      e.preventDefault();
      hidden.value = opt.dataset.v;
      inp.value = opt.dataset.v || 'כל האזורים / Israel';
      drop.style.display = 'none';
    }});

    inp.addEventListener('blur', () => setTimeout(() => {{
      drop.style.display = 'none';
      if (inp.value === '') inp.value = hidden.value || 'כל האזורים / Israel';
    }}, 200));

    function submitChart() {{
      document.getElementById('form').dispatchEvent(new Event('submit', {{bubbles: true, cancelable: true}}));
    }}

    document.getElementById('form').addEventListener('submit', function(e) {{
      e.preventDefault();
      const fd = new FormData(this);
      const params = new URLSearchParams();
      for (const [k, v] of fd.entries()) params.set(k, v);
      if (!params.has('forecast')) params.set('forecast', 'off');
      history.pushState(null, '', '?' + params);
      const wrap = document.getElementById('chart-wrap');
      wrap.innerHTML = '<p style="color:#888">Generating chart…</p>';
      fetch('/chart.svg?' + params).then(r => {{
        if (!r.ok) return r.text().then(t => {{ throw new Error(t); }});
        return r.blob();
      }}).then(blob => {{
        const url = URL.createObjectURL(blob);
        const obj = document.createElement('object');
        obj.type = 'image/svg+xml';
        obj.data = url;
        const chartContainer = document.createElement('div');
        chartContainer.style.cssText = 'position:relative;display:inline-block;';
        chartContainer.appendChild(obj);
        wrap.innerHTML = '';
        wrap.appendChild(chartContainer);

        async function svgToPng(scale) {{
          const svgText = await blob.text();
          let modSvg = svgText;
          try {{
            // Embed Google Fonts as base64 data URIs so canvas can render them
            const fontCssUrl = 'https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,700;1,400&display=swap';
            const css = await fetch(fontCssUrl).then(r => r.text());
            function toB64(buf) {{
              const bytes = new Uint8Array(buf);
              let s = '';
              for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
              return btoa(s);
            }}
            const fontUrls = [...new Set([...css.matchAll(/url\((https:\/\/[^)]+)\)/g)].map(m => m[1]))];
            let embCss = css;
            for (const u of fontUrls) {{
              const buf = await fetch(u).then(r => r.arrayBuffer());
              const mime = u.includes('.woff2') ? 'font/woff2' : 'font/woff';
              embCss = embCss.split(u).join('data:' + mime + ';base64,' + toB64(buf));
            }}
            modSvg = svgText.replace(/@import url\([^)]+\);?/, embCss);
          }} catch(e) {{ /* fall back to original SVG; fonts may not render */ }}
          return new Promise((resolve, reject) => {{
            const blobUrl = URL.createObjectURL(new Blob([modSvg], {{type: 'image/svg+xml'}}));
            const img = new Image();
            img.onload = () => {{
              const canvas = document.createElement('canvas');
              canvas.width  = (img.naturalWidth  || 800) * scale;
              canvas.height = (img.naturalHeight || 600) * scale;
              const ctx = canvas.getContext('2d');
              ctx.scale(scale, scale);
              ctx.drawImage(img, 0, 0);
              URL.revokeObjectURL(blobUrl);
              canvas.toBlob(resolve, 'image/png');
            }};
            img.onerror = reject;
            img.src = blobUrl;
          }});
        }}

        function mkBtn(label, action) {{
          const b = document.createElement('button');
          b.className = 'dl-btn';
          b.innerHTML = label;
          b.onclick = async () => {{
            const orig = b.innerHTML;
            b.disabled = true;
            try {{
              await action();
              b.innerHTML = '✓';
            }} catch(e) {{
              b.innerHTML = '✗';
            }}
            setTimeout(() => {{ b.innerHTML = orig; b.disabled = false; }}, 2000);
          }};
          return b;
        }}

        const dlSvg = document.createElement('a');
        dlSvg.className = 'dl-btn';
        dlSvg.href = url;
        dlSvg.download = 'alarms-chart.svg';
        dlSvg.textContent = '↓ SVG';

        const bar = document.createElement('div');
        bar.className = 'dl-bar';
        bar.appendChild(dlSvg);
        bar.appendChild(mkBtn('↓ PNG', async () => {{
          const png = await svgToPng(2);
          const pu = URL.createObjectURL(png);
          Object.assign(document.createElement('a'), {{href: pu, download: 'alarms-chart.png'}}).click();
          URL.revokeObjectURL(pu);
        }}));
        async function doCopy() {{
          const png = await svgToPng(2);
          await navigator.clipboard.write([new ClipboardItem({{'image/png': png}})]);
        }}
        bar.appendChild(mkBtn('⎘', doCopy));

        const overlayBtn = document.createElement('button');
        overlayBtn.className = 'dl-btn copy-overlay';
        overlayBtn.textContent = '⎘';
        overlayBtn.onclick = async () => {{
          const orig = overlayBtn.textContent;
          overlayBtn.disabled = true;
          try {{ await doCopy(); overlayBtn.textContent = '✓'; }}
          catch(e) {{ overlayBtn.textContent = '✗'; }}
          setTimeout(() => {{ overlayBtn.textContent = orig; overlayBtn.disabled = false; }}, 2000);
        }};
        chartContainer.appendChild(overlayBtn);
        wrap.appendChild(bar);
      }}).catch(err => {{
        wrap.innerHTML = `<p style="color:red">Error: ${{err.message}}</p>`;
      }});
    }});

    // Pre-fill form from URL params and auto-generate if any are present
    (function() {{
      const sp = new URLSearchParams(window.location.search);
      if (sp.has('area')) {{
        const a = sp.get('area');
        const entry = CITIES.find(([he]) => he === a);
        hidden.value = entry ? a : '';
        inp.value = entry ? (a || 'כל האזורים / Israel') : '';
      }}
      if (sp.has('start')) document.getElementById('start').value = sp.get('start');
      if (sp.has('style')) {{
        const s = sp.get('style');
        document.querySelectorAll('input[name=style]').forEach(r => r.checked = r.value === s);
      }}
      if (sp.has('threat')) {{
        const t = sp.get('threat');
        document.querySelectorAll('input[name=threat]').forEach(r => r.checked = r.value === t);
      }}
      if (sp.has('forecast')) {{
        const f = sp.get('forecast');
        document.querySelectorAll('input[name=forecast]').forEach(r => r.checked = r.value === f);
      }}
      if (sp.has('area') || sp.has('start') || sp.has('style') || sp.has('threat') || sp.has('forecast')) submitChart();
    }})();
  </script>
  <p style="margin-top:2em;font-size:0.78rem;color:#888;line-height:1.6">
    Data: <a href="https://github.com/yuval-harpaz/alarms" style="color:#888">yuval-harpaz/alarms</a>,
    <a href="https://www.tzevaadom.co.il/" style="color:#888;font-family:'Alef',sans-serif">צופר - צבע אדום</a> ·
    App: <a href="https://github.com/aviad/Israel-alarms-timeline-by-locality" style="color:#888">aviad/Israel-alarms-timeline-by-locality</a> ·
    Forecast model: <a href="https://github.com/ofir-reich/missile-alarms-prediction" style="color:#888">ofir-reich/missile-alarms-prediction</a> ·
    Chart design inspired by <a href="https://www.edwardtufte.com" style="color:#888">Edward Tufte</a>
  </p>
</body>
</html>"""


class Default(WorkerEntrypoint):

    async def _fetch_csv(self) -> tuple[str, str]:
        """Fetch alarms CSV from KV cache or GitHub. Filters to rows >= 2026-02-27."""
        cached = await self.env.CACHE.get("csv:alarms:v3")
        if cached:
            meta = await self.env.CACHE.get("csv:meta") or ""
            return cached, meta

        resp = await js_fetch(ALARMS_CSV_URL)
        text = await resp.text()
        last_mod = resp.headers.get("Last-Modified") or ""

        # Filter to recent rows to stay within the 128 MB worker memory limit.
        # The full CSV (~122K rows) is too large; we only need data from Feb 27.
        CUTOFF = "2026-02-27"
        buf = io.StringIO(text)
        header = buf.readline()
        cols = header.strip().split(",")
        try:
            time_idx = cols.index("time")
        except ValueError:
            time_idx = 1
        lines = [header]
        for line in buf:
            parts = line.split(",", time_idx + 1)
            if len(parts) > time_idx and parts[time_idx].strip('"')[:10] >= CUTOFF:
                lines.append(line)
        del text
        filtered = "".join(lines)

        await self.env.CACHE.put("csv:alarms:v3", filtered, to_js({"expirationTtl": 30 * 60}))
        await self.env.CACHE.put("csv:meta", last_mod, to_js({"expirationTtl": 30 * 60}))
        return filtered, last_mod

    async def _fetch_api_data(self) -> list[dict]:
        """Fetch recent alerts from tzevaadom API via KV cache (1-min buckets)."""
        import time
        bucket = int(time.time()) // 60  # changes every 1 minute
        cache_key = f"api:v1:{bucket}"

        cached = await self.env.CACHE.get(cache_key)
        if cached:
            return json.loads(cached)

        resp = await js_fetch(TZEVAADOM_API_URL)
        text = await resp.text()
        await self.env.CACHE.put(cache_key, text, to_js({"expirationTtl": 2 * 60}))
        return json.loads(text)

    async def fetch(self, request):
        url = urlparse(request.url)
        path = url.path

        if path in ("/", ""):
            return Response(_build_landing_html(), headers={"Content-Type": "text/html; charset=utf-8"})

        # Chart endpoint: /chart.png or /chart.svg
        if not (path.endswith(".png") or path.endswith(".svg")):
            return Response("Not found", status=404)

        params = parse_qs(url.query, keep_blank_values=True)
        area = (params.get("area", [DEFAULT_AREA_FILTER]) or [DEFAULT_AREA_FILTER])[0]
        label = (params.get("label", [""]) or [""])[0] or CITY_TRANSLATIONS.get(area, area or "Israel")
        start = (params.get("start", [DEFAULT_START]) or [DEFAULT_START])[0]
        style = (params.get("style", ["lines"]) or ["lines"])[0]
        threat = int((params.get("threat", ["0"]) or ["0"])[0])
        threat_label = {0: "Rocket alert", 5: "UAV alert", -1: "All threats alert"}.get(threat, "Rocket alert")
        bin_hours = int((params.get("bin_hours", [str(DEFAULT_BIN_HOURS)]) or [str(DEFAULT_BIN_HOURS)])[0])
        try:
            csv_text, _last_mod = await self._fetch_csv()

            try:
                api_data = await self._fetch_api_data()
            except Exception:
                api_data = []

            forecast = (params.get("forecast", ["off"]) or ["off"])[0]
            if forecast not in ("off", "simple", "advanced", "ridge"):
                forecast = "off"

            # Display loading (area-filtered, deduplicated per event)
            times, seen_ids = load_alerts(csv_text, area, threat, start)
            api_times = load_api_alerts(api_data, area, threat, start, seen_ids)
            times = sorted(times + api_times)

            # Rich loading and global features cache (only for ridge forecast)
            all_records = None
            global_feats = None
            if forecast == "ridge":
                all_records, rich_seen_ids = load_alerts_rich(csv_text, threat, start)
                api_rich = load_api_alerts_rich(api_data, threat, start, rich_seen_ids)
                all_records = all_records + api_rich

                _now = _now_israel()
                _gf_key = "global_features:v1"
                _gf_cached = await self.env.CACHE.get(_gf_key)
                if _gf_cached:
                    global_feats = json.loads(_gf_cached)
                else:
                    global_feats = _compute_global_features(all_records, _now)
                    await self.env.CACHE.put(
                        _gf_key, json.dumps(global_feats),
                        to_js({"expirationTtl": 30 * 60})
                    )

            svg = render_chart(
                times, label, bin_hours, start, None, style,
                threat_label=threat_label, forecast=forecast,
                all_records=all_records if forecast == "ridge" else None,
                city_filter=area if forecast == "ridge" else None,
                global_features_cache=global_feats,
            ).decode("utf-8")
        except ValueError as exc:
            return Response(str(exc), status=400)
        except Exception as exc:
            return Response(f"Internal error: {exc}", status=500)

        return Response(
            svg,
            headers={
                "Content-Type": "image/svg+xml; charset=utf-8",
                "Cache-Control": "public, max-age=120",
            },
        )
