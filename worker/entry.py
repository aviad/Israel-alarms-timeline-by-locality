"""
Cloudflare Python Worker entry point for alarms-graph.

Serves:
  GET /          → landing page (HTML form)
  GET /chart.png → PNG chart
  GET /chart.svg → SVG chart
"""

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
    render_chart,
)


def _build_landing_html() -> str:
    options = sorted(CITY_TRANSLATIONS.items(), key=lambda x: x[1])
    option_tags = "\n".join(
        f'<option value="{he}">{en}</option>' for he, en in options
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alarms Graph — Israel Rocket Alert Frequency</title>
  <style>
    body {{
      font-family: Georgia, Palatino, "DejaVu Serif", serif;
      background: #f0ede3;
      color: #333;
      max-width: 860px;
      margin: 40px auto;
      padding: 0 20px;
    }}
    h1 {{ font-size: 1.6rem; font-weight: bold; margin-bottom: 0.3em; }}
    p.sub {{ color: #888; margin-top: 0; }}
    form {{ margin: 1.5em 0; display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }}
    label {{ display: flex; flex-direction: column; font-size: 0.9rem; color: #555; gap: 4px; }}
    select, input[type=date] {{
      font-family: inherit;
      font-size: 1rem;
      padding: 4px 8px;
      border: 1px solid #ccc;
      background: #faf9f5;
      border-radius: 3px;
    }}
    .radios {{ display: flex; gap: 12px; align-items: center; }}
    button {{
      font-family: inherit;
      font-size: 1rem;
      padding: 6px 20px;
      background: #555;
      color: #f0ede3;
      border: none;
      border-radius: 3px;
      cursor: pointer;
    }}
    button:hover {{ background: #333; }}
    #chart-wrap {{ margin-top: 1.5em; }}
    #chart-wrap img {{ max-width: 100%; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Rocket Alert Frequency</h1>
  <p class="sub">Israel Civil Defense alert data · <a href="https://github.com/yuval-harpaz/alarms">yuval-harpaz/alarms</a></p>

  <form id="form">
    <label>
      Area
      <select name="area" id="area">
        <option value="">All Areas</option>
        {option_tags}
      </select>
    </label>

    <label>
      Start date
      <input type="date" name="start" id="start" value="{DEFAULT_START}">
    </label>

    <label>
      Style
      <div class="radios">
        <label><input type="radio" name="style" value="lines" checked> Lines</label>
        <label><input type="radio" name="style" value="dots"> Dots</label>
      </div>
    </label>

    <button type="submit">Generate chart</button>
  </form>

  <div id="chart-wrap"></div>

  <script>
    document.getElementById('form').addEventListener('submit', function(e) {{
      e.preventDefault();
      const fd = new FormData(this);
      const params = new URLSearchParams();
      for (const [k, v] of fd.entries()) if (v) params.set(k, v);
      const src = '/chart.png?' + params.toString();
      const wrap = document.getElementById('chart-wrap');
      wrap.innerHTML = '<p style="color:#888">Generating chart… (first load may take ~10s)</p>';
      const img = new Image();
      img.onload = () => {{ wrap.innerHTML = ''; wrap.appendChild(img); }};
      img.onerror = () => {{ wrap.innerHTML = '<p style="color:red">Error generating chart.</p>'; }};
      img.src = src;
      img.alt = 'Alarms frequency chart';
      img.style.maxWidth = '100%';
    }});
  </script>
</body>
</html>"""


class Default(WorkerEntrypoint):

    async def _fetch_csv(self) -> tuple[str, str]:
        """Fetch alarms CSV from KV cache or GitHub."""
        cached = await self.env.CACHE.get("csv:alarms")
        if cached:
            meta = await self.env.CACHE.get("csv:meta") or ""
            return cached, meta

        resp = await js_fetch(ALARMS_CSV_URL)
        text = await resp.text()
        last_mod = resp.headers.get("Last-Modified") or ""

        await self.env.CACHE.put("csv:alarms", text, to_js({"expirationTtl": 30 * 60}))
        await self.env.CACHE.put("csv:meta", last_mod, to_js({"expirationTtl": 30 * 60}))
        return text, last_mod

    async def _fetch_api_data(self) -> list[dict]:
        """Fetch recent alerts from tzevaadom API via KV cache."""
        cached = await self.env.CACHE.get("api:alerts")
        if cached:
            return json.loads(cached)

        resp = await js_fetch(TZEVAADOM_API_URL)
        text = await resp.text()
        await self.env.CACHE.put("api:alerts", text, to_js({"expirationTtl": 2 * 60}))
        return json.loads(text)

    async def fetch(self, request):
        url = urlparse(request.url)
        path = url.path

        if path in ("/", ""):
            return Response(_build_landing_html(), headers={"Content-Type": "text/html; charset=utf-8"})

        # Chart endpoint: /chart.png or /chart.svg
        if not (path.endswith(".png") or path.endswith(".svg")):
            return Response("Not found", status=404)

        params = parse_qs(url.query)
        area = (params.get("area", [DEFAULT_AREA_FILTER]) or [DEFAULT_AREA_FILTER])[0]
        label = (params.get("label", [""]) or [""])[0] or CITY_TRANSLATIONS.get(area, area or "All Areas")
        start = (params.get("start", [DEFAULT_START]) or [DEFAULT_START])[0]
        style = (params.get("style", ["lines"]) or ["lines"])[0]
        threat = int((params.get("threat", ["0"]) or ["0"])[0])
        bin_hours = int((params.get("bin_hours", [str(DEFAULT_BIN_HOURS)]) or [str(DEFAULT_BIN_HOURS)])[0])
        fmt = "svg" if path.endswith(".svg") else "png"

        try:
            csv_text, _last_mod = await self._fetch_csv()
            times, seen_ids = load_alerts(csv_text, area, threat, start)

            try:
                api_data = await self._fetch_api_data()
            except Exception:
                api_data = []
            api_times = load_api_alerts(api_data, area, threat, start, seen_ids)
            times = sorted(times + api_times)

            img_bytes = render_chart(times, label, bin_hours, start, None, style, fmt)
        except ValueError as exc:
            return Response(str(exc), status=400)
        except Exception as exc:
            return Response(f"Internal error: {exc}", status=500)

        content_type = "image/svg+xml" if fmt == "svg" else "image/png"
        return Response(
            img_bytes,
            headers={
                "Content-Type": content_type,
                "Cache-Control": "public, max-age=120",
            },
        )
