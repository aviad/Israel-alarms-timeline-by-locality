#!/usr/bin/env python3
"""
Parity harness: run Python (Pyodide) and Rust workers side-by-side and compare
/chart.svg output.

Usage
-----
  # Launch both wrangler devs, run 30 param sets, then kill:
  python tools/parity/diff.py

  # Both servers already running (skip launch):
  python tools/parity/diff.py --no-start

  # More params, include ridge forecast:
  python tools/parity/diff.py --count 50 --ridge

  # Verbose (show SVG element counts per test):
  python tools/parity/diff.py --verbose
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_WORKER_DIR = REPO_ROOT / "worker"
RUST_WORKER_DIR = REPO_ROOT / "worker-rs"

# ── Parameter space ────────────────────────────────────────────────────────────

AREAS = [
    "",           # Israel-wide
    "תל אביב",
    "אשקלון",
    "באר שבע",
    "ירושלים",
    "חיפה",
    "נתיבות",
    "שדרות",
]
STARTS = ["2026-02-28", "2026-03-01", "2026-03-15"]
STYLES = ["lines", "dots"]
THREATS = ["0", "5", "-1"]
FORECASTS_BASE = ["off", "simple"]
FORECASTS_RIDGE = ["off", "simple", "ridge"]


def build_param_sets(n: int, include_ridge: bool) -> list[dict]:
    """Return a deterministic list of N parameter dicts."""
    import random
    forecasts = FORECASTS_RIDGE if include_ridge else FORECASTS_BASE

    # Fixed cases that must always be covered
    fixed = [
        {"area": "",         "start": "2026-02-28", "style": "lines", "threat": "0",  "forecast": "off"},
        {"area": "תל אביב",  "start": "2026-02-28", "style": "lines", "threat": "0",  "forecast": "off"},
        {"area": "אשקלון",   "start": "2026-03-01", "style": "dots",  "threat": "0",  "forecast": "off"},
        {"area": "",         "start": "2026-03-01", "style": "lines", "threat": "5",  "forecast": "off"},
        {"area": "",         "start": "2026-02-28", "style": "lines", "threat": "-1", "forecast": "off"},
        {"area": "תל אביב",  "start": "2026-02-28", "style": "lines", "threat": "0",  "forecast": "simple"},
    ]
    if include_ridge:
        fixed.append(
            {"area": "אשקלון", "start": "2026-03-01", "style": "lines", "threat": "0", "forecast": "ridge"}
        )

    rng = random.Random(42)
    extra = []
    while len(fixed) + len(extra) < n:
        extra.append({
            "area":     rng.choice(AREAS),
            "start":    rng.choice(STARTS),
            "style":    rng.choice(STYLES),
            "threat":   rng.choice(THREATS),
            "forecast": rng.choice(forecasts),
        })

    return (fixed + extra)[:n]


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def fetch(port: int, path: str, params: dict, timeout: int = 90) -> tuple[int | None, str]:
    qs = urllib.parse.urlencode(params)
    url = f"http://localhost:{port}{path}?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return None, str(e)


def wait_for_port(port: int, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(2)
    return False


# ── SVG comparison ─────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(text)]


def _parse_pred_data(svg: str) -> tuple[float, float, str] | None:
    """Extract (remaining, sigma, label) from <desc id="pred-data">."""
    m = re.search(r'<desc id="pred-data">([^<]*)</desc>', svg)
    if not m:
        return None
    parts = m.group(1).split("|")
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), parts[2]
    except ValueError:
        return None


def compare_svgs(py_svg: str, rs_svg: str,
                 numeric_tol: float = 0.5) -> tuple[bool, list[str]]:
    """
    Compare two SVG strings.  Returns (ok, [issue_descriptions]).

    Checks:
    1. Both parse as valid XML.
    2. Element count matches.
    3. Tag sequence (element types in document order) matches.
    4. pred-data presence matches; if both present, values within tolerance.
    5. Numeric values in order — up to 500 numbers compared within tolerance.
       (Absolute OR relative ≤ numeric_tol, whichever is looser.)
    """
    issues: list[str] = []

    # 1. XML parse
    try:
        py_root = ET.fromstring(py_svg)
    except ET.ParseError as e:
        return False, [f"Python SVG XML error: {e}"]
    try:
        rs_root = ET.fromstring(rs_svg)
    except ET.ParseError as e:
        return False, [f"Rust SVG XML error: {e}"]

    # 2. Element count
    py_elems = list(py_root.iter())
    rs_elems = list(rs_root.iter())
    if len(py_elems) != len(rs_elems):
        issues.append(
            f"element count Python={len(py_elems)} Rust={len(rs_elems)}"
        )

    # 3. Tag sequence
    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    py_tags = [strip_ns(e.tag) for e in py_elems]
    rs_tags = [strip_ns(e.tag) for e in rs_elems]
    if py_tags != rs_tags:
        # Find first mismatch
        for i, (pt, rt) in enumerate(zip(py_tags, rs_tags)):
            if pt != rt:
                issues.append(
                    f"tag mismatch at pos {i}: Python={pt!r} Rust={rt!r}"
                )
                break
        if len(py_tags) != len(rs_tags):
            issues.append(
                f"tag count Python={len(py_tags)} Rust={len(rs_tags)}"
            )

    # 4. pred-data
    py_pred = _parse_pred_data(py_svg)
    rs_pred = _parse_pred_data(rs_svg)
    if (py_pred is None) != (rs_pred is None):
        issues.append(
            f"pred-data presence: Python={'yes' if py_pred else 'no'} "
            f"Rust={'yes' if rs_pred else 'no'}"
        )
    elif py_pred and rs_pred:
        # label must match
        if py_pred[2] != rs_pred[2]:
            issues.append(
                f"pred-data label: Python={py_pred[2]!r} Rust={rs_pred[2]!r}"
            )
        # remaining and sigma within tolerance (forecast model may differ slightly)
        for name, pv, rv in [
            ("remaining", py_pred[0], rs_pred[0]),
            ("sigma",     py_pred[1], rs_pred[1]),
        ]:
            diff = abs(pv - rv)
            rel = diff / max(abs(pv), abs(rv), 1e-9)
            if diff > 1.0 and rel > 0.01:  # allow ±1 count absolute
                issues.append(
                    f"pred-data {name}: Python={pv:.4f} Rust={rv:.4f} "
                    f"(diff={diff:.4f})"
                )

    # 5. Numeric values
    py_nums = _extract_numbers(py_svg)
    rs_nums = _extract_numbers(rs_svg)
    n_compare = min(len(py_nums), len(rs_nums), 500)
    num_diffs = 0
    for i in range(n_compare):
        p, r = py_nums[i], rs_nums[i]
        diff = abs(p - r)
        rel = diff / max(abs(p), abs(r), 1e-9)
        if diff > numeric_tol and rel > numeric_tol:
            if num_diffs < 5:
                issues.append(
                    f"numeric[{i}] Python={p} Rust={r} (diff={diff:.4g})"
                )
            num_diffs += 1
    if num_diffs > 5:
        issues.append(f"... and {num_diffs - 5} more numeric diffs")
    if len(py_nums) != len(rs_nums):
        issues.append(
            f"numeric count Python={len(py_nums)} Rust={len(rs_nums)}"
        )

    return len(issues) == 0, issues


# ── Worker launch ──────────────────────────────────────────────────────────────

def start_worker(worker_dir: Path, port: int, name: str) -> subprocess.Popen:
    """Launch wrangler dev in worker_dir on the given port."""
    env = os.environ.copy()
    env["WRANGLER_LOG"] = "error"
    proc = subprocess.Popen(
        ["npx", "wrangler", "dev", "--port", str(port), "--local"],
        cwd=worker_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    print(f"  Launched {name} (pid {proc.pid}) on :{port} ...")
    return proc


def stop_worker(proc: subprocess.Popen, name: str) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"  Stopped {name}.")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_parity(py_port: int, rs_port: int, param_sets: list[dict],
               verbose: bool, tolerance: float, timeout: int = 120) -> int:
    """Run parity checks. Returns exit code (0 = all pass)."""
    total = len(param_sets)
    passed = 0
    failed = 0
    errors = 0

    print(f"\nRunning {total} parity checks (Python :{py_port} vs Rust :{rs_port})...\n")

    # Hit both workers simultaneously so they see the same live alert data.
    # Note: Python/Pyodide in local dev has a ~30s CPU limit per request;
    # ridge forecasts on large areas can exceed this and crash wrangler dev.
    # Skip --ridge when the Python worker is unstable.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i, params in enumerate(param_sets, 1):
            tag = (
                f"[{i:>2}/{total}] area={params['area'] or '<all>'!r:12s} "
                f"start={params['start']} style={params['style']} "
                f"threat={params['threat']} forecast={params['forecast']}"
            )

            # Hit both workers at the same time
            py_fut = pool.submit(fetch, py_port, "/chart.svg", params, timeout)
            rs_fut = pool.submit(fetch, rs_port, "/chart.svg", params, timeout)
            py_status, py_body = py_fut.result()
            rs_status, rs_body = rs_fut.result()

            # Network/timeout error
            if py_status is None or rs_status is None:
                print(f"ERROR {tag}")
                if py_status is None:
                    print(f"        Python network error: {py_body}")
                if rs_status is None:
                    print(f"        Rust   network error: {rs_body}")
                errors += 1
                continue

            # Status code
            if py_status != rs_status:
                print(f"FAIL  {tag}")
                print(f"        Status: Python={py_status} Rust={rs_status}")
                if py_status >= 400:
                    print(f"        Python body: {py_body[:200]}")
                if rs_status >= 400:
                    print(f"        Rust   body: {rs_body[:200]}")
                failed += 1
                continue

            # Both errored (e.g. empty city) — that's a valid match
            if py_status >= 400:
                if verbose:
                    print(f"OK    {tag}  (both {py_status})")
                passed += 1
                continue

            # SVG comparison
            ok, issues = compare_svgs(py_body, rs_body, tolerance)
            if ok:
                if verbose:
                    py_elems = len(list(ET.fromstring(py_body).iter()))
                    print(f"OK    {tag}  ({py_elems} elements)")
                else:
                    print(f"OK    {tag}")
                passed += 1
            else:
                print(f"FAIL  {tag}")
                for iss in issues:
                    print(f"        {iss}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {errors} errors / {total} total")
    return 0 if (failed == 0 and errors == 0) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Parity harness: Python vs Rust worker")
    parser.add_argument("--py-port", type=int, default=8787)
    parser.add_argument("--rs-port", type=int, default=8788)
    parser.add_argument("--no-start", action="store_true",
                        help="Skip launching wrangler dev (assume both already running)")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of param sets to test (default 20)")
    parser.add_argument("--ridge", action="store_true",
                        help="Include ridge forecast tests (slow)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Absolute numeric tolerance for SVG values (default 0.5)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-request HTTP timeout in seconds (default 120)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    param_sets = build_param_sets(args.count, args.ridge)

    procs: list[tuple[subprocess.Popen, str]] = []

    if not args.no_start:
        print("Starting workers...")
        py_proc = start_worker(PYTHON_WORKER_DIR, args.py_port, "Python worker")
        rs_proc = start_worker(RUST_WORKER_DIR,   args.rs_port, "Rust worker")
        procs = [(py_proc, "Python"), (rs_proc, "Rust")]

        try:
            print(f"\nWaiting for Python worker on :{args.py_port} ...")
            if not wait_for_port(args.py_port, timeout=180):
                print("ERROR: Python worker did not start in time.", file=sys.stderr)
                return 1

            print(f"Waiting for Rust worker on :{args.rs_port} ...")
            if not wait_for_port(args.rs_port, timeout=180):
                print("ERROR: Rust worker did not start in time.", file=sys.stderr)
                return 1

            # Extra warm-up: first requests are slowest (Pyodide cold start)
            print("Warming up (first request to each worker) ...")
            warm_params = {"area": "", "start": "2026-03-01", "style": "lines",
                           "threat": "0", "forecast": "off"}
            s_py, _ = fetch(args.py_port, "/chart.svg", warm_params, timeout=120)
            s_rs, _ = fetch(args.rs_port, "/chart.svg", warm_params, timeout=120)
            print(f"  Warm-up: Python→{s_py}  Rust→{s_rs}")

            rc = run_parity(args.py_port, args.rs_port, param_sets,
                            args.verbose, args.tolerance, args.timeout)
        finally:
            print("\nStopping workers...")
            for proc, name in procs:
                stop_worker(proc, name)

        return rc
    else:
        return run_parity(args.py_port, args.rs_port, param_sets,
                          args.verbose, args.tolerance, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
