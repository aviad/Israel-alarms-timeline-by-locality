"""data_loading.py — CSV and API alert loading functions."""

import csv
import datetime
import io

from israel_time import _epoch_to_israel

ROCKET_DESC = "ירי רקטות וטילים"


def load_alerts(
    csv_text: str, area_filter: str, threat: int, start: str
) -> tuple[list[datetime.datetime], set[str]]:
    """Parse CSV and return (deduplicated alert times matching filters, seen ids)."""
    cutoff = datetime.datetime.strptime(start, "%Y-%m-%d")
    seen_ids: set[str] = set()
    times = []

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        dt = datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
        if dt < cutoff:
            continue

        if threat >= 0:
            try:
                row_threat = int(row["threat"])
            except (ValueError, KeyError):
                continue
            if row_threat != threat:
                continue

        if area_filter and area_filter not in row.get("cities", ""):
            continue

        alert_id = row.get("id", "")
        if alert_id in seen_ids:
            continue
        seen_ids.add(alert_id)

        times.append(dt)

    return sorted(times), seen_ids


def load_api_alerts(
    api_data: list[dict], area_filter: str, threat: int, start: str, seen_ids: set[str]
) -> list[datetime.datetime]:
    """Return alert times from API data not already in seen_ids."""
    cutoff = datetime.datetime.strptime(start, "%Y-%m-%d")
    times = []
    for group in api_data:
        gid = str(group["id"])
        if gid in seen_ids:
            continue
        for alert in group.get("alerts", []):
            ts = alert.get("time")
            if ts is None:
                continue
            dt = _epoch_to_israel(ts)
            if dt < cutoff:
                continue
            if threat >= 0 and alert.get("threat") != threat:
                continue
            cities = " ".join(alert.get("cities", []))
            if area_filter and area_filter not in cities:
                continue
            seen_ids.add(gid)
            times.append(dt)
            break
    return times


def load_alerts_rich(
    csv_text: str, threat: int, start: str
) -> tuple[list[dict], set[str]]:
    """Parse CSV and return (event-level records, seen_ids).

    Unlike load_alerts, no area_filter — all events are returned.
    Each dict: {time, cities: list[str], event_id, is_rocket}.
    One dict per unique event_id; all CSV rows for the same event_id are
    aggregated so that cities contains every city hit in that alert.
    Filters to origin == "Iran" when that column is present.
    """
    cutoff = datetime.datetime.strptime(start, "%Y-%m-%d")
    seen_ids: set[str] = set()
    # event_id -> {"time": dt, "cities": [str], "is_rocket": bool}
    by_event: dict = {}

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        try:
            dt = datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, KeyError):
            continue
        if dt < cutoff:
            continue

        if threat >= 0:
            try:
                if int(row["threat"]) != threat:
                    continue
            except (ValueError, KeyError):
                continue

        event_id = row.get("id", "")
        city = row.get("cities", "").strip()
        if not city:
            continue

        is_rocket = row.get("description", "") == ROCKET_DESC

        if event_id in by_event:
            by_event[event_id]["cities"].append(city)
        else:
            by_event[event_id] = {"time": dt, "cities": [city], "is_rocket": is_rocket}
            seen_ids.add(event_id)

    records = [
        {"time": data["time"], "cities": data["cities"], "event_id": eid, "is_rocket": data["is_rocket"]}
        for eid, data in by_event.items()
    ]
    return records, seen_ids


def load_api_alerts_rich(
    api_data: list[dict], threat: int, start: str, seen_ids: set[str]
) -> list[dict]:
    """Return event-level records from API data not already in seen_ids.

    Each dict: {time, cities: list[str], event_id, is_rocket}. No area_filter.
    """
    cutoff = datetime.datetime.strptime(start, "%Y-%m-%d")
    records: list[dict] = []
    for group in api_data:
        gid = str(group["id"])
        if gid in seen_ids:
            continue
        for alert in group.get("alerts", []):
            ts = alert.get("time")
            if ts is None:
                continue
            dt = _epoch_to_israel(ts)
            if dt < cutoff:
                continue
            if threat >= 0 and alert.get("threat") != threat:
                continue
            cities = alert.get("cities", [])
            if not cities:
                continue
            is_rocket = alert.get("threat") == 0
            seen_ids.add(gid)
            records.append({"time": dt, "cities": list(cities), "event_id": gid, "is_rocket": is_rocket})
            break  # one alert per group (same convention as load_api_alerts)
    return records
