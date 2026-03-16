"""israel_time.py — Israel UTC offset and epoch conversion."""

import datetime


def _israel_utc_offset(utc_dt: datetime.datetime) -> int:
    """Return Israel UTC offset (2 or 3) for a given UTC datetime.
    DST: last Friday of March 02:00 → last Sunday of October 02:00 (Israel rule).
    """
    y = utc_dt.year
    mar31_wd = datetime.date(y, 3, 31).weekday()  # Mon=0 … Sun=6
    dst_start = datetime.datetime(y, 3, 31 - (mar31_wd + 3) % 7, 2)  # last Friday
    oct31_wd = datetime.date(y, 10, 31).weekday()
    dst_end = datetime.datetime(y, 10, 31 - (oct31_wd + 1) % 7, 2)   # last Sunday
    return 3 if dst_start <= utc_dt < dst_end else 2


def _epoch_to_israel(ts: float) -> datetime.datetime:
    """Convert Unix timestamp to Israel local time."""
    utc = datetime.datetime.utcfromtimestamp(ts)
    return utc + datetime.timedelta(hours=_israel_utc_offset(utc))
