"""When the UCL pipeline should run, derived from the fixture list.

For every coming matchday (times from UEFA's fixture feed, in UTC):

* eve     - the evening before the deadline, 20:00 Central European time;
* pre     - 2 hours before the deadline (the first kick-off). This run
            freezes the recommendation. Line-ups come out only 1 hour
            before kick-off, i.e. after the deadline, so they can't be used
            for the squad itself.
* dayN-end - 3.5 hours after the last kick-off of each match day but the
            last (results and stats are in): the between-days advice;
* dayN-morning - 08:00 the next morning, a second between-days run with
            the final stats and the next day's team news.

No between-days runs on MD8 or the final (no changes allowed).
The gate (.github/scripts/ucl_gate.py) reads data/ucl/schedule.json and
data/ucl/runs.json from the repo, so it needs no Python packages.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from ucl import feed, rules

PATH = feed.DATA / "schedule.json"
RUNS = feed.DATA / "runs.json"
CET = ZoneInfo("Europe/Paris")


def _at_cet(day, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=CET).astimezone(timezone.utc)


def build(matches: pd.DataFrame) -> list[dict]:
    slots = []
    for md, g in matches[~matches["finished"] | (matches["kickoff"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2))].groupby("md"):
        md = int(md)
        dl = g["kickoff"].min().to_pydatetime()
        dl_local = dl.astimezone(CET)
        slots.append({"id": f"md{md}-eve", "md": md, "kind": "eve", "at": _at_cet(dl_local.date() - timedelta(days=1), 20)})
        slots.append({"id": f"md{md}-pre", "md": md, "kind": "pre", "at": dl - timedelta(hours=2)})
        if rules.between_days_allowed(md):
            days = sorted(g["gd"].unique())
            for n, gd in enumerate(days[:-1], 1):
                last_ko = g.loc[g["gd"] == gd, "kickoff"].max().to_pydatetime()
                slots.append({"id": f"md{md}-day{n}-end", "md": md, "kind": "between", "at": last_ko + timedelta(hours=3.5)})
                nxt = last_ko.astimezone(CET).date() + timedelta(days=1)
                slots.append({"id": f"md{md}-day{n}-morning", "md": md, "kind": "between", "at": _at_cet(nxt, 8)})
    slots.sort(key=lambda s: s["at"])
    return [s | {"at": s["at"].isoformat()} for s in slots]


def write(slots: list[dict]) -> None:
    PATH.write_text(json.dumps({"slots": slots}, indent=1))


def mark_done(slot: str, now: datetime) -> None:
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    runs[slot] = now.isoformat(timespec="seconds")
    RUNS.write_text(json.dumps(dict(sorted(runs.items())), indent=1))


def next_slots(slots: list[dict], now: datetime, n: int = 4) -> list[dict]:
    return [s for s in slots if datetime.fromisoformat(s["at"]) > now][:n]
