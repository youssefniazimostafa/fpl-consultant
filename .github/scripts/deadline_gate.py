#!/usr/bin/env python3
"""Hourly pre-deadline gate (standard library only -- runs before `uv sync`).

The hourly cron in update.yml calls this first. It decides whether this
run should be *the* pre-deadline snapshot run for the next gameweek, and
writes ``run=true`` or ``run=false`` to $GITHUB_OUTPUT.

Rule (exactly one snapshot run per deadline, however GitHub delays cron):

* Run if the deadline is 1-2 hours away **and** no recommendation snapshot
  taken in the final 2 hours already exists on origin/main.
* If cron was delayed so badly that no run landed between 2h and 1h before
  the deadline, still run as a late fallback down to 15 minutes before it
  (enough for the ~3-minute pipeline plus the commit), again only if no
  snapshot from the final 2 hours exists yet.
* Otherwise skip -- the workflow then ends quickly as a success, without
  deploying or committing.

"Snapshot exists" is read from data/recommendations/gw{N}.json's
``frozen_at`` on origin/main (``git show``), not the possibly stale
checkout. Once a run in the final 2 hours has frozen it, later hourly runs
skip, and ``fpl.consultant.freeze_recommendation`` won't overwrite it
either (same 2-hour constant: ``FINAL_SNAPSHOT_WINDOW`` there).

Test inputs (env): GATE_NOW, GATE_DEADLINE (ISO times), GATE_GW, and
GATE_FROZEN_AT ("" = no snapshot) replace the API / git lookups.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

WINDOW_START = timedelta(hours=2)  # keep in sync with fpl/consultant.py FINAL_SNAPSHOT_WINDOW
WINDOW_END = timedelta(hours=1)
LATEST_FALLBACK = timedelta(minutes=15)
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def next_deadline(now: datetime) -> tuple[int, datetime] | None:
    if os.environ.get("GATE_DEADLINE"):
        dl = parse(os.environ["GATE_DEADLINE"])
        return (int(os.environ.get("GATE_GW", "0")), dl) if dl > now else None
    # FPL sometimes refuses GitHub runners briefly (HTTP 403); retry, and
    # let main() turn a lasting failure into a quiet skip (the next hourly
    # run tries again, and no snapshot is marked as taken).
    events = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(BOOTSTRAP_URL, headers={"User-Agent": "fpl-consultant-gate"})
            with urllib.request.urlopen(req, timeout=30) as r:
                events = json.load(r)["events"]
            break
        except (OSError, ValueError, KeyError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    upcoming = [(e["id"], parse(e["deadline_time"])) for e in events if parse(e["deadline_time"]) > now]
    return min(upcoming, key=lambda x: x[1]) if upcoming else None


def frozen_at(gw: int) -> datetime | None:
    if "GATE_FROZEN_AT" in os.environ:
        v = os.environ["GATE_FROZEN_AT"]
        return parse(v) if v else None
    try:
        text = subprocess.run(
            ["git", "show", f"origin/main:data/recommendations/gw{gw}.json"],
            capture_output=True, text=True, check=True,
        ).stdout
        return parse(json.loads(text)["frozen_at"])
    except (subprocess.CalledProcessError, KeyError, ValueError):
        return None


def decide(now: datetime) -> tuple[bool, str]:
    nd = next_deadline(now)
    if nd is None:
        return False, "no upcoming deadline"
    gw, deadline = nd
    left = deadline - now
    mins = int(left.total_seconds() // 60)
    snap = frozen_at(gw)
    already = snap is not None and snap >= deadline - WINDOW_START
    where = f"GW{gw} deadline {deadline:%Y-%m-%d %H:%M}Z is {mins} min away"
    if already:
        return False, f"{where}; a final snapshot was already taken at {snap:%H:%M}Z -- skip"
    if WINDOW_END <= left <= WINDOW_START:
        return True, f"{where}; inside the 1-2h window and no final snapshot yet -- run"
    if LATEST_FALLBACK <= left < WINDOW_END:
        return True, f"{where}; cron was late and no final snapshot yet -- run (late fallback)"
    return False, f"{where}; outside the window -- skip"


def main() -> None:
    now = parse(os.environ["GATE_NOW"]) if os.environ.get("GATE_NOW") else datetime.now(timezone.utc)
    try:
        run, reason = decide(now)
    except (OSError, ValueError, KeyError) as exc:
        run, reason = False, f"couldn't read the next deadline from FPL ({exc}) -- skip; the next hourly run will try again"
        print(f"::warning::{reason}")
    print(f"gate: {reason}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"run={'true' if run else 'false'}\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
