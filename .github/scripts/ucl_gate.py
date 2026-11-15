#!/usr/bin/env python3
"""Gate for the UCL workflow (standard library only, runs before `uv sync`).

The hourly check (cron "15 * * * *" in ucl.yml, plus the cron-job.org
backup with slot_check=true) runs the UCL pipeline only when one of the
run slots in data/ucl/schedule.json is due and hasn't run yet:

* a slot is due from its time until DUE_FOR after it (so an hourly check
  that starts late still catches it);
* "has run" is read from data/ucl/runs.json on origin/main (the pipeline
  records the slot there when it finishes, via UCL_SLOT).

Writes run=true|false and slot=<id> to $GITHUB_OUTPUT. Test inputs (env):
GATE_NOW (ISO time), GATE_SCHEDULE and GATE_RUNS (JSON strings).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

DUE_FOR = timedelta(minutes=100)


def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def from_main(path: str, env: str) -> dict:
    if os.environ.get(env) is not None:
        return json.loads(os.environ[env] or "{}")
    try:
        text = subprocess.run(["git", "show", f"origin/main:{path}"], capture_output=True, text=True, check=True).stdout
        return json.loads(text)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def due(now: datetime, schedule: dict, runs: dict) -> dict | None:
    for s in schedule.get("slots", []):
        at = parse(s["at"])
        if at <= now < at + DUE_FOR and s["id"] not in runs:
            return s
    return None


def main() -> None:
    now = parse(os.environ["GATE_NOW"]) if os.environ.get("GATE_NOW") else datetime.now(timezone.utc)
    s = due(now, from_main("data/ucl/schedule.json", "GATE_SCHEDULE"), from_main("data/ucl/runs.json", "GATE_RUNS"))
    out = os.environ.get("GITHUB_OUTPUT")
    lines = [f"run={'true' if s else 'false'}", f"slot={s['id'] if s else ''}"]
    if out:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
    print(f"{now.isoformat(timespec='minutes')}: " + (f"slot {s['id']} is due -> run" if s else "no UCL slot due -> skip"))
    sys.exit(0)


if __name__ == "__main__":
    main()
