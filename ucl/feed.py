"""Polite access to UEFA's public UCL Fantasy feeds.

Endpoints (all public JSON, no login), relative to FEEDS:

* config/web.json                         - the game's own config (URL templates)
* constraints/constraints_{tour}.json     - current matchday, budget, club limit, deadline
* fixtures/fixtures_{tour}_en.json        - every matchday: deadline, match days, kick-offs (CET), scores, line-up flag
* players/players_{tour}_en_{md}.json     - all players: price, position, club, ownership, status, totals
* popupstats/popupstats_{tour}_{pid}.json - one player's per-match stats and points (minutes, goals, assists,
                                            balls recovered, saves, PotM, cards, penalties ...)
* playerofmatch/stats_{tour}_en_{md}.json - Player of the Match per match
* teams/teams_{tour}_en.json              - clubs

Behind a login (not used): Gameplay/user/{guid}/* (your team, transfers,
captain, substitutions; retVal -70 without one) and Leaderboard/* (the game
average and ranks). scoring/live-players-points_{tour}.json returns 403.

Politeness: a plain client name (a browser-like client was blocked), one
request at a time with REQUEST_GAP seconds between requests, timeouts and
retries with back-off. Every successful core feed is also saved as "last
good" in data/ucl/last_good/, which is used, and flagged as stale in
source health, when a later fetch fails.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ucl"
RAW = DATA / "raw"  # git-ignored cache
LAST_GOOD = DATA / "last_good"
HEALTH_PATH = DATA / "source_health.json"

FEEDS = "https://gaming.uefa.com/en/uclfantasy/services/feeds/"
TOUR = 90  # 2026/27
PREV_TOUR = 80  # 2025/26
USER_AGENT = "xpert-ucl/0.1 (personal fantasy helper; one request at a time)"
REQUEST_GAP = 1.5
TIMEOUT = 20
RETRIES = 3

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_request = 0.0
_health: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _wait() -> None:
    global _last_request
    gap = time.monotonic() - _last_request
    if gap < REQUEST_GAP:
        time.sleep(REQUEST_GAP - gap)
    _last_request = time.monotonic()


def fetch(path: str) -> Any:
    """GET FEEDS + path as JSON, with timeouts and retries. Raises on failure."""
    err: Exception | None = None
    for attempt in range(RETRIES):
        _wait()
        try:
            r = _session.get(FEEDS + path, timeout=TIMEOUT)
            if r.status_code in (401, 403, 404):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError, RuntimeError) as e:
            err = e
            if isinstance(e, RuntimeError):
                break  # 4xx: retrying won't help
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"{path}: {err}")


def value(doc: Any) -> Any:
    return doc["data"]["value"] if isinstance(doc, dict) and "data" in doc else doc


def core(name: str, path: str, check=None, save: bool = True) -> Any:
    """A core feed with a last-good fallback and a source-health entry.

    `check(value) -> str | None` is a sanity check; a failing check is
    treated like a failed fetch."""
    LAST_GOOD.mkdir(parents=True, exist_ok=True)
    lg = LAST_GOOD / f"{name}.json"
    try:
        v = value(fetch(path))
        problem = check(v) if check else None
        if problem:
            raise RuntimeError(f"sanity check: {problem}")
        if save:
            lg.write_text(json.dumps({"saved_at": _now(), "value": v}, separators=(",", ":")))
        _health[name] = {"status": "ok", "fetched_at": _now(), "error": None, "path": path}
        return v
    except Exception as e:  # noqa: BLE001 - any failure falls back
        if lg.exists():
            saved = json.loads(lg.read_text())
            _health[name] = {"status": "stale", "fetched_at": saved["saved_at"], "error": str(e)[:200], "path": path}
            return saved["value"]
        _health[name] = {"status": "failed", "fetched_at": None, "error": str(e)[:200], "path": path}
        raise


def clean(x: Any) -> Any:
    """JSON-safe: NaN/inf become null (the app's JSON.parse rejects NaN)."""
    import math

    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return clean(x.item())
        except (ValueError, AttributeError):
            return x
    return x


def note_health(name: str, status: str, error: str | None = None, **extra: Any) -> None:
    _health[name] = {"status": status, "fetched_at": _now() if status == "ok" else None, "error": error, **extra}


def health() -> dict[str, dict[str, Any]]:
    return dict(_health)


def write_health() -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_PATH.write_text(json.dumps(_health, indent=2))


# --------------------------------------------------------------- the feeds

def constraints(tour: int = TOUR) -> dict:
    return core(f"constraints_{tour}", f"constraints/constraints_{tour}.json",
                lambda v: None if v.get("matchdayId") else "no matchdayId")


def fixtures(tour: int = TOUR) -> list[dict]:
    return core(f"fixtures_{tour}", f"fixtures/fixtures_{tour}_en.json",
                lambda v: None if isinstance(v, list) and v and v[0].get("match") else "no matchdays")


def players(tour: int, md: int) -> list[dict]:
    def check(v):
        n = len(v.get("playerList") or [])
        return None if n >= 300 else f"only {n} players"
    # no JSON last-good copy (1.5 MB); data/ucl/history/players_{tour}.parquet is the fallback
    return core(f"players_{tour}", f"players/players_{tour}_en_{md}.json", check, save=False)["playerList"]


def popupstats(tour: int, pid: int, refresh: bool = False) -> dict | None:
    """One player's per-match stats, cached in data/ucl/raw/ (git-ignored)."""
    f = RAW / str(tour) / f"popup_{pid}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    try:
        v = value(fetch(f"popupstats/popupstats_{tour}_{pid}.json"))
    except RuntimeError:
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(v, separators=(",", ":")))
    return v


def cet_to_utc(s: str) -> datetime:
    """UEFA's feed times are Central European (CET/CEST): '11/24/2026 18:45:00'
    or '11/24/26 06:45:00 PM'."""
    from zoneinfo import ZoneInfo

    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p"):
        try:
            t = datetime.strptime(s.strip(), fmt)
            return t.replace(tzinfo=ZoneInfo("Europe/Paris")).astimezone(timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unknown time format: {s!r}")
