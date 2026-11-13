"""Shared source-health tracking and last-good-table fallback for the
external data sources this project depends on: the FPL API itself, The
Odds API, ClubElo, and Understat.

fpl.elo scrapes clubelo.com pages and fpl.understat calls an undocumented
understat.com endpoint -- both have broken once already, silently, with
the pipeline carrying on but fixture difficulty quietly losing that input.
This module gives every source a visible status (``ok`` / ``stale`` /
``failed``; fpl.elo adds ``fallback_derived`` for ratings derived from its
last-good seed plus FPL results), a last-successful-fetch time, a row count, and a short error
message, and -- when a fetch fails or fails a sanity check -- a small,
committed "last good" table to fall back to instead of nothing.

Persistence choice: the last good table for each source is committed to
``data/last_good/<source>.parquet`` (+ a tiny ``.json`` sidecar with the
timestamp it was saved), overwritten in place every time a fetch
succeeds, rather than using GitHub Actions' ``actions/cache``. Reasons:

- ``actions/cache`` entries are evicted (7 days unused, or a repo-wide
  10GB cap) and aren't readable locally or from a PR run the same way a
  committed file is -- a silent cache miss the one time a real outage
  hits would defeat the entire point of a fallback.
- These tables are tiny (each is a single current snapshot, not a growing
  log like the old prediction-log problem Fix 2 addressed -- elo.parquet
  is ~85KB, team_xg.parquet ~19KB, player_xg.parquet ~42KB,
  odds_goals.parquet a few KB), so committing them costs negligible repo
  size and, as a bonus, gives a visible git history of each source's
  health over time.

Sanity checks live here too: Elo and team-level xG must cover all 20 of
the *current* season's Premier League teams (from ``teams.parquet``,
which the FPL API itself -- the one source that can't silently drift --
always has right), and values must fall in a plausible range. A table
that fails a sanity check is treated exactly like a fetch that returned
nothing: it's discarded and the last-good fallback (if any) is used
instead, with ``status="stale"``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "data" / "tables"
LAST_GOOD_DIR = ROOT / "data" / "last_good"
HEALTH_PATH = TABLES_DIR / "source_health.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_pl_team_ids() -> set[int]:
    """The 20 team ids the FPL API itself currently lists -- the one source
    that can't silently go stale, so it's the ground truth other sources'
    coverage is checked against."""
    path = TABLES_DIR / "teams.parquet"
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["id"])["id"].tolist())


def check_team_coverage(df: pd.DataFrame, team_id_col: str) -> str | None:
    """None if `df` has at least one row for every current PL team id in
    `team_id_col`, else a short message naming what's missing."""
    current = current_pl_team_ids()
    if not current:
        return None  # teams.parquet not available yet -- nothing to check against
    covered = set(pd.to_numeric(df[team_id_col], errors="coerce").dropna().astype(int))
    missing = current - covered
    if missing:
        return f"missing {len(missing)}/{len(current)} current PL team(s): team_id {sorted(missing)}"
    return None


def check_value_range(df: pd.DataFrame, col: str, lo: float, hi: float) -> str | None:
    """None if every value in `col` is within [lo, hi], else a short message."""
    values = pd.to_numeric(df[col], errors="coerce")
    if values.isna().all():
        return f"{col} is entirely non-numeric/missing"
    out_of_range = values[(values < lo) | (values > hi)]
    if not out_of_range.empty:
        return (
            f"{len(out_of_range)} {col} value(s) outside the plausible range "
            f"[{lo}, {hi}] (e.g. {out_of_range.iloc[0]:.2f})"
        )
    return None


def combine_checks(*errors: str | None) -> str | None:
    """The first non-None error, or None if every check passed."""
    for error in errors:
        if error is not None:
            return error
    return None


# ---------------------------------------------------------------------------
# last-good persistence
# ---------------------------------------------------------------------------


def save_last_good(name: str, df: pd.DataFrame) -> None:
    LAST_GOOD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LAST_GOOD_DIR / f"{name}.parquet", index=False)
    meta_path = LAST_GOOD_DIR / f"{name}.json"
    meta_path.write_text(json.dumps({"saved_at": _now(), "row_count": len(df)}, indent=2))


def load_last_good(name: str) -> tuple[pd.DataFrame, str | None]:
    """The last table that passed its sanity check for `name`, and when it
    was saved -- an empty DataFrame and None if there's never been one."""
    path = LAST_GOOD_DIR / f"{name}.parquet"
    meta_path = LAST_GOOD_DIR / f"{name}.json"
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_parquet(path)
    saved_at = None
    if meta_path.exists():
        saved_at = json.loads(meta_path.read_text()).get("saved_at")
    return df, saved_at


# ---------------------------------------------------------------------------
# health record + fallback decision
# ---------------------------------------------------------------------------


def record(
    name: str,
    df: pd.DataFrame,
    sanity_check: Callable[[pd.DataFrame], str | None] | None = None,
    empty_message: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Checks a freshly-fetched `df` for `name` (empty-check first, then
    `sanity_check` if given) and decides what table to actually use:

    - sanity check passes -> save it as the new last-good table, status "ok".
    - it fails but a last-good table exists -> fall back to that, status "stale".
    - it fails and there's no last-good table either -> empty table, status "failed".

    `empty_message` overrides the default "fetch returned no rows" error text
    for the empty-df case -- e.g. distinguishing "ODDS_API_KEY not set" (a
    config choice) from a real fetch failure, even though both leave `df`
    empty.

    Returns (the table to use downstream, this source's health record).
    """
    error: str | None
    if df is None or df.empty:
        error = empty_message or "fetch returned no rows"
    elif sanity_check is not None:
        error = sanity_check(df)
    else:
        error = None

    if error is None:
        save_last_good(name, df)
        return df, {
            "source": name,
            "status": "ok",
            "last_success_at": _now(),
            "row_count": int(len(df)),
            "error": None,
        }

    fallback, saved_at = load_last_good(name)
    if fallback.empty:
        return pd.DataFrame(), {
            "source": name,
            "status": "failed",
            "last_success_at": None,
            "row_count": 0,
            "error": error,
        }
    return fallback, {
        "source": name,
        "status": "stale",
        "last_success_at": saved_at,
        "row_count": int(len(fallback)),
        "error": error,
    }


def write_health_record(health: dict[str, Any]) -> None:
    """Merges one source's health record into the shared
    data/tables/source_health.json, preserving the others -- each source
    module runs in its own process within the pipeline, so this is a
    read-modify-write rather than each module owning the whole file."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if HEALTH_PATH.exists():
        existing = {r["source"]: r for r in json.loads(HEALTH_PATH.read_text())}
    existing[health["source"]] = health
    HEALTH_PATH.write_text(json.dumps(list(existing.values()), indent=2))
