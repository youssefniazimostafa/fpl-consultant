"""Team strength from ClubElo — builds ``data/tables/elo.parquet``
(team, date, Elo). Elo moves after every match, so it reacts faster than
FPL's fixture difficulty ratings, which barely change all season.

ClubElo's documented CSV API (``api.clubelo.com/<date>``, ``.../<Club>``) is
currently dead: every ratings request answers HTTP 502, and
``api.clubelo.com/Fixtures`` answers "Fixtures API deactivated". This looks
like a permanent change (ClubElo moved behind a login), not a transient
outage. The ``clubelo.com`` *website* still publishes ratings, though: each
club's page (e.g. https://clubelo.com/Arsenal) embeds its full Elo history
as a Vega-Lite dataset inside a ``<script>`` tag. This module reads that
embedded JSON instead of the dead CSV endpoints.

This has broken silently once already, so the result is run through
``fpl.source_health``: it must cover all 20 current PL teams and carry
plausible Elo values.

When ClubElo is unavailable (it now sits behind a Cloudflare bot
challenge that a plain script can't pass -- and this module deliberately
doesn't try to bypass it), the fetch fails fast and a **derived** table
is used instead:

- seed: each current PL team's latest rating in the last real, sanity-
  checked ClubElo table (``data/last_good/elo.parquet``); the seed date is
  the latest rating date in that table (not when the file was saved -- a
  table saved today can still hold ratings from days ago).
- updates: every finished FPL Premier League fixture with a kickoff date
  strictly after the seed date, applied in kickoff order with a standard
  ClubElo-style update (constants below).

It's recomputed from the seed every run and never written back to
``data/last_good/``, so only a real ClubElo fetch ever moves the seed and
no match can be counted twice. Known limitation: ClubElo also moves
ratings after cup and European matches, which FPL fixtures don't include,
so the derived ratings drift from ClubElo's the longer an outage lasts.
The output has exactly the same schema as a real fetch, so nothing
downstream changes; ``source_health`` reports ``status="fallback_derived"``
with the seed date, matches applied and any unseeded teams.

Run with `uv run python -m fpl.elo`.
"""

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fpl.source_health import (
    check_team_coverage,
    check_value_range,
    combine_checks,
    current_pl_team_ids,
    load_last_good,
    record,
    write_health_record,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "data" / "tables"
ALIASES_PATH = Path(__file__).resolve().parent / "team_aliases.csv"

BASE = "https://clubelo.com"

# --- fetch: fail fast -------------------------------------------------------
CLUBELO_TIMEOUT_S = 10   # per request
CLUBELO_RETRIES = 1      # one retry, only for 5xx / network errors (never for 4xx or a challenge)
CLUBELO_RETRY_PAUSE_S = 2.0
# Markers of a Cloudflare bot-challenge page (served with HTTP 403/503).
CHALLENGE_MARKERS = ("Just a moment...", "challenge-platform", "cf_chl_opt")

# --- derived fallback: ClubElo-style update ----------------------------------
# For each match: expected home result  We = 1 / (1 + 10^(-(R_home - R_away + HOME_ADVANTAGE) / 400))
#                 rating change         d  = K_FACTOR * G * (result - We)   (home +d, away -d)
# with result = 1 / 0.5 / 0 for a home win / draw / loss, and G the
# goal-difference multiplier below (1 for a draw or one-goal margin,
# growing slowly for bigger wins, as in ClubElo/World Football Elo).
# Validated against ClubElo's own history (fpl/elo_backtest.py): the
# textbook K=20 overshot ClubElo -- with 2-3 gameweeks applied it was
# further from ClubElo's real ratings than not updating at all -- while
# K=10-12 was best at every seed point (1-6 gameweeks back). Home advantage
# barely mattered (0-100 all within ~0.5 Elo mean error); 60 is kept as a
# typical value.
K_FACTOR = 10.0
HOME_ADVANTAGE = 60.0


def goal_difference_multiplier(goal_diff: int) -> float:
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8

# Elo systems like ClubElo's have historically kept Premier League clubs
# somewhere around 1300-2100; padded generously to 1000-2300 so this only
# flags genuinely broken data (e.g. a parsing bug returning zeros or a
# constant placeholder), not a real top/bottom-of-division swing.
PLAUSIBLE_ELO_RANGE = (1000.0, 2300.0)


def _sanity_check(elo: pd.DataFrame) -> str | None:
    """Elo must cover every current PL team and every value must be in a
    plausible range -- a failed check is reported exactly like a failed
    fetch, so the site shows a warning either way."""
    return combine_checks(
        check_team_coverage(elo, "team_id"),
        check_value_range(elo, "elo", *PLAUSIBLE_ELO_RANGE),
    )

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fpl-consultant (ymostafa1@babson.edu)"})


def load_aliases() -> pd.DataFrame:
    return pd.read_csv(ALIASES_PATH)


def extract_vega_dataset(html: str) -> list[dict]:
    """Pull the Elo history out of a club page's embedded `var vegaJson = {...}`."""
    marker = "var vegaJson = "
    start = html.find(marker)
    if start == -1:
        raise ValueError("no embedded vegaJson block found (clubelo.com's layout may have changed)")
    start += len(marker)
    obj, _ = json.JSONDecoder().raw_decode(html, start)
    datasets = obj.get("datasets", {})
    if not datasets:
        raise ValueError("vegaJson has no datasets")
    # The club-history chart is the dataset with Date/Elo rows (the ranking
    # chart's dataset, if present, carries Name/Federation instead).
    for rows in datasets.values():
        if rows and "Date" in rows[0] and "Elo" in rows[0]:
            return rows
    raise ValueError("no Date/Elo dataset found among vegaJson's datasets")


class ClubEloUnavailable(Exception):
    """A clean, expected ClubElo failure (challenge page, HTTP error,
    timeout, or a page without an Elo dataset) -- triggers the fallback."""


def fetch_club_html(day_dir: Path, slug: str) -> str:
    """One club page, fail-fast: short timeout, at most one retry (5xx and
    network errors only), and a Cloudflare challenge, any 4xx, or a page
    with no Elo dataset is an immediate ClubEloUnavailable. Only pages that
    actually contain an Elo dataset are cached (per day)."""
    cache_file = day_dir / f"clubelo_{slug}.html"
    if cache_file.exists():
        html = cache_file.read_text()
        try:
            extract_vega_dataset(html)
            return html
        except ValueError:
            cache_file.unlink()  # stale/empty cache entry from an older run -- refetch

    url = f"{BASE}/{slug}"
    last_error = "no attempt made"
    for attempt in range(1 + CLUBELO_RETRIES):
        if attempt:
            time.sleep(CLUBELO_RETRY_PAUSE_S)
        try:
            r = SESSION.get(url, timeout=CLUBELO_TIMEOUT_S)
        except requests.RequestException as exc:
            last_error = f"{url}: network error ({type(exc).__name__})"
            continue
        if r.headers.get("cf-mitigated") or any(m in r.text[:5000] for m in CHALLENGE_MARKERS):
            raise ClubEloUnavailable(f"{url}: Cloudflare bot challenge (HTTP {r.status_code})")
        if 400 <= r.status_code < 500:
            raise ClubEloUnavailable(f"{url}: HTTP {r.status_code}")
        if r.status_code >= 500:
            last_error = f"{url}: HTTP {r.status_code}"
            continue
        try:
            extract_vega_dataset(r.text)
        except ValueError as exc:
            raise ClubEloUnavailable(f"{url}: {exc}") from exc
        cache_file.write_text(r.text)
        time.sleep(0.2)
        return r.text
    raise ClubEloUnavailable(last_error)


def fetch_club_elo(day_dir: Path, slug: str) -> pd.DataFrame:
    history = pd.DataFrame(extract_vega_dataset(fetch_club_html(day_dir, slug)))
    history["date"] = pd.to_datetime(history["Date"]).dt.date
    return history[["date", "Elo", "Golo"]].rename(columns={"Elo": "elo", "Golo": "golo"})


def build_elo_table(day_dir: Path) -> tuple[pd.DataFrame, str | None]:
    """(full ClubElo history for every aliased club, None) on success, or
    (empty table, reason) on the first club that fails -- no point asking
    for the other 19 pages once ClubElo is clearly unavailable."""
    aliases = load_aliases()
    tables = []
    for _, row in aliases.iterrows():
        try:
            history = fetch_club_elo(day_dir, row["clubelo_slug"])
        except ClubEloUnavailable as exc:
            return pd.DataFrame(), str(exc)
        history["team_id"] = row["fpl_team_id"]
        history["team"] = row["fpl_short_name"]
        tables.append(history)
    if not tables:
        return pd.DataFrame(), "team_aliases.csv lists no clubs"
    return pd.concat(tables, ignore_index=True)[["team_id", "team", "date", "elo", "golo"]], None


# ---------------------------------------------------------------------------
# derived fallback: last real ClubElo ratings + FPL results since
# ---------------------------------------------------------------------------


def latest_ratings(history: pd.DataFrame, as_of: date | None = None) -> pd.Series:
    """Each team's most recent rating (on or before ``as_of``), by team_id."""
    if as_of is not None:
        history = history[history["date"] <= as_of]
    return history.sort_values("date").groupby("team_id")["elo"].last()


def finished_results(fixtures: pd.DataFrame, after: date, until: date | None = None) -> pd.DataFrame:
    """Finished fixtures with a kickoff date strictly after ``after`` (and on
    or before ``until``, if given), in kickoff order."""
    f = fixtures[fixtures["finished"].astype(bool)].dropna(subset=["team_h_score", "team_a_score"]).copy()
    f["kickoff"] = pd.to_datetime(f["kickoff_time"], utc=True)
    f["kickoff_date"] = f["kickoff"].dt.date
    f = f[f["kickoff_date"] > after]
    if until is not None:
        f = f[f["kickoff_date"] <= until]
    return f.sort_values(["kickoff", "id"])


def apply_results(ratings: dict[int, float], results: pd.DataFrame) -> list[dict]:
    """Updates ``ratings`` in place, match by match; returns one
    ``{team_id, date, elo}`` row per team per match (its post-match rating)."""
    rows = []
    for m in results.itertuples():
        h, a = int(m.team_h), int(m.team_a)
        if h not in ratings or a not in ratings:
            continue  # not a current PL team (shouldn't happen for PL fixtures)
        gd = int(m.team_h_score) - int(m.team_a_score)
        result = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        expected = 1.0 / (1.0 + 10 ** (-(ratings[h] - ratings[a] + HOME_ADVANTAGE) / 400.0))
        delta = K_FACTOR * goal_difference_multiplier(gd) * (result - expected)
        ratings[h] += delta
        ratings[a] -= delta
        rows.append({"team_id": h, "date": m.kickoff_date, "elo": ratings[h]})
        rows.append({"team_id": a, "date": m.kickoff_date, "elo": ratings[a]})
    return rows


def derive_from_seed(seed: pd.DataFrame, fixtures: pd.DataFrame, team_ids: set[int],
                     names: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    """The seed's history for the current PL teams, plus one derived row per
    team per FPL result since the seed date -- same columns as a real fetch
    (``golo`` is NaN on derived rows: it's ClubElo's own field, not ours)."""
    seed = seed[seed["team_id"].isin(team_ids)].copy()
    seed_date = max(seed["date"])
    start = latest_ratings(seed)
    unseeded = sorted(team_ids - set(start.index))
    ratings = {int(t): float(r) for t, r in start.items()}
    floor = float(start.min())
    extra_rows = []
    for t in unseeded:
        ratings[t] = floor
        extra_rows.append({"team_id": t, "date": seed_date, "elo": floor})

    results = finished_results(fixtures, after=seed_date)
    extra_rows += apply_results(ratings, results)

    derived = pd.DataFrame(extra_rows, columns=["team_id", "date", "elo"])
    derived["golo"] = np.nan
    derived["team"] = derived["team_id"].map(names)
    out = pd.concat([seed, derived[seed.columns]], ignore_index=True)
    out["team_id"] = out["team_id"].astype(seed["team_id"].dtype)
    info = {
        "seed_date": seed_date.isoformat(),
        "matches_applied": int(len(results[results["team_h"].isin(ratings) & results["team_a"].isin(ratings)])),
        "unseeded_teams": [names.get(t, str(t)) for t in unseeded],
    }
    return out, info


def _team_names() -> dict[int, str]:
    names = dict(zip(load_aliases()["fpl_team_id"].astype(int), load_aliases()["fpl_short_name"]))
    teams_path = TABLES_DIR / "teams.parquet"
    if teams_path.exists():  # a newly promoted team not yet in team_aliases.csv
        teams = pd.read_parquet(teams_path, columns=["id", "short_name"])
        for t, n in zip(teams["id"].astype(int), teams["short_name"]):
            names.setdefault(t, n)
    return names


def build_derived_fallback(fetch_error: str) -> tuple[pd.DataFrame, dict] | None:
    """(table, health record) from the last-good seed + FPL results, or
    None if there's no seed / no fixtures / the result fails its sanity check."""
    seed, saved_at = load_last_good("elo")
    fixtures_path = TABLES_DIR / "fixtures.parquet"
    team_ids = current_pl_team_ids()
    if seed.empty or not fixtures_path.exists() or not team_ids:
        return None
    derived, info = derive_from_seed(seed, pd.read_parquet(fixtures_path), team_ids, _team_names())
    if _sanity_check(derived) is not None:
        return None
    return derived, {
        "source": "elo",
        "status": "fallback_derived",
        "last_success_at": saved_at,  # when the seed (a real ClubElo fetch) was saved
        "row_count": int(len(derived)),
        "error": fetch_error,
        **info,
    }


def main() -> None:
    day_dir = RAW_DIR / date.today().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching ClubElo ratings (per-club page scrape; the CSV API is dead)...")
    elo, fetch_error = build_elo_table(day_dir)
    if fetch_error:
        print(f"  ClubElo unavailable: {fetch_error}")

    # Only a real fetch goes through record(), which is the only thing that
    # ever saves data/last_good/elo.parquet (i.e. moves the seed).
    used, health = record("elo", elo, _sanity_check, empty_message=fetch_error)
    if health["status"] != "ok":
        fallback = build_derived_fallback(health["error"])
        if fallback is not None:
            used, health = fallback
        else:
            health["error"] += "; no derived fallback (no ClubElo seed in data/last_good/, or no FPL fixtures)"
    write_health_record(health)

    if health["status"] == "ok":
        print(f"  health: ok ({health['row_count']} rows)")
    elif health["status"] == "fallback_derived":
        print(f"  health: FALLBACK_DERIVED -- ClubElo ratings from {health['seed_date']} + "
              f"{health['matches_applied']} FPL result(s) since"
              + (f"; unseeded (started at lowest seeded rating): {health['unseeded_teams']}"
                 if health["unseeded_teams"] else ""))
    elif health["status"] == "stale":
        print(f"  health: STALE -- {health['error']}; falling back to the last good table "
              f"(saved {health['last_success_at']})")
    else:
        print(f"  health: FAILED -- {health['error']}; no last-good table to fall back to either")

    if used.empty:
        print("ClubElo was unavailable for every club, with no usable fallback; elo.parquet was not written.")
        return

    path = TABLES_DIR / "elo.parquet"
    used.to_parquet(path, index=False)
    print(f"  wrote {path} ({len(used)} rows, {used['team'].nunique()} teams)")


if __name__ == "__main__":
    main()
